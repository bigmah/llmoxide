# Plan: private file reading

Status: planned, not built (2026-09-24).

Let the chat window's model read files in a folder you choose, without
giving up the app's promise: after New chat or exit, nothing on the machine
records what was read, or that anything was read at all. Tools that write, run
commands or reach the network are a later, separate step. They can never be
private, so they must visibly take the app out of private mode (see
[Later: tools that are not private](#later-tools-that-are-not-private)).

## What "private" means here

This is the same threat model as [privacy.md](privacy.md). The adversary looks
at the machine after the fact: files, logs, databases, swap, snapshots taken
after the wipe. This includes root reading ordinary logs and the disk.

Out of scope, as for the rest of the app:

- **Anyone watching live.** Root, `lsof`, dtrace, or an Endpoint Security
  client (EDR, MDM agents) sees each `open` as it happens. After the wipe
  there is nothing left of it.
- **Forensic recovery of old APFS metadata blocks.** APFS is copy-on-write, so
  the briefly-updated access time is written somewhere before it is put back.
  Carving free space, or a local snapshot taken during that window of a few
  milliseconds, could find it. See [open questions](#open-questions).
- **The OS noticing that the app ran**, and the model file's own access time.
  Both are already listed in privacy.md.

## What was measured

Measured on this machine (Darwin 25.6, APFS Data volume) on 2026-09-24, in a
scratch folder. These results are the reason this plan exists; `read_check`
(below) makes them repeatable.

| Action | Result |
|---|---|
| `cat` a file | The file's access time was updated. Only the first read after a change updates it (relatime-like), so the trace is "read since last modified". |
| `ls` the folder | The folder's access time was updated. |
| Read, then `utimensat(atime = saved, mtime = UTIME_OMIT)` | Access time was back to the saved value to the nanosecond. mtime and ctime were unchanged. |
| FSEvents (file-level stream) during a plain read, a read with restore, and a listing | No events. A control that appended to the file produced `ItemModified`, so the watcher worked. Nothing reaches `/.fseventsd`. |

Conclusion: a read whose access times are put back leaves no timestamp and no
FSEvents record. The other known traces come from things around the read,
not the read itself: the folder picker, shell history, iCloud downloads,
network drives and TCC. The rules below avoid those.

## Design

### Tools

Private mode offers only read tools, all confined to one folder:

| Tool | Arguments | Returns |
|---|---|---|
| `list_dir` | `path` (relative to the folder) | names, kinds and sizes, one level deep |
| `read_file` | `path`, optional `offset` / `max_bytes` | UTF-8 text; refuses anything that looks binary |
| `search` | `pattern` (literal), optional `path` | matching lines with `file:line`, capped |

`search` is built on the same read path, so every file it opens gets the same
treatment. It is not a shelled-out `grep`: a child process would sit outside
the zeroing allocator and read files itself.

### Choosing the folder

- **You type or paste the path into the app.** Do not use the system
  folder picker: NSOpenPanel saves the last folder as a bookmark in a
  ViewBridge plist (privacy.md already documents this for the model picker).
  Do not use a command-line argument either, since that lands in shell history.
- The path is resolved with `realpath` once, and the grant lives only in
  memory. New chat and exit drop it. No "recent folders" list.
- Show the grant in the header next to the model ("Reading `~/proj/foo`"),
  with a button to revoke it.

### Rules checked before every file or folder is read

Any rule that fails means a refusal, reported to the model as a tool error,
not a read:

1. **Stays inside the folder.** Open each path component with `openat` from
   the root descriptor, using `O_NOFOLLOW`. Refuse a symlink that points out
   of the folder, and `..` escapes.
2. **On a local drive.** `fstatfs` must report `MNT_LOCAL`. On SMB/NFS and
   similar, the server sees every read.
3. **Not an iCloud placeholder.** Refuse if `st_flags & SF_DATALESS` is set.
   Reading one would download it: a network request plus cloud-sync logs.
4. **Owned by you.** Only the owner, or root, can set the access time back.
   Otherwise refuse, rather than leave a trace we can't undo.
5. **A regular file or folder, within size limits.** Sizes per read and per
   turn are capped to fit the context.
6. **Outside macOS-protected folders** (Documents, Desktop, Downloads, iCloud
   Drive, removable volumes, and others), unless you tick an explicit "macOS
   will record that this app was allowed into Documents" box. The TCC grant is
   recorded per app, not per file, but it is a lasting record, so it is your
   decision to make.

### How one file is read

```
fstatat(dir, name, O_NOFOLLOW)           → save atime (ns) of the file
open via openat(O_RDONLY | O_NOFOLLOW)
fcntl(F_NOCACHE, 1)                      → keep it out of the unified buffer cache
fstat, run the rules above
read into a SecretVec                    → mlocked, zeroed on drop
close
utimensat(file, [saved atime, UTIME_OMIT])
fstatat again                            → confirm atime == saved, else report it
```

Folders get the same save-and-restore around each `getdirentries` pass. The
restores run innermost first (the file, then its parent folders) in a
`Drop` guard, so an error or cancellation still puts every time back.

Race: if another process reads the same file in that window, our restore
also hides its access-time update. That is acceptable, and noted in
privacy.md.

### The tool loop in the engine

`crates/app/src/engine.rs` today sends `Request { tools: Vec::new(), .. }` and
returns `o.completion.content`. The chat crate already parses tool calls for
both formats (`Completion.tool_calls`; Qwen's `<tool_call>` in `qwen.rs`,
Gemma's in `lib.rs`), so the loop is:

1. `Cmd::Gen` carries the tool definitions whenever a folder is granted.
2. When a completion has `tool_calls`, run them on the engine thread. They
   are quick, bounded reads, and running them there keeps them behind the
   same `closing` flag. Stream a `Reply::ToolCall { name, args }` and then a
   `Reply::ToolResult { summary }` to the UI.
3. Append the assistant turn and a `role: "tool"` message with the result,
   then generate again. Stop after N steps (default 8), or when Stop is
   pressed: dropping the receiver ends the loop, the same as it cancels a
   reply today.

Where the file text lives: the `SecretVec` holds the bytes read, and the
tool message's `String` sits in the zeroing heap. That copy is not mlocked,
the same gap privacy.md already lists for the UI's history. The tokens go into
the KV cache, which `session.wipe()` already clears.

### Chat window

- A **tool-call row** under the reply: "Read `src/main.rs` · 4.2 KB", or
  "Listed `crates/` · 14 entries", collapsed by default. Refusals are shown
  too, with the reason ("iCloud placeholder, not downloaded").
- The **Private badge stays**, because nothing leaves. The folder grant
  shows next to the model name.
- **New chat** already wipes everything the tools produced (conversation,
  KV cache, GPU buffers) and now also drops the grant.
- The welcome text states the scope plainly: "The model can read files in
  this folder. Reading leaves no trace on disk; nothing is written or sent."

## Verification

- **`read_check` binary**, in the same spirit as `wipe_check`. It builds a
  fixture folder, ages all access times, runs the real tool code over it
  (list, read, search), then asserts:
  - every access time, mtime and ctime is unchanged, to the nanosecond, for
    the files and the folders;
  - a file-level FSEvents stream on the fixture saw nothing, with a
    positive-control write at the end that it must see;
  - every rule refuses its case: a symlink out of the folder, a file you don't
    own (no `sudo` needed: grant a root-owned folder such as `/usr/share/dict`), a
    dataless file where one can be made, a non-local mount if one is present.

  Run it in CI on macOS, and locally after every OS update. If a future macOS
  starts logging access-time restores, this check is what catches it.
- **Unit tests** for path confinement and the rule checks.
- **A GUI check** using the guarded CGEvent recipe in the app notes: grant a
  fixture folder, ask the model to summarise a file, then confirm
  `read_check`'s timestamp assertions still hold after New chat.

## Later: tools that are not private

This is not part of this plan, but the shape is reserved now so nothing has
to be redone later:

- Every tool declares an `Exposure`: `None` (the read tools),
  `Local(what persists)` for writes and edits, or `Remote(host, what is sent)`
  for web fetch and MCP.
- Enabling any tool whose exposure isn't `None` switches the badge from
  "Private" to "Not private · N exits", and clicking it lists them. **Read
  tools plus any exit is treated as not private.** A file the model reads can
  prompt-inject it into leaking the conversation through the exit.
- Each call to an exit tool gets an approval card showing the exact bytes and
  destination. "Allow for this chat" can be offered, "allow forever" never,
  because saved grants are disk writes.
- An in-memory "what left" list records each side effect in the session. New
  chat says what it could not erase.

## Milestones

1. `crates/app/src/read.rs`: the confined read path, the rules and the restore
   guard, plus unit tests.
2. `read_check`: the FSEvents and timestamp assertions, run against step 1.
3. The engine tool loop and the `Reply` variants, tested with Qwen 27B first,
   since E4B's tool calling is untested and likely weak.
4. The UI: typed folder grant, header indicator, tool-call rows, welcome text.
5. Docs: a "Reading files" section in privacy.md with the measured table and
   the out-of-scope list; a README bullet.

## Open questions

- **The access-time window versus Time Machine local snapshots.** Local
  snapshots are hourly while Time Machine is on. Should the tool check
  `tmutil` and warn, or pause reads while a snapshot is being taken? It is
  unlikely to matter, but it should be measured before being dismissed.
- **Files you don't own.** Refusing is the safe default. The alternative is a
  per-file "this read will be visible" confirmation.
- **Linux and Windows.** Unmeasured. Linux has `O_NOATIME` (owner only), which
  avoids the write entirely, but inotify reports reads (`IN_ACCESS`) to live
  watchers. On Windows, whether the last-access time updates depends on a
  volume setting. Until each is measured, private reading is macOS-only.
- **Unified log.** No read-related entries were seen, but that was not
  checked systematically. `read_check` could grep `log show --last 1m` for
  the fixture path.
