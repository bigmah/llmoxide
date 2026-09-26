//! Private file reading: the read-only tools the model gets once a folder is
//! granted. See `docs/private-reading-plan.md` for the measurements behind
//! this.
//!
//! Private means that once the chat is wiped, nothing on disk shows what was
//! read, or that anything was read at all. A read changes exactly one thing
//! on disk: the access time of the file or folder that was read. Opening,
//! `stat` and path lookups change nothing, and FSEvents reports none of it.
//! So every read here goes through [`Held`], which saves the access time
//! before the read and puts it back afterwards, to the nanosecond. The rules
//! in [`Grant::check`] refuse anything where that cannot work, or where the
//! read would be seen somewhere else: a network drive, an iCloud placeholder
//! (reading it downloads it), or a file someone else owns (only its owner can
//! set its access time).
//!
//! Every path is walked one component at a time with `openat` from the
//! granted folder's descriptor, with `O_NOFOLLOW`, so `..` and symlinks
//! cannot lead out of it. Symlinks are not followed at all, even ones that
//! point inside the folder.
//!
//! File contents are read into a [`SecretVec`] (locked, zeroed on drop), and
//! the text handed to the model is on the zeroing heap.
//!
//! macOS only: the rules and the restore were measured there. Elsewhere
//! [`Grant::open`] refuses.

use chat::Tool;
use serde_json::{json, Value};

/// Bytes `read_file` returns per call unless the model asks for fewer.
pub const READ_MAX: usize = 16 * 1024;
/// Bytes all tool results in one turn may add up to. The default context is
/// 16k tokens, about 64 KB of text; this leaves room for the conversation.
pub const TURN_BUDGET: usize = 48 * 1024;
const LIST_MAX: usize = 500;
const SEARCH_HITS: usize = 50;
const SEARCH_FILES: usize = 2000;
const SEARCH_FILE_MAX: u64 = 4 << 20;
const SEARCH_LINE_MAX: usize = 200;
/// Folders `search` does not descend into: build output and dependencies,
/// which are large and rarely what the question is about.
const SEARCH_SKIP_DIRS: &[&str] = &["target", "node_modules"];

/// The tool definitions sent with every turn while a folder is granted.
pub fn tools() -> Vec<Tool> {
    let defs = json!([
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List one folder, one level deep: names, kinds and sizes. Folders end in '/'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Folder, relative to the shared folder. \".\" for its top level." }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file. Returns at most max_bytes (default 16384) from offset; a longer file says where to continue.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File, relative to the shared folder." },
                        "offset": { "type": "integer", "description": "Byte offset to start at. Default 0." },
                        "max_bytes": { "type": "integer", "description": "Most bytes to return. Default and maximum 16384." }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Find lines containing a literal string (case-sensitive) in the text files under a folder. Returns file:line: text for each match, up to 50.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "The exact text to look for." },
                        "path": { "type": "string", "description": "Folder or file to search, relative to the shared folder. Default \".\"." }
                    },
                    "required": ["pattern"]
                }
            }
        }
    ]);
    serde_json::from_value(defs).expect("tool definitions")
}

/// The system message that goes with [`tools`].
pub fn system_prompt(g: &Grant) -> String {
    format!(
        "The user has shared the folder `{}` with you, read-only. Use the list_dir, \
         read_file and search tools to look at its files; paths are relative to that \
         folder. You cannot change files, run commands or reach the network. When a \
         question is about these files, look before answering.",
        g.display()
    )
}

/// What one tool call came to.
pub struct Outcome {
    /// For the model.
    pub text: String,
    /// One line for the chat window.
    pub summary: String,
    pub ok: bool,
}

/// Tracks how much one turn's tool results have added to the context.
pub struct Budget(usize);

impl Budget {
    pub fn new() -> Self {
        Self(TURN_BUDGET)
    }
}

/// Run the tool call `name(args)` against `g`.
pub fn call(g: &Grant, name: &str, args: &Value, budget: &mut Budget) -> Outcome {
    let s = |k: &str| args.get(k).and_then(Value::as_str);
    let n = |k: &str| args.get(k).and_then(Value::as_u64);
    let path = s("path").unwrap_or(".");
    let shown = if path.is_empty() { "." } else { path };
    if budget.0 == 0 {
        return refused(
            format!("Didn't run {name}: this turn's reading budget is used up"),
            "refused: this turn's reading budget is used up; answer with what you have".into(),
        );
    }
    let out = match name {
        "list_dir" => g.list_dir(path).map(|l| {
            let summary = format!("Listed `{shown}` · {}", count(l.total, "entry", "entries"));
            (l.text, summary)
        }),
        "read_file" => {
            let max = n("max_bytes").map_or(READ_MAX, |m| (m as usize).clamp(1, READ_MAX));
            let max = max.min(budget.0);
            g.read_file(path, n("offset").unwrap_or(0), max).map(|f| {
                let part = if f.partial { " (part)" } else { "" };
                let summary = format!("Read `{shown}` · {}{part}", size(f.shown as u64));
                (f.text, summary)
            })
        }
        "search" => match s("pattern").filter(|p| !p.is_empty()) {
            None => Err("`pattern` is required".to_string()),
            Some(pat) => g.search(pat, path).map(|r| {
                let summary = format!(
                    "Searched `{pat}` in `{shown}` · {}",
                    count(r.hits, "match", "matches")
                );
                (r.text, summary)
            }),
        },
        other => Err(format!("there is no tool called {other}")),
    };
    match out {
        Ok((mut text, summary)) => {
            if text.len() > budget.0 {
                let cut = floor_char(&text, budget.0);
                text.truncate(cut);
                text.push_str("\n[cut off: this turn's reading budget is used up]");
            }
            budget.0 = budget.0.saturating_sub(text.len());
            Outcome {
                text,
                summary,
                ok: true,
            }
        }
        Err(why) => refused(format!("Couldn't {} `{shown}`: {why}", verb(name)), format!("refused: {why}")),
    }
}

fn refused(summary: String, text: String) -> Outcome {
    Outcome {
        text,
        summary,
        ok: false,
    }
}

fn verb(tool: &str) -> &str {
    match tool {
        "list_dir" => "list",
        "read_file" => "read",
        "search" => "search",
        _ => "run",
    }
}

pub struct Listing {
    pub text: String,
    pub total: usize,
}

pub struct FileText {
    pub text: String,
    /// Bytes of the file the text covers.
    pub shown: usize,
    pub partial: bool,
}

pub struct Found {
    pub text: String,
    pub hits: usize,
}

fn count(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// `n` bytes, for people.
pub fn size(n: u64) -> String {
    match n {
        n if n < 1024 => format!("{n} B"),
        n if n < 1024 * 1024 => format!("{:.1} KB", n as f64 / 1024.0),
        n => format!("{:.1} MB", n as f64 / (1024.0 * 1024.0)),
    }
}

/// The largest char boundary in `s` at or below `i`.
fn floor_char(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Split a tool's `path` argument into components under the grant.
///
/// `rel` may also be absolute if it is inside `root`. `..` is refused
/// outright, rather than resolved: resolving it textually is only right when
/// nothing on the way is a symlink, and refusing is simpler than proving it.
fn components<'a>(root: &str, rel: &'a str) -> Result<Vec<&'a str>, String> {
    let rel = if rel.starts_with('/') {
        match rel.strip_prefix(root) {
            Some(r) if r.is_empty() || r.starts_with('/') => r,
            _ => return Err("that path is outside the shared folder".into()),
        }
    } else {
        rel
    };
    let mut out = Vec::new();
    for c in rel.split('/') {
        match c {
            "" | "." => {}
            ".." => return Err("`..` is not allowed; paths stay inside the shared folder".into()),
            c if c.contains('\0') => return Err("that path contains a NUL byte".into()),
            c => out.push(c),
        }
    }
    Ok(out)
}

/// Decode bytes read from `offset` as text, or say why they are not. Drops a
/// partial character at either end: the start may land mid-character when
/// continuing at an offset, and the end when the read was cut short.
fn decode(bytes: &[u8], cut_short: bool) -> Result<(&str, usize), &'static str> {
    if bytes.contains(&0) {
        return Err("it looks like a binary file, not text");
    }
    let skip = bytes.iter().take(3).take_while(|&&b| b & 0xC0 == 0x80).count();
    let body = &bytes[skip..];
    match std::str::from_utf8(body) {
        Ok(s) => Ok((s, skip)),
        // Incomplete at the very end, and only because the read stopped there.
        Err(e) if e.error_len().is_none() && cut_short => {
            Ok((std::str::from_utf8(&body[..e.valid_up_to()]).unwrap(), skip))
        }
        Err(_) => Err("it is not UTF-8 text"),
    }
}

#[cfg(target_os = "macos")]
pub use mac::Grant;

#[cfg(not(target_os = "macos"))]
pub use other::Grant;

#[cfg(not(target_os = "macos"))]
mod other {
    use super::*;

    /// Private reading was only measured on macOS, so it is not offered
    /// elsewhere. Uninhabited: no grant can exist.
    pub enum Grant {}

    impl Grant {
        pub fn open(_: &str, _: bool) -> Result<Grant, String> {
            Err("private file reading is macOS-only for now".into())
        }
        pub fn display(&self) -> String {
            match *self {}
        }
        pub fn list_dir(&self, _: &str) -> Result<Listing, String> {
            match *self {}
        }
        pub fn read_file(&self, _: &str, _: u64, _: usize) -> Result<FileText, String> {
            match *self {}
        }
        pub fn search(&self, _: &str, _: &str) -> Result<Found, String> {
            match *self {}
        }
    }
}

#[cfg(target_os = "macos")]
mod mac {
    use std::ffi::{CStr, CString};
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
    use std::path::PathBuf;

    use secret::SecretVec;

    use super::*;

    /// `st_flags`: the file's contents are not on this machine (an iCloud
    /// placeholder). Reading it fetches them. Not in `libc`.
    const SF_DATALESS: u32 = 0x4000_0000;

    /// A folder the model may read in. Lives only in memory: New chat and
    /// exit drop it, and nothing records it.
    pub struct Grant {
        root: OwnedFd,
        /// `realpath` of what was typed.
        path: String,
        home: Option<String>,
        allow_protected: bool,
    }

    /// An open file or folder whose access time is put back on drop, or
    /// earlier by [`Held::restore`], which also checks that it took.
    struct Held {
        fd: OwnedFd,
        atime: libc::timespec,
        restored: bool,
    }

    impl Held {
        fn new(fd: OwnedFd, st: &libc::stat) -> Self {
            Self {
                fd,
                atime: libc::timespec {
                    tv_sec: st.st_atime,
                    tv_nsec: st.st_atime_nsec,
                },
                restored: false,
            }
        }

        fn put_back(&self) -> bool {
            let times = [
                self.atime,
                libc::timespec {
                    tv_sec: 0,
                    tv_nsec: libc::UTIME_OMIT,
                },
            ];
            unsafe { libc::futimens(self.fd.as_raw_fd(), times.as_ptr()) == 0 }
        }

        /// Put the access time back and confirm it. Call once the reading is
        /// done; an error here means the read left a trace.
        fn restore(&mut self) -> Result<(), String> {
            self.restored = true;
            if !self.put_back() {
                return Err(format!(
                    "could not put its access time back ({})",
                    std::io::Error::last_os_error()
                ));
            }
            let st = fstat(self.fd.as_raw_fd())?;
            if (st.st_atime, st.st_atime_nsec) != (self.atime.tv_sec, self.atime.tv_nsec) {
                return Err("its access time did not go back to what it was".into());
            }
            Ok(())
        }
    }

    impl Drop for Held {
        fn drop(&mut self) {
            if !self.restored {
                self.put_back();
            }
        }
    }

    fn fstat(fd: RawFd) -> Result<libc::stat, String> {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut st) } != 0 {
            return Err(os_err("stat"));
        }
        Ok(st)
    }

    fn os_err(what: &str) -> String {
        let e = std::io::Error::last_os_error();
        match e.raw_os_error() {
            Some(libc::ENOENT) => "no such file or folder".into(),
            Some(libc::ENOTDIR) => "a part of that path is not a folder".into(),
            Some(libc::ELOOP) => "it is a symbolic link, and links are not followed".into(),
            Some(libc::EACCES) => "permission denied".into(),
            Some(libc::EPERM) => "macOS privacy protection denied access".into(),
            _ => format!("{what} failed: {e}"),
        }
    }

    fn is_dir(st: &libc::stat) -> bool {
        st.st_mode & libc::S_IFMT == libc::S_IFDIR
    }

    fn is_reg(st: &libc::stat) -> bool {
        st.st_mode & libc::S_IFMT == libc::S_IFREG
    }

    /// Folders macOS guards with TCC (or that hold other apps' private
    /// data). Reading in them makes macOS record, per app, that access was
    /// granted — a lasting record, so it takes an explicit opt-in.
    fn protected(home: Option<&str>) -> Vec<String> {
        let mut v = vec!["/Volumes".to_string()];
        if let Some(h) = home {
            for d in ["Desktop", "Documents", "Downloads", "Library", "Pictures", "Movies", "Music"] {
                v.push(format!("{h}/{d}"));
            }
        }
        v
    }

    fn under(path: &str, dir: &str) -> bool {
        path == dir || path.strip_prefix(dir).is_some_and(|r| r.starts_with('/'))
    }

    impl Grant {
        /// Grant the folder `input` (a typed path; `~` is expanded).
        /// Protected folders need `allow_protected`.
        pub fn open(input: &str, allow_protected: bool) -> Result<Grant, String> {
            let home = std::env::var("HOME")
                .ok()
                .and_then(|h| std::fs::canonicalize(h).ok())
                .map(|p| p.to_string_lossy().into_owned());
            let input = input.trim();
            if input.is_empty() {
                return Err("type the path of a folder".into());
            }
            let expanded: PathBuf = match (input.strip_prefix('~'), &home) {
                (Some(rest), Some(h)) if rest.is_empty() || rest.starts_with('/') => {
                    PathBuf::from(format!("{h}{rest}"))
                }
                _ => PathBuf::from(input),
            };
            if !expanded.is_absolute() {
                return Err("use a full path, such as ~/projects/foo".into());
            }
            // `realpath`: resolves symlinks with lstat/readlink, and does not
            // read any folder, so it changes no access time.
            let real = std::fs::canonicalize(&expanded)
                .map_err(|e| format!("{}: {e}", expanded.display()))?;
            let path = real.to_string_lossy().into_owned();
            if !allow_protected {
                if let Some(p) = protected(home.as_deref())
                    .iter()
                    .find(|p| under(&path, p))
                {
                    let (me, p) = (tilde(&path, home.as_deref()), tilde(p, home.as_deref()));
                    let what = if me == p {
                        format!("{me} is protected by macOS")
                    } else {
                        format!("{me} is inside {p}, which macOS protects")
                    };
                    return Err(format!(
                        "{what}. Tick \"Allow protected folders\" to grant it anyway"
                    ));
                }
            }
            let c = CString::new(path.as_bytes()).map_err(|_| "path contains a NUL byte")?;
            let fd = unsafe {
                libc::open(
                    c.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
                )
            };
            if fd < 0 {
                return Err(os_err("open"));
            }
            let root = unsafe { OwnedFd::from_raw_fd(fd) };
            let g = Grant {
                root,
                path,
                home,
                allow_protected,
            };
            let st = fstat(g.root.as_raw_fd())?;
            g.check(g.root.as_raw_fd(), &st)?;
            Ok(g)
        }

        /// The folder, with the home directory shown as `~`.
        pub fn display(&self) -> String {
            tilde(&self.path, self.home.as_deref())
        }

        /// The rules every file and folder must pass before it is read. A
        /// failure is a refusal, not a read.
        fn check(&self, fd: RawFd, st: &libc::stat) -> Result<(), String> {
            let mut fs: libc::statfs = unsafe { std::mem::zeroed() };
            if unsafe { libc::fstatfs(fd, &mut fs) } != 0 {
                return Err(os_err("statfs"));
            }
            if fs.f_flags & libc::MNT_LOCAL as u32 == 0 {
                let kind = unsafe { CStr::from_ptr(fs.f_fstypename.as_ptr()) };
                return Err(format!(
                    "it is on a network drive ({}), whose server sees every read",
                    kind.to_string_lossy()
                ));
            }
            if st.st_flags & SF_DATALESS != 0 {
                return Err("it is an iCloud placeholder; reading it would download it".into());
            }
            let me = unsafe { libc::geteuid() };
            if me != 0 && st.st_uid != me {
                return Err(
                    "it belongs to another user, so its access time could not be put back".into(),
                );
            }
            if !is_dir(st) && !is_reg(st) {
                return Err("it is not a regular file or folder".into());
            }
            Ok(())
        }

        /// Open `name` in `dir` without following a symlink, and without
        /// opening an iCloud placeholder (which may start its download).
        fn step(&self, dir: RawFd, name: &str, want_dir: bool, abs: &str) -> Result<(OwnedFd, libc::stat), String> {
            if !self.allow_protected {
                if let Some(p) = protected(self.home.as_deref()).iter().find(|p| under(abs, p)) {
                    return Err(format!("it is inside {}, which macOS protects", tilde(p, self.home.as_deref())));
                }
            }
            let c = CString::new(name).map_err(|_| "that path contains a NUL byte")?;
            let mut st: libc::stat = unsafe { std::mem::zeroed() };
            if unsafe { libc::fstatat(dir, c.as_ptr(), &mut st, libc::AT_SYMLINK_NOFOLLOW) } != 0 {
                return Err(os_err("stat"));
            }
            if st.st_mode & libc::S_IFMT == libc::S_IFLNK {
                return Err("it is a symbolic link, and links are not followed".into());
            }
            if st.st_flags & SF_DATALESS != 0 {
                return Err("it is an iCloud placeholder; reading it would download it".into());
            }
            let mut flags = libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK;
            if want_dir {
                flags |= libc::O_DIRECTORY;
            }
            let fd = unsafe { libc::openat(dir, c.as_ptr(), flags) };
            if fd < 0 {
                return Err(os_err("open"));
            }
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };
            let st = fstat(fd.as_raw_fd())?;
            Ok((fd, st))
        }

        /// Open `rel` under the grant and check it. Only the last component
        /// is checked against the rules: the folders on the way are opened
        /// but not read.
        fn open_rel(&self, rel: &str) -> Result<(OwnedFd, libc::stat, String), String> {
            let parts = components(&self.path, rel)?;
            let clean = parts.join("/");
            let mut cur: Option<OwnedFd> = None;
            let mut st = fstat(self.root.as_raw_fd())?;
            let mut abs = self.path.clone();
            for (i, name) in parts.iter().enumerate() {
                let dir = cur.as_ref().map_or(self.root.as_raw_fd(), |f| f.as_raw_fd());
                abs.push('/');
                abs.push_str(name);
                let (fd, s) = self.step(dir, name, i + 1 < parts.len(), &abs)?;
                cur = Some(fd);
                st = s;
            }
            let fd = match cur {
                Some(fd) => fd,
                None => self.root.try_clone().map_err(|e| e.to_string())?,
            };
            self.check(fd.as_raw_fd(), &st)?;
            Ok((fd, st, clean))
        }

        pub fn read_file(&self, rel: &str, offset: u64, max: usize) -> Result<FileText, String> {
            let (fd, st, _) = self.open_rel(rel)?;
            if is_dir(&st) {
                return Err("it is a folder; use list_dir".into());
            }
            let size = st.st_size.max(0) as u64;
            let start = offset.min(size);
            let want = (max as u64).min(size - start) as usize;
            let mut held = Held::new(fd, &st);
            let buf = read_at(&held.fd, start, want)?;
            held.restore()?;
            let end = start + buf.len() as u64;
            let (text, skip) = decode(&buf, end < size)?;
            let from = start + skip as u64;
            let to = from + text.len() as u64;
            let mut out = String::with_capacity(text.len() + 120);
            out.push_str(text);
            let partial = from > 0 || to < size;
            if to < size {
                out.push_str(&format!(
                    "\n[showing bytes {from}–{to} of {size}; call read_file with offset={to} for more]"
                ));
            } else if from > 0 {
                out.push_str(&format!("\n[showing bytes {from}–{to} of {size}; end of file]"));
            }
            if size == 0 {
                out.push_str("[empty file]");
            }
            Ok(FileText {
                shown: text.len(),
                text: out,
                partial,
            })
        }

        pub fn list_dir(&self, rel: &str) -> Result<Listing, String> {
            let (fd, st, _) = self.open_rel(rel)?;
            if !is_dir(&st) {
                return Err("it is a file; use read_file".into());
            }
            let mut held = Held::new(fd, &st);
            let names = read_names(&held.fd)?;
            held.restore()?;
            let total = names.len();
            let mut rows: Vec<(bool, String)> = Vec::new();
            for name in names.iter().take(LIST_MAX) {
                let row = match lstat_at(held.fd.as_raw_fd(), name) {
                    None => (false, format!("{name}  (unreadable)")),
                    Some(s) if is_dir(&s) => (true, format!("{name}/")),
                    Some(s) if s.st_mode & libc::S_IFMT == libc::S_IFLNK => {
                        (false, format!("{name}  (symbolic link, not followed)"))
                    }
                    Some(s) if s.st_flags & SF_DATALESS != 0 => {
                        (false, format!("{name}  (iCloud, not downloaded)"))
                    }
                    Some(s) => (false, format!("{name}  ({})", size(s.st_size.max(0) as u64))),
                };
                rows.push(row);
            }
            rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let mut text: String = rows.into_iter().map(|r| r.1 + "\n").collect();
            if total > LIST_MAX {
                text.push_str(&format!("[{} more not shown]\n", total - LIST_MAX));
            }
            if total == 0 {
                text.push_str("[empty folder]\n");
            }
            Ok(Listing { text, total })
        }

        pub fn search(&self, pattern: &str, rel: &str) -> Result<Found, String> {
            let (fd, st, clean) = self.open_rel(rel)?;
            let mut s = Search {
                pattern: pattern.as_bytes(),
                lines: Vec::new(),
                hits: 0,
                files: 0,
                skipped: 0,
            };
            let base = self.path.clone() + if clean.is_empty() { "" } else { "/" } + &clean;
            if is_dir(&st) {
                self.search_dir(fd, &st, &clean, &base, &mut s)?;
            } else {
                s.file(fd, &st, &clean)?;
            }
            let mut text = s.lines.join("\n");
            if s.lines.is_empty() {
                text.push_str("no matches");
            }
            if s.hits > SEARCH_HITS {
                text.push_str(&format!("\n[{} more matches not shown]", s.hits - SEARCH_HITS));
            }
            if s.files >= SEARCH_FILES {
                text.push_str(&format!("\n[stopped after {SEARCH_FILES} files; search a smaller folder]"));
            }
            if s.skipped > 0 {
                text.push_str(&format!(
                    "\n[{} files or folders skipped: binary, too large, or not readable privately]",
                    s.skipped
                ));
            }
            Ok(Found { text, hits: s.hits })
        }

        /// Depth first. Each folder is listed and its access time put back
        /// before any child is opened, so restores never interleave.
        fn search_dir(&self, fd: OwnedFd, st: &libc::stat, rel: &str, abs: &str, s: &mut Search) -> Result<(), String> {
            let mut held = Held::new(fd, st);
            let names = read_names(&held.fd)?;
            held.restore()?;
            for name in names {
                if s.files >= SEARCH_FILES || s.hits > SEARCH_HITS {
                    break;
                }
                if name.starts_with('.') {
                    continue;
                }
                let child_rel = if rel.is_empty() { name.clone() } else { format!("{rel}/{name}") };
                let child_abs = format!("{abs}/{name}");
                let Some(ls) = lstat_at(held.fd.as_raw_fd(), &name) else {
                    s.skipped += 1;
                    continue;
                };
                let dir = is_dir(&ls);
                if dir && SEARCH_SKIP_DIRS.contains(&name.as_str()) {
                    continue;
                }
                if !dir && !is_reg(&ls) {
                    continue; // symlinks and the like: not followed, not counted
                }
                let opened = self
                    .step(held.fd.as_raw_fd(), &name, dir, &child_abs)
                    .and_then(|(cfd, cst)| self.check(cfd.as_raw_fd(), &cst).map(|()| (cfd, cst)));
                let Ok((cfd, cst)) = opened else {
                    s.skipped += 1;
                    continue;
                };
                // A restore failure is a trace left behind: stop and say so
                // rather than keep reading.
                if dir {
                    self.search_dir(cfd, &cst, &child_rel, &child_abs, s)?;
                } else {
                    s.file(cfd, &cst, &child_rel)?;
                }
            }
            Ok(())
        }
    }

    struct Search<'a> {
        pattern: &'a [u8],
        lines: Vec<String>,
        hits: usize,
        files: usize,
        skipped: usize,
    }

    impl Search<'_> {
        fn file(&mut self, fd: OwnedFd, st: &libc::stat, rel: &str) -> Result<(), String> {
            self.files += 1;
            let size = st.st_size.max(0) as u64;
            if size > SEARCH_FILE_MAX {
                self.skipped += 1;
                return Ok(());
            }
            let mut held = Held::new(fd, st);
            let buf = read_at(&held.fd, 0, size as usize)?;
            held.restore().map_err(|e| format!("{rel}: {e}"))?;
            if buf.contains(&0) {
                self.skipped += 1;
                return Ok(());
            }
            for (i, line) in buf.split(|&b| b == b'\n').enumerate() {
                if !contains(line, self.pattern) {
                    continue;
                }
                self.hits += 1;
                if self.hits > SEARCH_HITS {
                    continue;
                }
                let text = String::from_utf8_lossy(line);
                let text = text.trim();
                let cut = floor_char(text, SEARCH_LINE_MAX);
                let more = if cut < text.len() { "…" } else { "" };
                self.lines.push(format!("{rel}:{}: {}{more}", i + 1, &text[..cut]));
            }
            Ok(())
        }
    }

    fn contains(hay: &[u8], needle: &[u8]) -> bool {
        hay.windows(needle.len()).any(|w| w == needle)
    }

    fn lstat_at(dir: RawFd, name: &str) -> Option<libc::stat> {
        let c = CString::new(name).ok()?;
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        (unsafe { libc::fstatat(dir, c.as_ptr(), &mut st, libc::AT_SYMLINK_NOFOLLOW) } == 0).then_some(st)
    }

    /// Read `len` bytes from `offset` into locked memory, bypassing the
    /// unified buffer cache.
    fn read_at(fd: &OwnedFd, offset: u64, len: usize) -> Result<SecretVec<u8>, String> {
        let fd = fd.as_raw_fd();
        unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
        let mut buf = SecretVec::new();
        buf.resize(len, 0u8);
        let mut got = 0;
        while got < len {
            let n = unsafe {
                libc::pread(fd, buf[got..].as_mut_ptr().cast(), len - got, (offset + got as u64) as libc::off_t)
            };
            match n {
                n if n < 0 => {
                    if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                        continue;
                    }
                    return Err(os_err("read"));
                }
                0 => break,
                n => got += n as usize,
            }
        }
        buf.resize(got, 0);
        Ok(buf)
    }

    /// The names in a folder, sorted, without `.` and `..`. Reading a folder
    /// is what updates its access time; the caller's [`Held`] puts it back.
    fn read_names(fd: &OwnedFd) -> Result<Vec<String>, String> {
        // `fdopendir` takes ownership of the descriptor it is given, so give
        // it a duplicate. It shares the read offset, which starts at 0.
        let dup = unsafe { libc::dup(fd.as_raw_fd()) };
        if dup < 0 {
            return Err(os_err("dup"));
        }
        let dir = unsafe { libc::fdopendir(dup) };
        if dir.is_null() {
            unsafe { libc::close(dup) };
            return Err(os_err("opendir"));
        }
        unsafe { libc::rewinddir(dir) };
        let mut names = Vec::new();
        loop {
            let ent = unsafe { libc::readdir(dir) };
            if ent.is_null() {
                break;
            }
            let name = unsafe { CStr::from_ptr((*ent).d_name.as_ptr()) };
            let name = name.to_string_lossy();
            if name != "." && name != ".." {
                names.push(name.into_owned());
            }
        }
        unsafe { libc::closedir(dir) };
        names.sort();
        Ok(names)
    }

    fn tilde(path: &str, home: Option<&str>) -> String {
        match home.and_then(|h| path.strip_prefix(h)) {
            Some(rest) if rest.is_empty() || rest.starts_with('/') => format!("~{rest}"),
            _ => path.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_stay_inside() {
        assert_eq!(components("/r", "a/b").unwrap(), ["a", "b"]);
        assert_eq!(components("/r", "./a//b/").unwrap(), ["a", "b"]);
        assert!(components("/r", ".").unwrap().is_empty());
        assert!(components("/r", "").unwrap().is_empty());
        assert!(components("/r", "../x").is_err());
        assert!(components("/r", "a/../../x").is_err());
        assert!(components("/r", "a/..").is_err());
        assert_eq!(components("/r", "/r/a").unwrap(), ["a"]);
        assert!(components("/r", "/r").unwrap().is_empty());
        assert!(components("/r", "/rx/a").is_err());
        assert!(components("/r", "/etc/passwd").is_err());
    }

    #[test]
    fn decode_text_and_binary() {
        assert_eq!(decode(b"hi", false).unwrap(), ("hi", 0));
        assert!(decode(b"a\0b", false).is_err());
        assert!(decode(b"\xff\xfe", false).is_err());
        // "é" is C3 A9: cut after C3, then resumed at A9.
        assert_eq!(decode(b"ab\xC3", true).unwrap(), ("ab", 0));
        assert!(decode(b"ab\xC3", false).is_err());
        assert_eq!(decode(b"\xA9cd", false).unwrap(), ("cd", 1));
    }

    #[test]
    fn budget_caps_a_turn() {
        let mut b = Budget(10);
        // No grant needed to see the budget refuse before touching disk.
        #[cfg(target_os = "macos")]
        {
            let dir = std::env::temp_dir().join(format!("llmoxide-budget-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("f.txt"), "0123456789abcdef").unwrap();
            let g = Grant::open(dir.to_str().unwrap(), true).unwrap();
            let out = call(&g, "read_file", &json!({"path": "f.txt"}), &mut b);
            assert!(out.ok);
            assert!(out.text.starts_with("0123456789"));
            let out = call(&g, "read_file", &json!({"path": "f.txt"}), &mut b);
            assert!(!out.ok, "{}", out.text);
            std::fs::remove_dir_all(dir).unwrap();
        }
        let _ = &mut b;
    }

    #[test]
    fn sizes() {
        assert_eq!(size(12), "12 B");
        assert_eq!(size(4300), "4.2 KB");
    }

    #[test]
    fn tool_defs_parse() {
        let t = tools();
        let names: Vec<_> = t.iter().map(|t| t.function.name.as_str()).collect();
        assert_eq!(names, ["list_dir", "read_file", "search"]);
    }
}
