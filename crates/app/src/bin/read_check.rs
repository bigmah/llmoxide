//! Verify that the app's file tools read without leaving a trace on disk.
//!
//! The claim in `read.rs` rests on two measurements: a read whose access time
//! is put back leaves no timestamp behind, and FSEvents reports none of it.
//! Either could change with a macOS update, so this runs the real tool code
//! over a fixture folder and checks both, plus every refusal rule:
//!
//!   read_check [parent-folder]      # default: $TMPDIR
//!
//! 1. Builds a fixture and ages every access time to before its mtime, so any
//!    read the OS notices would move it.
//! 2. Watches the fixture with a file-level FSEvents stream.
//! 3. Grants it, then lists, reads and searches through the tools, and tries
//!    each thing a rule must refuse.
//! 4. Checks every atime, mtime and ctime is unchanged to the nanosecond, and
//!    that FSEvents saw nothing — then two positive controls: a plain read
//!    must move an access time, and an append must reach the stream. Without
//!    those, a pass could just mean the check was blind.
//!
//! Exits non-zero on any failure. macOS only, as the tools are.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("read_check: private file reading is macOS-only");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
#[path = "../read.rs"]
#[allow(dead_code)]
mod read;

#[cfg(target_os = "macos")]
fn main() {
    mac::main()
}

#[cfg(target_os = "macos")]
mod mac {
    use std::ffi::{c_char, c_void, CString};
    use std::os::unix::fs::MetadataExt;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::time::Duration;

    use serde_json::json;

    use crate::read::{self, Budget, Grant};

    // --- FSEvents -------------------------------------------------------

    type Callback =
        extern "C" fn(*const c_void, *mut c_void, usize, *mut c_void, *const u32, *const u64);

    #[repr(C)]
    struct Context {
        version: isize,
        info: *mut c_void,
        retain: *const c_void,
        release: *const c_void,
        copy_description: *const c_void,
    }

    #[link(name = "CoreServices", kind = "framework")]
    extern "C" {
        fn FSEventStreamCreate(
            alloc: *const c_void,
            cb: Callback,
            ctx: *const Context,
            paths: *const c_void,
            since: u64,
            latency: f64,
            flags: u32,
        ) -> *mut c_void;
        fn FSEventStreamSetDispatchQueue(s: *mut c_void, q: *mut c_void);
        fn FSEventStreamStart(s: *mut c_void) -> u8;
        fn FSEventStreamFlushSync(s: *mut c_void);
        fn FSEventStreamStop(s: *mut c_void);
        fn FSEventStreamInvalidate(s: *mut c_void);
        fn FSEventStreamRelease(s: *mut c_void);
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFStringCreateWithCString(a: *const c_void, s: *const c_char, enc: u32) -> *const c_void;
        fn CFArrayCreate(a: *const c_void, v: *const *const c_void, n: isize, cb: *const c_void) -> *const c_void;
        static kCFTypeArrayCallBacks: c_void;
    }

    extern "C" {
        fn dispatch_queue_create(label: *const c_char, attr: *const c_void) -> *mut c_void;
    }

    const SINCE_NOW: u64 = u64::MAX;
    const NO_DEFER: u32 = 0x02;
    const FILE_EVENTS: u32 = 0x10;
    const UTF8: u32 = 0x0800_0100;

    static EVENTS: Mutex<Vec<(String, u32)>> = Mutex::new(Vec::new());

    extern "C" fn on_events(
        _: *const c_void,
        _: *mut c_void,
        n: usize,
        paths: *mut c_void,
        flags: *const u32,
        _: *const u64,
    ) {
        let paths = paths as *const *const c_char;
        let mut ev = EVENTS.lock().unwrap();
        for i in 0..n {
            let p = unsafe { std::ffi::CStr::from_ptr(*paths.add(i)) };
            ev.push((p.to_string_lossy().into_owned(), unsafe { *flags.add(i) }));
        }
    }

    struct Watch(*mut c_void);

    impl Watch {
        fn start(dir: &Path) -> Watch {
            let c = CString::new(dir.to_str().unwrap()).unwrap();
            unsafe {
                let s = CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), UTF8);
                let arr = CFArrayCreate(std::ptr::null(), &s, 1, &kCFTypeArrayCallBacks as *const _);
                let ctx = Context {
                    version: 0,
                    info: std::ptr::null_mut(),
                    retain: std::ptr::null(),
                    release: std::ptr::null(),
                    copy_description: std::ptr::null(),
                };
                let stream = FSEventStreamCreate(
                    std::ptr::null(),
                    on_events,
                    &ctx,
                    arr,
                    SINCE_NOW,
                    0.05,
                    NO_DEFER | FILE_EVENTS,
                );
                assert!(!stream.is_null(), "FSEventStreamCreate failed");
                let q = dispatch_queue_create(c"read_check.fsevents".as_ptr(), std::ptr::null());
                FSEventStreamSetDispatchQueue(stream, q);
                assert!(FSEventStreamStart(stream) != 0, "FSEventStreamStart failed");
                Watch(stream)
            }
        }

        /// Everything the stream has for paths under `dir` so far.
        fn drain(&self, dir: &Path) -> Vec<(String, u32)> {
            // Give fseventsd time to publish, then force delivery.
            std::thread::sleep(Duration::from_millis(1500));
            unsafe { FSEventStreamFlushSync(self.0) };
            std::thread::sleep(Duration::from_millis(200));
            let prefix = dir.to_str().unwrap();
            let mut ev = EVENTS.lock().unwrap();
            let out = ev.iter().filter(|(p, _)| p.starts_with(prefix)).cloned().collect();
            ev.clear();
            out
        }
    }

    impl Drop for Watch {
        fn drop(&mut self) {
            unsafe {
                FSEventStreamStop(self.0);
                FSEventStreamInvalidate(self.0);
                FSEventStreamRelease(self.0);
            }
        }
    }

    // --- Timestamps ---------------------------------------------------------

    type Stamp = (i64, i64, i64, i64, i64, i64);

    fn stamp(p: &Path) -> Stamp {
        let m = std::fs::symlink_metadata(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        (m.atime(), m.atime_nsec(), m.mtime(), m.mtime_nsec(), m.ctime(), m.ctime_nsec())
    }

    /// Set atime and mtime of `p` (not following a link) to fixed, old values
    /// with atime before mtime: the case where macOS always updates atime.
    fn age(p: &Path) {
        let c = CString::new(p.to_str().unwrap()).unwrap();
        let t = [
            libc::timespec { tv_sec: 1_609_459_200, tv_nsec: 123_456_789 }, // 2021-01-01
            libc::timespec { tv_sec: 1_609_545_600, tv_nsec: 987_654_321 }, // 2021-01-02
        ];
        let r = unsafe { libc::utimensat(libc::AT_FDCWD, c.as_ptr(), t.as_ptr(), libc::AT_SYMLINK_NOFOLLOW) };
        assert_eq!(r, 0, "age {}: {}", p.display(), std::io::Error::last_os_error());
    }

    // --- The check ----------------------------------------------------------

    struct Report {
        failed: usize,
    }

    impl Report {
        fn ok(&mut self, cond: bool, what: &str) {
            println!("  [{}] {what}", if cond { "ok" } else { "FAIL" });
            if !cond {
                self.failed += 1;
            }
        }
        fn skip(&self, what: &str) {
            println!("  [skip] {what}");
        }
    }

    pub fn main() {
        let parent = std::env::args()
            .nth(1)
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        // `.noindex`: Spotlight reads new text files in indexed folders within
        // a second or two of their creation, which moves their access times
        // and would be blamed on the tools.
        let root = parent.join(format!("llmoxide-read-check-{}.noindex", std::process::id()));
        std::fs::create_dir_all(&root).expect("create fixture");
        let root = std::fs::canonicalize(&root).unwrap();
        let mut r = Report { failed: 0 };
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&root, &mut r)));
        let _ = std::fs::remove_dir_all(&root);
        if res.is_err() {
            r.failed += 1;
        }
        if r.failed == 0 {
            println!("\nread_check: pass");
        } else {
            println!("\nread_check: {} FAILED", r.failed);
            std::process::exit(1);
        }
    }

    fn run(root: &Path, r: &mut Report) {
        println!("fixture: {}", root.display());

        // Build it. Folders last in the aging order, since creating their
        // children changed their times.
        let w = |rel: &str, body: &[u8]| std::fs::write(root.join(rel), body).unwrap();
        std::fs::create_dir_all(root.join("sub/deep")).unwrap();
        w("a.txt", b"alpha\nthe needle is here\n");
        w("empty.txt", b"");
        w("bin.dat", b"\x7fELF\0\0\0binary");
        w("sub/b.rs", b"fn main() {\n    let needle = 1;\n}\n");
        w("sub/deep/c.md", "# Café\n\nnothing to find\n".as_bytes());
        let big: String = (0..2000).map(|i| format!("line {i:05} of the big file ✓\n")).collect();
        w("big.txt", big.as_bytes());
        w("control.txt", b"control\n");
        std::os::unix::fs::symlink("/etc/hosts", root.join("link_out")).unwrap();
        std::os::unix::fs::symlink("a.txt", root.join("link_in")).unwrap();
        let files = [
            "a.txt", "empty.txt", "bin.dat", "sub/b.rs", "sub/deep/c.md", "big.txt", "control.txt",
            "link_out", "link_in",
        ];
        let dirs = ["sub/deep", "sub", ""];
        let all: Vec<PathBuf> = files.iter().chain(&dirs).map(|p| root.join(p)).collect();
        // Age only after anything else interested in new files (an indexer,
        // a virus scanner) has had its look.
        std::thread::sleep(Duration::from_millis(1500));
        for p in &all {
            age(p);
        }
        let before: Vec<Stamp> = all.iter().map(|p| stamp(p)).collect();

        // Building and aging the fixture produced events of its own, and
        // fseventsd can publish them after the stream's "since now". Drain
        // whatever still arrives before the tools run.
        let watch = Watch::start(root);
        let stale = watch.drain(root);
        println!("({} events from building the fixture, discarded)", stale.len());
        let quiet = all.iter().zip(&before).all(|(p, b)| stamp(p) == *b);
        r.ok(quiet, "nothing else touched the fixture before the tools ran");

        println!("\ntools:");
        let g = Grant::open(root.to_str().unwrap(), false).expect("grant the fixture");
        let mut budget = Budget::new();
        let mut run = |name: &str, args: serde_json::Value| {
            let out = read::call(&g, name, &args, &mut budget);
            (out.ok, out.text)
        };
        let (ok, t) = run("list_dir", json!({"path": "."}));
        r.ok(ok && t.contains("sub/") && t.contains("big.txt") && t.contains("link_out  (symbolic link"), "list_dir .");
        let (ok, t) = run("list_dir", json!({"path": "sub/deep"}));
        r.ok(ok && t.contains("c.md"), "list_dir sub/deep");
        let (ok, t) = run("read_file", json!({"path": "a.txt"}));
        r.ok(ok && t.starts_with("alpha\n"), "read_file a.txt");
        let (ok, t) = run("read_file", json!({"path": "sub/deep/c.md"}));
        r.ok(ok && t.contains("Café"), "read_file sub/deep/c.md");
        let (ok, t) = run("read_file", json!({"path": "empty.txt"}));
        r.ok(ok && t.contains("empty"), "read_file empty.txt");
        let (ok, t) = run("read_file", json!({"path": "big.txt"}));
        r.ok(ok && t.contains("call read_file with offset="), "read_file big.txt, first part");
        let (ok, t) = run("read_file", json!({"path": "big.txt", "offset": 16385}));
        r.ok(ok && t.contains("showing bytes"), "read_file big.txt from mid-character offset");
        let (ok, t) = run("search", json!({"pattern": "needle"}));
        r.ok(ok && t.contains("a.txt:2:") && t.contains("sub/b.rs:2:"), "search needle");
        let (ok, t) = run("search", json!({"pattern": "Café", "path": "sub/deep/c.md"}));
        r.ok(ok && t.contains("c.md:1:"), "search in one file");

        println!("\nrefusals:");
        let mut refuse = |name: &str, args: serde_json::Value, why: &str, label: &str| {
            let out = read::call(&g, name, &args, &mut Budget::new());
            r.ok(!out.ok && out.text.contains(why), &format!("{label}: {}", out.text));
        };
        refuse("read_file", json!({"path": "link_out"}), "symbolic link", "symlink out of the folder");
        refuse("read_file", json!({"path": "link_in"}), "symbolic link", "symlink inside the folder");
        refuse("read_file", json!({"path": "../x"}), "..", "`..` escape");
        refuse("read_file", json!({"path": "/etc/hosts"}), "outside", "absolute path outside");
        refuse("read_file", json!({"path": "bin.dat"}), "binary", "binary file");
        refuse("read_file", json!({"path": "sub"}), "folder", "folder as a file");
        refuse("read_file", json!({"path": "nope.txt"}), "no such", "missing file");
        drop(g);

        let events = watch.drain(root);
        let after: Vec<Stamp> = all.iter().map(|p| stamp(p)).collect();

        println!("\ntraces:");
        for ((p, b), a) in all.iter().zip(&before).zip(&after) {
            if b != a {
                println!("    {}: {b:?} -> {a:?}", p.display());
            }
        }
        let changed = before.iter().zip(&after).filter(|(b, a)| b != a).count();
        r.ok(changed == 0, &format!("atime, mtime and ctime unchanged on all {} paths ({changed} changed)", all.len()));
        for (p, f) in &events {
            println!("    event {p} flags {f:#x}");
        }
        r.ok(events.is_empty(), &format!("FSEvents saw nothing ({} events)", events.len()));

        println!("\ncontrols:");
        let control = root.join("control.txt");
        let a0 = stamp(&control);
        let _ = std::fs::read(&control).unwrap();
        let a1 = stamp(&control);
        r.ok((a1.0, a1.1) != (a0.0, a0.1), "a plain read moves the access time (the check is not blind)");
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&control)
            .unwrap()
            .write_all(b"more\n")
            .unwrap();
        let events = watch.drain(root);
        r.ok(events.iter().any(|(p, _)| p.ends_with("control.txt")), "an append reaches the FSEvents stream");
        drop(watch);

        println!("\nrules outside the fixture:");
        match Grant::open("/usr/share/dict", false) {
            Err(e) => r.ok(e.contains("another user"), &format!("root-owned folder: {e}")),
            Ok(g) => {
                let out = read::call(&g, "read_file", &json!({"path": "words"}), &mut Budget::new());
                r.ok(!out.ok && out.text.contains("another user"), &format!("root-owned file: {}", out.text));
            }
        }
        if Path::new(&format!("{}/Documents", std::env::var("HOME").unwrap_or_default())).is_dir() {
            match Grant::open("~/Documents", false) {
                Err(e) => r.ok(e.contains("macOS"), &format!("~/Documents without opt-in: {e}")),
                Ok(_) => r.ok(false, "~/Documents without opt-in was granted"),
            }
        }
        match nonlocal_mount() {
            Some(m) => match Grant::open(&m, true) {
                Err(e) => r.ok(e.contains("network"), &format!("network mount {m}: {e}")),
                Ok(_) => r.ok(false, &format!("network mount {m} was granted")),
            },
            None => r.skip("network drive: none mounted"),
        }
        r.skip("iCloud placeholder: cannot be made without iCloud Drive; SF_DATALESS is checked on every path");
    }

    fn nonlocal_mount() -> Option<String> {
        let n = unsafe { libc::getfsstat(std::ptr::null_mut(), 0, libc::MNT_NOWAIT) };
        if n <= 0 {
            return None;
        }
        let mut v: Vec<libc::statfs> = vec![unsafe { std::mem::zeroed() }; n as usize];
        let size = (v.len() * std::mem::size_of::<libc::statfs>()) as i32;
        let n = unsafe { libc::getfsstat(v.as_mut_ptr(), size, libc::MNT_NOWAIT) };
        v.truncate(n.max(0) as usize);
        v.iter()
            .find(|s| s.f_flags & libc::MNT_LOCAL as u32 == 0 && {
                let t = unsafe { std::ffi::CStr::from_ptr(s.f_fstypename.as_ptr()) };
                !matches!(t.to_bytes(), b"devfs" | b"autofs" | b"nullfs")
            })
            .map(|s| unsafe { std::ffi::CStr::from_ptr(s.f_mntonname.as_ptr()) }.to_string_lossy().into_owned())
    }
}
