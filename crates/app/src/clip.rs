//! The copy button's clipboard.
//!
//! The clipboard is shared with every other app, so this is the one place a
//! reply leaves the process — only when Copy is pressed. Two things limit
//! that: the copy is marked for clipboard managers to skip (the
//! `org.nspasteboard.ConcealedType` convention on macOS, its equivalents on
//! Windows and Linux), and leaving — New chat, or quitting — takes it back
//! off the clipboard, unless something else has been copied since.

use std::sync::Mutex;

use arboard::Clipboard;
#[cfg(target_os = "macos")]
use arboard::SetExtApple as _;
#[cfg(all(unix, not(any(target_os = "macos", target_os = "android", target_os = "emscripten"))))]
use arboard::SetExtLinux as _;
#[cfg(windows)]
use arboard::SetExtWindows as _;

/// The clipboard handle and what was last copied from here. The handle is
/// kept, not reopened: on X11 the copy lives only as long as its owner.
static STATE: Mutex<Option<(Clipboard, String)>> = Mutex::new(None);

/// Put `text` on the clipboard. False if there is no clipboard to use.
pub fn copy(text: &str) -> bool {
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let mut clipboard = match state.take() {
        Some((c, _)) => c,
        None => match Clipboard::new() {
            Ok(c) => c,
            Err(_) => return false,
        },
    };
    let ok = clipboard
        .set()
        .exclude_from_history()
        .text(text.to_string())
        .is_ok();
    *state = Some((clipboard, if ok { text.to_string() } else { String::new() }));
    ok
}

/// Clear the clipboard if it still holds what was copied from here.
pub fn forget() {
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((mut clipboard, copied)) = state.take() {
        if !copied.is_empty() && clipboard.get_text().is_ok_and(|now| now == copied) {
            let _ = clipboard.clear();
        }
    }
}
