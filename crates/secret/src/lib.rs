//! Memory that leaves nothing behind.
//!
//! Wiping after the fact is not enough on its own: a page that reached swap or
//! the hibernation image has already been written to disk, and zeroing the RAM
//! copy afterwards does not reach it. So this crate is arranged around
//! *preventing* the durable copy first, and zeroing second:
//!
//! * [`SecretVec`] — page-aligned, `mlock`ed, explicitly zeroed. Locked pages
//!   are never paged out and never enter `/var/vm/sleepimage`, so the only copy
//!   is the one in RAM, and that copy is overwritten on wipe or on drop.
//! * [`ZeroizingAlloc`] — a global allocator wrapper that overwrites every heap
//!   block as it is freed. Prompt text passes through many short-lived
//!   allocations that no explicit wipe could ever chase — `String`s from the
//!   chat template, the decoded output, per-token logit vectors. This catches
//!   them all at the one point they have in common.
//! * [`harden`] — turns off core dumps and debugger attach, the two ways a
//!   third party reads this memory without touching disk at all.
//!
//! What this does *not* claim: the zeroing allocator only covers Rust's heap,
//! `mlock` only covers pages this process owns, and neither reaches GPU
//! buffers (see `gpu::forward::GpuModel::wipe`) or anything already written to
//! a terminal. Root on a live machine can still read process memory.

use std::alloc::{GlobalAlloc, Layout};
use std::ffi::c_void;
use std::sync::atomic::{compiler_fence, AtomicBool, Ordering};

extern "C" {
    fn mlock(addr: *const c_void, len: usize) -> i32;
    fn munlock(addr: *const c_void, len: usize) -> i32;
    /// C11 Annex K. Unlike `memset`, the standard forbids optimizing this away
    /// even when the buffer is provably dead — which is exactly the case for
    /// every wipe here.
    fn memset_s(s: *mut c_void, smax: usize, c: i32, n: usize) -> i32;
    fn getpagesize() -> i32;
    fn setrlimit(resource: i32, rlp: *const Rlimit) -> i32;
    fn ptrace(request: i32, pid: i32, addr: *mut c_void, data: i32) -> i32;
}

#[repr(C)]
struct Rlimit {
    rlim_cur: u64,
    rlim_max: u64,
}

const RLIMIT_CORE: i32 = 4;
const PT_DENY_ATTACH: i32 = 31;

/// Overwrite `len` bytes at `ptr`, with a guarantee the write survives the
/// optimizer.
///
/// # Safety
/// `ptr` must be valid for writes of `len` bytes.
pub unsafe fn zero(ptr: *mut u8, len: usize) {
    if len == 0 {
        return;
    }
    memset_s(ptr.cast(), len, 0, len);
    // memset_s alone is enough by the letter of the standard; the fence also
    // stops surrounding stores being reordered past the wipe.
    compiler_fence(Ordering::SeqCst);
}

/// Overwrite a slice in place.
pub fn zero_slice<T: Copy>(s: &mut [T]) {
    let bytes = std::mem::size_of_val(s);
    unsafe { zero(s.as_mut_ptr().cast(), bytes) }
}

// ---------------------------------------------------------------------------
// Locked, self-zeroing storage
// ---------------------------------------------------------------------------

/// A growable buffer whose pages are locked into RAM and zeroed before release.
///
/// Used for the state that lives longest — the resident prompt token ids, which
/// the server keeps between turns so prefix reuse works, and which decode
/// losslessly back to the conversation text. That is precisely the allocation
/// that would otherwise sit in memory for hours and land in a sleep image.
///
/// The allocation is page-aligned so that `munlock` on drop cannot unlock a
/// page that some other secret still occupies.
pub struct SecretVec<T: Copy> {
    ptr: *mut T,
    cap: usize,
    len: usize,
}

unsafe impl<T: Copy + Send> Send for SecretVec<T> {}

impl<T: Copy> SecretVec<T> {
    pub const fn new() -> Self {
        Self {
            ptr: std::ptr::NonNull::<T>::dangling().as_ptr(),
            cap: 0,
            len: 0,
        }
    }

    fn layout(cap: usize) -> Layout {
        let page = unsafe { getpagesize() } as usize;
        let bytes = cap * std::mem::size_of::<T>();
        // Round up so the lock covers whole pages and nothing else.
        let bytes = bytes.div_ceil(page) * page;
        Layout::from_size_align(bytes, page).expect("secret layout")
    }

    fn grow(&mut self, need: usize) {
        if need <= self.cap {
            return;
        }
        let cap = need.next_power_of_two().max(1024);
        let layout = Self::layout(cap);
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        // Best effort: a failed lock is worth reporting but not worth refusing
        // to run over, since the wipe path still works without it.
        if unsafe { mlock(ptr.cast(), layout.size()) } != 0 {
            eprintln!("warning: could not lock {} bytes into RAM", layout.size());
        }
        let ptr = ptr.cast::<T>();
        unsafe {
            if self.len > 0 {
                std::ptr::copy_nonoverlapping(self.ptr, ptr, self.len);
            }
            self.release();
        }
        self.ptr = ptr;
        self.cap = cap;
    }

    /// Zero, unlock and free the current allocation. `len` is left alone; the
    /// caller fixes it up.
    unsafe fn release(&mut self) {
        if self.cap == 0 {
            return;
        }
        let layout = Self::layout(self.cap);
        zero(self.ptr.cast(), layout.size());
        munlock(self.ptr.cast(), layout.size());
        std::alloc::dealloc(self.ptr.cast(), layout);
        self.cap = 0;
    }

    pub fn push(&mut self, v: T) {
        self.grow(self.len + 1);
        unsafe { self.ptr.add(self.len).write(v) };
        self.len += 1;
    }

    pub fn extend_from_slice(&mut self, s: &[T]) {
        self.grow(self.len + s.len());
        unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), self.ptr.add(self.len), s.len()) };
        self.len += s.len();
    }

    /// Wipe whatever is here and take `s` instead.
    pub fn replace(&mut self, s: &[T]) {
        self.wipe();
        self.extend_from_slice(s);
    }

    /// Overwrite the whole allocation, not just the live prefix — a shorter
    /// prompt must not leave the tail of a longer one behind it.
    pub fn wipe(&mut self) {
        if self.cap > 0 {
            unsafe { zero(self.ptr.cast(), Self::layout(self.cap).size()) };
        }
        self.len = 0;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: Copy> Default for SecretVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy> std::ops::Deref for SecretVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Copy> Drop for SecretVec<T> {
    fn drop(&mut self) {
        unsafe { self.release() }
    }
}

/// A locked, self-zeroing string, for accumulating generated text.
pub struct SecretString(SecretVec<u8>);

impl SecretString {
    pub const fn new() -> Self {
        Self(SecretVec::new())
    }

    pub fn push_str(&mut self, s: &str) {
        self.0.extend_from_slice(s.as_bytes());
    }

    pub fn as_str(&self) -> &str {
        // Only ever fed whole `&str`s, so the bytes stay valid UTF-8.
        std::str::from_utf8(&self.0).unwrap_or("")
    }

    pub fn contains(&self, needle: &str) -> bool {
        self.as_str().contains(needle)
    }

    pub fn wipe(&mut self) {
        self.0.wipe();
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Default for SecretString {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Zeroing allocator
// ---------------------------------------------------------------------------

static ARMED: AtomicBool = AtomicBool::new(false);

/// Start overwriting freed heap blocks. Idempotent.
///
/// Arm this *after* the model is resident: weight staging buffers are gigabytes
/// of non-secret data, and zeroing them on the way out only slows down load.
pub fn arm() {
    ARMED.store(true, Ordering::SeqCst);
}

pub fn armed() -> bool {
    ARMED.load(Ordering::Relaxed)
}

/// Wraps another allocator, overwriting every block as it is freed.
///
/// Install with `#[global_allocator]`. Note that `realloc` is deliberately left
/// as the trait's default implementation, which routes through `alloc` + copy +
/// `dealloc` here — delegating to the inner allocator's `realloc` would hand
/// the old block back to the system with the plaintext still in it, which is
/// exactly how a growing `String` leaks every intermediate copy of itself.
pub struct ZeroizingAlloc<A>(pub A);

unsafe impl<A: GlobalAlloc> GlobalAlloc for ZeroizingAlloc<A> {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        self.0.alloc(l)
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        self.0.alloc_zeroed(l)
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if ARMED.load(Ordering::Relaxed) {
            zero(p, l.size());
        }
        self.0.dealloc(p, l)
    }
}

// ---------------------------------------------------------------------------
// Process hardening
// ---------------------------------------------------------------------------

/// Close the two routes that read this process's memory without touching disk.
///
/// * `RLIMIT_CORE = 0` — a crash cannot write a core file containing the
///   prompt. (macOS defaults to this, but it is inherited from the shell and
///   a `ulimit -c unlimited` upstream would silently undo it.)
/// * `PT_DENY_ATTACH` — no debugger can attach and read memory out of a live
///   session. Root can still do so through other means; this raises the bar,
///   it does not close the door.
pub fn harden() {
    let rl = Rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    unsafe {
        setrlimit(RLIMIT_CORE, &rl);
        ptrace(PT_DENY_ATTACH, 0, std::ptr::null_mut(), 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::System;
    use std::sync::atomic::AtomicUsize;

    /// An inner allocator that snapshots each block *at the moment it is handed
    /// back*, which is after [`ZeroizingAlloc`] has had its turn. Checking the
    /// snapshot rather than reading the freed pointer keeps the test honest:
    /// nothing here reads memory it does not own.
    struct Spy;

    static SPY_NONZERO: AtomicUsize = AtomicUsize::new(0);
    static SPY_WATCH: AtomicUsize = AtomicUsize::new(0);

    unsafe impl GlobalAlloc for Spy {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            System.alloc(l)
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            if SPY_WATCH.load(Ordering::Relaxed) == l.size() {
                let seen = std::slice::from_raw_parts(p, l.size());
                SPY_NONZERO.store(seen.iter().filter(|b| **b != 0).count(), Ordering::SeqCst);
            }
            System.dealloc(p, l)
        }
    }

    #[test]
    fn allocator_zeroes_before_release() {
        let alloc = ZeroizingAlloc(Spy);
        let layout = Layout::from_size_align(4096, 8).unwrap();

        // Disarmed: the block reaches the inner allocator with its contents.
        SPY_WATCH.store(4096, Ordering::SeqCst);
        unsafe {
            let p = alloc.alloc(layout);
            std::ptr::write_bytes(p, 0xAB, 4096);
            alloc.dealloc(p, layout);
        }
        assert_eq!(SPY_NONZERO.load(Ordering::SeqCst), 4096, "disarmed should pass through");

        // Armed: the inner allocator only ever sees zeros.
        arm();
        unsafe {
            let p = alloc.alloc(layout);
            std::ptr::write_bytes(p, 0xAB, 4096);
            alloc.dealloc(p, layout);
        }
        assert_eq!(SPY_NONZERO.load(Ordering::SeqCst), 0, "armed should zero before free");
        SPY_WATCH.store(0, Ordering::SeqCst);
    }

    #[test]
    fn wipe_clears_the_whole_allocation_not_just_the_live_prefix() {
        let mut v = SecretVec::<u32>::new();
        v.extend_from_slice(&[0xDEAD_BEEF; 2000]);
        let ptr = v.ptr;
        let cap = v.cap;
        v.wipe();
        // A shorter follow-up must not leave the tail of the longer one behind.
        v.extend_from_slice(&[1, 2, 3]);
        let all = unsafe { std::slice::from_raw_parts(ptr, cap) };
        assert_eq!(&all[..3], &[1, 2, 3]);
        assert!(
            all[3..].iter().all(|w| *w == 0),
            "tail of the wiped conversation survived"
        );
    }

    #[test]
    fn secret_vec_round_trips() {
        let mut v = SecretVec::<u32>::new();
        for i in 0..5000u32 {
            v.push(i);
        }
        assert_eq!(v.len(), 5000);
        assert_eq!(v[4999], 4999);
        v.replace(&[7, 8, 9]);
        assert_eq!(&v[..], &[7, 8, 9]);
    }

    #[test]
    fn secret_string_accumulates_and_wipes() {
        let mut s = SecretString::new();
        s.push_str("the quick ");
        s.push_str("brown fox");
        assert!(s.contains("quick brown"));
        s.wipe();
        assert!(s.is_empty());
        assert_eq!(s.as_str(), "");
    }

    #[test]
    fn zero_slice_overwrites() {
        let mut v = vec![0xFFu8; 100];
        zero_slice(&mut v);
        assert!(v.iter().all(|b| *b == 0));
    }
}
