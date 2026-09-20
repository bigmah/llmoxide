//! Where the resident conversation is held.
//!
//! A [`Session`] keeps two things that are the conversation itself rather than
//! anything derived from it: the token ids currently in the cache, decodable
//! straight back to plaintext by the tokenizer sitting next to them, and the
//! response text accumulated for stop-string matching.
//!
//! With the `private` feature those live in `llmoxide-secret`'s locked,
//! self-zeroing allocations. Without it they are an ordinary `Vec` and
//! `String`, and `llmoxide-secret` is not built at all.
//!
//! The feature is off by default, because most callers want inference and
//! nothing else, and locked memory is not free for them: `SecretVec` allocates
//! page-aligned, zeroes on every reallocation, and — the part that actually
//! bites an embedded consumer — prints a warning to stderr each time `mlock`
//! fails, which is every time under a low `RLIMIT_MEMLOCK` or in a container
//! that does not grant `IPC_LOCK`. A library has no business writing to stderr
//! on a machine that never asked for the guarantee.
//!
//! What is *not* conditional: [`Backend::wipe`] always overwrites the KV cache,
//! the recurrent state and the device buffers, because that is a cache-clearing
//! operation the correctness of prefix reuse depends on, not a privacy feature.
//! The `private` feature only decides whether the token-id copy is additionally
//! locked into RAM and zeroed.
//!
//! [`Session`]: crate::Session
//! [`Backend::wipe`]: crate::Backend::wipe

#[cfg(feature = "private")]
pub(crate) use locked::{TextStore, TokenStore};
#[cfg(not(feature = "private"))]
pub(crate) use plain::{TextStore, TokenStore};

#[cfg(feature = "private")]
mod locked {
    pub(crate) type TokenStore = secret::SecretVec<u32>;
    pub(crate) type TextStore = secret::SecretString;
}

#[cfg(not(feature = "private"))]
mod plain {
    /// The resident token ids, as a plain `Vec`.
    ///
    /// `wipe` clears rather than overwrites. It cannot honestly do more: a
    /// `Vec` that has grown leaves its old buffer behind on reallocation, and
    /// zeroing only the live one would suggest a guarantee this build does not
    /// make. Build with the `private` feature for the one that does.
    #[derive(Default)]
    pub(crate) struct TokenStore(Vec<u32>);

    impl TokenStore {
        pub(crate) const fn new() -> Self {
            Self(Vec::new())
        }

        pub(crate) fn push(&mut self, v: u32) {
            self.0.push(v);
        }

        pub(crate) fn replace(&mut self, s: &[u32]) {
            self.0.clear();
            self.0.extend_from_slice(s);
        }

        pub(crate) fn wipe(&mut self) {
            self.0.clear();
        }
    }

    impl std::ops::Deref for TokenStore {
        type Target = [u32];
        fn deref(&self) -> &[u32] {
            &self.0
        }
    }

    /// The generated text so far, for stop-string matching.
    #[derive(Default)]
    pub(crate) struct TextStore(String);

    impl TextStore {
        pub(crate) const fn new() -> Self {
            Self(String::new())
        }

        pub(crate) fn push_str(&mut self, s: &str) {
            self.0.push_str(s);
        }

        pub(crate) fn contains(&self, needle: &str) -> bool {
            self.0.contains(needle)
        }
    }
}
