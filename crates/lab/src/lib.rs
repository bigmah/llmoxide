//! The engine behind the browser demo in the llmoxide blog post.
//!
//! [`engine`] is a whole dense transformer forward pass on the CPU, with
//! probes. [`session`] is what the page does with it: tokenize, inspect a
//! prompt layer by layer, step it a token at a time, and chat. `web` is the
//! wasm-bindgen face of `session`, compiled only for the browser.

pub mod engine;
pub mod session;

#[cfg(target_arch = "wasm32")]
mod web;
