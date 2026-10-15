//! The page's handle on [`Lab`]. Runs inside a Web Worker (`lab-worker.js`),
//! so a forward pass never blocks the page's own thread.
//!
//! Results that are lists of numbers cross as `Float32Array`s; everything
//! else as small JSON strings. Timing is the worker's job, not this one's.

use serde_json::json;
use wasm_bindgen::prelude::*;

use crate::session::Lab;

/// Overwrite every heap block as it is freed: prompt text, token ids, decoded
/// pieces. It cannot reach the browser's own copies (the text box, the DOM,
/// the message the page posted to this worker), which is why the post is
/// careful to claim less for the demo than for the app.
#[global_allocator]
static ALLOC: secret::ZeroizingAlloc<std::alloc::System> = secret::ZeroizingAlloc(std::alloc::System);

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen]
pub struct Engine {
    lab: Lab<'static>,
    file_bytes: usize,
}

#[wasm_bindgen]
impl Engine {
    /// Take ownership of the whole checkpoint. The bytes are leaked on
    /// purpose: the weights borrow from them for as long as the worker lives,
    /// which is as long as the page does.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: Vec<u8>, chat_ctx: usize, probe_ctx: usize) -> Result<Engine, JsValue> {
        let file_bytes = bytes.len();
        let g: &'static gguf::Gguf = Box::leak(Box::new(gguf::Gguf::from_bytes(bytes).map_err(err)?));
        let lab = Lab::new(g, chat_ctx, probe_ctx).map_err(err)?;
        Ok(Self { lab, file_bytes })
    }

    pub fn config(&self) -> String {
        let c = &self.lab.model.cfg;
        let per_layer = 2 * c.d_model * c.q_dim() + 2 * c.d_model * c.kv_dim() + 3 * c.d_model * c.ffn_dim;
        json!({
            "arch": c.arch, "n_layers": c.n_layers, "d_model": c.d_model, "ffn_dim": c.ffn_dim,
            "n_heads": c.n_heads, "n_kv_heads": c.n_kv_heads, "head_dim": c.head_dim,
            "vocab": c.vocab, "rope_base": c.rope_base,
            "params": c.n_layers * per_layer + c.vocab * c.d_model,
            "file_bytes": self.file_bytes,
            "kv_bytes_per_token": 2 * c.n_layers * c.kv_dim() * 4,
            "chat_ctx": self.lab.chat_cache.capacity, "probe_ctx": self.lab.probe_cache.capacity,
        })
        .to_string()
    }

    /// `[[id, piece], ...]`
    pub fn tokenize(&self, text: &str) -> String {
        let ids = self.lab.tok.encode(text, true);
        json!(ids.iter().map(|&i| json!([i, self.lab.piece(i)])).collect::<Vec<_>>()).to_string()
    }

    pub fn embedding(&self, id: u32) -> Vec<f32> {
        self.lab.embedding(id)
    }

    /// `[[piece, cosine], ...]` for the nearest tokens to `id`.
    pub fn neighbors(&self, id: u32, k: usize) -> String {
        let near = self.lab.neighbors(id, k);
        json!(near.iter().map(|(i, s)| json!([self.lab.piece(*i), s])).collect::<Vec<_>>()).to_string()
    }

    /// Prefill `text` into the inspector with probes on. `[[id, piece], ...]`.
    pub fn inspect(&mut self, text: &str, chat: bool) -> Result<String, JsValue> {
        self.lab.inspect(text, chat).map_err(err)?;
        let ids = &self.lab.probe_ids;
        Ok(json!(ids.iter().map(|&i| json!([i, self.lab.piece(i)])).collect::<Vec<_>>()).to_string())
    }

    /// One more token in the inspector. `{id, piece, done}`.
    pub fn step(&mut self, temperature: f32) -> Result<String, JsValue> {
        let s = self.lab.probe_step(temperature).map_err(err)?;
        Ok(json!({ "id": s.id, "piece": self.lab.piece(s.id), "done": s.done }).to_string())
    }

    /// `[layers, heads, rows, ctx]` of the last probed forward.
    pub fn probe_shape(&self) -> Vec<u32> {
        let p = &self.lab.probe;
        let c = &self.lab.model.cfg;
        vec![c.n_layers as u32, c.n_heads as u32, p.t as u32, p.ctx as u32]
    }

    pub fn attention(&self) -> Vec<f32> {
        self.lab.probe.attn.clone()
    }

    /// Residual-stream RMS, `[boundary][row]` of the last probed forward.
    pub fn rms(&self) -> Vec<f32> {
        self.lab.probe.rms.clone()
    }

    /// `[[id, piece, p], ...]`, the inspector's next-token distribution.
    pub fn top(&self, temperature: f32, k: usize) -> String {
        let top = self.lab.probe_top(temperature, k);
        json!(top.iter().map(|(i, p)| json!([i, self.lab.piece(*i), p])).collect::<Vec<_>>()).to_string()
    }

    /// The logit lens: per layer boundary, `[[piece, p], ...]`.
    pub fn lens(&mut self, k: usize) -> String {
        let rows = self.lab.lens(k);
        json!(rows
            .iter()
            .map(|r| r.iter().map(|(i, p)| json!([self.lab.piece(*i), p])).collect::<Vec<_>>())
            .collect::<Vec<_>>())
        .to_string()
    }

    /// `{ran, reused}`: tokens prefilled, tokens the cache already held.
    pub fn chat_begin(&mut self, text: &str) -> Result<String, JsValue> {
        let (ran, reused) = self.lab.chat_begin(text).map_err(err)?;
        Ok(json!({ "ran": ran, "reused": reused }).to_string())
    }

    /// The next bit of the reply, or `undefined` once it is over.
    pub fn chat_next(&mut self) -> Option<String> {
        let (text, done) = self.lab.chat_next();
        (!done).then_some(text)
    }

    pub fn chat_stop(&mut self) {
        self.lab.finish();
    }

    /// `{tokens, kv_bytes, nonzero}` for the chat's cache.
    pub fn chat_stats(&self) -> String {
        let c = &self.lab.chat_cache;
        json!({
            "tokens": c.len,
            "kv_bytes": c.used_bytes(self.lab.model.cfg.n_layers),
            "nonzero": c.nonzero() + self.lab.probe_cache.nonzero(),
        })
        .to_string()
    }

    /// Forget everything. `{before, after}`: nonzero floats across both caches.
    pub fn wipe(&mut self) -> String {
        let count = |l: &Lab| l.chat_cache.nonzero() + l.probe_cache.nonzero();
        let before = count(&self.lab);
        self.lab.wipe();
        json!({ "before": before, "after": count(&self.lab) }).to_string()
    }
}
