//! A dense transformer, start to finish, on the CPU.
//!
//! This is the forward pass of `qwen3` and `llama`-family checkpoints written
//! to be read rather than to be fast. It is llmoxide's own CPU reference path
//! (`model::qwen35::Cpu`) with the hybrid delta-net layers taken out, and with
//! probes put in: a forward can record every attention pattern, the size of
//! the residual stream after each layer, and the stream itself at the last
//! position, which is what the logit lens reads.
//!
//! ```text
//! h = embed(tok)
//! per layer:
//!   x = rms(h) * attn_norm
//!   q, k, v = Wq x, Wk x, Wv x          // GQA: fewer K/V heads than Q heads
//!   q, k = rope(norm?(q)), rope(norm?(k)) // qwen3 RMS-norms each head first
//!   h = h + Wo softmax(q·kᵀ / √d) v      // causal: row i sees 0..=i
//!   y = rms(h) * ffn_norm
//!   h = h + Wd (silu(Wg y) * Wu y)        // SwiGLU
//! logits = W_out · (rms(h) * output_norm)  // W_out is the embedding when tied
//! ```
//!
//! The two families differ in two places only. qwen3 normalizes each query
//! and key head before rotating it; llama does not. And they pair different
//! dimensions for RoPE: qwen3 rotates `(i, i + d/2)`, llama `(2i, 2i + 1)`.
//! (llama's converter permutes Q and K so that ggml's adjacent-pair rotation
//! reproduces the original model, so getting this wrong does not crash, it
//! produces fluent nonsense.)

use gguf::{Gguf, TensorView};
use model::ops;

/// Which dimensions RoPE rotates together.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rope {
    /// `(2i, 2i+1)`: ggml's "normal" mode, which llama checkpoints use.
    Adjacent,
    /// `(i, i + d/2)`: NeoX mode, which every Qwen uses.
    Halves,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: String,
    pub n_layers: usize,
    pub d_model: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
    pub rope: Rope,
    /// Tokens that end the assistant's turn.
    pub eog: Vec<u32>,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Self> {
        let arch = g.str("general.architecture")?.to_string();
        let rope = match arch.as_str() {
            "llama" => Rope::Adjacent,
            "qwen3" => Rope::Halves,
            other => anyhow::bail!("the lab runs dense llama and qwen3 checkpoints, not {other:?}"),
        };
        let k = |s: &str| format!("{arch}.{s}");
        let d_model = g.usize(&k("embedding_length"))?;
        let n_heads = g.usize(&k("attention.head_count"))?;
        // llama files often omit the head width; it is then d_model / heads.
        let head_dim = g.usize(&k("attention.key_length")).unwrap_or(d_model / n_heads);

        let tokens = g.string_array("tokenizer.ggml.tokens").unwrap_or(&[]);
        let mut eog: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|n| tokens.iter().position(|t| t == n).map(|i| i as u32))
            .collect();
        if let Ok(eos) = g.u64("tokenizer.ggml.eos_token_id") {
            eog.push(eos as u32);
        }
        eog.sort_unstable();
        eog.dedup();

        Ok(Self {
            n_layers: g.usize(&k("block_count"))?,
            d_model,
            ffn_dim: g.usize(&k("feed_forward_length"))?,
            n_heads,
            n_kv_heads: g.usize(&k("attention.head_count_kv")).unwrap_or(n_heads),
            head_dim,
            vocab: g.info("token_embd.weight")?.out_dim(),
            rms_eps: g.f32(&k("attention.layer_norm_rms_epsilon"))?,
            rope_base: g.f32(&k("rope.freq_base")).unwrap_or(10_000.0),
            rope,
            eog,
            arch,
        })
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
}

struct Layer<'a> {
    attn_norm: Vec<f32>,
    q: TensorView<'a>,
    k: TensorView<'a>,
    v: TensorView<'a>,
    o: TensorView<'a>,
    /// Per-head RMS gains on Q and K: qwen3 has them, llama does not.
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    gate: TensorView<'a>,
    up: TensorView<'a>,
    down: TensorView<'a>,
}

/// The weights, borrowed from the file they live in. Norms are copied out to
/// `Vec<f32>` since they are tiny; everything else is read quantized, in place.
pub struct Weights<'a> {
    pub embed: TensorView<'a>,
    layers: Vec<Layer<'a>>,
    out_norm: Vec<f32>,
    /// The output head. Small models tie it to the embedding, and their files
    /// then have no `output.weight` at all.
    pub output: TensorView<'a>,
}

impl<'a> Weights<'a> {
    pub fn load(g: &'a Gguf, cfg: &Config) -> anyhow::Result<Self> {
        let norm = |name: &str| -> anyhow::Result<Vec<f32>> { Ok(g.tensor(name)?.to_f32()) };
        let embed = g.tensor("token_embd.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            let layer = Layer {
                attn_norm: norm(&p("attn_norm"))?,
                q: g.tensor(&p("attn_q"))?,
                k: g.tensor(&p("attn_k"))?,
                v: g.tensor(&p("attn_v"))?,
                o: g.tensor(&p("attn_output"))?,
                q_norm: g.tensor_opt(&p("attn_q_norm")).map(|t| t.to_f32()),
                k_norm: g.tensor_opt(&p("attn_k_norm")).map(|t| t.to_f32()),
                ffn_norm: norm(&p("ffn_norm"))?,
                gate: g.tensor(&p("ffn_gate"))?,
                up: g.tensor(&p("ffn_up"))?,
                down: g.tensor(&p("ffn_down"))?,
            };
            for (what, got, want) in [
                ("attn_q", layer.q.out_dim(), cfg.q_dim()),
                ("attn_k", layer.k.out_dim(), cfg.kv_dim()),
                ("attn_v", layer.v.out_dim(), cfg.kv_dim()),
                ("attn_output", layer.o.in_dim(), cfg.q_dim()),
                ("ffn_gate", layer.gate.out_dim(), cfg.ffn_dim),
                ("ffn_down", layer.down.out_dim(), cfg.d_model),
            ] {
                anyhow::ensure!(got == want, "blk.{i}.{what}: {got} != expected {want}");
            }
            layers.push(layer);
        }
        Ok(Self {
            embed,
            layers,
            out_norm: norm("output_norm.weight")?,
            output: g.tensor_opt("output.weight").unwrap_or(embed),
        })
    }
}

/// Keys and values for every position seen so far, so that a new token only
/// has to compute its own. Layout `[layer][position][kv_dim]`.
pub struct Cache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// Positions filled.
    pub len: usize,
    pub capacity: usize,
    kv_dim: usize,
}

impl Cache {
    pub fn new(cfg: &Config, capacity: usize) -> Self {
        let n = cfg.n_layers * capacity * cfg.kv_dim();
        Self {
            k: vec![0.0; n],
            v: vec![0.0; n],
            len: 0,
            capacity,
            kv_dim: cfg.kv_dim(),
        }
    }

    fn at(&self, layer: usize, pos: usize) -> usize {
        (layer * self.capacity + pos) * self.kv_dim
    }

    /// Bytes in use: what this conversation costs in memory right now.
    pub fn used_bytes(&self, n_layers: usize) -> usize {
        2 * n_layers * self.len * self.kv_dim * std::mem::size_of::<f32>()
    }

    /// Overwrite everything, not just reset the length. A `len = 0` would make
    /// the old keys and values unreachable to the model, and leave every one
    /// of them in memory for anything else that looks.
    pub fn wipe(&mut self) {
        secret::zero_slice(&mut self.k);
        secret::zero_slice(&mut self.v);
        self.len = 0;
    }

    /// How many floats in the whole cache are not zero.
    pub fn nonzero(&self) -> usize {
        self.k.iter().chain(&self.v).filter(|x| **x != 0.0).count()
    }
}

/// What a probed forward pass leaves behind for the page to draw.
#[derive(Default)]
pub struct Probe {
    /// Tokens in the probed batch, and the context length they ended at.
    pub t: usize,
    pub ctx: usize,
    /// Attention weights, `[layer][head][query row 0..t][key 0..ctx]`. Row `i`
    /// is zero past key `ctx - t + i`: that is the causal mask.
    pub attn: Vec<f32>,
    /// RMS of the residual stream, `[boundary 0..=n_layers][token 0..t]`.
    /// Boundary 0 is the embedding, boundary `l + 1` is after layer `l`.
    pub rms: Vec<f32>,
    /// The residual stream at the last position, `[boundary][d_model]`.
    pub last: Vec<f32>,
}

/// Scratch reused across calls, sized for the largest batch seen.
#[derive(Default)]
struct Scratch {
    h: Vec<f32>,
    x: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    scores: Vec<f32>,
    row: Vec<f32>,
}

pub struct Model<'a> {
    pub cfg: Config,
    pub w: Weights<'a>,
    s: Scratch,
}

impl<'a> Model<'a> {
    pub fn load(g: &'a Gguf) -> anyhow::Result<Self> {
        let cfg = Config::from_gguf(g)?;
        let w = Weights::load(g, &cfg)?;
        Ok(Self {
            cfg,
            w,
            s: Scratch::default(),
        })
    }

    /// Run `tokens` from `cache.len` onward and return the logits for the
    /// last one. Many tokens at once is *prefill*; one at a time is *decode*.
    /// It is the same function: the only difference is how many rows `t` the
    /// matrix multiplies have.
    pub fn forward(&mut self, cache: &mut Cache, tokens: &[u32], probe: Option<&mut Probe>) -> Vec<f32> {
        let cfg = &self.cfg;
        let w = &self.w;
        let s = &mut self.s;
        let (t, d, hd) = (tokens.len(), cfg.d_model, cfg.head_dim);
        let (q_dim, kv_dim) = (cfg.q_dim(), cfg.kv_dim());
        let base = cache.len;
        assert!(t > 0, "forward with no tokens");
        assert!(base + t <= cache.capacity, "context full: {base} + {t} > {}", cache.capacity);

        let mut probe = probe;
        if let Some(p) = probe.as_deref_mut() {
            p.t = t;
            p.ctx = base + t;
            p.attn.clear();
            p.attn.resize(cfg.n_layers * cfg.n_heads * t * (base + t), 0.0);
            p.rms.clear();
            p.last.clear();
        }

        for (buf, n) in [
            (&mut s.h, t * d),
            (&mut s.x, t * d),
            (&mut s.q, t * q_dim),
            (&mut s.k, t * kv_dim),
            (&mut s.v, t * kv_dim),
            (&mut s.attn, t * q_dim),
            (&mut s.proj, t * d),
            (&mut s.gate, t * cfg.ffn_dim),
            (&mut s.up, t * cfg.ffn_dim),
        ] {
            buf.resize(n, 0.0);
        }
        s.row.resize(d.max(cfg.ffn_dim).max(q_dim), 0.0);

        // Embedding: one row of the table per token. No arithmetic at all.
        for (i, &tok) in tokens.iter().enumerate() {
            w.embed.dequant_row_into(tok as usize, &mut s.h[i * d..(i + 1) * d]);
        }
        record(probe.as_deref_mut(), &s.h, t, d);

        let group = cfg.n_heads / cfg.n_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();

        for (il, lw) in w.layers.iter().enumerate() {
            // --- attention -------------------------------------------------
            s.x.copy_from_slice(&s.h);
            for row in s.x.chunks_exact_mut(d) {
                ops::rms_norm_mul(row, &lw.attn_norm, cfg.rms_eps);
            }
            matmul(&mut s.q, &lw.q, &s.x, t, &mut s.row);
            matmul(&mut s.k, &lw.k, &s.x, t, &mut s.row);
            matmul(&mut s.v, &lw.v, &s.x, t, &mut s.row);

            for (i, head) in s.q.chunks_exact_mut(hd).enumerate() {
                if let Some(n) = &lw.q_norm {
                    ops::rms_norm_mul(head, n, cfg.rms_eps);
                }
                rope(head, base + i / cfg.n_heads, cfg.rope_base, cfg.rope);
            }
            for (i, head) in s.k.chunks_exact_mut(hd).enumerate() {
                if let Some(n) = &lw.k_norm {
                    ops::rms_norm_mul(head, n, cfg.rms_eps);
                }
                rope(head, base + i / cfg.n_kv_heads, cfg.rope_base, cfg.rope);
            }

            // Append this batch's keys and values to the cache, then attend
            // over everything in it.
            let at = cache.at(il, base);
            cache.k[at..at + t * kv_dim].copy_from_slice(&s.k);
            cache.v[at..at + t * kv_dim].copy_from_slice(&s.v);
            let layer_k = &cache.k[cache.at(il, 0)..];
            let layer_v = &cache.v[cache.at(il, 0)..];

            for i in 0..t {
                let seen = base + i + 1; // causal: position base+i sees 0..=base+i
                s.scores.resize(seen, 0.0);
                for h in 0..cfg.n_heads {
                    let kvh = h / group; // GQA: `group` query heads share a K/V head
                    let q = &s.q[i * q_dim + h * hd..][..hd];
                    for (p, sc) in s.scores.iter_mut().enumerate() {
                        *sc = dot(q, &layer_k[p * kv_dim + kvh * hd..][..hd]) * scale;
                    }
                    ops::softmax(&mut s.scores);

                    let out = &mut s.attn[i * q_dim + h * hd..][..hd];
                    out.fill(0.0);
                    for (p, &a) in s.scores.iter().enumerate() {
                        let v = &layer_v[p * kv_dim + kvh * hd..][..hd];
                        for (o, &vv) in out.iter_mut().zip(v) {
                            *o += a * vv;
                        }
                    }

                    if let Some(p) = probe.as_deref_mut() {
                        let ctx = p.ctx;
                        let row = ((il * cfg.n_heads + h) * t + i) * ctx;
                        p.attn[row..row + seen].copy_from_slice(&s.scores);
                    }
                }
            }
            matmul(&mut s.proj, &lw.o, &s.attn, t, &mut s.row);
            for (hv, pv) in s.h.iter_mut().zip(&s.proj) {
                *hv += pv;
            }

            // --- feed-forward ----------------------------------------------
            s.x.copy_from_slice(&s.h);
            for row in s.x.chunks_exact_mut(d) {
                ops::rms_norm_mul(row, &lw.ffn_norm, cfg.rms_eps);
            }
            matmul(&mut s.gate, &lw.gate, &s.x, t, &mut s.row);
            matmul(&mut s.up, &lw.up, &s.x, t, &mut s.row);
            for (g, u) in s.gate.iter_mut().zip(&s.up) {
                *g = ops::silu(*g) * u;
            }
            matmul(&mut s.proj, &lw.down, &s.gate, t, &mut s.row);
            for (hv, pv) in s.h.iter_mut().zip(&s.proj) {
                *hv += pv;
            }
            record(probe.as_deref_mut(), &s.h, t, d);
        }
        cache.len = base + t;

        // Only the last position's logits are needed to pick the next token.
        let last = &mut s.h[(t - 1) * d..];
        ops::rms_norm_mul(last, &w.out_norm, cfg.rms_eps);
        let mut logits = vec![0.0; cfg.vocab];
        matmul(&mut logits, &w.output, last, 1, &mut s.row);
        logits
    }

    /// The logit lens: what the model would predict if it stopped after
    /// `boundary` layers, by sending that point of the residual stream
    /// straight to the output head.
    pub fn lens(&mut self, residual: &[f32]) -> Vec<f32> {
        let mut x = residual.to_vec();
        ops::rms_norm_mul(&mut x, &self.w.out_norm, self.cfg.rms_eps);
        let mut logits = vec![0.0; self.cfg.vocab];
        matmul(&mut logits, &self.w.output, &x, 1, &mut self.s.row);
        logits
    }
}

/// Note the residual stream after a layer (or after the embedding).
fn record(probe: Option<&mut Probe>, h: &[f32], t: usize, d: usize) {
    let Some(p) = probe else { return };
    for row in h.chunks_exact(d) {
        p.rms.push((row.iter().map(|v| v * v).sum::<f32>() / d as f32).sqrt());
    }
    p.last.extend_from_slice(&h[(t - 1) * d..]);
}

/// `Y = W X` for `t` input rows, `W` quantized with `ne = [in, out]`.
///
/// Each weight row is dequantized once into `row` and then dotted with all
/// `t` inputs. That is the whole reason prefill is faster per token than
/// decode: the expensive part, reading the weights, is shared by every token
/// in the batch, and decode has a batch of one.
pub fn matmul(y: &mut [f32], w: &TensorView<'_>, x: &[f32], t: usize, row: &mut Vec<f32>) {
    let (n_in, n_out) = (w.in_dim(), w.out_dim());
    debug_assert_eq!(x.len(), t * n_in);
    debug_assert_eq!(y.len(), t * n_out);
    row.resize(row.len().max(n_in), 0.0);
    let row = &mut row[..n_in];
    for o in 0..n_out {
        w.dequant_row_into(o, row);
        for i in 0..t {
            y[i * n_out + o] = dot(row, &x[i * n_in..(i + 1) * n_in]);
        }
    }
}

/// A dot product with eight independent running sums.
///
/// The compiler may not reorder float additions on its own, so a plain
/// `zip().map().sum()` is one long dependency chain and runs a lane at a
/// time. Eight explicit lanes give it permission to use SIMD.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) {
        for j in 0..8 {
            acc[j] += x[j] * y[j];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// Rotary position embedding: rotate pairs of dimensions by an angle that
/// grows with the position, each pair at its own frequency. After this, the
/// dot product of a query and a key depends on how far apart they are.
pub fn rope(head: &mut [f32], pos: usize, base: f32, mode: Rope) {
    let d = head.len();
    let half = d / 2;
    let theta_scale = (base as f64).powf(-2.0 / d as f64);
    let mut theta = pos as f64;
    for i in 0..half {
        let (sin, cos) = (theta.sin() as f32, theta.cos() as f32);
        let (a, b) = match mode {
            Rope::Adjacent => (2 * i, 2 * i + 1),
            Rope::Halves => (i, i + half),
        };
        let (x0, x1) = (head[a], head[b]);
        head[a] = x0 * cos - x1 * sin;
        head[b] = x0 * sin + x1 * cos;
        theta *= theta_scale;
    }
}
