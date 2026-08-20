//! The qwen35 forward pass on wgpu.
//!
//! Mirrors `model::qwen35::cpu` step for step — that implementation is
//! validated tensor-by-tensor against llama.cpp, so any divergence here is a
//! kernel bug, not an architecture question. See ARCHITECTURE-qwen35.md.
//!
//! The hybrid stack needs two kinds of per-layer state: full-attention layers
//! keep an ordinary KV cache, gated-delta-net layers keep a conv window plus
//! one s_dim x s_dim matrix per v-head. Both live in GPU buffers; the delta
//! recurrence runs inside a single dispatch per layer with the token loop in
//! the kernel, since tokens are inherently sequential there.

use std::collections::HashMap;

use gguf::Gguf;
use model::qwen35::config::Config;

use crate::{Gpu, MatvecParams, QuantKernels, Weights};

/// Uniform slots are padded to the alignment dynamic offsets require.
const UNIFORM_SLOT: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Op {
    n_rows: u32,
    dim: u32,
    off0: u32,
    off1: u32,
    f0: f32,
    f1: f32,
    u0: u32,
    u1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Attn {
    n_tokens: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    base_pos: u32,
    window: u32,
    max_vis: u32,
    scale: f32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct DOp {
    n_tok: u32,
    conv_dim: u32,
    key_dim: u32,
    d_inner: u32,
    s_dim: u32,
    n_vh: u32,
    n_kh: u32,
    kernel: u32,
    off_taps: u32,
    off_a: u32,
    off_dt: u32,
    eps: f32,
    scale: f32,
    _pad: [u32; 3],
}

/// All F32 model tensors concatenated into one buffer, addressed by offset.
struct ParamArena {
    data: Vec<f32>,
    offsets: HashMap<String, u32>,
}

impl ParamArena {
    fn build(g: &Gguf, cfg: &Config) -> anyhow::Result<Self> {
        let mut data = Vec::new();
        let mut offsets = HashMap::new();
        let push = |name: String, data: &mut Vec<f32>, offsets: &mut HashMap<String, u32>| -> anyhow::Result<()> {
            let t = g.tensor(&name)?;
            offsets.insert(name, data.len() as u32);
            data.extend_from_slice(&t.to_f32());
            Ok(())
        };

        push("output_norm.weight".into(), &mut data, &mut offsets)?;
        for i in 0..cfg.n_layers {
            let mut names = vec![
                format!("blk.{i}.attn_norm.weight"),
                format!("blk.{i}.post_attention_norm.weight"),
            ];
            if cfg.recurrent[i] {
                names.extend([
                    format!("blk.{i}.ssm_conv1d.weight"),
                    format!("blk.{i}.ssm_dt.bias"),
                    format!("blk.{i}.ssm_a"),
                    format!("blk.{i}.ssm_norm.weight"),
                ]);
            } else {
                names.extend([
                    format!("blk.{i}.attn_q_norm.weight"),
                    format!("blk.{i}.attn_k_norm.weight"),
                ]);
            }
            for n in names {
                push(n, &mut data, &mut offsets)?;
            }
        }
        Ok(Self { data, offsets })
    }

    fn at(&self, name: &str) -> anyhow::Result<u32> {
        self.offsets
            .get(name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("param {name:?} missing"))
    }
}

struct Pipelines {
    ops_layout: wgpu::BindGroupLayout,
    rms_norm: wgpu::ComputePipeline,
    add_scale: wgpu::ComputePipeline,
    copy: wgpu::ComputePipeline,
    swiglu: wgpu::ComputePipeline,
    mul_silu: wgpu::ComputePipeline,
    mul_sigmoid: wgpu::ComputePipeline,
    split_half: wgpu::ComputePipeline,
    rope_partial: wgpu::ComputePipeline,
    write_cache: wgpu::ComputePipeline,

    attn_layout: wgpu::BindGroupLayout,
    scores: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    weighted_v: wgpu::ComputePipeline,

    delta_layout: wgpu::BindGroupLayout,
    conv_silu: wgpu::ComputePipeline,
    conv_shift: wgpu::ComputePipeline,
    conv_commit: wgpu::ComputePipeline,
    l2norm_qk: wgpu::ComputePipeline,
    delta_recur: wgpu::ComputePipeline,
}

impl Pipelines {
    fn new(gpu: &Gpu) -> Self {
        let dev = &gpu.device;
        let storage = |ro: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: None,
        };
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };

        let ops_layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qwen-ops"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(3, storage(false)),
                entry(4, uniform),
            ],
        });
        let attn_layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qwen-attn"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(3, storage(false)),
                entry(4, storage(false)),
                entry(5, uniform),
            ],
        });
        let delta_layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qwen-delta"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(3, storage(true)),
                entry(4, storage(false)),
                entry(5, storage(false)),
                entry(6, uniform),
            ],
        });

        let build = |name: &str, src: &str, layout: &wgpu::BindGroupLayout, entries: &[&str]| {
            let module = gpu.shader(name, src);
            let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            entries
                .iter()
                .map(|e| {
                    dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(e),
                        layout: Some(&pl),
                        module: &module,
                        entry_point: Some(e),
                        compilation_options: Default::default(),
                        cache: None,
                    })
                })
                .collect::<Vec<_>>()
        };

        let mut ops = build(
            "qwen-ops",
            include_str!("shaders/ops.wgsl"),
            &ops_layout,
            &[
                "rms_norm",
                "add_scale",
                "copy",
                "swiglu",
                "mul_silu",
                "mul_sigmoid",
                "split_half",
                "rope_partial",
                "write_cache",
            ],
        )
        .into_iter();
        let mut at = build(
            "qwen-attn",
            include_str!("shaders/attn.wgsl"),
            &attn_layout,
            &["scores_pass", "softmax_pass", "weighted_v"],
        )
        .into_iter();
        let mut dl = build(
            "qwen-delta",
            include_str!("shaders/deltanet.wgsl"),
            &delta_layout,
            &["conv_silu", "conv_shift", "conv_commit", "l2norm_qk", "delta_recur"],
        )
        .into_iter();

        Self {
            rms_norm: ops.next().unwrap(),
            add_scale: ops.next().unwrap(),
            copy: ops.next().unwrap(),
            swiglu: ops.next().unwrap(),
            mul_silu: ops.next().unwrap(),
            mul_sigmoid: ops.next().unwrap(),
            split_half: ops.next().unwrap(),
            rope_partial: ops.next().unwrap(),
            write_cache: ops.next().unwrap(),
            ops_layout,
            scores: at.next().unwrap(),
            softmax: at.next().unwrap(),
            weighted_v: at.next().unwrap(),
            attn_layout,
            conv_silu: dl.next().unwrap(),
            conv_shift: dl.next().unwrap(),
            conv_commit: dl.next().unwrap(),
            l2norm_qk: dl.next().unwrap(),
            delta_recur: dl.next().unwrap(),
            delta_layout,
        }
    }
}

/// Per-layer GPU state, matching `model::qwen35::state::LayerState`.
enum LayerState {
    Linear {
        /// `(kernel-1) * conv_dim`, time-major, oldest row first.
        conv: wgpu::Buffer,
        /// `n_v_heads * s_dim * s_dim`, rows are value dims.
        s: wgpu::Buffer,
    },
    Attn {
        k: wgpu::Buffer,
        v: wgpu::Buffer,
    },
}

pub struct Qwen35Gpu {
    gpu: Gpu,
    cfg: Config,
    weights: Weights,
    quant: QuantKernels,
    pipes: Pipelines,

    params: wgpu::Buffer,
    param_off: ParamArena,

    // Activations, sized for max_batch tokens.
    h: wgpu::Buffer,
    x: wgpu::Buffer,
    /// Fused projection: q|k|v channels on delta layers, [query|gate] pairs on
    /// attention layers.
    fused: wgpu::Buffer,
    conv: wgpu::Buffer,
    conv_scratch: wgpu::Buffer,
    z: wgpu::Buffer,
    alpha: wgpu::Buffer,
    beta: wgpu::Buffer,
    q: wgpu::Buffer,
    gate_a: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    attn: wgpu::Buffer,
    proj: wgpu::Buffer,
    gate: wgpu::Buffer,
    up: wgpu::Buffer,
    scores: wgpu::Buffer,
    logits: wgpu::Buffer,
    tokens: wgpu::Buffer,
    zero: wgpu::Buffer,
    /// Dummy for unused read-write bindings; `zero` covers the read-only ones.
    /// One buffer cannot serve both roles within a dispatch.
    sink: wgpu::Buffer,

    state: Vec<LayerState>,
    uniforms: wgpu::Buffer,

    max_batch: usize,
    n_ctx: usize,
    pub pos: usize,
    /// When set, `forward` stops at the named checkpoint and returns that
    /// buffer's contents instead of logits, for bisecting against the CPU path.
    pub debug_stop: Option<String>,
    /// Cached single-token plan; building one costs ~1500 bind groups.
    decode: Option<CachedPlan>,
}

impl Qwen35Gpu {
    pub fn load(
        gpu: Gpu,
        g: &Gguf,
        cfg: Config,
        n_ctx: usize,
        max_batch: usize,
    ) -> anyhow::Result<Self> {
        // Matvec weights go to the arena — by explicit name so the trailing
        // NextN/MTP block is never uploaded and the all-F32 synthetic
        // checkpoints work the same way as quantized ones.
        let mut names = vec!["token_embd.weight".to_string()];
        if g.tensor_opt("output.weight").is_some() {
            names.push("output.weight".to_string());
        }
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            if cfg.recurrent[i] {
                names.extend([
                    p("attn_qkv.weight"),
                    p("attn_gate.weight"),
                    p("ssm_alpha.weight"),
                    p("ssm_beta.weight"),
                    p("ssm_out.weight"),
                ]);
            } else {
                names.extend([
                    p("attn_q.weight"),
                    p("attn_k.weight"),
                    p("attn_v.weight"),
                    p("attn_output.weight"),
                ]);
            }
            names.extend([p("ffn_gate.weight"), p("ffn_up.weight"), p("ffn_down.weight")]);
        }
        let weights = Weights::upload(&gpu, g, names)?;

        let param_off = ParamArena::build(g, &cfg)?;
        let params = gpu.upload_f32("params", &param_off.data);

        let d = cfg.d_model;
        let conv_dim = cfg.conv_dim();
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.kv_dim();
        let fused_dim = conv_dim.max(2 * q_dim);
        let f32s = |n: usize| (n * 4) as u64;

        let state = (0..cfg.n_layers)
            .map(|i| {
                if cfg.recurrent[i] {
                    LayerState::Linear {
                        conv: gpu.storage(
                            &format!("conv{i}"),
                            f32s((cfg.conv_kernel - 1) * conv_dim),
                        ),
                        s: gpu.storage(
                            &format!("s{i}"),
                            f32s(cfg.n_v_heads * cfg.lin_head_dim * cfg.lin_head_dim),
                        ),
                    }
                } else {
                    LayerState::Attn {
                        k: gpu.storage(&format!("k{i}"), f32s(n_ctx * kv_dim)),
                        v: gpu.storage(&format!("v{i}"), f32s(n_ctx * kv_dim)),
                    }
                }
            })
            .collect();

        let uniforms = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: UNIFORM_SLOT * 4096,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            h: gpu.storage("h", f32s(max_batch * d)),
            x: gpu.storage("x", f32s(max_batch * d)),
            fused: gpu.storage("fused", f32s(max_batch * fused_dim)),
            conv: gpu.storage("conv", f32s(max_batch * conv_dim)),
            conv_scratch: gpu.storage("conv_scratch", f32s((cfg.conv_kernel - 1) * conv_dim)),
            z: gpu.storage("z", f32s(max_batch * cfg.d_inner)),
            alpha: gpu.storage("alpha", f32s(max_batch * cfg.n_v_heads)),
            beta: gpu.storage("beta", f32s(max_batch * cfg.n_v_heads)),
            q: gpu.storage("q", f32s(max_batch * q_dim)),
            gate_a: gpu.storage("gate_a", f32s(max_batch * q_dim)),
            k: gpu.storage("k", f32s(max_batch * kv_dim)),
            v: gpu.storage("v", f32s(max_batch * kv_dim)),
            attn: gpu.storage("attn", f32s(max_batch * q_dim.max(cfg.d_inner))),
            proj: gpu.storage("proj", f32s(max_batch * d)),
            gate: gpu.storage("gate", f32s(max_batch * cfg.ffn_dim)),
            up: gpu.storage("up", f32s(max_batch * cfg.ffn_dim)),
            scores: gpu.storage("scores", f32s(max_batch * cfg.n_heads * n_ctx)),
            logits: gpu.storage("logits", f32s(cfg.vocab)),
            tokens: gpu.storage("tokens", f32s(max_batch)),
            zero: gpu.upload_f32("zero", &[0.0]),
            sink: gpu.storage("sink", 4),
            params,
            param_off,
            quant: QuantKernels::new(&gpu),
            pipes: Pipelines::new(&gpu),
            weights,
            state,
            uniforms,
            cfg,
            gpu,
            max_batch,
            n_ctx,
            pos: 0,
            debug_stop: None,
            decode: None,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn context_len(&self) -> usize {
        self.n_ctx
    }

    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Rewind to an empty context. Unlike the KV cache, recurrent state is
    /// cumulative, so it must be zeroed rather than just overwritten.
    pub fn reset(&mut self) {
        self.pos = 0;
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for l in &self.state {
            if let LayerState::Linear { conv, s } = l {
                enc.clear_buffer(conv, 0, None);
                enc.clear_buffer(s, 0, None);
            }
        }
        self.gpu.queue.submit([enc.finish()]);
    }
}

/// One uniform slot; position-dependent values are materialized per call so
/// the cached decode plan stays valid across steps.
enum Slot {
    Fixed(Vec<u8>),
    /// `u1` becomes the batch's base position.
    Positioned(Op),
    /// `base_pos` and the visible length are recomputed per call.
    Attention(Attn),
}

impl Slot {
    fn emit(&self, base_pos: u32, n_tokens: u32, buf: &mut Vec<u8>) {
        let start = buf.len();
        match self {
            Slot::Fixed(b) => buf.extend_from_slice(b),
            Slot::Positioned(op) => {
                let mut op = *op;
                op.u1 = base_pos;
                buf.extend_from_slice(bytemuck::bytes_of(&op));
            }
            Slot::Attention(a) => {
                let mut a = *a;
                a.base_pos = base_pos;
                a.max_vis = base_pos + n_tokens;
                buf.extend_from_slice(bytemuck::bytes_of(&a));
            }
        }
        buf.resize(start + UNIFORM_SLOT as usize, 0);
    }
}

struct CachedPlan {
    plan: Vec<Dispatch>,
    slots: Vec<Slot>,
    tail: Vec<Dispatch>,
    tail_slots: Vec<Slot>,
    checkpoints: Vec<(String, usize, usize)>,
}

struct Dispatch {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    offset: u32,
    groups: (u32, u32, u32),
}

const WG_OPS: u32 = 256;

/// Workgroups for an elementwise pass; the kernels grid-stride, so a clamped
/// grid still covers the range.
fn cells(n: u32, wg: u32) -> u32 {
    n.div_ceil(wg).clamp(1, 65535)
}

fn bind<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn bind_slot<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: std::num::NonZeroU64::new(UNIFORM_SLOT),
        }),
    }
}

impl Qwen35Gpu {
    fn ops_bind(&self, a: &wgpu::Buffer, b: &wgpu::Buffer, out: &wgpu::Buffer) -> wgpu::BindGroup {
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes.ops_layout,
            entries: &[
                bind(0, &self.params),
                bind(1, a),
                bind(2, b),
                bind(3, out),
                bind_slot(4, &self.uniforms),
            ],
        })
    }

    fn delta_bind(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        c: &wgpu::Buffer,
        state: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes.delta_layout,
            entries: &[
                bind(0, &self.params),
                bind(1, a),
                bind(2, b),
                bind(3, c),
                bind(4, state),
                bind(5, out),
                bind_slot(6, &self.uniforms),
            ],
        })
    }

    fn attn_bind(&self, k: &wgpu::Buffer, v: &wgpu::Buffer) -> wgpu::BindGroup {
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes.attn_layout,
            entries: &[
                bind(0, &self.q),
                bind(1, k),
                bind(2, v),
                bind(3, &self.scores),
                bind(4, &self.attn),
                bind_slot(5, &self.uniforms),
            ],
        })
    }

    fn matvec_bind(
        &self,
        w_buffer: usize,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.quant.layout,
            entries: &[
                bind(0, &self.weights.buffers[w_buffer]),
                bind(1, x),
                bind(2, y),
                bind_slot(3, &self.uniforms),
                bind(4, &self.tokens),
            ],
        })
    }
}

impl Qwen35Gpu {
    fn build_plan(&self, t: usize) -> anyhow::Result<CachedPlan> {
        let cfg = self.cfg.clone();
        let d = cfg.d_model;
        let conv_dim = cfg.conv_dim();
        let key_dim = cfg.n_k_heads * cfg.lin_head_dim;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.kv_dim();
        let max_groups = self.gpu.limits.max_compute_workgroups_per_dimension;

        let mut u: Vec<Slot> = Vec::new();
        let mut plan: Vec<Dispatch> = Vec::new();
        let mut checkpoints: Vec<(String, usize, usize)> = Vec::new();

        fn slot(u: &mut Vec<Slot>, bytes: &[u8]) -> u32 {
            let off = u.len() as u32 * UNIFORM_SLOT as u32;
            u.push(Slot::Fixed(bytes.to_vec()));
            off
        }
        fn slot_of(u: &mut Vec<Slot>, s: Slot) -> u32 {
            let off = u.len() as u32 * UNIFORM_SLOT as u32;
            u.push(s);
            off
        }

        macro_rules! op {
            ($u:ident, $($f:tt)*) => { slot(&mut $u, bytemuck::bytes_of(&Op { $($f)* ..Default::default() })) };
        }

        // Matvec dispatch, shared by every projection.
        let matvec = |u: &mut Vec<Slot>,
                          plan: &mut Vec<Dispatch>,
                          name: &str,
                          x: &wgpu::Buffer,
                          y: &wgpu::Buffer,
                          out_dim: u32|
         -> anyhow::Result<()> {
            let h = self.weights.get(name)?;
            let off = slot(
                u,
                bytemuck::bytes_of(&MatvecParams {
                    w_base: h.base_u32,
                    in_dim: h.in_dim,
                    out_dim,
                    n_tokens: t as u32,
                }),
            );
            plan.push(Dispatch {
                pipeline: self.quant.pipeline_for(h.ty, t as u32)?.clone(),
                bind: self.matvec_bind(h.buffer, x, y),
                offset: off,
                groups: (
                    crate::row_groups(out_dim, max_groups),
                    crate::token_groups(t as u32),
                    1,
                ),
            });
            Ok(())
        };

        // --- embeddings (no scale) -----------------------------------------
        let embd = self.weights.get("token_embd.weight")?;
        let off = slot(
            &mut u,
            bytemuck::bytes_of(&MatvecParams {
                w_base: embd.base_u32,
                in_dim: d as u32,
                out_dim: cfg.vocab as u32,
                n_tokens: t as u32,
            }),
        );
        plan.push(Dispatch {
            pipeline: self.quant.embed_for(embd.ty)?.clone(),
            bind: self.matvec_bind(embd.buffer, &self.zero, &self.h),
            offset: off,
            groups: (16, t as u32, 1),
        });
        checkpoints.push(("model.input_embed".into(), plan.len(), 0));

        for il in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{il}.{s}");

            // pre-attention norm (copy first: rms_norm works in place)
            let off = op!(u, n_rows: t as u32, dim: d as u32,);
            plan.push(Dispatch {
                pipeline: self.pipes.copy.clone(),
                bind: self.ops_bind(&self.h, &self.zero, &self.x),
                offset: off,
                groups: (cells((t * d) as u32, WG_OPS), 1, 1),
            });
            let off = op!(u,
                n_rows: t as u32, dim: d as u32,
                off0: self.param_off.at(&p("attn_norm.weight"))?,
                f0: cfg.rms_eps, u0: 1,
            );
            plan.push(Dispatch {
                pipeline: self.pipes.rms_norm.clone(),
                bind: self.ops_bind(&self.zero, &self.zero, &self.x),
                offset: off,
                groups: (cells(t as u32, 1), 1, 1),
            });
            checkpoints.push((format!("attn_norm-{il}"), plan.len(), 1));

            if cfg.recurrent[il] {
                let LayerState::Linear { conv: conv_state, s } = &self.state[il] else {
                    unreachable!("layer {il}: state kind disagrees with config");
                };

                matvec(&mut u, &mut plan, &p("attn_qkv.weight"), &self.x, &self.fused, conv_dim as u32)?;
                checkpoints.push((format!("linear_attn_qkv_mixed-{il}"), plan.len(), 2));
                matvec(&mut u, &mut plan, &p("attn_gate.weight"), &self.x, &self.z, cfg.d_inner as u32)?;
                checkpoints.push((format!("z-{il}"), plan.len(), 4));
                matvec(&mut u, &mut plan, &p("ssm_alpha.weight"), &self.x, &self.alpha, cfg.n_v_heads as u32)?;
                checkpoints.push((format!("alpha-{il}"), plan.len(), 5));
                matvec(&mut u, &mut plan, &p("ssm_beta.weight"), &self.x, &self.beta, cfg.n_v_heads as u32)?;
                checkpoints.push((format!("beta-{il}"), plan.len(), 6));

                // Conv + state shift + commit + head norms + the recurrence
                // share one uniform slot: they read the same geometry.
                let dop = slot(
                    &mut u,
                    bytemuck::bytes_of(&DOp {
                        n_tok: t as u32,
                        conv_dim: conv_dim as u32,
                        key_dim: key_dim as u32,
                        d_inner: cfg.d_inner as u32,
                        s_dim: cfg.lin_head_dim as u32,
                        n_vh: cfg.n_v_heads as u32,
                        n_kh: cfg.n_k_heads as u32,
                        kernel: cfg.conv_kernel as u32,
                        off_taps: self.param_off.at(&p("ssm_conv1d.weight"))?,
                        off_a: self.param_off.at(&p("ssm_a"))?,
                        off_dt: self.param_off.at(&p("ssm_dt.bias"))?,
                        eps: cfg.rms_eps,
                        scale: 1.0 / (cfg.lin_head_dim as f32).sqrt(),
                        _pad: [0; 3],
                    }),
                );

                for (pipeline, bind, n) in [
                    (
                        &self.pipes.conv_silu,
                        self.delta_bind(&self.fused, &self.zero, &self.zero, conv_state, &self.conv),
                        (t * conv_dim) as u32,
                    ),
                    (
                        &self.pipes.conv_shift,
                        self.delta_bind(&self.fused, &self.zero, &self.zero, conv_state, &self.conv_scratch),
                        ((cfg.conv_kernel - 1) * conv_dim) as u32,
                    ),
                    (
                        &self.pipes.conv_commit,
                        self.delta_bind(&self.conv_scratch, &self.zero, &self.zero, conv_state, &self.sink),
                        ((cfg.conv_kernel - 1) * conv_dim) as u32,
                    ),
                ] {
                    plan.push(Dispatch {
                        pipeline: pipeline.clone(),
                        bind,
                        offset: dop,
                        groups: (cells(n, WG_OPS), 1, 1),
                    });
                }

                plan.push(Dispatch {
                    pipeline: self.pipes.l2norm_qk.clone(),
                    bind: self.delta_bind(&self.zero, &self.zero, &self.zero, &self.sink, &self.conv),
                    offset: dop,
                    groups: (cells((t * 2 * cfg.n_k_heads) as u32, 1), 1, 1),
                });

                plan.push(Dispatch {
                    pipeline: self.pipes.delta_recur.clone(),
                    bind: self.delta_bind(&self.alpha, &self.beta, &self.conv, s, &self.attn),
                    offset: dop,
                    groups: (cfg.n_v_heads as u32, 1, 1),
                });
                checkpoints.push((format!("attn_output-{il}"), plan.len(), 11));

                // Gated RMS norm per v-head, then the silu(z) gate.
                let off = op!(u,
                    n_rows: (t * cfg.n_v_heads) as u32, dim: cfg.lin_head_dim as u32,
                    off0: self.param_off.at(&p("ssm_norm.weight"))?,
                    f0: cfg.rms_eps, u0: 1,
                );
                plan.push(Dispatch {
                    pipeline: self.pipes.rms_norm.clone(),
                    bind: self.ops_bind(&self.zero, &self.zero, &self.attn),
                    offset: off,
                    groups: (cells((t * cfg.n_v_heads) as u32, 1), 1, 1),
                });
                let off = op!(u, n_rows: t as u32, dim: cfg.d_inner as u32,);
                plan.push(Dispatch {
                    pipeline: self.pipes.mul_silu.clone(),
                    bind: self.ops_bind(&self.zero, &self.z, &self.attn),
                    offset: off,
                    groups: (cells((t * cfg.d_inner) as u32, WG_OPS), 1, 1),
                });
                checkpoints.push((format!("final_output-{il}"), plan.len(), 11));

                matvec(&mut u, &mut plan, &p("ssm_out.weight"), &self.attn, &self.proj, d as u32)?;
                checkpoints.push((format!("linear_attn_out-{il}"), plan.len(), 12));
            } else {
                let LayerState::Attn { k: kc, v: vc } = &self.state[il] else {
                    unreachable!("layer {il}: state kind disagrees with config");
                };

                matvec(&mut u, &mut plan, &p("attn_q.weight"), &self.x, &self.fused, (2 * q_dim) as u32)?;
                checkpoints.push((format!("Qcur_full-{il}"), plan.len(), 2));
                matvec(&mut u, &mut plan, &p("attn_k.weight"), &self.x, &self.k, kv_dim as u32)?;
                checkpoints.push((format!("Kcur-{il}"), plan.len(), 9));
                matvec(&mut u, &mut plan, &p("attn_v.weight"), &self.x, &self.v, kv_dim as u32)?;
                checkpoints.push((format!("Vcur-{il}"), plan.len(), 10));

                // Deinterleave the fused per-head [query | gate] pairs.
                for (dst, half) in [(&self.q, 0u32), (&self.gate_a, cfg.head_dim as u32)] {
                    let off = op!(u,
                        n_rows: (t * cfg.n_heads) as u32, dim: cfg.head_dim as u32,
                        u0: half,
                    );
                    plan.push(Dispatch {
                        pipeline: self.pipes.split_half.clone(),
                        bind: self.ops_bind(&self.fused, &self.zero, dst),
                        offset: off,
                        groups: (cells((t * q_dim) as u32, WG_OPS), 1, 1),
                    });
                }

                // Per-head QK norms (learned per-dim gains), then partial RoPE.
                for (buf, gain, heads) in [
                    (&self.q, p("attn_q_norm.weight"), cfg.n_heads as u32),
                    (&self.k, p("attn_k_norm.weight"), cfg.n_kv_heads as u32),
                ] {
                    let off = op!(u,
                        n_rows: t as u32 * heads, dim: cfg.head_dim as u32,
                        off0: self.param_off.at(&gain)?, f0: cfg.rms_eps, u0: 1,
                    );
                    plan.push(Dispatch {
                        pipeline: self.pipes.rms_norm.clone(),
                        bind: self.ops_bind(&self.zero, &self.zero, buf),
                        offset: off,
                        groups: (cells(t as u32 * heads, 1), 1, 1),
                    });
                    let off = slot_of(&mut u, Slot::Positioned(Op {
                        n_rows: t as u32 * heads, dim: cfg.head_dim as u32,
                        off0: cfg.n_rot as u32,
                        f0: cfg.rope_base,
                        u0: heads,
                        ..Default::default()
                    }));
                    plan.push(Dispatch {
                        pipeline: self.pipes.rope_partial.clone(),
                        bind: self.ops_bind(&self.zero, &self.zero, buf),
                        offset: off,
                        groups: (cells(t as u32 * heads, 1), 1, 1),
                    });
                }
                checkpoints.push((format!("Qcur-{il}"), plan.len(), 7));
                checkpoints.push((format!("Kcur_pos-{il}"), plan.len(), 9));

                // KV cache append, then the three attention passes.
                for (src, dst) in [(&self.k, kc), (&self.v, vc)] {
                    let off = slot_of(&mut u, Slot::Positioned(Op {
                        n_rows: t as u32, dim: kv_dim as u32, u0: 0,
                        ..Default::default()
                    }));
                    plan.push(Dispatch {
                        pipeline: self.pipes.write_cache.clone(),
                        bind: self.ops_bind(src, &self.zero, dst),
                        offset: off,
                        groups: (cells((t * kv_dim) as u32, WG_OPS), 1, 1),
                    });
                }

                let attn_off = slot_of(&mut u, Slot::Attention(Attn {
                    n_tokens: t as u32,
                    n_heads: cfg.n_heads as u32,
                    n_kv_heads: cfg.n_kv_heads as u32,
                    head_dim: cfg.head_dim as u32,
                    kv_dim: kv_dim as u32,
                    window: 0,
                    scale: cfg.attn_scale(),
                    ..Default::default()
                }));
                let attn_bind = self.attn_bind(kc, vc);
                for pipeline in [&self.pipes.scores, &self.pipes.softmax, &self.pipes.weighted_v] {
                    plan.push(Dispatch {
                        pipeline: pipeline.clone(),
                        bind: attn_bind.clone(),
                        offset: attn_off,
                        groups: (cfg.n_heads as u32, t as u32, 1),
                    });
                }

                // The fused projection's gate halves scale the head outputs.
                let off = op!(u, n_rows: t as u32, dim: q_dim as u32,);
                plan.push(Dispatch {
                    pipeline: self.pipes.mul_sigmoid.clone(),
                    bind: self.ops_bind(&self.zero, &self.gate_a, &self.attn),
                    offset: off,
                    groups: (cells((t * q_dim) as u32, WG_OPS), 1, 1),
                });
                checkpoints.push((format!("attn_gated-{il}"), plan.len(), 11));

                matvec(&mut u, &mut plan, &p("attn_output.weight"), &self.attn, &self.proj, d as u32)?;
                checkpoints.push((format!("attn_output-{il}"), plan.len(), 12));
            }

            // --- residual, pre-FFN norm, FFN, residual ------------------
            let off = op!(u, n_rows: t as u32, dim: d as u32, f0: 1.0,);
            plan.push(Dispatch {
                pipeline: self.pipes.add_scale.clone(),
                bind: self.ops_bind(&self.zero, &self.proj, &self.h),
                offset: off,
                groups: (cells((t * d) as u32, WG_OPS), 1, 1),
            });
            checkpoints.push((format!("attn_residual-{il}"), plan.len(), 0));

            let off = op!(u, n_rows: t as u32, dim: d as u32,);
            plan.push(Dispatch {
                pipeline: self.pipes.copy.clone(),
                bind: self.ops_bind(&self.h, &self.zero, &self.x),
                offset: off,
                groups: (cells((t * d) as u32, WG_OPS), 1, 1),
            });
            let off = op!(u,
                n_rows: t as u32, dim: d as u32,
                off0: self.param_off.at(&p("post_attention_norm.weight"))?,
                f0: cfg.rms_eps, u0: 1,
            );
            plan.push(Dispatch {
                pipeline: self.pipes.rms_norm.clone(),
                bind: self.ops_bind(&self.zero, &self.zero, &self.x),
                offset: off,
                groups: (cells(t as u32, 1), 1, 1),
            });
            checkpoints.push((format!("attn_post_norm-{il}"), plan.len(), 1));

            matvec(&mut u, &mut plan, &p("ffn_gate.weight"), &self.x, &self.gate, cfg.ffn_dim as u32)?;
            matvec(&mut u, &mut plan, &p("ffn_up.weight"), &self.x, &self.up, cfg.ffn_dim as u32)?;

            let off = op!(u, n_rows: t as u32, dim: cfg.ffn_dim as u32,);
            plan.push(Dispatch {
                pipeline: self.pipes.swiglu.clone(),
                bind: self.ops_bind(&self.zero, &self.up, &self.gate),
                offset: off,
                groups: (cells((t * cfg.ffn_dim) as u32, WG_OPS), 1, 1),
            });

            matvec(&mut u, &mut plan, &p("ffn_down.weight"), &self.gate, &self.proj, d as u32)?;
            checkpoints.push((format!("ffn_out-{il}"), plan.len(), 12));

            let off = op!(u, n_rows: t as u32, dim: d as u32, f0: 1.0,);
            plan.push(Dispatch {
                pipeline: self.pipes.add_scale.clone(),
                bind: self.ops_bind(&self.zero, &self.proj, &self.h),
                offset: off,
                groups: (cells((t * d) as u32, WG_OPS), 1, 1),
            });
            checkpoints.push((format!("l_out-{il}"), plan.len(), 0));
        }

        // --- output head ---------------------------------------------------
        let off = op!(u,
            n_rows: t as u32, dim: d as u32,
            off0: self.param_off.at("output_norm.weight")?,
            f0: cfg.rms_eps, u0: 1,
        );
        plan.push(Dispatch {
            pipeline: self.pipes.rms_norm.clone(),
            bind: self.ops_bind(&self.zero, &self.zero, &self.h),
            offset: off,
            groups: (cells(t as u32, 1), 1, 1),
        });
        checkpoints.push(("result_norm".into(), plan.len(), 0));

        let out = self
            .weights
            .get("output.weight")
            .or_else(|_| self.weights.get("token_embd.weight"))?;
        let mut tail: Vec<Slot> = Vec::new();
        let off = slot(
            &mut tail,
            bytemuck::bytes_of(&MatvecParams {
                w_base: out.base_u32,
                in_dim: d as u32,
                out_dim: cfg.vocab as u32,
                n_tokens: 1,
            }),
        );
        let tail_plan = vec![Dispatch {
            pipeline: self.quant.pipeline_for(out.ty, 1)?.clone(),
            bind: self.matvec_bind(out.buffer, &self.x, &self.logits),
            offset: off,
            groups: (crate::row_groups(cfg.vocab as u32, max_groups), 1, 1),
        }];

        Ok(CachedPlan {
            plan,
            slots: u,
            tail: tail_plan,
            tail_slots: tail,
            checkpoints,
        })
    }

    /// Run `tokens` starting at the current position and return the final
    /// logits. Mirrors `model::qwen35::cpu::Cpu::forward`.
    pub fn forward(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let t = tokens.len();
        anyhow::ensure!(t > 0 && t <= self.max_batch, "batch of {t} out of range");
        let base_pos = self.pos;
        anyhow::ensure!(base_pos + t <= self.n_ctx, "context overflow");

        self.gpu
            .queue
            .write_buffer(&self.tokens, 0, bytemuck::cast_slice(tokens));

        if t == 1 && self.debug_stop.is_none() {
            if self.decode.is_none() {
                self.decode = Some(self.build_plan(1)?);
            }
            let plan = self.decode.take().expect("decode plan");
            let r = self.execute(&plan, t, base_pos);
            self.decode = Some(plan);
            r
        } else {
            let plan = self.build_plan(t)?;
            self.execute(&plan, t, base_pos)
        }
    }

    fn execute(&mut self, cp: &CachedPlan, t: usize, base_pos: usize) -> anyhow::Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let mut bytes = Vec::with_capacity(cp.slots.len() * UNIFORM_SLOT as usize);
        for s in &cp.slots {
            s.emit(base_pos as u32, t as u32, &mut bytes);
        }

        if let Some(stop) = self.debug_stop.clone() {
            let (at, buf) = cp
                .checkpoints
                .iter()
                .find(|(n, _, _)| *n == stop)
                .map(|(_, i, b)| (*i, *b))
                .ok_or_else(|| anyhow::anyhow!("unknown checkpoint {stop:?}"))?;
            self.gpu.queue.write_buffer(&self.uniforms, 0, &bytes);
            self.run(&cp.plan[..at]);
            self.pos += t;
            let b = self.debug_buffer(buf);
            let n = (b.size() / 4) as usize;
            return Ok(self.gpu.read_f32(b, n));
        }

        self.gpu.queue.write_buffer(&self.uniforms, 0, &bytes);
        self.run(&cp.plan);

        // Only the last token's row feeds the output projection.
        self.copy_at(&self.h, ((t - 1) * d * 4) as u64, &self.x, 0, (d * 4) as u64);

        let mut tail_bytes = Vec::new();
        for s in &cp.tail_slots {
            s.emit(base_pos as u32, t as u32, &mut tail_bytes);
        }
        self.gpu.queue.write_buffer(&self.uniforms, 0, &tail_bytes);
        self.run(&cp.tail);

        self.pos += t;
        let timing = std::env::var("LLMOXIDE_TIMING").is_ok();
        if timing {
            // Force the queued compute to finish before timing the readback,
            // so the two phases are attributed separately.
            let t_c = std::time::Instant::now();
            self.gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            eprintln!("  compute {:?}", t_c.elapsed());
        }
        let t_r = std::time::Instant::now();
        let logits = self.gpu.read_f32(&self.logits, self.cfg.vocab);
        if timing {
            eprintln!("  readback {:?}", t_r.elapsed());
        }
        Ok(logits)
    }

    fn debug_buffer(&self, i: usize) -> &wgpu::Buffer {
        match i {
            1 => &self.x,
            2 => &self.fused,
            3 => &self.conv,
            4 => &self.z,
            5 => &self.alpha,
            6 => &self.beta,
            7 => &self.q,
            8 => &self.gate_a,
            9 => &self.k,
            10 => &self.v,
            11 => &self.attn,
            12 => &self.proj,
            13 => &self.gate,
            14 => &self.up,
            _ => &self.h,
        }
    }

    /// Dispatches per submitted command buffer. One giant buffer trips Metal's
    /// GPU watchdog — the work is killed and every buffer silently reads back
    /// zeros — both on the first cold-pipeline run and on long prefills, where
    /// this model's 25 GB of weight traffic adds up to tens of seconds.
    /// Ordering between submits is still guaranteed, so splitting is free
    /// apart from ~0.1 ms of per-submit overhead.
    const CHUNK: usize = 64;

    fn run(&self, plan: &[Dispatch]) {
        for chunk in plan.chunks(Self::CHUNK) {
            let mut enc = self
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                for d in chunk {
                    pass.set_pipeline(&d.pipeline);
                    pass.set_bind_group(0, &d.bind, &[d.offset]);
                    pass.dispatch_workgroups(d.groups.0, d.groups.1, d.groups.2);
                }
            }
            self.gpu.queue.submit([enc.finish()]);
        }
    }

    fn copy_at(&self, src: &wgpu::Buffer, so: u64, dst: &wgpu::Buffer, dof: u64, bytes: u64) {
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, so, dst, dof, bytes);
        self.gpu.queue.submit([enc.finish()]);
    }
}
