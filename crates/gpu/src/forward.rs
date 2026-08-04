//! The gemma4 forward pass on wgpu.
//!
//! Mirrors `model::cpu` step for step — that implementation is byte-exact
//! against llama.cpp, so any divergence here is a kernel bug, not an
//! architecture question. See ARCHITECTURE.md for the graph.

use std::collections::HashMap;

use gguf::Gguf;
use model::config::Config;
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

/// All F32 model tensors concatenated into one buffer, addressed by offset.
struct ParamArena {
    data: Vec<f32>,
    offsets: HashMap<String, u32>,
}

impl ParamArena {
    fn build(g: &Gguf, cfg: &Config) -> anyhow::Result<Self> {
        let mut data = Vec::new();
        let mut offsets = HashMap::new();
        let mut push = |name: &str, values: &[f32], offsets: &mut HashMap<String, u32>, data: &mut Vec<f32>| {
            offsets.insert(name.to_string(), data.len() as u32);
            data.extend_from_slice(values);
        };

        for name in ["output_norm.weight", "rope_freqs.weight"] {
            if let Some(t) = g.tensor_opt(name) {
                push(name, &t.to_f32(), &mut offsets, &mut data);
            }
        }
        for i in 0..cfg.n_layers {
            for suffix in [
                "attn_norm.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "post_attention_norm.weight",
                "ffn_norm.weight",
                "post_ffw_norm.weight",
            ] {
                let name = format!("blk.{i}.{suffix}");
                let t = g.tensor(&name)?;
                push(&name, &t.to_f32(), &mut offsets, &mut data);
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
    geglu: wgpu::ComputePipeline,
    add_scale: wgpu::ComputePipeline,
    scale: wgpu::ComputePipeline,
    copy: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    soft_cap: wgpu::ComputePipeline,
    write_cache: wgpu::ComputePipeline,

    attn_layout: wgpu::BindGroupLayout,
    scores: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    weighted_v: wgpu::ComputePipeline,
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
            label: Some("ops"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(3, storage(false)),
                entry(4, uniform),
            ],
        });
        let attn_layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attn"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(3, storage(false)),
                entry(4, storage(false)),
                entry(5, uniform),
            ],
        });

        let build = |name: &str, src: &str, layout: &wgpu::BindGroupLayout, entries: &[&str]| {
            let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
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
            "ops",
            include_str!("shaders/ops.wgsl"),
            &ops_layout,
            &[
                "rms_norm",
                "geglu",
                "add_scale",
                "scale",
                "copy",
                "rope",
                "soft_cap",
                "write_cache",
            ],
        )
        .into_iter();
        let mut at = build(
            "attn",
            include_str!("shaders/attn.wgsl"),
            &attn_layout,
            &["scores_pass", "softmax_pass", "weighted_v"],
        )
        .into_iter();

        Self {
            rms_norm: ops.next().unwrap(),
            geglu: ops.next().unwrap(),
            add_scale: ops.next().unwrap(),
            scale: ops.next().unwrap(),
            copy: ops.next().unwrap(),
            rope: ops.next().unwrap(),
            soft_cap: ops.next().unwrap(),
            write_cache: ops.next().unwrap(),
            ops_layout,
            scores: at.next().unwrap(),
            softmax: at.next().unwrap(),
            weighted_v: at.next().unwrap(),
            attn_layout,
        }
    }
}

/// Per-layer KV storage. Sliding-window layers get a ring of `window` rows.
struct LayerCache {
    k: wgpu::Buffer,
    v: wgpu::Buffer,
}

pub struct GpuModel {
    gpu: Gpu,
    cfg: Config,
    weights: Weights,
    quant: QuantKernels,
    pipes: Pipelines,

    params: wgpu::Buffer,
    param_off: ParamArena,

    // Activations, sized for the largest layer geometry.
    h: wgpu::Buffer,
    x: wgpu::Buffer,
    q: wgpu::Buffer,
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

    cache: Vec<LayerCache>,
    uniforms: wgpu::Buffer,
    /// Per-layer scalar applied to the whole residual stream after the FFN.
    layer_scales: Vec<f32>,

    max_batch: usize,
    n_ctx: usize,
    pub pos: usize,
    /// When set, `forward` stops at the named checkpoint and returns that
    /// buffer's contents instead of logits, for bisecting against `model::cpu`.
    pub debug_stop: Option<String>,
    /// Cached single-token plan. Building it costs ~800 bind groups, which
    /// dominates decode if repeated every step.
    decode: Option<CachedPlan>,
}

impl GpuModel {
    pub fn load(
        gpu: Gpu,
        g: &Gguf,
        cfg: Config,
        n_ctx: usize,
        max_batch: usize,
    ) -> anyhow::Result<Self> {
        // Quantized tensors go to the arena; F32 norms to the param buffer.
        let quant_names: Vec<String> = g
            .tensors
            .iter()
            .filter(|t| t.ty.is_quantized())
            .map(|t| t.name.clone())
            .collect();
        let weights = Weights::upload(&gpu, g, quant_names)?;

        let param_off = ParamArena::build(g, &cfg)?;
        let params = gpu.upload_f32("params", &param_off.data);

        let d = cfg.d_model;
        let head_dim = cfg.layers.iter().map(|l| l.head_dim).max().unwrap();
        let kv_dim = cfg.layers.iter().map(|l| l.kv_dim()).max().unwrap();
        let q_dim = cfg.n_heads * head_dim;
        let f32s = |n: usize| (n * 4) as u64;

        // Scores are bounded by the window on SWA layers but by the context on
        // global ones, so this is the dominant scratch allocation.
        let max_vis = n_ctx;
        let scores_len = max_batch * cfg.n_heads * max_vis;

        let cache = (0..cfg.n_layers)
            .map(|i| {
                let slots = cfg.kv_slots(i, n_ctx);
                let bytes = f32s(slots * cfg.layers[i].kv_dim());
                LayerCache {
                    k: gpu.storage(&format!("k{i}"), bytes),
                    v: gpu.storage(&format!("v{i}"), bytes),
                }
            })
            .collect();

        let layer_scales = (0..cfg.n_layers)
            .map(|i| {
                let t = g.tensor(&format!("blk.{i}.layer_output_scale.weight"))?;
                Ok(t.to_f32()[0])
            })
            .collect::<anyhow::Result<Vec<f32>>>()?;

        let uniforms = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: UNIFORM_SLOT * 2048,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            h: gpu.storage("h", f32s(max_batch * d)),
            x: gpu.storage("x", f32s(max_batch * d)),
            q: gpu.storage("q", f32s(max_batch * q_dim)),
            k: gpu.storage("k", f32s(max_batch * kv_dim)),
            v: gpu.storage("v", f32s(max_batch * kv_dim)),
            attn: gpu.storage("attn", f32s(max_batch * q_dim)),
            proj: gpu.storage("proj", f32s(max_batch * d)),
            gate: gpu.storage("gate", f32s(max_batch * cfg.ffn_dim)),
            up: gpu.storage("up", f32s(max_batch * cfg.ffn_dim)),
            scores: gpu.storage("scores", f32s(scores_len)),
            logits: gpu.storage("logits", f32s(cfg.vocab)),
            tokens: gpu.storage("tokens", f32s(max_batch)),
            zero: gpu.upload_f32("zero", &[0.0]),
            params,
            param_off,
            quant: QuantKernels::new(&gpu),
            pipes: Pipelines::new(&gpu),
            weights,
            cache,
            uniforms,
            layer_scales,
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

    /// Largest prefill batch the scratch buffers were sized for.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

}

/// One uniform slot. Decode reuses a cached dispatch plan across tokens, so
/// anything that varies with position is stored symbolically and materialized
/// per call rather than baked into bytes.
enum Slot {
    Fixed(Vec<u8>),
    /// `u1` becomes the batch's base position.
    Positioned(Op),
    /// `base_pos` and the visible-window length are recomputed per call.
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
                let end = base_pos + n_tokens;
                a.max_vis = if a.window == 0 { end } else { a.window.min(end) };
                buf.extend_from_slice(bytemuck::bytes_of(&a));
            }
        }
        buf.resize(start + UNIFORM_SLOT as usize, 0);
    }
}

/// A dispatch plan plus its uniform slots, reusable across decode steps.
struct CachedPlan {
    plan: Vec<Dispatch>,
    slots: Vec<Slot>,
    tail: Vec<Dispatch>,
    tail_slots: Vec<Slot>,
    checkpoints: Vec<(String, usize, usize)>,
}

/// A recorded dispatch: the pipeline, its bind group, and the grid.
struct Dispatch {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    offset: u32,
    groups: (u32, u32, u32),
}

const WG_OPS: u32 = 256;

/// Workgroups for an elementwise pass, clamped to the per-dimension dispatch
/// limit. The kernels grid-stride, so a clamped grid still covers the range.
fn cells(n: u32, wg: u32) -> u32 {
    n.div_ceil(wg).clamp(1, 65535)
}

impl GpuModel {
    fn ops_bind(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipes.ops_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.uniforms,
                            offset: 0,
                            size: std::num::NonZeroU64::new(UNIFORM_SLOT),
                        }),
                    },
                ],
            })
    }

    fn matvec_bind(
        &self,
        w_buffer: usize,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        params: &wgpu::Buffer,
        tokens: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.quant.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.weights.buffers[w_buffer].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: y.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        // One slot, so the dynamic offset can walk the buffer.
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: params,
                            offset: 0,
                            size: std::num::NonZeroU64::new(UNIFORM_SLOT),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: tokens.as_entire_binding(),
                    },
                ],
            })
    }
}

impl GpuModel {
    fn build_plan(&self, t: usize) -> anyhow::Result<CachedPlan> {
        let cfg = self.cfg.clone();
        let d = cfg.d_model;
        let max_groups = self.gpu.limits.max_compute_workgroups_per_dimension;

        let mut u: Vec<Slot> = Vec::new();
        let mut plan: Vec<Dispatch> = Vec::new();
        let mut checkpoints: Vec<(String, usize, usize)> = Vec::new();

        // Slots are 256-byte aligned so a dynamic offset can select one,
        // rather than allocating a uniform buffer per dispatch.
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

        // --- embeddings ----------------------------------------------------
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
            pipeline: self.quant.embed.clone(),
            bind: self.matvec_bind(embd.buffer, &self.zero, &self.h, &self.uniforms, &self.tokens),
            offset: off,
            groups: (16, t as u32, 1),
        });

        let off = op!(u, n_rows: t as u32, dim: d as u32, f0: cfg.embed_scale,);
        plan.push(Dispatch {
            pipeline: self.pipes.scale.clone(),
            bind: self.ops_bind(&self.zero, &self.zero, &self.h),
            offset: off,
            groups: (cells((t * d) as u32, WG_OPS), 1, 1),
        });
        checkpoints.push(("inp_scaled".into(), plan.len(), 0));

        for il in 0..cfg.n_layers {
            let lc = &cfg.layers[il];
            let head_dim = lc.head_dim as u32;
            let n_kv = lc.n_kv_heads as u32;
            let kv_dim = lc.kv_dim() as u32;
            let q_dim = (cfg.n_heads * lc.head_dim) as u32;
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

            // Q, K, and V projections. On global layers V has no weight of its
            // own and reuses the K projection output.
            for (name, dst, out_dim) in [
                (p("attn_q.weight"), &self.q, q_dim),
                (p("attn_k.weight"), &self.k, kv_dim),
            ] {
                let h = self.weights.get(&name)?;
                let off = slot(
                    &mut u,
                    bytemuck::bytes_of(&MatvecParams {
                        w_base: h.base_u32,
                        in_dim: d as u32,
                        out_dim,
                        n_tokens: t as u32,
                    }),
                );
                plan.push(Dispatch {
                    pipeline: self.quant.pipeline_for(h.ty)?.clone(),
                    bind: self.matvec_bind(h.buffer, &self.x, dst, &self.uniforms, &self.tokens),
                    offset: off,
                    groups: (crate::row_groups(out_dim, max_groups), crate::token_groups(t as u32), 1),
                });
            }
            if lc.v_from_k {
                // Global layers have no attn_v: V is a copy of the raw K
                // projection, which diverges only in the normalization below.
                let off = op!(u, n_rows: t as u32, dim: kv_dim,);
                plan.push(Dispatch {
                    pipeline: self.pipes.copy.clone(),
                    bind: self.ops_bind(&self.k, &self.zero, &self.v),
                    offset: off,
                    groups: (cells(t as u32 * kv_dim, WG_OPS), 1, 1),
                });
            } else {
                let h = self.weights.get(&p("attn_v.weight"))?;
                let off = slot(
                    &mut u,
                    bytemuck::bytes_of(&MatvecParams {
                        w_base: h.base_u32,
                        in_dim: d as u32,
                        out_dim: kv_dim,
                        n_tokens: t as u32,
                    }),
                );
                plan.push(Dispatch {
                    pipeline: self.quant.pipeline_for(h.ty)?.clone(),
                    bind: self.matvec_bind(h.buffer, &self.x, &self.v, &self.uniforms, &self.tokens),
                    offset: off,
                    groups: (crate::row_groups(kv_dim, max_groups), crate::token_groups(t as u32), 1),
                });
            }

            checkpoints.push((format!("Vcur-{il}"), plan.len(), 4));
            checkpoints.push((format!("Qcur-{il}"), plan.len(), 2));

            // Per-head QK norms, then RoPE. V gets a bare norm and no rotation.
            for (buf, gain, heads) in [
                (&self.q, Some(p("attn_q_norm.weight")), cfg.n_heads as u32),
                (&self.k, Some(p("attn_k_norm.weight")), n_kv),
                (&self.v, None, n_kv),
            ] {
                let (off0, u0) = match &gain {
                    Some(n) => (self.param_off.at(n)?, 1),
                    None => (0, 0),
                };
                let off = op!(u,
                    n_rows: t as u32 * heads, dim: head_dim,
                    off0: off0, f0: cfg.rms_eps, u0: u0,
                );
                plan.push(Dispatch {
                    pipeline: self.pipes.rms_norm.clone(),
                    bind: self.ops_bind(&self.zero, &self.zero, buf),
                    offset: off,
                    groups: (cells(t as u32 * heads, 1), 1, 1),
                });
            }

            checkpoints.push((format!("Vcur_normed-{il}"), plan.len(), 4));
            checkpoints.push((format!("Qcur_normed-{il}"), plan.len(), 2));

            let rope_off = self.param_off.at("rope_freqs.weight").unwrap_or(0);
            for (buf, heads) in [(&self.q, cfg.n_heads as u32), (&self.k, n_kv)] {
                let off = slot_of(&mut u, Slot::Positioned(Op {
                    n_rows: t as u32 * heads, dim: head_dim,
                    off0: rope_off,
                    off1: u32::from(lc.rope_factors),
                    f0: lc.rope_base,
                    u0: heads,
                    ..Default::default()
                }));
                plan.push(Dispatch {
                    pipeline: self.pipes.rope.clone(),
                    bind: self.ops_bind(&self.zero, &self.zero, buf),
                    offset: off,
                    groups: (cells(t as u32 * heads, 1), 1, 1),
                });
            }

            checkpoints.push((format!("Qcur_pos-{il}"), plan.len(), 2));

            // KV cache write, then the three attention passes.
            let window = lc.window.map_or(0, |w| w.min(self.n_ctx)) as u32;
            for (src, dst) in [(&self.k, 0), (&self.v, 1)] {
                let off = slot_of(&mut u, Slot::Positioned(Op {
                    n_rows: t as u32, dim: kv_dim, u0: window,
                    ..Default::default()
                }));
                let cache = if dst == 0 {
                    &self.cache[il].k
                } else {
                    &self.cache[il].v
                };
                plan.push(Dispatch {
                    pipeline: self.pipes.write_cache.clone(),
                    bind: self.ops_bind(src, &self.zero, cache),
                    offset: off,
                    groups: (cells(t as u32 * kv_dim, WG_OPS), 1, 1),
                });
            }

            let attn_off = slot_of(&mut u, Slot::Attention(Attn {
                n_tokens: t as u32,
                n_heads: cfg.n_heads as u32,
                n_kv_heads: n_kv,
                head_dim,
                kv_dim,
                window,
                scale: cfg.attn_scale(il),
                ..Default::default()
            }));
            let attn_bind = self.attn_bind(il);
            for pipeline in [
                &self.pipes.scores,
                &self.pipes.softmax,
                &self.pipes.weighted_v,
            ] {
                plan.push(Dispatch {
                    pipeline: pipeline.clone(),
                    bind: attn_bind.clone(),
                    offset: attn_off,
                    groups: (cfg.n_heads as u32, t as u32, 1),
                });
            }

            checkpoints.push((format!("kqv_out-{il}"), plan.len(), 5));

            // output projection, post-norm, residual
            let ho = self.weights.get(&p("attn_output.weight"))?;
            let off = slot(
                &mut u,
                bytemuck::bytes_of(&MatvecParams {
                    w_base: ho.base_u32,
                    in_dim: q_dim,
                    out_dim: d as u32,
                    n_tokens: t as u32,
                }),
            );
            plan.push(Dispatch {
                pipeline: self.quant.pipeline_for(ho.ty)?.clone(),
                bind: self.matvec_bind(ho.buffer, &self.attn, &self.proj, &self.uniforms, &self.tokens),
                offset: off,
                groups: (crate::row_groups(d as u32, max_groups), crate::token_groups(t as u32), 1),
            });

            let off = op!(u,
                n_rows: t as u32, dim: d as u32,
                off0: self.param_off.at(&p("post_attention_norm.weight"))?,
                f0: cfg.rms_eps, u0: 1,
            );
            plan.push(Dispatch {
                pipeline: self.pipes.rms_norm.clone(),
                bind: self.ops_bind(&self.zero, &self.zero, &self.proj),
                offset: off,
                groups: (cells(t as u32, 1), 1, 1),
            });

            let off = op!(u, n_rows: t as u32, dim: d as u32, f0: 1.0,);
            plan.push(Dispatch {
                pipeline: self.pipes.add_scale.clone(),
                bind: self.ops_bind(&self.zero, &self.proj, &self.h),
                offset: off,
                groups: (cells((t * d) as u32, WG_OPS), 1, 1),
            });

            checkpoints.push((format!("attn_out-{il}"), plan.len(), 0));

            // feed-forward
            let off = op!(u, n_rows: t as u32, dim: d as u32,);
            plan.push(Dispatch {
                pipeline: self.pipes.copy.clone(),
                bind: self.ops_bind(&self.h, &self.zero, &self.x),
                offset: off,
                groups: (cells((t * d) as u32, WG_OPS), 1, 1),
            });
            let off = op!(u,
                n_rows: t as u32, dim: d as u32,
                off0: self.param_off.at(&p("ffn_norm.weight"))?,
                f0: cfg.rms_eps, u0: 1,
            );
            plan.push(Dispatch {
                pipeline: self.pipes.rms_norm.clone(),
                bind: self.ops_bind(&self.zero, &self.zero, &self.x),
                offset: off,
                groups: (cells(t as u32, 1), 1, 1),
            });

            for (name, dst) in [(p("ffn_gate.weight"), &self.gate), (p("ffn_up.weight"), &self.up)] {
                let hw = self.weights.get(&name)?;
                let off = slot(
                    &mut u,
                    bytemuck::bytes_of(&MatvecParams {
                        w_base: hw.base_u32,
                        in_dim: d as u32,
                        out_dim: cfg.ffn_dim as u32,
                        n_tokens: t as u32,
                    }),
                );
                plan.push(Dispatch {
                    pipeline: self.quant.pipeline_for(hw.ty)?.clone(),
                    bind: self.matvec_bind(hw.buffer, &self.x, dst, &self.uniforms, &self.tokens),
                    offset: off,
                    groups: (crate::row_groups(cfg.ffn_dim as u32, max_groups), crate::token_groups(t as u32), 1),
                });
            }

            let off = op!(u, n_rows: t as u32, dim: cfg.ffn_dim as u32,);
            plan.push(Dispatch {
                pipeline: self.pipes.geglu.clone(),
                bind: self.ops_bind(&self.zero, &self.up, &self.gate),
                offset: off,
                groups: (cells((t * cfg.ffn_dim) as u32, WG_OPS), 1, 1),
            });

            checkpoints.push((format!("ffn_geglu-{il}"), plan.len(), 7));

            let hd = self.weights.get(&p("ffn_down.weight"))?;
            let off = slot(
                &mut u,
                bytemuck::bytes_of(&MatvecParams {
                    w_base: hd.base_u32,
                    in_dim: cfg.ffn_dim as u32,
                    out_dim: d as u32,
                    n_tokens: t as u32,
                }),
            );
            plan.push(Dispatch {
                pipeline: self.quant.pipeline_for(hd.ty)?.clone(),
                bind: self.matvec_bind(hd.buffer, &self.gate, &self.proj, &self.uniforms, &self.tokens),
                offset: off,
                groups: (crate::row_groups(d as u32, max_groups), crate::token_groups(t as u32), 1),
            });

            let off = op!(u,
                n_rows: t as u32, dim: d as u32,
                off0: self.param_off.at(&p("post_ffw_norm.weight"))?,
                f0: cfg.rms_eps, u0: 1,
            );
            plan.push(Dispatch {
                pipeline: self.pipes.rms_norm.clone(),
                bind: self.ops_bind(&self.zero, &self.zero, &self.proj),
                offset: off,
                groups: (cells(t as u32, 1), 1, 1),
            });

            // residual add fused with the layer's output scale
            let off = op!(u,
                n_rows: t as u32, dim: d as u32,
                f0: self.layer_scale(il),
            );
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

        checkpoints.push(("h_nextn".into(), plan.len(), 0));

        let out = self.weights.get("token_embd.weight")?;
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
        let cap_off = slot(
            &mut tail,
            bytemuck::bytes_of(&Op {
                dim: cfg.vocab as u32,
                f0: cfg.logit_softcap.unwrap_or(0.0),
                ..Default::default()
            }),
        );

        let mut tail_plan = vec![Dispatch {
            pipeline: self.quant.pipeline_for(out.ty)?.clone(),
            bind: self.matvec_bind(out.buffer, &self.x, &self.logits, &self.uniforms, &self.tokens),
            offset: off,
            groups: (crate::row_groups(cfg.vocab as u32, max_groups), 1, 1),
        }];
        if cfg.logit_softcap.is_some() {
            tail_plan.push(Dispatch {
                pipeline: self.pipes.soft_cap.clone(),
                bind: self.ops_bind(&self.zero, &self.zero, &self.logits),
                offset: cap_off,
                groups: (cells(cfg.vocab as u32, WG_OPS), 1, 1),
            });
        }

        Ok(CachedPlan {
            plan,
            slots: u,
            tail: tail_plan,
            tail_slots: tail,
            checkpoints,
        })
    }

    /// Run `tokens` starting at the current position and return the final
    /// logits. Mirrors `model::cpu::Cpu::forward`.
    pub fn forward(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let t = tokens.len();
        anyhow::ensure!(t > 0 && t <= self.max_batch, "batch of {t} out of range");
        let base_pos = self.pos;
        anyhow::ensure!(base_pos + t <= self.n_ctx, "context overflow");

        self.gpu
            .queue
            .write_buffer(&self.tokens, 0, bytemuck::cast_slice(tokens));

        // Decode is one token at a time with a fixed shape, so its plan — and
        // the ~800 bind groups in it — is built once and reused. Prefill
        // batches vary in length and rebuild.
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
        let t_sync = std::time::Instant::now();
        if timing {
            // Force completion of the compute before timing the readback, so
            // the two phases are attributed separately.
            self.gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            eprintln!("  compute {:?}", t_sync.elapsed());
        }
        let t_read = std::time::Instant::now();
        let mut logits = self.gpu.read_f32(&self.logits, self.cfg.vocab);
        if timing {
            eprintln!("  readback {:?}", t_read.elapsed());
        }
        for &tok in &self.cfg.suppress_tokens {
            if let Some(l) = logits.get_mut(tok as usize) {
                *l = f32::NEG_INFINITY;
            }
        }
        Ok(logits)
    }

    fn debug_buffer(&self, i: usize) -> &wgpu::Buffer {
        match i {
            1 => &self.x,
            2 => &self.q,
            3 => &self.k,
            4 => &self.v,
            5 => &self.attn,
            6 => &self.proj,
            7 => &self.gate,
            _ => &self.h,
        }
    }

    fn layer_scale(&self, il: usize) -> f32 {
        self.layer_scales[il]
    }

    fn run(&self, plan: &[Dispatch]) {
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for d in plan {
                pass.set_pipeline(&d.pipeline);
                pass.set_bind_group(0, &d.bind, &[d.offset]);
                pass.dispatch_workgroups(d.groups.0, d.groups.1, d.groups.2);
            }
        }
        self.gpu.queue.submit([enc.finish()]);
    }

    fn copy_at(&self, src: &wgpu::Buffer, so: u64, dst: &wgpu::Buffer, dof: u64, bytes: u64) {
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, so, dst, dof, bytes);
        self.gpu.queue.submit([enc.finish()]);
    }

    fn attn_bind(&self, il: usize) -> wgpu::BindGroup {
        self.gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipes.attn_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.q.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.cache[il].k.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.cache[il].v.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.scores.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.attn.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.uniforms,
                            offset: 0,
                            size: std::num::NonZeroU64::new(UNIFORM_SLOT),
                        }),
                    },
                ],
            })
    }
}
