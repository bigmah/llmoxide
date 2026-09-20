//! The gemma4v vision tower on the GPU.
//!
//! Mirrors `llmoxide-vision`'s CPU forward step for step; that implementation
//! is the oracle this one is checked against, exactly as `forward.rs` is
//! checked against `model::cpu`.
//!
//! Most of it is the text stack's kernels pointed at patches instead of
//! tokens. Three things are specific enough to be worth naming:
//!
//! - **Attention reuses `attn.wgsl` unchanged.** A tower has no KV cache, but
//!   a cache with `window = 0` addressed at `base_pos = 0` *is* a flat
//!   `[n_patches, kv_dim]` buffer, and the `bidi` flag added for image spans
//!   in the text model already makes every query see every key. So K and V are
//!   bound straight into the cache slots and the mask comes out right.
//! - **The patch convolution is a matmul.** With the image lowered to
//!   `[n_patches, 16*16*3]` on the way in (`im2col`), the 4-D filter is a
//!   row-major `[768, 768]` weight and `matvec_f32` is the convolution. The
//!   GGUF's declared shape is never consulted by the kernel — only the
//!   `MatvecParams` this file builds.
//! - **Positions are gathered on the CPU.** The lookup tables are 63 MB of
//!   F32 for 10 240 positions, of which one image uses a few dozen; uploading
//!   the rows it actually needs beats uploading the table.
//!
//! Scratch is allocated per image rather than reserved for the largest one:
//! the scores buffer alone is `n^2 * n_heads` floats, a quarter of a gigabyte
//! at the token ceiling and a fifth of that for a typical photo. This runs
//! once per image, so the allocation is not on any hot path.

use std::collections::HashMap;

use gguf::{Gguf, GgmlType};
use model::vision::Config;

use crate::{Gpu, MatvecParams, QuantKernels, Weights};

const UNIFORM_SLOT: u64 = 256;
const WG_OPS: u32 = 256;

fn cells(n: u32, wg: u32) -> u32 {
    n.div_ceil(wg).max(1)
}

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
    bidi: u32,
    _pad: [u32; 2],
}

/// Calibration range carried beside a weight (`Gemma4ClippableLinear`).
#[derive(Clone, Copy)]
struct Clamp {
    in_lo: f32,
    in_hi: f32,
    out_lo: f32,
    out_hi: f32,
}

impl Clamp {
    fn any(&self) -> bool {
        self.in_lo.is_finite()
            || self.in_hi.is_finite()
            || self.out_lo.is_finite()
            || self.out_hi.is_finite()
    }
    /// Infinities are legal in WGSL but needlessly close to the edge of what
    /// a driver will constant-fold; the finite extremes clamp identically.
    fn finite(lo: f32, hi: f32) -> (f32, f32) {
        (
            if lo.is_finite() { lo } else { f32::MIN },
            if hi.is_finite() { hi } else { f32::MAX },
        )
    }
}

struct Pipelines {
    ops_layout: wgpu::BindGroupLayout,
    rms_norm: wgpu::ComputePipeline,
    geglu: wgpu::ComputePipeline,
    add_scale: wgpu::ComputePipeline,
    copy: wgpu::ComputePipeline,
    rope_2d: wgpu::ComputePipeline,
    clamp_range: wgpu::ComputePipeline,
    pool_avg: wgpu::ComputePipeline,

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
            label: Some("vision-ops"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(3, storage(false)),
                entry(4, uniform),
            ],
        });
        let attn_layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vision-attn"),
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
            "ops",
            include_str!("shaders/ops.wgsl"),
            &ops_layout,
            &[
                "rms_norm",
                "geglu",
                "add_scale",
                "copy",
                "rope_2d",
                "clamp_range",
                "pool_avg",
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
            copy: ops.next().unwrap(),
            rope_2d: ops.next().unwrap(),
            clamp_range: ops.next().unwrap(),
            pool_avg: ops.next().unwrap(),
            ops_layout,
            scores: at.next().unwrap(),
            softmax: at.next().unwrap(),
            weighted_v: at.next().unwrap(),
            attn_layout,
        }
    }
}

struct Dispatch {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    offset: u32,
    groups: (u32, u32, u32),
}

/// Scratch for one image, sized to it.
struct Scratch {
    x: wgpu::Buffer,
    h: wgpu::Buffer,
    pos: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    attn: wgpu::Buffer,
    proj: wgpu::Buffer,
    gate: wgpu::Buffer,
    up: wgpu::Buffer,
    scores: wgpu::Buffer,
    pooled: wgpu::Buffer,
    out: wgpu::Buffer,
}

pub struct VisionGpu {
    gpu: Gpu,
    g: Gguf,
    pub cfg: Config,
    weights: Weights,
    quant: QuantKernels,
    pipes: Pipelines,
    /// Every F32 norm gain, concatenated; `off0` in an `Op` indexes it.
    params: wgpu::Buffer,
    param_off: HashMap<String, u32>,
    clamp: HashMap<String, Clamp>,
    /// Bound where the text model binds token ids. Nothing here reads it.
    zero: wgpu::Buffer,
}

impl VisionGpu {
    pub fn load(gpu: Gpu, g: Gguf, cfg: Config) -> anyhow::Result<Self> {
        // The quantized/BF16 side: every projection, plus the patch filter,
        // which is F32 but is consumed as a [768, 768] matmul.
        let mut names = vec![
            "v.patch_embd.weight".to_string(),
            "mm.input_projection.weight".to_string(),
        ];
        for il in 0..cfg.n_layers {
            for t in [
                "attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up", "ffn_down",
            ] {
                names.push(format!("v.blk.{il}.{t}.weight"));
            }
        }
        let weights = Weights::upload(&gpu, &g, names)?;

        // The F32 side: norm gains, small enough to live in one flat buffer.
        let mut data: Vec<f32> = Vec::new();
        let mut param_off = HashMap::new();
        for il in 0..cfg.n_layers {
            for t in [
                "ln1",
                "ln2",
                "attn_post_norm",
                "ffn_post_norm",
                "attn_q_norm",
                "attn_k_norm",
            ] {
                let name = format!("v.blk.{il}.{t}.weight");
                let t = g.tensor(&name)?;
                let v = t
                    .as_f32()
                    .ok_or_else(|| anyhow::anyhow!("{name} is not F32"))?;
                param_off.insert(name, data.len() as u32);
                data.extend_from_slice(v);
            }
        }
        let params = gpu.upload_f32("vision-params", &data);

        Ok(Self {
            clamp: clamp_table(&g),
            quant: QuantKernels::new(&gpu),
            pipes: Pipelines::new(&gpu),
            zero: gpu.upload_u32("vision-zero", &[0u32]),
            params,
            param_off,
            weights,
            gpu,
            g,
            cfg,
        })
    }

    fn gain(&self, name: &str) -> anyhow::Result<u32> {
        self.param_off
            .get(name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("gain {name:?} not resident"))
    }

    /// Lower the image to one row per patch: `[c][ky][kx]` within a row, which
    /// is the filter's own layout, so the convolution becomes a matmul.
    fn im2col(&self, img: &[f32], w: usize, h: usize, nx: usize, ny: usize) -> Vec<f32> {
        let p = self.cfg.patch_size;
        let plane = w * h;
        let row = p * p * 3;
        let mut out = vec![0f32; nx * ny * row];
        for pi in 0..nx * ny {
            let (px, py) = (pi % nx, pi / nx);
            let dst = pi * row;
            for c in 0..3 {
                for ky in 0..p {
                    let src = c * plane + (py * p + ky) * w + px * p;
                    let d = dst + c * p * p + ky * p;
                    out[d..d + p].copy_from_slice(&img[src..src + p]);
                }
            }
        }
        out
    }

    /// The learned `(x, y)` vectors for this grid, summed per patch.
    fn positions(&self, nx: usize, ny: usize) -> anyhow::Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let len = self.cfg.pos_table_len;
        let t = self.g.tensor("v.position_embd.weight")?;
        let p = t
            .as_f32()
            .ok_or_else(|| anyhow::anyhow!("v.position_embd.weight is not F32"))?;
        let mut out = vec![0f32; nx * ny * d];
        for pi in 0..nx * ny {
            let (col, row) = (pi % nx, pi / nx);
            let (tx, ty) = (col * d, (len + row) * d);
            let dst = pi * d;
            for k in 0..d {
                out[dst + k] = p[tx + k] + p[ty + k];
            }
        }
        Ok(out)
    }

    fn alloc(&self, n: usize, n_out: usize) -> Scratch {
        let c = &self.cfg;
        let f32s = |x: usize| (x * 4) as u64;
        let s = |label: &str, x: usize| self.gpu.storage(label, f32s(x).max(4));
        Scratch {
            x: s("v.x", n * c.d_model),
            h: s("v.h", n * c.d_model),
            pos: s("v.pos", n * c.d_model),
            q: s("v.q", n * c.d_model),
            k: s("v.k", n * c.d_model),
            v: s("v.v", n * c.d_model),
            attn: s("v.attn", n * c.d_model),
            proj: s("v.proj", n * c.d_model),
            gate: s("v.gate", n * c.ffn_dim),
            up: s("v.up", n * c.ffn_dim),
            scores: s("v.scores", n * c.n_heads * n),
            pooled: s("v.pooled", n_out * c.d_model),
            out: s("v.out", n_out * c.proj_dim),
        }
    }

    /// Encode one prepared image into residual-stream rows.
    ///
    /// `img` is channel-planar `[c][y][x]` in `[-1, 1]` — what
    /// `llmoxide-vision`'s preprocessing produces.
    pub fn encode(&self, img: &[f32], w: usize, h: usize) -> anyhow::Result<Vec<f32>> {
        let c = &self.cfg;
        let (nx, ny) = (w / c.patch_size, h / c.patch_size);
        let n = nx * ny;
        anyhow::ensure!(n > 0, "image is smaller than one patch");
        anyhow::ensure!(
            nx % c.n_merge == 0 && ny % c.n_merge == 0,
            "patch grid {nx}x{ny} is not a multiple of the {} pooling kernel",
            c.n_merge
        );
        anyhow::ensure!(
            nx.max(ny) <= c.pos_table_len,
            "patch grid {nx}x{ny} exceeds the {}-entry position table",
            c.pos_table_len
        );
        let (ox, oy) = (nx / c.n_merge, ny / c.n_merge);
        let n_out = ox * oy;

        let sc = self.alloc(n, n_out);
        let q = &self.gpu.queue;
        q.write_buffer(
            &sc.x,
            0,
            bytemuck::cast_slice(&self.im2col(img, w, h, nx, ny)),
        );
        q.write_buffer(&sc.pos, 0, bytemuck::cast_slice(&self.positions(nx, ny)?));

        let uniforms = self.gpu.buffer(
            "vision-uniforms",
            UNIFORM_SLOT * 4096,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let mut u: Vec<u8> = Vec::new();
        let mut plan: Vec<Dispatch> = Vec::new();
        let max_groups = self.gpu.limits.max_compute_workgroups_per_dimension;

        // --- helpers -------------------------------------------------------
        let slot = |u: &mut Vec<u8>, bytes: &[u8]| {
            let off = u.len() as u32;
            u.extend_from_slice(bytes);
            u.resize(off as usize + UNIFORM_SLOT as usize, 0);
            off
        };
        let ops_bind = |a: &wgpu::Buffer, b: &wgpu::Buffer, out: &wgpu::Buffer| {
            self.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipes.ops_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.params.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: a.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &uniforms,
                                offset: 0,
                                size: std::num::NonZeroU64::new(UNIFORM_SLOT),
                            }),
                        },
                    ],
                })
        };
        let mv_bind = |wb: usize, x: &wgpu::Buffer, y: &wgpu::Buffer| {
            self.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.quant.layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.weights.buffers[wb].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &uniforms,
                                offset: 0,
                                size: std::num::NonZeroU64::new(UNIFORM_SLOT),
                            }),
                        },
                        wgpu::BindGroupEntry { binding: 4, resource: self.zero.as_entire_binding() },
                    ],
                })
        };

        macro_rules! op {
            ($($f:tt)*) => { slot(&mut u, bytemuck::bytes_of(&Op { $($f)* ..Default::default() })) };
        }

        // `y = W x`, clamping around it where the weight carries a range.
        macro_rules! linear {
            ($name:expr, $src:expr, $dst:expr, $out_dim:expr, $in_dim:expr) => {{
                let name: String = $name;
                let cl = self.clamp.get(&name).copied();
                if let Some(cl) = cl.filter(Clamp::any) {
                    let (lo, hi) = Clamp::finite(cl.in_lo, cl.in_hi);
                    let off = op!(n_rows: n as u32, dim: $in_dim as u32, f0: lo, f1: hi,);
                    plan.push(Dispatch {
                        pipeline: self.pipes.clamp_range.clone(),
                        bind: ops_bind(&self.zero, &self.zero, $src),
                        offset: off,
                        groups: (cells((n * $in_dim) as u32, WG_OPS), 1, 1),
                    });
                }
                let hw = self.weights.get(&name)?;
                let off = slot(
                    &mut u,
                    bytemuck::bytes_of(&MatvecParams {
                        w_base: hw.base_u32,
                        in_dim: $in_dim as u32,
                        out_dim: $out_dim as u32,
                        n_tokens: n as u32,
                    }),
                );
                plan.push(Dispatch {
                    pipeline: self.quant.pipeline_for(hw.ty, n as u32)?.clone(),
                    bind: mv_bind(hw.buffer, $src, $dst),
                    offset: off,
                    groups: (
                        crate::row_groups($out_dim as u32, max_groups),
                        crate::token_groups(n as u32),
                        1,
                    ),
                });
                if let Some(cl) = cl.filter(Clamp::any) {
                    let (lo, hi) = Clamp::finite(cl.out_lo, cl.out_hi);
                    let off = op!(n_rows: n as u32, dim: $out_dim as u32, f0: lo, f1: hi,);
                    plan.push(Dispatch {
                        pipeline: self.pipes.clamp_range.clone(),
                        bind: ops_bind(&self.zero, &self.zero, $dst),
                        offset: off,
                        groups: (cells((n * $out_dim) as u32, WG_OPS), 1, 1),
                    });
                }
            }};
        }

        let d = c.d_model;
        let hd = c.head_dim;
        let nh = c.n_heads;

        // --- patch embedding + positions -----------------------------------
        linear!("v.patch_embd.weight".to_string(), &sc.x, &sc.h, d, c.patch_size * c.patch_size * 3);
        let off = op!(n_rows: n as u32, dim: d as u32, f0: 1.0,);
        plan.push(Dispatch {
            pipeline: self.pipes.add_scale.clone(),
            bind: ops_bind(&self.zero, &sc.pos, &sc.h),
            offset: off,
            groups: (cells((n * d) as u32, WG_OPS), 1, 1),
        });

        for il in 0..c.n_layers {
            let t = |s: &str| format!("v.blk.{il}.{s}.weight");

            // --- pre-attention norm ----------------------------------------
            let off = op!(n_rows: n as u32, dim: d as u32,);
            plan.push(Dispatch {
                pipeline: self.pipes.copy.clone(),
                bind: ops_bind(&sc.h, &self.zero, &sc.x),
                offset: off,
                groups: (cells((n * d) as u32, WG_OPS), 1, 1),
            });
            let off = op!(n_rows: n as u32, dim: d as u32, off0: self.gain(&t("ln1"))?, f0: c.eps, u0: 1,);
            plan.push(Dispatch {
                pipeline: self.pipes.rms_norm.clone(),
                bind: ops_bind(&self.zero, &self.zero, &sc.x),
                offset: off,
                groups: (cells(n as u32, 1).min(max_groups), 1, 1),
            });

            // --- q / k / v --------------------------------------------------
            linear!(t("attn_q"), &sc.x, &sc.q, d, d);
            linear!(t("attn_k"), &sc.x, &sc.k, d, d);
            linear!(t("attn_v"), &sc.x, &sc.v, d, d);

            // Per-head norms, then the 2-D rotation.
            for (buf, gain) in [(&sc.q, t("attn_q_norm")), (&sc.k, t("attn_k_norm"))] {
                let off = op!(n_rows: (n * nh) as u32, dim: hd as u32, off0: self.gain(&gain)?, f0: c.eps, u0: 1,);
                plan.push(Dispatch {
                    pipeline: self.pipes.rms_norm.clone(),
                    bind: ops_bind(&self.zero, &self.zero, buf),
                    offset: off,
                    groups: (((n * nh) as u32).min(max_groups).max(1), 1, 1),
                });
                let off = op!(n_rows: (n * nh) as u32, dim: hd as u32, f0: c.rope_theta, u0: nh as u32, u1: nx as u32,);
                plan.push(Dispatch {
                    pipeline: self.pipes.rope_2d.clone(),
                    bind: ops_bind(&self.zero, &self.zero, buf),
                    offset: off,
                    groups: (((n * nh) as u32).min(max_groups).max(1), 1, 1),
                });
            }
            // V is normalized per head with no gain and never rotated.
            let off = op!(n_rows: (n * nh) as u32, dim: hd as u32, f0: c.eps,);
            plan.push(Dispatch {
                pipeline: self.pipes.rms_norm.clone(),
                bind: ops_bind(&self.zero, &self.zero, &sc.v),
                offset: off,
                groups: (((n * nh) as u32).min(max_groups).max(1), 1, 1),
            });

            // --- attention ---------------------------------------------------
            // `window = 0` makes the cache index the position directly, so the
            // flat K/V buffers are the cache; `bidi` opens the mask both ways.
            let off = slot(
                &mut u,
                bytemuck::bytes_of(&Attn {
                    n_tokens: n as u32,
                    n_heads: nh as u32,
                    n_kv_heads: nh as u32,
                    head_dim: hd as u32,
                    kv_dim: d as u32,
                    base_pos: 0,
                    window: 0,
                    max_vis: n as u32,
                    scale: 1.0,
                    bidi: 1,
                    _pad: [0; 2],
                }),
            );
            let abind = self
                .gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipes.attn_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: sc.q.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: sc.k.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: sc.v.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: sc.scores.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: sc.attn.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &uniforms,
                                offset: 0,
                                size: std::num::NonZeroU64::new(UNIFORM_SLOT),
                            }),
                        },
                    ],
                });
            for pipeline in [&self.pipes.scores, &self.pipes.softmax, &self.pipes.weighted_v] {
                plan.push(Dispatch {
                    pipeline: pipeline.clone(),
                    bind: abind.clone(),
                    offset: off,
                    groups: (nh as u32, n as u32, 1),
                });
            }

            // --- output projection, post-norm, residual ----------------------
            linear!(t("attn_out"), &sc.attn, &sc.proj, d, d);
            let off = op!(n_rows: n as u32, dim: d as u32, off0: self.gain(&t("attn_post_norm"))?, f0: c.eps, u0: 1,);
            plan.push(Dispatch {
                pipeline: self.pipes.rms_norm.clone(),
                bind: ops_bind(&self.zero, &self.zero, &sc.proj),
                offset: off,
                groups: ((n as u32).min(max_groups).max(1), 1, 1),
            });
            let off = op!(n_rows: n as u32, dim: d as u32, f0: 1.0,);
            plan.push(Dispatch {
                pipeline: self.pipes.add_scale.clone(),
                bind: ops_bind(&self.zero, &sc.proj, &sc.h),
                offset: off,
                groups: (cells((n * d) as u32, WG_OPS), 1, 1),
            });

            // --- feed-forward -------------------------------------------------
            let off = op!(n_rows: n as u32, dim: d as u32,);
            plan.push(Dispatch {
                pipeline: self.pipes.copy.clone(),
                bind: ops_bind(&sc.h, &self.zero, &sc.x),
                offset: off,
                groups: (cells((n * d) as u32, WG_OPS), 1, 1),
            });
            let off = op!(n_rows: n as u32, dim: d as u32, off0: self.gain(&t("ln2"))?, f0: c.eps, u0: 1,);
            plan.push(Dispatch {
                pipeline: self.pipes.rms_norm.clone(),
                bind: ops_bind(&self.zero, &self.zero, &sc.x),
                offset: off,
                groups: ((n as u32).min(max_groups).max(1), 1, 1),
            });
            linear!(t("ffn_gate"), &sc.x, &sc.gate, c.ffn_dim, d);
            linear!(t("ffn_up"), &sc.x, &sc.up, c.ffn_dim, d);
            let off = op!(n_rows: n as u32, dim: c.ffn_dim as u32,);
            plan.push(Dispatch {
                pipeline: self.pipes.geglu.clone(),
                bind: ops_bind(&self.zero, &sc.up, &sc.gate),
                offset: off,
                groups: (cells((n * c.ffn_dim) as u32, WG_OPS), 1, 1),
            });
            linear!(t("ffn_down"), &sc.gate, &sc.proj, d, c.ffn_dim);
            let off = op!(n_rows: n as u32, dim: d as u32, off0: self.gain(&t("ffn_post_norm"))?, f0: c.eps, u0: 1,);
            plan.push(Dispatch {
                pipeline: self.pipes.rms_norm.clone(),
                bind: ops_bind(&self.zero, &self.zero, &sc.proj),
                offset: off,
                groups: ((n as u32).min(max_groups).max(1), 1, 1),
            });
            let off = op!(n_rows: n as u32, dim: d as u32, f0: 1.0,);
            plan.push(Dispatch {
                pipeline: self.pipes.add_scale.clone(),
                bind: ops_bind(&self.zero, &sc.proj, &sc.h),
                offset: off,
                groups: (cells((n * d) as u32, WG_OPS), 1, 1),
            });
        }

        // --- pool, normalize, project ---------------------------------------
        let off = op!(
            n_rows: n_out as u32, dim: d as u32,
            f0: (d as f32).sqrt(), u0: c.n_merge as u32, u1: nx as u32,
        );
        plan.push(Dispatch {
            pipeline: self.pipes.pool_avg.clone(),
            bind: ops_bind(&sc.h, &self.zero, &sc.pooled),
            offset: off,
            groups: (cells((n_out * d) as u32, WG_OPS), 1, 1),
        });
        // The embedder's norm has no gain tensor.
        let off = op!(n_rows: n_out as u32, dim: d as u32, f0: c.eps,);
        plan.push(Dispatch {
            pipeline: self.pipes.rms_norm.clone(),
            bind: ops_bind(&self.zero, &self.zero, &sc.pooled),
            offset: off,
            groups: ((n_out as u32).min(max_groups).max(1), 1, 1),
        });
        {
            // The projection runs over pooled tokens, not patches, so it
            // cannot go through `linear!` — that macro is fixed at `n` rows.
            let hw = self.weights.get("mm.input_projection.weight")?;
            let off = slot(
                &mut u,
                bytemuck::bytes_of(&MatvecParams {
                    w_base: hw.base_u32,
                    in_dim: d as u32,
                    out_dim: c.proj_dim as u32,
                    n_tokens: n_out as u32,
                }),
            );
            plan.push(Dispatch {
                pipeline: self.quant.pipeline_for(hw.ty, n_out as u32)?.clone(),
                bind: mv_bind(hw.buffer, &sc.pooled, &sc.out),
                offset: off,
                groups: (
                    crate::row_groups(c.proj_dim as u32, max_groups),
                    crate::token_groups(n_out as u32),
                    1,
                ),
            });
        }

        anyhow::ensure!(
            u.len() as u64 <= UNIFORM_SLOT * 4096,
            "vision plan needs {} uniform slots",
            u.len() as u64 / UNIFORM_SLOT
        );
        q.write_buffer(&uniforms, 0, &u);

        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vision"),
                timestamp_writes: None,
            });
            for dsp in &plan {
                pass.set_pipeline(&dsp.pipeline);
                pass.set_bind_group(0, &dsp.bind, &[dsp.offset]);
                pass.dispatch_workgroups(dsp.groups.0, dsp.groups.1, dsp.groups.2);
            }
        }
        self.gpu.queue.submit([enc.finish()]);

        Ok(self.gpu.read_f32(&sc.out, n_out * c.proj_dim))
    }
}

/// Collect the per-weight calibration ranges. A weight with no scalars beside
/// it simply does not appear.
fn clamp_table(g: &Gguf) -> HashMap<String, Clamp> {
    let scalar = |name: &str| -> Option<f32> {
        g.tensor_opt(name)?.as_f32().and_then(|v| v.first().copied())
    };
    let mut out = HashMap::new();
    for i in 0..g.header().tensors.len() {
        let name = g.tensor_at(i).info.name.clone();
        let Some(stem) = name.strip_suffix(".weight") else {
            continue;
        };
        let c = Clamp {
            in_lo: scalar(&format!("{stem}.input_min")).unwrap_or(f32::NEG_INFINITY),
            in_hi: scalar(&format!("{stem}.input_max")).unwrap_or(f32::INFINITY),
            out_lo: scalar(&format!("{stem}.output_min")).unwrap_or(f32::NEG_INFINITY),
            out_hi: scalar(&format!("{stem}.output_max")).unwrap_or(f32::INFINITY),
        };
        if c.any() {
            out.insert(name, c);
        }
    }
    let _ = GgmlType::F32;
    out
}
