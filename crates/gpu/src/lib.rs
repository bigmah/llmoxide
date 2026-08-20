//! wgpu device, weight residency, and the quantized matvec pipelines.

pub mod arena;
pub mod forward;
pub mod qwen35;

use std::collections::HashMap;

use arena::{Handle, Plan};
use gguf::{GgmlType, Gguf};
use wgpu::util::DeviceExt;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub limits: wgpu::Limits,
    /// Whether subgroup intrinsics are available in WGSL.
    pub subgroups: bool,
}

impl Gpu {
    pub async fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await?;

        // Ask for everything the adapter will give us: the defaults cap storage
        // bindings at 128 MB, far below the 826 MB embedding table.
        let limits = adapter.limits();
        // Naga rejects `enable subgroups;` — it is still on its unimplemented
        // list — but the builtins themselves compile without the directive and
        // the Metal backend lowers subgroupAdd to simd_sum. That turns the
        // matvec row reduction from a five-round barrier tree into one
        // instruction, so take it whenever the adapter offers it.
        // `LLMOXIDE_NO_SUBGROUP` forces the barrier fallback, which is
        // otherwise unreachable on hardware that has subgroups — and so would
        // never be exercised.
        let subgroups = adapter.features().contains(wgpu::Features::SUBGROUP)
            && std::env::var_os("LLMOXIDE_NO_SUBGROUP").is_none();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("llmoxide"),
                required_features: if subgroups {
                    wgpu::Features::SUBGROUP
                } else {
                    wgpu::Features::empty()
                },
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await?;

        // A lost device (e.g. Metal's watchdog killing an over-long command
        // buffer) otherwise fails *silently*: every readback just turns to
        // zeros. Make it loud.
        device.set_device_lost_callback(|reason, msg| {
            eprintln!("wgpu device lost ({reason:?}): {msg}");
        });

        // `subgroups` above only says the feature exists. The row reduction is
        // only correct if a subgroup is exactly ROWS_PER_GROUP-wide and its
        // lanes are the workgroup's threads in order, and wgpu reports a
        // supported *range* (4..64 on Apple) rather than the real width — so
        // ask the hardware before trusting it.
        let subgroups = subgroups && subgroup_layout_matches(&device, &queue);

        Ok(Self {
            adapter_name: adapter.get_info().name,
            device,
            queue,
            limits,
            subgroups,
        })
    }

    pub fn blocking_new() -> anyhow::Result<Self> {
        pollster::block_on(Self::new())
    }

    /// Compile a compute shader with Naga's loop-termination guards disabled.
    ///
    /// Naga otherwise wraps *every* loop in a decrementing 64-bit counter. That
    /// costs several ALU ops per iteration, and worse, it hides constant trip
    /// counts from the Metal compiler — so the token tile never unrolls and its
    /// accumulator lands in scratch memory instead of registers. Every loop in
    /// these kernels is bounded by a tensor dimension or a compile-time
    /// constant, so none of them can run away. Bounds checks stay on.
    pub fn shader(&self, label: &str, source: &str) -> wgpu::ShaderModule {
        let desc = wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        };
        // SAFETY: the only requirement is that no loop in `source` is infinite.
        unsafe {
            self.device.create_shader_module_trusted(
                desc,
                wgpu::ShaderRuntimeChecks {
                    bounds_checks: true,
                    force_loop_bounding: false,
                },
            )
        }
    }

    pub fn buffer(&self, label: &str, bytes: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(4),
            usage,
            mapped_at_creation: false,
        })
    }

    pub fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.buffer(
            label,
            bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )
    }

    pub fn upload_f32(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
    }

    pub fn upload_u32(&self, label: &str, data: &[u32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
    }

    pub fn uniform<T: bytemuck::Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(value),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Read a storage buffer back to the host. Synchronous and slow — for
    /// validation and final logits, not the hot path.
    pub fn read_f32(&self, src: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let bytes = (len * 4) as u64;
        let staging = self.buffer(
            "readback",
            bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        self.queue.submit([enc.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().expect("map channel").expect("buffer map failed");

        let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        out
    }
}

/// Run one workgroup and report whether each subgroup is exactly [`LANES`] wide
/// with `subgroup_invocation_id == local_invocation_index % LANES`. Both hold
/// on Apple silicon; the matvec reduction is wrong without them, so a device
/// that disagrees gets the barrier fallback instead.
fn subgroup_layout_matches(device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    const PROBE: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) tid: u32,
    @builtin(subgroup_size) sz: u32,
    @builtin(subgroup_invocation_id) id: u32,
) {
    out[tid] = id;
    out[64u + tid] = sz;
}
"#;
    let n = 64usize;
    let bytes = (n * 2 * 4) as u64;

    // A malformed probe must not take the process down with it.
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("subgroup probe"),
        source: wgpu::ShaderSource::Wgsl(PROBE.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("subgroup probe"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buf.as_entire_binding(),
        }],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&buf, 0, &staging, 0, bytes);
    queue.submit([enc.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = rx.recv().is_ok_and(|r| r.is_ok());
    let ok = mapped && {
        let data = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        let (ids, sizes) = data.split_at(n);
        sizes.iter().all(|&s| s == LANES)
            && ids
                .iter()
                .enumerate()
                .all(|(i, &id)| id == i as u32 % LANES)
    };

    // A validation error means subgroups are not usable here, whatever the
    // feature bit said.
    pollster::block_on(device.pop_error_scope()).is_none() && ok
}

/// Every model weight, resident on the GPU.
pub struct Weights {
    pub buffers: Vec<wgpu::Buffer>,
    pub handles: HashMap<String, Handle>,
    pub bytes: u64,
}

impl Weights {
    /// Upload all tensors named by `names`.
    ///
    /// Buffers are filled and flushed **one at a time**. Mapping every buffer
    /// at once looks harmless but is not: wgpu backs each mapped-at-creation
    /// buffer with a staging shadow until the next submit, so a 25 GB model
    /// briefly wants 50 GB — past Metal's working set, and the writes then
    /// vanish *silently* (the buffers read back as zeros, no error anywhere).
    /// Sequential fill + poll caps the transient at one buffer's shadow.
    pub fn upload(
        gpu: &Gpu,
        g: &Gguf,
        names: impl IntoIterator<Item = String>,
    ) -> anyhow::Result<Self> {
        let plan = Plan::build(g, names)?;

        let mut by_buffer: Vec<Vec<&(String, Handle)>> = vec![Vec::new(); plan.buffer_sizes.len()];
        for entry in &plan.handles {
            by_buffer[entry.1.buffer].push(entry);
        }

        let mut buffers = Vec::with_capacity(plan.buffer_sizes.len());
        let mut spans: Vec<(usize, u64, Vec<u8>)> = Vec::new();
        for (i, &size) in plan.buffer_sizes.iter().enumerate() {
            let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("weights{i}")),
                size: size.max(4),
                // COPY_SRC so upload integrity is checkable (see upload_check).
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            {
                let mut view = buf.slice(..size.max(4)).get_mapped_range_mut();
                for (name, h) in &by_buffer[i] {
                    let t = g.tensor(name)?;
                    let start = h.base_u32 as usize * 4;
                    let len = arena::gpu_bytes(t.ty(), t.elem_count());
                    arena::encode_tensor(&t, &mut view[start..start + len]);
                }
                // Remember what the head and tail should read back as.
                let n = view.len().min(4096);
                spans.push((i, 0, view[..n].to_vec()));
                if view.len() > n {
                    let off = view.len() - n;
                    spans.push((i, off as u64, view[off..].to_vec()));
                }
            }
            buf.unmap();
            // Push the staging copy through now, releasing the shadow before
            // the next buffer allocates its own.
            gpu.queue.submit([]);
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            buffers.push(buf);
        }

        // Read the sampled spans back. Under memory pressure Metal can drop
        // the staged copies without raising any error — the buffers just read
        // as zeros — and 60 layers of silent garbage is much worse than
        // failing the load. Observed on a 25 GB model on a 48 GB machine.
        for (i, offset, expect) in &spans {
            let got = read_back(gpu, &buffers[*i], *offset, expect.len());
            anyhow::ensure!(
                got == *expect,
                "weight upload verification failed: buffer {i} at byte {offset} \
                 read back wrong (likely GPU memory pressure — close other \
                 applications and retry)"
            );
        }

        Ok(Self {
            bytes: plan.total_bytes(),
            handles: plan.handles.into_iter().collect(),
            buffers,
        })
    }

    pub fn get(&self, name: &str) -> anyhow::Result<Handle> {
        self.handles
            .get(name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("weight {name:?} not resident"))
    }
}

/// Synchronous readback of `len` bytes at `offset`, for upload verification.
fn read_back(gpu: &Gpu, src: &wgpu::Buffer, offset: u64, len: usize) -> Vec<u8> {
    let staging = gpu.buffer(
        "verify",
        len as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(src, offset, &staging, 0, len as u64);
    gpu.queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    if rx.recv().map(|r| r.is_err()).unwrap_or(true) {
        return Vec::new();
    }
    let out = slice.get_mapped_range().to_vec();
    staging.unmap();
    out
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatvecParams {
    pub w_base: u32,
    pub in_dim: u32,
    pub out_dim: u32,
    pub n_tokens: u32,
}

/// Reduce a row's partial products across its lane group. Selected at startup:
/// the subgroup form is one instruction, the fallback is a barrier tree.
const REDUCE_SUBGROUP: &str = "\
// Naga lowers this to Metal's simd_sum. The lane group and the hardware
// subgroup are the same threads — verified at startup, see Gpu::new.
fn reduce_row(tid: u32, lane: u32, v: f32) -> f32 {
    return subgroupAdd(v);
}";

const REDUCE_BARRIER: &str = "\
// No usable subgroups: reduce through workgroup memory instead. All ROWS lane
// groups reduce in lockstep, so the barrier count is per workgroup rather than
// per row.
var<workgroup> partial: array<f32, WG>;
fn reduce_row(tid: u32, lane: u32, v: f32) -> f32 {
    // Keeps this call from overwriting values the previous one is still
    // reading; every caller reaches it in uniform control flow.
    workgroupBarrier();
    partial[tid] = v;
    workgroupBarrier();
    var s = LANES / 2u;
    loop {
        if (s == 0u) { break; }
        if (lane < s) { partial[tid] = partial[tid] + partial[tid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    return partial[tid - lane];
}";

/// Compute pipelines for the quantized kernels. The `_t` variants take TILE
/// tokens per dispatch and are used for prefill; the plain ones are the
/// single-token decode path.
pub struct QuantKernels {
    pub layout: wgpu::BindGroupLayout,
    pub q4k: wgpu::ComputePipeline,
    pub q6k: wgpu::ComputePipeline,
    pub q8_0: wgpu::ComputePipeline,
    pub f32: wgpu::ComputePipeline,
    pub q4k_t: wgpu::ComputePipeline,
    pub q6k_t: wgpu::ComputePipeline,
    pub q8_0_t: wgpu::ComputePipeline,
    pub f32_t: wgpu::ComputePipeline,
    pub embed: wgpu::ComputePipeline,
    pub embed_f32: wgpu::ComputePipeline,
}

/// The quant shader with its shape constants and row reduction filled in. The
/// constants have their single source of truth in this module rather than in
/// the WGSL, since the dispatch grid has to agree with them.
pub fn quant_shader_source(subgroups: bool) -> String {
    include_str!("shaders/quant.wgsl")
        .replace(
            "// @@REDUCE@@",
            if subgroups {
                REDUCE_SUBGROUP
            } else {
                REDUCE_BARRIER
            },
        )
        .replace("@@LANES@@", &LANES.to_string())
        .replace("@@ROWS@@", &ROWS_PER_GROUP.to_string())
        .replace("@@TILE@@", &token_tile().to_string())
}

/// Tokens per tiled dispatch. `LLMOXIDE_TILE` overrides the default so the
/// sweep behind it can be rerun on another GPU without a rebuild; the tile
/// trades weight traffic against the accumulator's register footprint.
pub fn token_tile() -> u32 {
    std::env::var("LLMOXIDE_TILE")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| [1, 2, 4, 8, 16, 32].contains(v))
        .unwrap_or(TOKEN_TILE)
}



impl QuantKernels {
    pub fn new(gpu: &Gpu) -> Self {
        let module = gpu.shader("quant", &quant_shader_source(gpu.subgroups));

        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |ro: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false,
            min_binding_size: None,
        };

        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("quant"),
                entries: &[
                    entry(0, storage(true)),  // weights
                    entry(1, storage(true)),  // x
                    entry(2, storage(false)), // y
                    entry(
                        3,
                        wgpu::BindingType::Buffer {
                            // Dynamic so a single uniform buffer can carry the
                            // parameters for every dispatch in a forward pass.
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: None,
                        },
                    ),
                    entry(4, storage(true)), // token ids (embed only)
                ],
            });

        let pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("quant"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                });

        let make = |entry_point: &str| {
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };

        Self {
            q4k: make("matvec_q4k"),
            q6k: make("matvec_q6k"),
            q8_0: make("matvec_q8_0"),
            f32: make("matvec_f32"),
            q4k_t: make("matvec_q4k_t"),
            q6k_t: make("matvec_q6k_t"),
            q8_0_t: make("matvec_q8_0_t"),
            f32_t: make("matvec_f32_t"),
            embed: make("embed_q6k"),
            embed_f32: make("embed_f32"),
            layout,
        }
    }

    /// The matvec pipeline for a weight type. `n_tokens == 1` takes the decode
    /// kernel, whose accumulator is a scalar rather than a tile array.
    pub fn pipeline_for(
        &self,
        ty: GgmlType,
        n_tokens: u32,
    ) -> anyhow::Result<&wgpu::ComputePipeline> {
        match (ty, n_tokens > 1) {
            (GgmlType::Q4K, false) => Ok(&self.q4k),
            (GgmlType::Q6K, false) => Ok(&self.q6k),
            (GgmlType::Q8_0, false) => Ok(&self.q8_0),
            (GgmlType::F32, false) => Ok(&self.f32),
            (GgmlType::Q4K, true) => Ok(&self.q4k_t),
            (GgmlType::Q6K, true) => Ok(&self.q6k_t),
            (GgmlType::Q8_0, true) => Ok(&self.q8_0_t),
            (GgmlType::F32, true) => Ok(&self.f32_t),
            (other, _) => anyhow::bail!("no matvec kernel for {}", other.name()),
        }
    }

    /// The embedding-gather pipeline for the token_embd type.
    pub fn embed_for(&self, ty: GgmlType) -> anyhow::Result<&wgpu::ComputePipeline> {
        match ty {
            GgmlType::Q6K => Ok(&self.embed),
            GgmlType::F32 => Ok(&self.embed_f32),
            other => anyhow::bail!("no embedding kernel for {}", other.name()),
        }
    }
}

/// Threads cooperating on one output row. Must equal the hardware subgroup
/// width for the subgroup reduction to be selected; see `Gpu::new`.
pub const LANES: u32 = 32;
/// Rows per workgroup.
pub const ROWS_PER_GROUP: u32 = 8;
/// Default tokens per dispatch of the tiled (prefill) kernels. Read through
/// [`token_tile`], which honours `LLMOXIDE_TILE`.
pub const TOKEN_TILE: u32 = 2;

/// Rows are strided across the grid because the output projection has 262 144
/// of them, well past the 65 535-per-dimension dispatch limit.
pub fn row_groups(out_dim: u32, limit: u32) -> u32 {
    out_dim.div_ceil(ROWS_PER_GROUP).clamp(1, limit.max(1))
}

/// Dispatches needed along the token axis.
pub fn token_groups(n_tokens: u32) -> u32 {
    n_tokens.div_ceil(token_tile()).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_groups_respects_dispatch_limit() {
        assert_eq!(token_groups(1), 1);
        assert_eq!(token_groups(token_tile()), 1);
        assert_eq!(token_groups(token_tile() + 1), 2);
        assert_eq!(row_groups(3840, 65535), 480);
        assert_eq!(row_groups(262144, 65535), 32768);
        assert_eq!(row_groups(0, 65535), 1);
        // Never fewer groups than the dispatch limit allows.
        assert!(row_groups(1_000_000, 65535) <= 65535);
    }
}
