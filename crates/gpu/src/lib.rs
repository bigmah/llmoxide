//! wgpu device, weight residency, and the quantized matvec pipelines.

pub mod arena;
pub mod forward;

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
        // Naga does not implement the `subgroups` enable-extension yet, so the
        // matvec kernels amortize their tree reduction over several rows per
        // workgroup instead.
        let subgroups = false;
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

/// Every model weight, resident on the GPU.
pub struct Weights {
    pub buffers: Vec<wgpu::Buffer>,
    pub handles: HashMap<String, Handle>,
    pub bytes: u64,
}

impl Weights {
    /// Upload all tensors named by `names`. Buffers are created mapped so the
    /// 7.4 GB goes out in one copy per buffer rather than through staging.
    pub fn upload(
        gpu: &Gpu,
        g: &Gguf,
        names: impl IntoIterator<Item = String>,
    ) -> anyhow::Result<Self> {
        let plan = Plan::build(g, names)?;
        let mut buffers = Vec::with_capacity(plan.buffer_sizes.len());

        for (i, &size) in plan.buffer_sizes.iter().enumerate() {
            buffers.push(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("weights{i}")),
                size: size.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            }));
        }

        {
            // Hold one mapped view per buffer and scatter tensors into it.
            let mut views: Vec<_> = buffers
                .iter()
                .zip(&plan.buffer_sizes)
                .map(|(b, &size)| b.slice(..size.max(4)).get_mapped_range_mut())
                .collect();

            for (name, h) in &plan.handles {
                let t = g.tensor(name)?;
                let start = h.base_u32 as usize * 4;
                let len = arena::gpu_bytes(t.ty(), t.elem_count());
                arena::encode_tensor(&t, &mut views[h.buffer][start..start + len]);
            }
        }
        for b in &buffers {
            b.unmap();
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatvecParams {
    pub w_base: u32,
    pub in_dim: u32,
    pub out_dim: u32,
    pub n_tokens: u32,
}

/// Compute pipelines for the quantized kernels.
pub struct QuantKernels {
    pub layout: wgpu::BindGroupLayout,
    pub q4k: wgpu::ComputePipeline,
    pub q6k: wgpu::ComputePipeline,
    pub embed: wgpu::ComputePipeline,
}

impl QuantKernels {
    pub fn new(gpu: &Gpu) -> Self {
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("quant"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/quant.wgsl").into()),
            });

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
            embed: make("embed_q6k"),
            layout,
        }
    }

    pub fn pipeline_for(&self, ty: GgmlType) -> anyhow::Result<&wgpu::ComputePipeline> {
        match ty {
            GgmlType::Q4K => Ok(&self.q4k),
            GgmlType::Q6K => Ok(&self.q6k),
            other => anyhow::bail!("no matvec kernel for {}", other.name()),
        }
    }
}

/// Rows per workgroup — must match ROWS in quant.wgsl.
pub const ROWS_PER_GROUP: u32 = 8;
/// Tokens per dispatch — must match TILE in quant.wgsl.
pub const TOKEN_TILE: u32 = 32;

/// Rows are strided across the grid because the output projection has 262 144
/// of them, well past the 65 535-per-dimension dispatch limit.
pub fn row_groups(out_dim: u32, limit: u32) -> u32 {
    out_dim.div_ceil(ROWS_PER_GROUP).clamp(1, limit.max(1))
}

/// Dispatches needed along the token axis.
pub fn token_groups(n_tokens: u32) -> u32 {
    n_tokens.div_ceil(TOKEN_TILE).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_groups_respects_dispatch_limit() {
        assert_eq!(token_groups(1), 1);
        assert_eq!(token_groups(TOKEN_TILE), 1);
        assert_eq!(token_groups(TOKEN_TILE + 1), 2);
        assert_eq!(row_groups(3840, 65535), 480);
        assert_eq!(row_groups(262144, 65535), 32768);
        assert_eq!(row_groups(0, 65535), 1);
        // Never fewer groups than the dispatch limit allows.
        assert!(row_groups(1_000_000, 65535) <= 65535);
    }
}
