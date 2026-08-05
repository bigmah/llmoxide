//! Check each GPU kernel against the CPU reference on real model weights.
//!
//! Synthetic data would not exercise the awkward parts — the 6-bit packed
//! scales in Q4_K, the interleaved sub-blocks in Q6_K — so this runs against
//! actual tensors from the checkpoint.

use gpu::{Gpu, MatvecParams, QuantKernels, Weights};
use wgpu::util::DeviceExt;

fn main() -> anyhow::Result<()> {
    let model = std::env::args()
        .nth(1)
        .expect("usage: kernels <model.gguf> [n_layers]");
    let n_layers: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    let g = gguf::Gguf::open(&model)?;
    let gpu = Gpu::blocking_new()?;
    println!("adapter: {}", gpu.adapter_name);

    // A representative slice: both quant types, both layer geometries, plus the
    // embedding table.
    let mut names = vec!["token_embd.weight".to_string()];
    for i in 0..n_layers {
        for t in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_down.weight",
        ] {
            let n = format!("blk.{i}.{t}");
            if g.has_tensor(&n) {
                names.push(n);
            }
        }
        let v = format!("blk.{i}.attn_v.weight");
        if g.has_tensor(&v) {
            names.push(v);
        }
    }

    let t0 = std::time::Instant::now();
    let w = Weights::upload(&gpu, &g, names.clone())?;
    println!(
        "uploaded {:.2} GB across {} buffer(s) in {:?}\n",
        w.bytes as f64 / 1e9,
        w.buffers.len(),
        t0.elapsed()
    );

    let kernels = QuantKernels::new(&gpu);
    let mut failures = 0;

    for name in &names {
        let h = w.get(name)?;
        let t = g.tensor(name)?;
        let in_dim = h.in_dim as usize;
        // The embedding table's 262144 rows would dominate runtime; a prefix is
        // enough to exercise the kernel.
        let out_dim = (h.out_dim as usize).min(4096);

        // Deterministic, non-trivial activations: constant input would hide
        // per-element indexing errors.
        let x: Vec<f32> = (0..in_dim)
            .map(|i| ((i % 61) as f32 - 30.0) * 0.031)
            .collect();

        let x_buf = gpu.upload_f32("x", &x);
        let y_buf = gpu.storage("y", (out_dim * 4) as u64);
        // Padded to the dynamic-offset alignment the layout now requires.
        let mut pbytes = vec![0u8; 256];
        pbytes[..16].copy_from_slice(bytemuck::bytes_of(&MatvecParams {
            w_base: h.base_u32,
            in_dim: in_dim as u32,
            out_dim: out_dim as u32,
            n_tokens: 1,
        }));
        let params = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: &pbytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let dummy = gpu.upload_u32("tokens", &[0u32]);

        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernels.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w.buffers[h.buffer].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: std::num::NonZeroU64::new(256),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dummy.as_entire_binding(),
                },
            ],
        });

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(name),
                timestamp_writes: None,
            });
            pass.set_pipeline(kernels.pipeline_for(h.ty, 1)?);
            pass.set_bind_group(0, &bind, &[0]);
            pass.dispatch_workgroups(
                gpu::row_groups(out_dim as u32, gpu.limits.max_compute_workgroups_per_dimension),
                1,
                1,
            );
        }
        gpu.queue.submit([enc.finish()]);
        let got = gpu.read_f32(&y_buf, out_dim);

        // Reference: the same fused dequant-dot the CPU forward pass uses.
        let row_bytes = t.ty().bytes_for(in_dim);
        let mut worst: (f64, usize) = (0.0, 0);
        let mut scale = 1e-6f64;
        for o in 0..out_dim {
            let want =
                gguf::quant::dot_row(t.ty(), &t.data[o * row_bytes..(o + 1) * row_bytes], &x) as f64;
            scale = scale.max(want.abs());
            let d = (want - got[o] as f64).abs();
            if d > worst.0 {
                worst = (d, o);
            }
        }
        let rel = worst.0 / scale;
        let ok = rel < 1e-4;
        failures += usize::from(!ok);
        println!(
            "{:>10}  {:<28} {:>5}x{:<6} max|Δ|={:.3e} rel={:.2e}  {}",
            h.ty.name(),
            name,
            in_dim,
            out_dim,
            worst.0,
            rel,
            if ok { "ok" } else { "FAIL" }
        );
    }

    println!();
    if failures == 0 {
        println!("all kernels match the CPU reference");
        Ok(())
    } else {
        anyhow::bail!("{failures} kernel(s) diverged")
    }
}
