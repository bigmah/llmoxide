//! Time the prefill GEMM against the tiled matvec it replaces, and check both
//! against the CPU reference, on real tensors from a checkpoint.
//!
//!   gemm <model.gguf> [n_tokens] [tensor ...]
//!
//! Loads only the named tensors, so it runs in seconds even against the 22 GB
//! qwen35 27B — the fast loop for kernel work, where a full load is minutes.

use llmoxide_gpu as gpu;

use gpu::{Gpu, MatvecParams, QuantKernels, Weights};
use wgpu::util::DeviceExt;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = args.first().expect("usage: gemm <model.gguf> [n_tokens] [tensor ...]");
    let t: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(301);
    let mut names: Vec<String> = args.iter().skip(2).cloned().collect();
    if names.is_empty() {
        names = ["blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.0.attn_qkv.weight", "blk.0.ssm_alpha.weight"]
            .map(String::from)
            .to_vec();
    }

    let g = gguf::Gguf::open(model)?;
    let gpu = Gpu::blocking_new()?;
    println!("adapter: {}  tokens: {t}", gpu.adapter_name);
    let w = Weights::upload(&gpu, &g, names.clone())?;
    let kernels = QuantKernels::new(&gpu);
    let mut failures = 0;

    for name in &names {
        let h = w.get(name)?;
        let tensor = g.tensor(name)?;
        let (in_dim, out_dim) = (h.in_dim as usize, h.out_dim as usize);

        let x: Vec<f32> = (0..t * in_dim)
            .map(|i| (((i * 7919) % 113) as f32 - 56.0) * 0.017)
            .collect();
        let x_buf = gpu.upload_f32("x", &x);
        let y_buf = gpu.storage("y", (t * out_dim * 4) as u64);
        let mut pbytes = vec![0u8; 256];
        pbytes[..16].copy_from_slice(bytemuck::bytes_of(&MatvecParams {
            w_base: h.base_u32,
            in_dim: in_dim as u32,
            out_dim: out_dim as u32,
            n_tokens: t as u32,
        }));
        let params = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: &pbytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let dummy = gpu.upload_u32("tokens", &[0u32]);
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernels.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w.buffers[h.buffer].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: std::num::NonZeroU64::new(256),
                    }),
                },
                wgpu::BindGroupEntry { binding: 4, resource: dummy.as_entire_binding() },
            ],
        });

        let max_groups = gpu.limits.max_compute_workgroups_per_dimension;
        let matvec = (
            kernels.pipeline_for(h.ty, t as u32)?,
            (gpu::row_groups(out_dim as u32, max_groups), gpu::token_groups(t as u32), 1),
        );
        let mut variants = vec![("matvec", matvec)];
        // `GEMM_READBW=1`: a kernel that only streams the tensor's bytes, the
        // ceiling any decode matvec on this GPU can reach.
        let readbw;
        if std::env::var_os("GEMM_READBW").is_some() && gpu.msl {
            let src = r#"
                #include <metal_stdlib>
                using namespace metal;
                struct P { uint w_base; uint in_dim; uint out_dim; uint n_tokens; };
                kernel void readbw(device const uint4 *w [[buffer(0)]], device const float4 *x [[buffer(1)]],
                                   device float *y [[buffer(2)]], constant P &p [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
                    uint words = p.out_dim * (p.in_dim / 256) * 14;   // uint4 per Q6_K block = 14
                    uint base = p.w_base / 4;
                    uint4 a = 0;
                    for (uint i = gid; i < words; i += n) a ^= w[base + i];
                    if ((a.x ^ a.y ^ a.z ^ a.w) == 0x12345678u) y[0] = 1.0f;
                }"#;
            let m = unsafe {
                gpu.device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                    entry_point: "readbw".into(),
                    label: Some("readbw"),
                    num_workgroups: (256, 1, 1),
                    msl: Some(src.into()),
                    ..Default::default()
                })
            };
            let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&kernels.layout],
                push_constant_ranges: &[],
            });
            readbw = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("readbw"),
                layout: Some(&pl),
                module: &m,
                entry_point: Some("readbw"),
                compilation_options: Default::default(),
                cache: None,
            });
            variants.push(("readbw", (&readbw, (40 * 32, 1, 1))));
        }
        if let Some(p) = kernels.gemm_for(h.ty, t as u32) {
            variants.push(("gemm", (p, kernels.gemm_groups(out_dim as u32, t as u32))));
        }

        // Reference rows for a spread of tokens, including the last (partial) tile.
        let row_bytes = tensor.ty().bytes_for(in_dim);
        let check_toks: Vec<usize> = [0, 1, t / 2, t.saturating_sub(1)]
            .into_iter()
            .filter(|&i| i < t)
            .collect();
        let want: Vec<Vec<f32>> = check_toks
            .iter()
            .map(|&ti| {
                (0..out_dim)
                    .map(|o| {
                        gguf::quant::dot_row(
                            tensor.ty(),
                            &tensor.data[o * row_bytes..(o + 1) * row_bytes],
                            &x[ti * in_dim..(ti + 1) * in_dim],
                        )
                    })
                    .collect()
            })
            .collect();

        for (label, (pipeline, groups)) in variants {
            let run = |iters: usize| {
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bind, &[0]);
                    for _ in 0..iters {
                        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
                    }
                }
                gpu.queue.submit([enc.finish()]);
                gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            };
            // Warm up until the GPU clock has ramped; the first few hundred
            // milliseconds otherwise read as a slow kernel.
            let warm = std::time::Instant::now();
            while warm.elapsed().as_millis() < 300 {
                run(if t == 1 { 50 } else { 1 });
            }
            let iters = if t == 1 { 50 } else { 5 };
            let t0 = std::time::Instant::now();
            run(iters);
            let dt = t0.elapsed().as_secs_f64() / iters as f64;

            let got = gpu.read_f32(&y_buf, t * out_dim);
            let mut worst = 0f64;
            let mut scale = 1e-6f64;
            for (ci, &ti) in check_toks.iter().enumerate() {
                for o in 0..out_dim {
                    let wv = want[ci][o] as f64;
                    scale = scale.max(wv.abs());
                    worst = worst.max((wv - got[ti * out_dim + o] as f64).abs());
                }
            }
            let rel = worst / scale;
            if std::env::var_os("GEMM_DEBUG").is_some() {
                println!("  got {:?}\n  want {:?}", &got[..6], &want[0][..6]);
            }
            let ok = rel < 1e-4 || label == "readbw";
            failures += usize::from(!ok);
            let flops = 2.0 * (t * in_dim * out_dim) as f64;
            let bytes = (tensor.ty().bytes_for(in_dim) * out_dim) as f64;
            println!(
                "{:<26} {:>5}x{:<6} {:<7} {:>8.3} ms  {:>6.0} GFLOP/s {:>5.0} GB/s  rel={:.2e} {}",
                name,
                in_dim,
                out_dim,
                label,
                dt * 1e3,
                flops / dt / 1e9,
                bytes / dt / 1e9,
                rel,
                if ok { "ok" } else { "FAIL" }
            );
            gpu.queue.write_buffer(&y_buf, 0, &vec![0u8; t * out_dim * 4]);
        }
    }
    anyhow::ensure!(failures == 0, "{failures} variant(s) diverged");
    Ok(())
}
