//! Verify weight-arena upload integrity by reading spans back and comparing
//! against a fresh CPU-side encode. Diagnoses silent upload failures that only
//! appear at multi-buffer scale.

use gpu::arena;
use gpu::{Gpu, Weights};

fn main() -> anyhow::Result<()> {
    let model = std::env::args()
        .nth(1)
        .expect("usage: upload_check <model.gguf>");
    let g = gguf::Gguf::open(&model)?;
    let cfg = model::qwen35::Config::from_gguf(&g)?;

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

    let gpu = Gpu::blocking_new()?;
    println!("adapter: {}", gpu.adapter_name);
    let t0 = std::time::Instant::now();
    let w = Weights::upload(&gpu, &g, names.clone())?;
    println!(
        "uploaded {:.2} GB across {} buffer(s) in {:?}",
        w.bytes as f64 / 1e9,
        w.buffers.len(),
        t0.elapsed()
    );

    // Check the head of every tensor (cheap: 1 KB each).
    let mut bad = 0usize;
    for name in &names {
        let h = w.get(name)?;
        let t = g.tensor(name)?;
        let len = arena::gpu_bytes(t.ty(), t.elem_count());
        let mut expect = vec![0u8; len];
        arena::encode_tensor(&t, &mut expect);

        let probe = 1024.min(len);
        let got = read_bytes(&gpu, &w.buffers[h.buffer], h.base_u32 as u64 * 4, probe);
        if got != expect[..probe] {
            bad += 1;
            let first = got
                .iter()
                .zip(&expect[..probe])
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            println!(
                "{name}: MISMATCH in buffer {} at byte {first} (got {:02x?} want {:02x?})",
                h.buffer,
                &got[first..(first + 8).min(probe)],
                &expect[first..(first + 8).min(probe)]
            );
        }
    }
    if bad == 0 {
        println!("all {} tensor heads match", names.len());
    } else {
        println!("{bad}/{} tensors corrupt", names.len());
    }
    Ok(())
}

fn read_bytes(gpu: &Gpu, src: &wgpu::Buffer, offset: u64, len: usize) -> Vec<u8> {
    let staging = gpu.buffer(
        "readback",
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
    rx.recv().expect("map channel").expect("map failed");
    let out = slice.get_mapped_range().to_vec();
    staging.unmap();
    out
}
