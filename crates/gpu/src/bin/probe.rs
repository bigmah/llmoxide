//! Report what this adapter actually allows, so buffer layout decisions are
//! made against real limits rather than wgpu's conservative defaults.

fn main() -> anyhow::Result<()> {
    pollster::block_on(run())
}

async fn run() -> anyhow::Result<()> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await?;

    let info = adapter.get_info();
    println!("adapter : {} ({:?}, {:?})", info.name, info.device_type, info.backend);
    println!("driver  : {} {}", info.driver, info.driver_info);

    let l = adapter.limits();
    println!("\n-- limits --");
    println!("max_buffer_size                 {:>15} ({:.2} GB)", l.max_buffer_size, l.max_buffer_size as f64 / 1e9);
    println!("max_storage_buffer_binding_size {:>15} ({:.2} GB)", l.max_storage_buffer_binding_size, l.max_storage_buffer_binding_size as f64 / 1e9);
    println!("max_storage_buffers_per_shader_stage {:>10}", l.max_storage_buffers_per_shader_stage);
    println!("max_compute_workgroup_size_x    {:>15}", l.max_compute_workgroup_size_x);
    println!("max_compute_invocations_per_workgroup {:>9}", l.max_compute_invocations_per_workgroup);
    println!("max_compute_workgroups_per_dimension  {:>9}", l.max_compute_workgroups_per_dimension);
    println!("max_compute_workgroup_storage_size    {:>9}", l.max_compute_workgroup_storage_size);
    println!("max_bind_groups                 {:>15}", l.max_bind_groups);

    let f = adapter.features();
    println!("\n-- relevant features --");
    for (name, feat) in [
        ("SHADER_F16", wgpu::Features::SHADER_F16),
        ("SUBGROUP", wgpu::Features::SUBGROUP),
        ("TIMESTAMP_QUERY", wgpu::Features::TIMESTAMP_QUERY),
        ("MAPPABLE_PRIMARY_BUFFERS", wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
        ("BUFFER_BINDING_ARRAY", wgpu::Features::BUFFER_BINDING_ARRAY),
    ] {
        println!("{name:28} {}", if f.contains(feat) { "yes" } else { "NO" });
    }

    // The embedding table is the largest single tensor; check it fits a binding.
    let embd = 3840u64 * 262144 / 256 * 210;
    println!("\ntoken_embd Q6_K = {embd} bytes ({:.0} MB)", embd as f64 / 1e6);
    println!("fits in one storage binding: {}", embd <= l.max_storage_buffer_binding_size as u64);
    Ok(())
}
