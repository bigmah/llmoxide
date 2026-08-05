//! Translate the quant kernels to Metal and print them.
//!
//! Kernel performance here turns on whether the per-token accumulator tile
//! lands in registers or in `thread` scratch memory, which the WGSL does not
//! say. Reading the generated MSL is how that gets settled.

fn main() -> anyhow::Result<()> {
    let src = gpu::quant_shader_source(!std::env::args().any(|a| a == "--barrier"));
    let module = naga::front::wgsl::parse_str(&src).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut opts = naga::back::msl::Options::default();
    opts.lang_version = (3, 0);
    // Match what Gpu::shader asks wgpu for: bindings resolve, out-of-range
    // indices are clamped, and loops carry no termination counter.
    opts.fake_missing_bindings = true;
    opts.bounds_check_policies.buffer = naga::proc::BoundsCheckPolicy::Restrict;
    opts.force_loop_bounding = false;

    let (out, _) = naga::back::msl::write_string(
        &module,
        &info,
        &opts,
        &naga::back::msl::PipelineOptions::default(),
    )
    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    println!("{out}");
    Ok(())
}
