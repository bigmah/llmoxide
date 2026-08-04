//! Serve a GGUF model over an OpenAI-compatible API.

use server::engine::Engine;
use server::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .unwrap_or_else(|| "gemma4-v2-Q4_K_M.gguf".to_string());
    let port: u16 = std::env::var("LLMOXIDE_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);
    let n_ctx: usize = std::env::var("LLMOXIDE_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16384);
    // Prefill batch: bigger is faster but grows the activation scratch.
    let batch: usize = std::env::var("LLMOXIDE_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    tracing::info!(model, n_ctx, batch, "loading");
    let t0 = std::time::Instant::now();
    let engine = Engine::load(&model, n_ctx, batch)?;
    tracing::info!(elapsed = ?t0.elapsed(), "ready");

    let model_id = std::path::Path::new(&model)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "llmoxide".into());

    let app = server::router(AppState::spawn(engine, model_id.clone()));
    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("listening on http://{addr}  (model id: {model_id})");
    axum::serve(listener, app).await?;
    Ok(())
}
