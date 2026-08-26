//! Serve a GGUF model over an OpenAI-compatible API.
//!
//! This process holds the same secrets as `llmoxide-private` and gets the same
//! treatment for them: the resident prompt ids and the response accumulator are
//! locked and zeroed by the engine itself, and the heap is overwritten as it is
//! freed, since prompt text passes through many transient copies here that no
//! explicit wipe could chase — the parsed JSON, the chat template's output,
//! decoded token pieces, the SSE frames.
//!
//! What it cannot do anything about is the client. A client that keeps a
//! transcript — opencode writes every message to `opencode.db` in plaintext —
//! is where the conversation actually persists, and nothing on this side of the
//! socket changes that. `llmoxide-private` exists because the only way to close
//! that gap is not to have a client at all.

use server::engine::Engine;
use server::AppState;

/// Overwrite freed heap blocks once armed. Disable with `LLMOXIDE_NO_ZEROIZE=1`
/// — the cost is a memset per free, immaterial against a forward pass, but the
/// escape hatch is there.
#[global_allocator]
static ALLOC: secret::ZeroizingAlloc<std::alloc::System> =
    secret::ZeroizingAlloc(std::alloc::System);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    secret::no_core_dumps();
    // Opt-in, unlike in `llmoxide-private`: this blocks profilers too, and a
    // server is the thing you profile.
    if std::env::var_os("LLMOXIDE_PRIVATE").is_some() {
        secret::no_debugger();
    }
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
    // Only now: loading stages gigabytes of weights through the heap, and those
    // are not secret. Nothing has been served yet.
    let zeroize = std::env::var_os("LLMOXIDE_NO_ZEROIZE").is_none();
    if zeroize {
        secret::arm();
    }
    tracing::info!(elapsed = ?t0.elapsed(), zeroize, "ready");

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
