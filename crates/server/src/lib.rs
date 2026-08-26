//! OpenAI-compatible HTTP surface.
//!
//! Scoped to what an agentic coding client actually uses: `/v1/models`,
//! `/v1/chat/completions` with SSE streaming, and tool calling. Requests are
//! serialized onto one engine thread.

pub mod engine;

use std::sync::mpsc;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use chat::{ApiToolCall, FunctionCall, Message, Tool};
use engine::{Engine, Event, FinishReason, GenerateRequest};
use model::sample::Sampling;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::{oneshot, Mutex};

/// Work for the engine thread. The engine is not `Sync`, so everything that
/// touches it — generation and wiping alike — arrives down this one channel.
enum Job {
    /// A queued generation: the request plus where to stream events.
    Generate {
        req: GenerateRequest,
        tx: mpsc::Sender<Event>,
        ready: oneshot::Sender<()>,
    },
    /// Overwrite the resident conversation. Acknowledged only once the wipe has
    /// actually run, so a client can rely on the response meaning it is done.
    Wipe(oneshot::Sender<()>),
}

#[derive(Clone)]
pub struct AppState {
    jobs: tokio::sync::mpsc::UnboundedSender<Job>,
    model_id: String,
    context_len: usize,
    default_sampling: Arc<Sampling>,
    /// Serializes clients so streams do not interleave on one engine.
    slot: Arc<Mutex<()>>,
}

impl AppState {
    /// Move the engine onto its own thread and return a handle to it.
    pub fn spawn(mut engine: Engine, model_id: String) -> Self {
        let (jobs, mut rx) = tokio::sync::mpsc::unbounded_channel::<Job>();
        let context_len = engine.context_len();
        let default_sampling = Arc::new(engine.default_sampling.clone());

        std::thread::Builder::new()
            .name("llmoxide-engine".into())
            .spawn(move || {
                while let Some(job) = rx.blocking_recv() {
                    match job {
                        Job::Generate { req, tx, ready } => {
                            let _ = ready.send(());
                            engine.generate(req, &tx);
                        }
                        Job::Wipe(done) => {
                            engine.wipe();
                            let _ = done.send(());
                        }
                    }
                }
            })
            .expect("spawn engine thread");

        Self {
            jobs,
            model_id,
            context_len,
            default_sampling,
            slot: Arc::new(Mutex::new(())),
        }
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/wipe", post(wipe))
        .route("/health", get(|| async { "ok" }))
        .fallback(unmatched)
        .layer(axum::middleware::from_fn(log_request))
        .with_state(state)
}

/// Log every request; a client hitting an unexpected path should be obvious
/// rather than silently 404ing.
async fn log_request(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let (method, uri) = (req.method().clone(), req.uri().clone());
    let res = next.run(req).await;
    tracing::info!(%method, %uri, status = res.status().as_u16(), "request");
    res
}

async fn unmatched(req: axum::extract::Request) -> impl IntoResponse {
    tracing::warn!(method = %req.method(), uri = %req.uri(), "no route");
    (
        StatusCode::NOT_FOUND,
        Json(json!({"error": {"message": format!("no route for {} {}", req.method(), req.uri())}})),
    )
}

/// Forget the resident conversation.
///
/// Queued behind any in-flight generation like everything else, so a wipe
/// cannot land halfway through a stream and leave the cache describing a
/// conversation the engine is still extending.
async fn wipe(State(s): State<AppState>) -> Json<Value> {
    let (done, wait) = oneshot::channel();
    let _ = s.jobs.send(Job::Wipe(done));
    let ok = wait.await.is_ok();
    tracing::info!(ok, "wiped");
    Json(json!({"wiped": ok}))
}

async fn list_models(State(s): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": s.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "llmoxide",
            "context_length": s.context_len,
        }]
    }))
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<Tool>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default, alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stop: StopField,
    /// Non-standard, but the natural switch for this model's thought channel.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
}

/// `stop` may be a string or an array of them.
#[derive(Debug, Default, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    #[default]
    None,
    One(String),
    Many(Vec<String>),
}

impl StopField {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::None => Vec::new(),
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

fn api_tool_calls(calls: &[chat::ToolCall]) -> Vec<ApiToolCall> {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| ApiToolCall {
            // Clients correlate results by id; the model's DSL has no notion of
            // one, so we mint a stable per-response id.
            id: format!("call_{i}"),
            kind: "function".into(),
            function: FunctionCall {
                name: c.name.clone(),
                arguments: c.arguments.to_string(),
            },
        })
        .collect()
}

fn build_job(s: &AppState, req: ChatRequest) -> (GenerateRequest, bool) {
    let d = s.default_sampling.as_ref();
    let sampling = Sampling {
        temperature: req.temperature.unwrap_or(d.temperature),
        top_k: req.top_k.unwrap_or(d.top_k),
        top_p: req.top_p.unwrap_or(d.top_p),
        seed: req.seed.unwrap_or(0),
        ..d.clone()
    };
    let stream = req.stream;
    (
        GenerateRequest {
            messages: req.messages,
            tools: req.tools,
            sampling,
            max_tokens: req.max_tokens.unwrap_or(2048),
            // Thinking costs tokens and opencode does not surface it, so it is
            // off unless asked for.
            enable_thinking: req.enable_thinking.unwrap_or(false),
            stop: req.stop.into_vec(),
        },
        stream,
    )
}

async fn submit(s: &AppState, req: GenerateRequest) -> mpsc::Receiver<Event> {
    let (tx, rx) = mpsc::channel();
    let (ready, started) = oneshot::channel();
    let _ = s.jobs.send(Job::Generate { req, tx, ready });
    let _ = started.await;
    rx
}

async fn chat_completions(
    State(s): State<AppState>,
    body: axum::body::Bytes,
) -> Result<axum::response::Response, ApiError> {
    // Parse by hand so a schema mismatch reports the offending payload instead
    // of axum rejecting it with an opaque 422.
    let req: ChatRequest = serde_json::from_slice(&body).map_err(|e| {
        // Neither the body nor serde's message reaches the log: the message
        // quotes the offending value, so it carries prompt text of its own.
        // Position and category are enough to debug a schema mismatch.
        tracing::error!(
            line = e.line(),
            column = e.column(),
            kind = ?e.classify(),
            bytes = body.len(),
            "bad request"
        );
        ApiError::split(
            format!("could not parse request: {e}"),
            format!(
                "could not parse request: {:?} at line {} column {}",
                e.classify(),
                e.line(),
                e.column()
            ),
        )
    })?;
    tracing::info!(
        messages = req.messages.len(),
        tools = req.tools.len(),
        stream = req.stream,
        max_tokens = ?req.max_tokens,
        "chat request"
    );
    let (job, stream) = build_job(&s, req);
    if stream {
        Ok(stream_completion(s, job).await.into_response())
    } else {
        Ok(Json(complete(s, job).await?).into_response())
    }
}

async fn complete(s: AppState, job: GenerateRequest) -> Result<Value, ApiError> {
    let _guard = s.slot.clone().lock_owned().await;
    let rx = submit(&s, job).await;

    let mut prompt_tokens = 0;
    let mut completion_tokens = 0;
    let mut done = None;

    // Receiving is blocking; keep it off the async runtime's worker.
    let collected = tokio::task::spawn_blocking(move || {
        let mut err = None;
        while let Ok(ev) = rx.recv() {
            match ev {
                Event::Prefill { tokens, .. } => prompt_tokens = tokens,
                Event::Token { .. } => {}
                Event::Done {
                    completion,
                    generated,
                    reason,
                } => {
                    completion_tokens = generated;
                    done = Some((completion, reason));
                    break;
                }
                Event::Error(e) => {
                    err = Some(e);
                    break;
                }
            }
        }
        (prompt_tokens, completion_tokens, done, err)
    })
    .await
    .map_err(|e| ApiError::new(e.to_string()))?;

    let (prompt_tokens, completion_tokens, done, err) = collected;
    if let Some(e) = err {
        return Err(ApiError::new(e));
    }
    let (completion, reason) = done.ok_or_else(|| ApiError::new("generation produced no result"))?;

    let calls = api_tool_calls(&completion.tool_calls);
    let mut message = json!({ "role": "assistant", "content": completion.content });
    if !calls.is_empty() {
        message["tool_calls"] = serde_json::to_value(&calls).unwrap_or(Value::Null);
        // OpenAI clients expect null content when a turn is only tool calls.
        if completion.content.is_empty() {
            message["content"] = Value::Null;
        }
    }
    if !completion.reasoning.is_empty() {
        message["reasoning_content"] = json!(completion.reasoning);
    }

    Ok(json!({
        "id": "chatcmpl-llmoxide",
        "object": "chat.completion",
        "created": 0,
        "model": s.model_id,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": reason.as_str(),
        }],
        "usage": Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

async fn stream_completion(s: AppState, job: GenerateRequest) -> impl IntoResponse {
    let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel::<SseEvent>();
    let model_id = s.model_id.clone();

    tokio::spawn(async move {
        let _guard = s.slot.clone().lock_owned().await;
        let rx = submit(&s, job).await;

        let chunk = move |delta: Value, finish: Option<&str>| {
            json!({
                "id": "chatcmpl-llmoxide",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model_id,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
            })
        };

        let send = |v: Value| SseEvent::default().data(v.to_string());
        let _ = out_tx.send(send(chunk(json!({"role": "assistant"}), None)));

        let forward = out_tx.clone();
        let _ = tokio::task::spawn_blocking(move || {
            while let Ok(ev) = rx.recv() {
                match ev {
                    Event::Token { text, .. } if !text.is_empty() => {
                        if forward
                            .send(send(chunk(json!({ "content": text }), None)))
                            .is_err()
                        {
                            return;
                        }
                    }
                    Event::Token { .. } | Event::Prefill { .. } => {}
                    Event::Done {
                        completion, reason, ..
                    } => {
                        // Tool calls only become well-formed once the closing
                        // delimiter arrives, so they are emitted whole at the
                        // end rather than streamed as partial JSON.
                        if !completion.tool_calls.is_empty() {
                            let calls = api_tool_calls(&completion.tool_calls);
                            let _ = forward.send(send(chunk(
                                json!({ "tool_calls": calls.iter().enumerate().map(|(i, c)| {
                                    json!({
                                        "index": i,
                                        "id": c.id,
                                        "type": "function",
                                        "function": {
                                            "name": c.function.name,
                                            "arguments": c.function.arguments,
                                        }
                                    })
                                }).collect::<Vec<_>>() }),
                                None,
                            )));
                        }
                        let _ = forward.send(send(chunk(json!({}), Some(reason.as_str()))));
                        break;
                    }
                    Event::Error(e) => {
                        let _ = forward.send(send(json!({ "error": { "message": e } })));
                        break;
                    }
                }
            }
            let _ = forward.send(SseEvent::default().data("[DONE]"));
        })
        .await;
    });

    let stream = futures::stream::unfold(out_rx, |mut rx| async move {
        rx.recv().await.map(|e| (Ok::<_, std::convert::Infallible>(e), rx))
    });
    Sse::new(stream)
}

/// An error with two faces: what the client is told, and what gets logged.
///
/// They have to differ. A `serde_json` parse error quotes the offending value
/// back at you — `invalid type: string "..."` — so logging the message that goes
/// to the client would write a fragment of the prompt into the log, which is the
/// exact leak that not logging the body was supposed to close. The client
/// already has its own payload, so it gets the detailed message; the log gets
/// the shape of the failure and nothing from inside it.
pub struct ApiError {
    client: String,
    log: String,
}

impl ApiError {
    /// Same text both ways, for errors generated here rather than from input.
    fn new(msg: impl Into<String>) -> Self {
        let msg = msg.into();
        Self {
            client: msg.clone(),
            log: msg,
        }
    }

    /// A detailed message for the client, a sanitized one for the log.
    fn split(client: impl Into<String>, log: impl Into<String>) -> Self {
        Self {
            client: client.into(),
            log: log.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        tracing::error!(error = %self.log, "request failed");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": { "message": self.client, "type": "server_error" } })),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_accepts_string_or_array() {
        let one: StopField = serde_json::from_str(r#""END""#).unwrap();
        assert_eq!(one.into_vec(), vec!["END".to_string()]);
        let many: StopField = serde_json::from_str(r#"["a","b"]"#).unwrap();
        assert_eq!(many.into_vec(), vec!["a".to_string(), "b".to_string()]);
        let none: StopField = serde_json::from_str("null").unwrap_or_default();
        assert!(none.into_vec().is_empty());
    }

    #[test]
    fn tool_calls_get_distinct_ids_and_json_arguments() {
        let calls = api_tool_calls(&[
            chat::ToolCall {
                name: "a".into(),
                arguments: json!({"x": 1}),
            },
            chat::ToolCall {
                name: "b".into(),
                arguments: json!({}),
            },
        ]);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[1].id, "call_1");
        // Clients parse `arguments` as a JSON string, not an object.
        assert_eq!(calls[0].function.arguments, r#"{"x":1}"#);
    }

    #[test]
    fn request_parses_openai_shape() {
        let r: ChatRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "stream":true,"max_completion_tokens":32}"#,
        )
        .unwrap();
        assert!(r.stream);
        assert_eq!(r.max_tokens, Some(32));
        assert_eq!(r.messages[0].text(), "hi");
    }

    #[test]
    fn content_parts_flatten_to_text() {
        let m: Message = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}"#,
        )
        .unwrap();
        assert_eq!(m.text(), "ab");
    }
}
