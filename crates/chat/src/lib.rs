//! Prompt assembly and response parsing for gemma4's chat format.
//!
//! The checkpoint ships a Jinja template; rather than embed a Jinja engine we
//! reimplement it directly against token ids. That buys two things beyond
//! speed: control tokens are inserted *by id*, so nothing in user text or a
//! tool result can forge a turn boundary, and the tool-call DSL round-trips
//! through the typed encoder in [`dsl`].

pub mod dsl;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokenizer::Tokenizer;

pub use dsl::ToolCall;

/// Control tokens, resolved by spelling at load time.
pub struct Special {
    pub bos: u32,
    pub turn_open: u32,
    pub turn_close: u32,
    pub channel_open: u32,
    pub channel_close: u32,
    pub think: u32,
    pub tool_open: u32,
    pub tool_close: u32,
    pub call_open: u32,
    pub call_close: u32,
    pub response_open: u32,
    pub response_close: u32,
    pub quote: u32,
}

impl Special {
    pub fn new(t: &Tokenizer) -> anyhow::Result<Self> {
        let get = |s: &str| {
            t.id_of(s)
                .ok_or_else(|| anyhow::anyhow!("vocabulary is missing control token {s:?}"))
        };
        Ok(Self {
            bos: t.bos,
            turn_open: get("<|turn>")?,
            turn_close: get("<turn|>")?,
            channel_open: get("<|channel>")?,
            channel_close: get("<channel|>")?,
            think: get("<|think|>")?,
            tool_open: get("<|tool>")?,
            tool_close: get("<tool|>")?,
            call_open: get("<|tool_call>")?,
            call_close: get("<tool_call|>")?,
            response_open: get("<|tool_response>")?,
            response_close: get("<tool_response|>")?,
            quote: get(dsl::QUOTE)?,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ApiToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional prior reasoning, replayed only alongside tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", default = "default_call_type")]
    pub kind: String,
    pub function: FunctionCall,
}

fn default_call_type() -> String {
    "function".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// OpenAI sends arguments as a JSON *string*.
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Tool {
    #[serde(rename = "type", default = "default_call_type")]
    pub kind: String,
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: Value,
}

impl Message {
    /// Flatten `content`, which may be a string or an array of content parts.
    pub fn text(&self) -> String {
        match &self.content {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter(|p| p.get("type").and_then(Value::as_str) != Some("image"))
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(""),
            Some(Value::Null) | None => String::new(),
            Some(other) => other.to_string(),
        }
    }
}

pub struct PromptBuilder<'a> {
    tok: &'a Tokenizer,
    sp: &'a Special,
    ids: Vec<u32>,
}

impl<'a> PromptBuilder<'a> {
    pub fn new(tok: &'a Tokenizer, sp: &'a Special) -> Self {
        Self {
            tok,
            sp,
            ids: vec![sp.bos],
        }
    }

    fn push(&mut self, id: u32) -> &mut Self {
        self.ids.push(id);
        self
    }

    /// Encode text as ordinary content. Control-token spellings inside it are
    /// tokenized as plain text, never as control tokens.
    fn text(&mut self, s: &str) -> &mut Self {
        self.tok.encode_into(s, &mut self.ids);
        self
    }

    /// Encode DSL text, honouring `<|"|>` as the single quote token.
    ///
    /// The rest of the DSL is ordinary text; only the delimiter must land as
    /// its own id, which is exactly what makes strings unambiguous.
    fn dsl(&mut self, s: &str) -> &mut Self {
        for (i, part) in s.split(dsl::QUOTE).enumerate() {
            if i > 0 {
                self.ids.push(self.sp.quote);
            }
            self.tok.encode_into(part, &mut self.ids);
        }
        self
    }

    fn open_turn(&mut self, role: &str) -> &mut Self {
        self.push(self.sp.turn_open).text(role).text("\n")
    }

    fn close_turn(&mut self) -> &mut Self {
        self.push(self.sp.turn_close).text("\n")
    }

    pub fn finish(self) -> Vec<u32> {
        self.ids
    }
}

/// Build the full prompt for a request.
///
/// Mirrors the checkpoint's template: a leading system turn carrying the system
/// message and any tool declarations, then one turn per message, then the
/// generation prompt. When thinking is disabled the generation prompt opens and
/// immediately closes an empty thought channel — that is how the template
/// suppresses reasoning, and omitting it makes the model think anyway.
pub fn build_prompt(
    tok: &Tokenizer,
    sp: &Special,
    messages: &[Message],
    tools: &[Tool],
    enable_thinking: bool,
) -> Vec<u32> {
    let mut b = PromptBuilder::new(tok, sp);

    let leading_system = messages
        .first()
        .filter(|m| m.role == "system" || m.role == "developer");

    if enable_thinking || !tools.is_empty() || leading_system.is_some() {
        b.open_turn("system");
        if enable_thinking {
            b.push(sp.think).text("\n");
        }
        if let Some(m) = leading_system {
            b.text(m.text().trim());
        }
        for t in tools {
            b.push(sp.tool_open)
                .dsl(&dsl::encode_declaration(
                    &t.function.name,
                    &t.function.description,
                    &t.function.parameters,
                ))
                .push(sp.tool_close);
        }
        b.close_turn();
    }

    let rest = if leading_system.is_some() {
        &messages[1..]
    } else {
        messages
    };

    for (i, m) in rest.iter().enumerate() {
        // Tool results are emitted as part of the assistant turn that called
        // them, so they are skipped here.
        if m.role == "tool" {
            continue;
        }
        let role = if m.role == "assistant" { "model" } else { &m.role };
        b.open_turn(role);

        // Replay an assistant turn exactly as it was generated, including the
        // empty thought channel the generation prompt opens when thinking is
        // off. Without this the next turn's prompt diverges from the KV cache
        // at the first assistant message, and every agentic step re-prefills
        // the whole conversation instead of extending it.
        if role == "model" && !enable_thinking && m.reasoning.is_none() {
            b.push(sp.channel_open).text("thought\n").push(sp.channel_close);
        }

        if !m.tool_calls.is_empty() {
            if let Some(r) = m.reasoning.as_deref().filter(|s| !s.trim().is_empty()) {
                b.push(sp.channel_open)
                    .text("thought\n")
                    .text(r)
                    .text("\n")
                    .push(sp.channel_close);
            }
            for c in &m.tool_calls {
                let args = serde_json::from_str::<Value>(&c.function.arguments)
                    .unwrap_or_else(|_| Value::Object(Default::default()));
                b.push(sp.call_open)
                    .dsl(&dsl::encode_call(&c.function.name, &args))
                    .push(sp.call_close);
            }

            // Then the results, taken from the run of `tool` messages that
            // follows, matched back to their call by id.
            let mut any = false;
            for follow in rest[i + 1..].iter().take_while(|f| f.role == "tool") {
                let name = m
                    .tool_calls
                    .iter()
                    .find(|c| Some(&c.id) == follow.tool_call_id.as_ref())
                    .map(|c| c.function.name.clone())
                    .or_else(|| follow.name.clone())
                    .unwrap_or_else(|| "unknown".into());
                b.push(sp.response_open)
                    .dsl(&dsl::encode_response(&name, &follow.text()))
                    .push(sp.response_close);
                any = true;
            }
            if !any {
                // Calls with no results yet: leave the response block open so
                // the model continues from where it stopped.
                b.push(sp.response_open);
                continue;
            }
            b.close_turn();
            continue;
        }

        let content = m.text();
        if m.role == "assistant" {
            b.text(strip_thinking(&content).trim());
        } else {
            b.text(content.trim());
        }
        b.close_turn();
    }

    b.open_turn("model");
    if !enable_thinking {
        b.push(sp.channel_open).text("thought\n").push(sp.channel_close);
    }
    b.finish()
}

/// Drop any `thought` channel from replayed assistant text.
pub fn strip_thinking(text: &str) -> String {
    let mut out = String::new();
    for (i, part) in text.split("<channel|>").enumerate() {
        if i > 0 || !part.contains("<|channel>") {
            match part.split_once("<|channel>") {
                Some((before, _)) => out.push_str(before),
                None => out.push_str(part),
            }
        } else if let Some((before, _)) = part.split_once("<|channel>") {
            out.push_str(before);
        }
    }
    out
}

/// What a completed generation decoded to.
#[derive(Debug, Default, Clone)]
pub struct Completion {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Split a generated token sequence into reasoning, content, and tool calls.
///
/// Works on ids rather than decoded text because the `<|"|>` delimiter and the
/// channel markers are control tokens that a text decoder deliberately drops.
pub fn parse_completion(tok: &Tokenizer, sp: &Special, ids: &[u32]) -> Completion {
    let mut out = Completion::default();
    let mut i = 0;

    // Raw spelling, so `<|"|>` survives into the DSL parser.
    let raw = |ids: &[u32]| -> String {
        ids.iter()
            .map(|&id| tok.token_text(id).replace('\u{2581}', " "))
            .collect()
    };

    while i < ids.len() {
        let id = ids[i];
        if id == sp.channel_open {
            // `<|channel>thought ... <channel|>`
            let end = ids[i + 1..]
                .iter()
                .position(|&x| x == sp.channel_close)
                .map(|p| i + 1 + p)
                .unwrap_or(ids.len());
            let body = tok.decode(&ids[i + 1..end]);
            out.reasoning
                .push_str(body.strip_prefix("thought\n").unwrap_or(&body));
            i = end + 1;
        } else if id == sp.call_open {
            let end = ids[i + 1..]
                .iter()
                .position(|&x| x == sp.call_close)
                .map(|p| i + 1 + p)
                .unwrap_or(ids.len());
            if let Some(c) = dsl::parse_call(raw(&ids[i + 1..end]).trim()) {
                out.tool_calls.push(c);
            }
            i = end + 1;
        } else if id == sp.response_open || id == sp.turn_close {
            // The model opens a response block to signal it is waiting on us.
            break;
        } else {
            let start = i;
            while i < ids.len()
                && ids[i] != sp.channel_open
                && ids[i] != sp.call_open
                && ids[i] != sp.response_open
                && ids[i] != sp.turn_close
            {
                i += 1;
            }
            out.content.push_str(&tok.decode(&ids[start..i]));
        }
    }
    out.content = out.content.trim().to_string();
    out.reasoning = out.reasoning.trim().to_string();
    out
}
