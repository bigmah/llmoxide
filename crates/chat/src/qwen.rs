//! Prompt assembly and response parsing for qwen35's chat format.
//!
//! Reimplements the checkpoint's Jinja template directly against token ids,
//! like the gemma4 path: control tokens are inserted *by id*, so nothing in
//! user text or a tool result can forge a turn boundary. (llama.cpp's server
//! tokenizes the rendered template with special parsing enabled, which makes
//! `<|im_end|>` in user content a real control token — we deliberately don't.)
//!
//! The format is ChatML plus Qwen's XML-ish tool-call convention:
//!
//! ```text
//! <|im_start|>system
//! ...instructions, # Tools block...<|im_end|>
//! <|im_start|>user
//! ...<|im_end|>
//! <|im_start|>assistant
//! <think>
//! ...reasoning...
//! </think>
//!
//! ...content...<tool_call>
//! <function=name>
//! <parameter=key>
//! value
//! </parameter>
//! </function>
//! </tool_call><|im_end|>
//! ```
//!
//! Tool *results* ride inside `user` turns as `<tool_response>` blocks.
//! `<think>`, `<tool_call>`, `<tool_response>` and their closers are single
//! vocabulary tokens (type 4, "user-defined"), inserted by id.

use serde_json::Value;
use tokenizer::Tokenizer;

use crate::dsl::ToolCall;
use crate::{Completion, Message, Tool};

/// Control tokens, resolved by spelling at load time.
pub struct Special {
    pub im_start: u32,
    pub im_end: u32,
    pub think_open: u32,
    pub think_close: u32,
    pub call_open: u32,
    pub call_close: u32,
    pub response_open: u32,
    pub response_close: u32,
}

impl Special {
    pub fn new(t: &Tokenizer) -> anyhow::Result<Self> {
        let get = |s: &str| {
            t.id_of(s)
                .ok_or_else(|| anyhow::anyhow!("vocabulary is missing control token {s:?}"))
        };
        Ok(Self {
            im_start: get("<|im_start|>")?,
            im_end: get("<|im_end|>")?,
            think_open: get("<think>")?,
            think_close: get("</think>")?,
            call_open: get("<tool_call>")?,
            call_close: get("</tool_call>")?,
            response_open: get("<tool_response>")?,
            response_close: get("</tool_response>")?,
        })
    }
}

/// The template's default reasoning-effort preamble (`xhigh`). It is part of
/// the trained prompt format, not advice we made up.
const XHIGH_INSTRUCTIONS: &str = "Reasoning effort is set to xhigh. Please think carefully \
     through the task, validate key assumptions, consider plausible alternatives, and \
     prioritize correctness, consistency, and clarity in the final answer.";

struct Builder<'a> {
    tok: &'a Tokenizer,
    sp: &'a Special,
    ids: Vec<u32>,
}

impl<'a> Builder<'a> {
    fn push(&mut self, id: u32) -> &mut Self {
        self.ids.push(id);
        self
    }

    /// Encode as ordinary content — control-token spellings stay text.
    fn text(&mut self, s: &str) -> &mut Self {
        self.tok.encode_into(s, &mut self.ids);
        self
    }

    fn open(&mut self, role: &str) -> &mut Self {
        self.push(self.sp.im_start).text(role).text("\n")
    }

    fn close(&mut self) -> &mut Self {
        self.push(self.sp.im_end).text("\n")
    }

    /// One `<tool_call>` block for an assistant replay.
    fn tool_call(&mut self, name: &str, arguments: &Value) -> &mut Self {
        let (open, close) = (self.sp.call_open, self.sp.call_close);
        self.push(open).text(&format!("\n<function={name}>\n"));
        if let Value::Object(map) = arguments {
            for (k, v) in map {
                let rendered = match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                self.text(&format!("<parameter={k}>\n{rendered}\n</parameter>\n"));
            }
        }
        self.text("</function>\n").push(close)
    }
}

/// Build the full prompt for a request, mirroring the checkpoint's template
/// (with its defaults: reasoning effort `xhigh`, thinking replayed verbatim).
pub fn build_prompt(
    tok: &Tokenizer,
    sp: &Special,
    messages: &[Message],
    tools: &[Tool],
    enable_thinking: bool,
) -> Vec<u32> {
    let mut b = Builder {
        tok,
        sp,
        ids: Vec::new(),
    };

    let leading_system = messages
        .first()
        .filter(|m| m.role == "system" || m.role == "developer");
    let system_text = leading_system.map(|m| m.text());
    let system_text = system_text.as_deref().map(str::trim).unwrap_or("");
    let reasoning = enable_thinking.then_some(XHIGH_INSTRUCTIONS).unwrap_or("");

    if !tools.is_empty() {
        b.open("system");
        if !reasoning.is_empty() {
            b.text(reasoning).text("\n\n");
        }
        b.text("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for t in tools {
            // The template runs each tool through `tojson`; we serialize the
            // OpenAI-shaped declaration the same way.
            let decl = serde_json::json!({
                "type": t.kind,
                "function": {
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters,
                },
            });
            b.text("\n").text(&decl.to_string());
        }
        b.text("\n</tools>");
        b.text(
            "\n\nIf you choose to call a function ONLY reply in the following format \
             with NO suffix:\n\n",
        );
        // The format example embeds the real <tool_call> tokens, exactly as
        // special-parsing the rendered template would.
        b.push(sp.call_open).text(
            "\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n\
             </parameter>\n<parameter=example_parameter_2>\nThis is the value for the second \
             parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n",
        );
        b.push(sp.call_close).text(
            "\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: \
             an inner <function=...></function> block must be nested within ",
        );
        b.push(sp.call_open);
        b.push(sp.call_close);
        b.text(
            " XML tags\n- Required parameters MUST be specified\n- You may provide optional \
             reasoning for your function call in natural language BEFORE the function call, \
             but NOT after\n- If there is no function call available, answer the question like \
             normal with your current knowledge and do not tell the user about function calls\n\
             </IMPORTANT>",
        );
        if !system_text.is_empty() {
            b.text("\n\n").text(system_text);
        }
        b.close();
    } else if !system_text.is_empty() {
        b.open("system");
        if !reasoning.is_empty() {
            b.text(reasoning).text("\n\n");
        }
        b.text(system_text);
        b.close();
    } else if !reasoning.is_empty() {
        b.open("system");
        b.text(reasoning);
        b.close();
    }

    let rest = if leading_system.is_some() {
        &messages[1..]
    } else {
        messages
    };

    for (i, m) in rest.iter().enumerate() {
        match m.role.as_str() {
            "user" => {
                b.open("user");
                b.text(m.text().trim());
                b.close();
            }
            "assistant" => {
                b.open("assistant");
                // The template replays thinking on every assistant turn
                // (`preserve_thinking` defaults to true).
                let reasoning = m.reasoning.as_deref().unwrap_or("").trim();
                b.push(sp.think_open)
                    .text("\n")
                    .text(reasoning)
                    .text("\n")
                    .push(sp.think_close)
                    .text("\n\n");
                let content = m.text();
                let content = content.trim();
                b.text(content);
                for (ci, c) in m.tool_calls.iter().enumerate() {
                    if ci > 0 || !content.is_empty() {
                        b.text(if ci == 0 { "\n\n" } else { "\n" });
                    }
                    let args = serde_json::from_str::<Value>(&c.function.arguments)
                        .unwrap_or_else(|_| Value::Object(Default::default()));
                    b.tool_call(&c.function.name, &args);
                }
                b.close();
            }
            // Tool results ride inside user turns; consecutive ones share it.
            "tool" => {
                let first = i == 0 || rest[i - 1].role != "tool";
                let last = rest.get(i + 1).is_none_or(|n| n.role != "tool");
                if first {
                    b.push(sp.im_start).text("user");
                }
                b.text("\n")
                    .push(sp.response_open)
                    .text("\n")
                    .text(m.text().trim())
                    .text("\n")
                    .push(sp.response_close);
                if last {
                    b.close();
                }
            }
            // Stray system/developer messages mid-conversation: the template
            // raises; we fold them in as user turns instead of failing.
            _ => {
                b.open("user");
                b.text(m.text().trim());
                b.close();
            }
        }
    }

    b.open("assistant");
    if enable_thinking {
        b.push(sp.think_open).text("\n");
    } else {
        b.push(sp.think_open)
            .text("\n\n")
            .push(sp.think_close)
            .text("\n\n");
    }
    b.ids
}

/// Split a generated token sequence into reasoning, content, and tool calls.
///
/// `thinking_open` says whether the generation prompt left a `<think>` block
/// open, i.e. the stream *starts* inside reasoning.
pub fn parse_completion(
    tok: &Tokenizer,
    sp: &Special,
    ids: &[u32],
    thinking_open: bool,
) -> Completion {
    let mut out = Completion::default();
    let mut i = 0;
    let mut in_think = thinking_open;

    while i < ids.len() {
        let id = ids[i];
        if id == sp.im_end {
            break;
        }
        if in_think || id == sp.think_open {
            let start = i + usize::from(!in_think);
            let end = ids[start..]
                .iter()
                .position(|&x| x == sp.think_close)
                .map(|p| start + p)
                .unwrap_or(ids.len());
            out.reasoning.push_str(&tok.decode(&ids[start..end]));
            in_think = false;
            i = (end + 1).min(ids.len());
        } else if id == sp.call_open {
            let end = ids[i + 1..]
                .iter()
                .position(|&x| x == sp.call_close)
                .map(|p| i + 1 + p)
                .unwrap_or(ids.len());
            if let Some(c) = parse_function_block(&tok.decode(&ids[i + 1..end])) {
                out.tool_calls.push(c);
            }
            i = (end + 1).min(ids.len());
        } else {
            let start = i;
            while i < ids.len()
                && ids[i] != sp.think_open
                && ids[i] != sp.call_open
                && ids[i] != sp.im_end
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

/// Parse one `<function=...>` block into a call.
///
/// Values that parse as JSON are kept typed; anything else stays a string —
/// the template renders string arguments raw, so the reverse mapping is
/// necessarily heuristic.
pub fn parse_function_block(text: &str) -> Option<ToolCall> {
    let text = text.trim();
    let rest = text.strip_prefix("<function=")?;
    let (name, mut rest) = rest.split_once('>')?;

    let mut args = serde_json::Map::new();
    while let Some(p) = rest.find("<parameter=") {
        let after = &rest[p + "<parameter=".len()..];
        let Some((key, val_rest)) = after.split_once('>') else {
            break;
        };
        let end = val_rest.find("</parameter>").unwrap_or(val_rest.len());
        let raw = val_rest[..end]
            .strip_prefix('\n')
            .unwrap_or(&val_rest[..end]);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let value = serde_json::from_str::<Value>(raw).unwrap_or(Value::String(raw.to_string()));
        args.insert(key.to_string(), value);
        rest = &val_rest[end..];
    }

    Some(ToolCall {
        name: name.to_string(),
        arguments: Value::Object(args),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_block_round_trip() {
        let c = parse_function_block(
            "<function=read_file>\n<parameter=path>\nsrc/main.rs\n</parameter>\n\
             <parameter=limit>\n42\n</parameter>\n</function>",
        )
        .unwrap();
        assert_eq!(c.name, "read_file");
        assert_eq!(c.arguments["path"], "src/main.rs");
        assert_eq!(c.arguments["limit"], 42);
    }

    #[test]
    fn multiline_parameter_values_survive() {
        let c = parse_function_block(
            "<function=write>\n<parameter=text>\nline one\nline two\n</parameter>\n</function>",
        )
        .unwrap();
        assert_eq!(c.arguments["text"], "line one\nline two");
    }

    #[test]
    fn malformed_block_is_none() {
        assert!(parse_function_block("not a call").is_none());
    }
}
