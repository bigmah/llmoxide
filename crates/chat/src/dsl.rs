//! The gemma4 tool-calling DSL.
//!
//! This model does not emit JSON for tool calls. It uses a compact custom
//! syntax where keys are bare, strings are delimited by the single token
//! `<|"|>`, and everything else is written literally:
//!
//! ```text
//! <|tool_call>call:read_file{path:<|"|>src/main.rs<|"|>,limit:20}<tool_call|>
//! ```
//!
//! Tool *declarations* use a matching shape, which is what the model was
//! trained to read:
//!
//! ```text
//! <|tool>declaration:read_file{description:<|"|>…<|"|>,parameters:{...}}<tool|>
//! ```
//!
//! Translating both directions is what lets an OpenAI-shaped client drive it.

use serde_json::{Map, Value};

/// The string-quote marker. It is a single vocabulary token (id 52), so it can
/// never be produced by ordinary text and is unambiguous as a delimiter.
pub const QUOTE: &str = "<|\"|>";

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Render a JSON value in the model's argument syntax.
pub fn encode_value(v: &Value, out: &mut String) {
    match v {
        Value::String(s) => {
            out.push_str(QUOTE);
            out.push_str(s);
            out.push_str(QUOTE);
        }
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Null => out.push_str("null"),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::Array(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                encode_value(item, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, val)) in sorted(m).into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                // Object keys inside argument values are quoted; parameter
                // names in declarations are not. See `encode_properties`.
                out.push_str(QUOTE);
                out.push_str(k);
                out.push_str(QUOTE);
                out.push(':');
                encode_value(val, out);
            }
            out.push('}');
        }
    }
}

/// The template sorts object keys, so declarations are stable across runs and
/// stay cache-friendly.
fn sorted(m: &Map<String, Value>) -> Vec<(&String, &Value)> {
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by(|a, b| a.0.cmp(b.0));
    v
}

/// Encode a JSON-Schema `properties` map as the model's parameter syntax.
fn encode_properties(props: &Map<String, Value>, out: &mut String) {
    for (i, (name, spec)) in sorted(props).into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(name);
        out.push_str(":{");
        let mut wrote = false;
        let mut comma = |out: &mut String, wrote: &mut bool| {
            if *wrote {
                out.push(',');
            }
            *wrote = true;
        };

        if let Some(Value::String(d)) = spec.get("description") {
            comma(out, &mut wrote);
            out.push_str("description:");
            out.push_str(QUOTE);
            out.push_str(d);
            out.push_str(QUOTE);
        }

        let ty = spec
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("string")
            .to_uppercase();

        if ty == "STRING" {
            if let Some(Value::Array(en)) = spec.get("enum") {
                comma(out, &mut wrote);
                out.push_str("enum:");
                encode_value(&Value::Array(en.clone()), out);
            }
        } else if ty == "ARRAY" {
            if let Some(Value::Object(items)) = spec.get("items") {
                comma(out, &mut wrote);
                out.push_str("items:{type:");
                out.push_str(QUOTE);
                out.push_str(
                    &items
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("string")
                        .to_uppercase(),
                );
                out.push_str(QUOTE);
                out.push('}');
            }
        } else if ty == "OBJECT" {
            if let Some(Value::Object(inner)) = spec.get("properties") {
                comma(out, &mut wrote);
                out.push_str("properties:{");
                encode_properties(inner, out);
                out.push('}');
            }
        }

        comma(out, &mut wrote);
        out.push_str("type:");
        out.push_str(QUOTE);
        out.push_str(&ty);
        out.push_str(QUOTE);
        out.push('}');
    }
}

/// Encode one tool declaration body (without the surrounding control tokens).
pub fn encode_declaration(name: &str, description: &str, params: &Value) -> String {
    let mut out = String::new();
    out.push_str("declaration:");
    out.push_str(name);
    out.push_str("{description:");
    out.push_str(QUOTE);
    out.push_str(description);
    out.push_str(QUOTE);

    if let Some(obj) = params.as_object() {
        out.push_str(",parameters:{");
        if let Some(Value::Object(props)) = obj.get("properties") {
            out.push_str("properties:{");
            encode_properties(props, &mut out);
            out.push_str("},");
        }
        if let Some(Value::Array(req)) = obj.get("required") {
            out.push_str("required:[");
            for (i, r) in req.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(QUOTE);
                out.push_str(r.as_str().unwrap_or_default());
                out.push_str(QUOTE);
            }
            out.push_str("],");
        }
        out.push_str("type:");
        out.push_str(QUOTE);
        out.push_str(
            &obj.get("type")
                .and_then(Value::as_str)
                .unwrap_or("object")
                .to_uppercase(),
        );
        out.push_str(QUOTE);
        out.push('}');
    }
    out.push('}');
    out
}

/// Encode a tool call the model previously made, for conversation replay.
pub fn encode_call(name: &str, args: &Value) -> String {
    let mut out = String::new();
    out.push_str("call:");
    out.push_str(name);
    out.push('{');
    if let Some(m) = args.as_object() {
        for (i, (k, v)) in sorted(m).into_iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(k);
            out.push(':');
            encode_value(v, &mut out);
        }
    }
    out.push('}');
    out
}

/// Encode a tool result block body.
pub fn encode_response(name: &str, content: &str) -> String {
    let mut out = String::new();
    out.push_str("response:");
    out.push_str(name);
    // Non-object results are wrapped in `{value:...}`, matching the template.
    match serde_json::from_str::<Value>(content) {
        Ok(Value::Object(m)) => {
            out.push('{');
            for (i, (k, v)) in sorted(&m).into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(k);
                out.push(':');
                encode_value(v, &mut out);
            }
            out.push('}');
        }
        _ => {
            out.push_str("{value:");
            encode_value(&Value::String(content.to_string()), &mut out);
            out.push('}');
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Parse a `call:NAME{...}` body into a name and a JSON argument object.
///
/// The grammar is small but not JSON: keys are bare, and only `<|"|>`-delimited
/// runs are strings. Anything unquoted that does not parse as a literal is kept
/// as a string so a malformed call degrades instead of being dropped.
pub fn parse_call(body: &str) -> Option<ToolCall> {
    let rest = body.strip_prefix("call:")?;
    let open = rest.find('{')?;
    let name = rest[..open].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let inner = rest[open + 1..].strip_suffix('}').unwrap_or(&rest[open + 1..]);

    let mut p = Parser { s: inner, i: 0 };
    let args = p.object_body();
    Some(ToolCall {
        name,
        arguments: Value::Object(args),
    })
}

/// Find `call:NAME{...}` sequences in ordinary text.
///
/// The model sometimes emits the call DSL without wrapping it in the
/// `<|tool_call>` control tokens, in which case it arrives as plain content.
/// Returns byte ranges alongside the parsed calls so the caller can excise them
/// — leaking raw DSL to a user is worse than dropping it.
pub fn find_bare_calls(text: &str) -> Vec<(std::ops::Range<usize>, ToolCall)> {
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(rel) = text[from..].find("call:") {
        let start = from + rel;
        let Some(open_rel) = text[start..].find('{') else { break };
        let open = start + open_rel;

        // Walk to the matching brace; arguments nest.
        let mut depth = 0i32;
        let mut end = None;
        for (off, c) in text[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(open + off + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end) = end else { break };
        match parse_call(&text[start..end]) {
            Some(c) => {
                out.push((start..end, c));
                from = end;
            }
            None => from = open + 1,
        }
    }
    out
}

struct Parser<'a> {
    s: &'a str,
    i: usize,
}

impl<'a> Parser<'a> {
    fn rest(&self) -> &'a str {
        &self.s[self.i..]
    }

    fn eat(&mut self, tok: &str) -> bool {
        if self.rest().starts_with(tok) {
            self.i += tok.len();
            true
        } else {
            false
        }
    }

    /// `key:value` pairs until the input runs out or a closing brace is hit.
    fn object_body(&mut self) -> Map<String, Value> {
        let mut m = Map::new();
        loop {
            self.skip_ws();
            if self.rest().is_empty() || self.rest().starts_with('}') {
                break;
            }
            let key = if self.eat(QUOTE) {
                let k = self.until(QUOTE);
                self.eat(QUOTE);
                k
            } else {
                self.until_any(&[':'])
            };
            if !self.eat(":") {
                break;
            }
            let v = self.value();
            m.insert(key.trim().to_string(), v);
            self.skip_ws();
            if !self.eat(",") {
                break;
            }
        }
        m
    }

    fn value(&mut self) -> Value {
        self.skip_ws();
        if self.eat(QUOTE) {
            let s = self.until(QUOTE);
            self.eat(QUOTE);
            return Value::String(s);
        }
        if self.eat("[") {
            let mut a = Vec::new();
            loop {
                self.skip_ws();
                if self.rest().is_empty() || self.eat("]") {
                    break;
                }
                a.push(self.value());
                self.skip_ws();
                if !self.eat(",") {
                    self.eat("]");
                    break;
                }
            }
            return Value::Array(a);
        }
        if self.eat("{") {
            let m = self.object_body();
            self.eat("}");
            return Value::Object(m);
        }
        // Bare token: a literal if it parses as one, otherwise a string.
        let raw = self.until_any(&[',', '}', ']']);
        let t = raw.trim();
        match t {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            "null" => Value::Null,
            _ => t
                .parse::<i64>()
                .map(Value::from)
                .or_else(|_| t.parse::<f64>().map(Value::from))
                .unwrap_or_else(|_| Value::String(t.to_string())),
        }
    }

    /// Consume up to `end`, respecting nothing — the delimiter is a token that
    /// cannot occur inside text.
    fn until(&mut self, end: &str) -> String {
        match self.rest().find(end) {
            Some(n) => {
                let s = self.rest()[..n].to_string();
                self.i += n;
                s
            }
            None => {
                let s = self.rest().to_string();
                self.i = self.s.len();
                s
            }
        }
    }

    /// Consume until one of `ends`, tracking nesting so a bare value containing
    /// a nested structure is not cut short.
    fn until_any(&mut self, ends: &[char]) -> String {
        let start = self.i;
        let mut depth = 0i32;
        for (off, c) in self.rest().char_indices() {
            match c {
                '{' | '[' => depth += 1,
                '}' | ']' if depth > 0 => depth -= 1,
                _ => {}
            }
            if depth == 0 && ends.contains(&c) {
                self.i = start + off;
                return self.s[start..self.i].to_string();
            }
        }
        self.i = self.s.len();
        self.s[start..].to_string()
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.rest().chars().next() {
            if c.is_whitespace() {
                self.i += c.len_utf8();
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_string_and_number_arguments() {
        let c = parse_call(&format!(
            "call:read_file{{path:{QUOTE}src/main.rs{QUOTE},limit:20}}"
        ))
        .unwrap();
        assert_eq!(c.name, "read_file");
        assert_eq!(c.arguments, json!({"path": "src/main.rs", "limit": 20}));
    }

    #[test]
    fn parses_booleans_arrays_and_nested_objects() {
        let c = parse_call(&format!(
            "call:x{{flag:true,items:[1,2],opts:{{{QUOTE}a{QUOTE}:{QUOTE}b{QUOTE}}}}}"
        ))
        .unwrap();
        assert_eq!(
            c.arguments,
            json!({"flag": true, "items": [1, 2], "opts": {"a": "b"}})
        );
    }

    #[test]
    fn string_arguments_may_contain_delimiters() {
        // Commas and braces inside a quoted run must not terminate the value —
        // this is why the quote marker is its own vocabulary token.
        let c = parse_call(&format!(
            "call:run{{cmd:{QUOTE}ls -la, then {{echo}}{QUOTE},n:1}}"
        ))
        .unwrap();
        assert_eq!(c.arguments["cmd"], json!("ls -la, then {echo}"));
        assert_eq!(c.arguments["n"], json!(1));
    }

    #[test]
    fn call_with_no_arguments() {
        let c = parse_call("call:list_files{}").unwrap();
        assert_eq!(c.name, "list_files");
        assert_eq!(c.arguments, json!({}));
    }

    #[test]
    fn rejects_non_call_bodies() {
        assert!(parse_call("response:foo{a:1}").is_none());
        assert!(parse_call("call:{}").is_none());
    }

    #[test]
    fn unquoted_non_literal_survives_as_string() {
        let c = parse_call("call:x{path:/tmp/a.txt}").unwrap();
        assert_eq!(c.arguments["path"], json!("/tmp/a.txt"));
    }

    #[test]
    fn finds_bare_calls_in_text() {
        let t = format!("Sure.\ncall:search{{q:{QUOTE}rust{QUOTE}}} done");
        let found = find_bare_calls(&t);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].1.name, "search");
        assert_eq!(&t[found[0].0.clone()], "call:search{q:<|\"|>rust<|\"|>}");
    }

    #[test]
    fn bare_call_scan_handles_nested_braces() {
        let t = "call:x{a:{b:1},c:2} tail";
        let f = find_bare_calls(t);
        assert_eq!(f.len(), 1);
        assert_eq!(&t[f[0].0.clone()], "call:x{a:{b:1},c:2}");
    }

    #[test]
    fn bare_call_scan_ignores_ordinary_prose() {
        assert!(find_bare_calls("I will call: the function later.").is_empty());
        assert!(find_bare_calls("no calls here at all").is_empty());
    }

    #[test]
    fn round_trips_through_encode() {
        let args = json!({"a": "hi", "b": 2, "c": true});
        let encoded = encode_call("t", &args);
        assert_eq!(parse_call(&encoded).unwrap().arguments, args);
    }

    #[test]
    fn declaration_shape_matches_template() {
        let d = encode_declaration(
            "read_file",
            "Read a file",
            &json!({
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path"}},
                "required": ["path"]
            }),
        );
        assert!(d.starts_with("declaration:read_file{description:"));
        assert!(d.contains("path:{description:"));
        assert!(d.contains("type:<|\"|>STRING<|\"|>"));
        assert!(d.contains("required:[<|\"|>path<|\"|>]"));
        assert!(d.ends_with("type:<|\"|>OBJECT<|\"|>}}"));
    }

    #[test]
    fn response_wraps_scalars_but_not_objects() {
        assert!(encode_response("t", "plain text").contains("{value:"));
        assert!(encode_response("t", r#"{"ok":true}"#).contains("ok:true"));
    }
}
