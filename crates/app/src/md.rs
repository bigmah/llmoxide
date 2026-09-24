//! Markdown replies, rendered as elements.
//!
//! The parser's events are folded into a small tree and the tree is turned
//! into RSX. No HTML string is ever built: raw HTML in a reply is shown as
//! text, and links are styled but not clickable (there is nothing to open
//! them with — the renderer is built without a network stack).

use dioxus_native::prelude::*;
use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag};

#[derive(Clone, PartialEq)]
enum Kind {
    Root,
    P,
    H(u8),
    Quote,
    Code(String),
    Ul,
    Ol(u64),
    Li,
    Em,
    Strong,
    Del,
    Link,
    Table,
    Row { head: bool },
    Cell,
    Span,
}

#[derive(Clone, PartialEq)]
enum Node {
    El(Kind, Vec<Node>),
    Text(String),
    Code(String),
    Break,
    Rule,
}

fn parse(src: &str) -> Vec<Node> {
    let opts = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
    let mut stack: Vec<(Kind, Vec<Node>)> = vec![(Kind::Root, Vec::new())];
    let push = |stack: &mut Vec<(Kind, Vec<Node>)>, n: Node| stack.last_mut().unwrap().1.push(n);

    for ev in Parser::new_ext(src, opts) {
        match ev {
            Event::Start(tag) => {
                let kind = match tag {
                    Tag::Paragraph => Kind::P,
                    Tag::Heading { level, .. } => Kind::H(match level {
                        HeadingLevel::H1 => 1,
                        HeadingLevel::H2 => 2,
                        HeadingLevel::H3 => 3,
                        _ => 4,
                    }),
                    Tag::BlockQuote(_) => Kind::Quote,
                    Tag::CodeBlock(CodeBlockKind::Fenced(lang)) => {
                        Kind::Code(lang.split_whitespace().next().unwrap_or("").to_string())
                    }
                    Tag::CodeBlock(CodeBlockKind::Indented) => Kind::Code(String::new()),
                    Tag::List(None) => Kind::Ul,
                    Tag::List(Some(start)) => Kind::Ol(start),
                    Tag::Item => Kind::Li,
                    Tag::Emphasis => Kind::Em,
                    Tag::Strong => Kind::Strong,
                    Tag::Strikethrough => Kind::Del,
                    Tag::Link { .. } => Kind::Link,
                    Tag::Table(_) => Kind::Table,
                    Tag::TableHead => Kind::Row { head: true },
                    Tag::TableRow => Kind::Row { head: false },
                    Tag::TableCell => Kind::Cell,
                    _ => Kind::Span,
                };
                stack.push((kind, Vec::new()));
            }
            Event::End(_) => {
                if stack.len() > 1 {
                    let (kind, children) = stack.pop().unwrap();
                    push(&mut stack, Node::El(kind, children));
                }
            }
            Event::Text(t) | Event::Html(t) | Event::InlineHtml(t) => {
                push(&mut stack, Node::Text(t.into_string()))
            }
            Event::InlineMath(t) | Event::DisplayMath(t) => push(&mut stack, Node::Text(t.into_string())),
            Event::Code(t) => push(&mut stack, Node::Code(t.into_string())),
            Event::SoftBreak => push(&mut stack, Node::Text(" ".into())),
            Event::HardBreak => push(&mut stack, Node::Break),
            Event::Rule => push(&mut stack, Node::Rule),
            Event::TaskListMarker(done) => {
                push(&mut stack, Node::Text(if done { "☑ " } else { "☐ " }.into()))
            }
            Event::FootnoteReference(t) => push(&mut stack, Node::Text(format!("[{t}]"))),
        }
    }
    // A reply still streaming in can leave elements open; close them.
    while stack.len() > 1 {
        let (kind, children) = stack.pop().unwrap();
        push(&mut stack, Node::El(kind, children));
    }
    stack.pop().unwrap().1
}

/// A reply's Markdown as elements.
#[component]
pub fn Markdown(text: String) -> Element {
    let nodes = parse(&text);
    rsx! {
        div { class: "md", {nodes.iter().map(render)} }
    }
}

fn render(node: &Node) -> Element {
    match node {
        Node::Text(t) => rsx! { "{t}" },
        Node::Code(t) => rsx! { span { class: "icode", "{t}" } },
        Node::Break => rsx! { br {} },
        Node::Rule => rsx! { div { class: "hr" } },
        Node::El(kind, kids) => {
            let inner = kids.iter().map(render);
            match kind {
                Kind::Root | Kind::Span => rsx! { span { {inner} } },
                Kind::P => rsx! { p { {inner} } },
                Kind::H(n) => rsx! { div { class: "h h{n}", {inner} } },
                Kind::Quote => rsx! { div { class: "quote", {inner} } },
                Kind::Code(lang) => {
                    let code: String = kids
                        .iter()
                        .filter_map(|k| match k {
                            Node::Text(t) => Some(t.as_str()),
                            _ => None,
                        })
                        .collect();
                    let code = code.strip_suffix('\n').unwrap_or(&code).to_string();
                    let label = if lang.is_empty() { "code" } else { lang.as_str() };
                    rsx! {
                        div { class: "codeblock",
                            div { class: "codehead", "{label}" }
                            pre { "{code}" }
                        }
                    }
                }
                Kind::Ul | Kind::Ol(_) => {
                    let items = kids.iter().enumerate().map(|(i, item)| {
                        let marker = match kind {
                            Kind::Ol(start) => format!("{}.", start + i as u64),
                            _ => "•".to_string(),
                        };
                        let body = match item {
                            Node::El(Kind::Li, c) => c.iter().map(render).collect::<Vec<_>>(),
                            other => vec![render(other)],
                        };
                        rsx! {
                            div { key: "{i}", class: "li",
                                span { class: "marker", "{marker}" }
                                div { class: "libody", {body.into_iter()} }
                            }
                        }
                    });
                    rsx! { div { class: "list", {items} } }
                }
                Kind::Li => rsx! { div { {inner} } },
                Kind::Em => rsx! { em { {inner} } },
                Kind::Strong => rsx! { strong { {inner} } },
                Kind::Del => rsx! { span { class: "del", {inner} } },
                Kind::Link => rsx! { span { class: "link", {inner} } },
                Kind::Table => rsx! { div { class: "table", {inner} } },
                Kind::Row { head } => rsx! {
                    div { class: if *head { "tr th" } else { "tr" }, {inner} }
                },
                Kind::Cell => rsx! { div { class: "td", {inner} } },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unterminated_fence_still_closes() {
        let nodes = parse("Here:\n\n```rust\nfn main() {");
        assert!(matches!(&nodes[1], Node::El(Kind::Code(l), _) if l == "rust"));
    }

    #[test]
    fn ordered_list_keeps_its_start() {
        let nodes = parse("3. a\n4. b");
        assert!(matches!(&nodes[0], Node::El(Kind::Ol(3), items) if items.len() == 2));
    }
}
