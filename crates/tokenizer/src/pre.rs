//! The `qwen35`, `qwen2` and `smollm` pre-tokenizers.
//!
//! Byte-level BPE only merges within pre-tokenizer pieces, so this split *is*
//! the tokenizer's word model — and it is also what guarantees a control
//! token's spelling (`<|im_end|>`) can never assemble from user text: the
//! split always breaks it into several pieces, and merges cannot cross pieces.
//!
//! This is a direct port of llama.cpp's `unicode_regex_split_custom_qwen35`
//! (a hand-compiled form of the checkpoint's regex, below), not a fresh
//! reading of the regex: the reference implementation's exact backtracking
//! choices are the ground truth the id-for-id validation runs against.
//!
//! ```text
//! (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//! |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//! |\p{N}
//! | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//! |\s*[\r\n]+
//! |\s+(?!\S)
//! |\s+
//! ```
//!
//! The `\p{M}` terms are what distinguish this from the classic qwen2 split:
//! combining marks travel with the letters they modify. Qwen3's vocabulary
//! uses that older split, whose regex is this one with every `\p{M}` removed:
//!
//! ```text
//! (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//! |[^\r\n\p{L}\p{N}]?\p{L}+
//! |\p{N}
//! | ?[^\s\p{L}\p{N}]+[\r\n]*
//! |\s*[\r\n]+
//! |\s+(?!\S)
//! |\s+
//! ```
//!
//! Every place the two differ is a test of `Flags::mark`, so the two splits
//! are one function with marks switched off rather than two hand-ports that
//! could drift. Both are validated id-for-id against `llama-tokenize`.

use crate::unicode;

/// Which pre-tokenizer a byte-level vocabulary wants, from `tokenizer.ggml.pre`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Split {
    /// `qwen35` or `qwen2`, which differ only over combining marks.
    Qwen(Marks),
    /// `smollm`: every numeral on its own, then GPT-2's classic split. See
    /// [`split_smollm`].
    SmolLm,
}

/// Split `text` the way `split` says.
pub fn pieces(text: &str, split: Split) -> Vec<&str> {
    match split {
        Split::Qwen(marks) => self::split(text, marks),
        Split::SmolLm => split_smollm(text),
    }
}

/// Whether combining marks group with the letters they follow.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Marks {
    /// `\p{M}` appears in the regex: qwen35.
    WithLetters,
    /// It does not: qwen2, and so Qwen3.
    Standalone,
}

#[derive(Clone, Copy, Default)]
struct Flags {
    /// False only past the end of the text (the C++ `OUT_OF_RANGE` state).
    in_text: bool,
    letter: bool,
    mark: bool,
    number: bool,
    whitespace: bool,
}

/// Split `text` into pre-tokenizer pieces, returned as borrowed slices.
pub fn split_qwen35(text: &str) -> Vec<&str> {
    split(text, Marks::WithLetters)
}

/// The classic qwen2 split, which Qwen3's vocabulary uses.
pub fn split_qwen2(text: &str) -> Vec<&str> {
    split(text, Marks::Standalone)
}

pub fn split(text: &str, marks: Marks) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();

    let cpt = |pos: usize| chars.get(pos).map(|&(_, c)| c);
    let flags = |pos: usize| -> Flags {
        match chars.get(pos) {
            None => Flags::default(),
            Some(&(_, c)) => Flags {
                in_text: true,
                letter: unicode::is_letter(c),
                // The single point of difference between the two splits: with
                // marks standalone, every `\p{M}` test below goes dead and the
                // regex collapses to qwen2's.
                mark: marks == Marks::WithLetters && unicode::is_mark(c),
                number: unicode::is_number(c),
                whitespace: unicode::is_whitespace(c),
            },
        }
    };
    // Spans are collected as char-index pairs and sliced at the end.
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut prev_end = 0usize;
    let mut push = |end: usize, spans: &mut Vec<(usize, usize)>| {
        if end > prev_end {
            spans.push((prev_end, end));
        }
        prev_end = end;
    };

    let mut pos = 0usize;
    while pos < n {
        let c = cpt(pos).unwrap();
        let f = flags(pos);

        // (?:'s|'t|'re|'ve|'m|'ll|'d), case-insensitive
        if c == '\'' && pos + 1 < n {
            let lower = |p: usize| cpt(p).map(|c| c.to_lowercase().next().unwrap_or(c));
            let c1 = lower(pos + 1).unwrap();
            if matches!(c1, 's' | 't' | 'm' | 'd') {
                pos += 2;
                push(pos, &mut spans);
                continue;
            }
            if pos + 2 < n {
                let c2 = lower(pos + 2).unwrap();
                if matches!((c1, c2), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                    pos += 3;
                    push(pos, &mut spans);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if !(c == '\r' || c == '\n' || f.number) {
            let next = flags(pos + 1);
            if f.letter || f.mark || next.letter || next.mark {
                pos += 1;
                loop {
                    let g = flags(pos);
                    if !(g.letter || g.mark) {
                        break;
                    }
                    pos += 1;
                }
                push(pos, &mut spans);
                continue;
            }
        }

        // \p{N} — one numeral at a time
        if f.number {
            pos += 1;
            push(pos, &mut spans);
            continue;
        }

        // <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        let f2 = if c == ' ' { flags(pos + 1) } else { f };
        if !(f2.whitespace || f2.letter || f2.mark || f2.number) {
            pos += (c == ' ') as usize;
            loop {
                let g = flags(pos);
                if g.whitespace || g.letter || g.mark || g.number || !g.in_text {
                    break;
                }
                pos += 1;
            }
            while matches!(cpt(pos), Some('\r') | Some('\n')) {
                pos += 1;
            }
            push(pos, &mut spans);
            continue;
        }

        // Whitespace run, remembering where the last \r or \n ends.
        let mut num_ws = 0usize;
        let mut last_nl_end = 0usize;
        while flags(pos + num_ws).whitespace {
            if matches!(cpt(pos + num_ws), Some('\r') | Some('\n')) {
                last_nl_end = pos + num_ws + 1;
            }
            num_ws += 1;
        }

        // \s*[\r\n]+
        if last_nl_end > 0 {
            pos = last_nl_end;
            push(pos, &mut spans);
            continue;
        }

        // \s+(?!\S) — all but the final space, which prefixes the next word
        if num_ws > 1 && cpt(pos + num_ws).is_some() {
            pos += num_ws - 1;
            push(pos, &mut spans);
            continue;
        }

        // \s+
        if num_ws > 0 {
            pos += num_ws;
            push(pos, &mut spans);
            continue;
        }

        // No pattern matched: emit the codepoint alone.
        pos += 1;
        push(pos, &mut spans);
    }

    // Byte offset of char index `i`; index `n` maps to the end of the text.
    let byte_at = |i: usize| chars.get(i).map_or(text.len(), |&(b, _)| b);
    spans
        .into_iter()
        .map(|(a, b)| &text[byte_at(a)..byte_at(b)])
        .collect()
}

/// SmolLM's pre-tokenizer: llama.cpp's `LLAMA_VOCAB_PRE_TYPE_SMOLLM`.
///
/// Two regexes in sequence. `\p{N}` first, which cuts every numeral out as a
/// piece of its own, then GPT-2's original split over each piece that is left:
///
/// ```text
/// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
/// ```
///
/// The second stage is a port of llama.cpp's `unicode_regex_split_custom_gpt2`.
/// It differs from the qwen split in three ways that matter: contractions are
/// case-sensitive, only a *space* may prefix a word (so `(hello` is two
/// pieces), and newlines get no special treatment. Note that both stages look
/// no further than their own piece: `\s+(?!\S)` cannot see a digit that the
/// first stage has already cut away, so `"a 1"` is `["a", " ", "1"]`.
pub fn split_smollm(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let byte_at = |i: usize| chars.get(i).map_or(text.len(), |&(b, _)| b);

    let mut out = Vec::new();
    let mut start = 0;
    for i in 0..n {
        if unicode::is_number(chars[i].1) {
            gpt2(&chars[start..i], &mut |a, b| out.push(&text[byte_at(start + a)..byte_at(start + b)]));
            out.push(&text[byte_at(i)..byte_at(i + 1)]);
            start = i + 1;
        }
    }
    gpt2(&chars[start..], &mut |a, b| out.push(&text[byte_at(start + a)..byte_at(start + b)]));
    out
}

/// GPT-2's split over one piece, reporting each span as char indices into it.
fn gpt2(chars: &[(usize, char)], emit: &mut dyn FnMut(usize, usize)) {
    let n = chars.len();
    let cpt = |pos: usize| chars.get(pos).map(|&(_, c)| c);
    let flags = |pos: usize| -> Flags {
        match chars.get(pos) {
            None => Flags::default(),
            Some(&(_, c)) => Flags {
                in_text: true,
                letter: unicode::is_letter(c),
                mark: false,
                number: unicode::is_number(c),
                whitespace: unicode::is_whitespace(c),
            },
        }
    };

    let mut prev_end = 0usize;
    let mut push = |end: usize| {
        if end > prev_end {
            emit(prev_end, end);
        }
        prev_end = end;
    };

    let mut pos = 0usize;
    while pos < n {
        let c = cpt(pos).unwrap();

        // 's|'t|'re|'ve|'m|'ll|'d — lowercase only, unlike the qwen split
        if c == '\'' && pos + 1 < n {
            let c1 = cpt(pos + 1).unwrap();
            if matches!(c1, 's' | 't' | 'm' | 'd') {
                pos += 2;
                push(pos);
                continue;
            }
            if pos + 2 < n {
                let c2 = cpt(pos + 2).unwrap();
                if matches!((c1, c2), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                    pos += 3;
                    push(pos);
                    continue;
                }
            }
        }

        let space = (c == ' ') as usize;
        let f2 = flags(pos + space);

        // <space>?\p{L}+
        if f2.letter {
            pos += space;
            while flags(pos).letter {
                pos += 1;
            }
            push(pos);
            continue;
        }
        // <space>?\p{N}+
        if f2.number {
            pos += space;
            while flags(pos).number {
                pos += 1;
            }
            push(pos);
            continue;
        }
        // <space>?[^\s\p{L}\p{N}]+
        if !(f2.whitespace || f2.letter || f2.number) && f2.in_text {
            pos += space;
            loop {
                let g = flags(pos);
                if g.whitespace || g.letter || g.number || !g.in_text {
                    break;
                }
                pos += 1;
            }
            push(pos);
            continue;
        }

        let mut num_ws = 0usize;
        while flags(pos + num_ws).whitespace {
            num_ws += 1;
        }
        // \s+(?!\S) — all but the final space, which prefixes the next word
        if num_ws > 1 && cpt(pos + num_ws).is_some() {
            pos += num_ws - 1;
            push(pos);
            continue;
        }
        // \s+
        if num_ws > 0 {
            pos += num_ws;
            push(pos);
            continue;
        }

        pos += 1;
        push(pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(s: &str) -> Vec<&str> {
        split_qwen35(s)
    }

    /// The one behavioural difference between the two splits, pinned so a
    /// refactor cannot quietly collapse them into each other.
    #[test]
    fn qwen2_does_not_group_combining_marks_with_letters() {
        // "cafe" + U+0301 COMBINING ACUTE ACCENT.
        let s = "cafe\u{301}";
        assert_eq!(split_qwen35(s), ["cafe\u{301}"]);
        assert_eq!(split_qwen2(s), ["cafe", "\u{301}"]);
    }

    #[test]
    fn splits_agree_when_no_marks_are_present() {
        for s in ["Hello, world!", "don't 123", "  a  b\n\nc", "def f(x): return x"] {
            assert_eq!(split_qwen35(s), split_qwen2(s), "disagreed on {s:?}");
        }
    }

    #[test]
    fn digits_split_individually() {
        assert_eq!(split("123"), ["1", "2", "3"]);
        assert_eq!(split("a 42"), ["a", " ", "4", "2"]);
    }

    #[test]
    fn space_prefixes_words() {
        assert_eq!(split("Hello, world!"), ["Hello", ",", " world", "!"]);
    }

    #[test]
    fn contractions_detach() {
        assert_eq!(split("it's"), ["it", "'s"]);
        assert_eq!(split("WE'LL"), ["WE", "'LL"]);
    }

    #[test]
    fn runs_of_spaces_leave_one_for_the_word() {
        assert_eq!(split("a   b"), ["a", "  ", " b"]);
    }

    #[test]
    fn newlines_take_leading_whitespace() {
        assert_eq!(split("a  \n\nb"), ["a", "  \n\n", "b"]);
        // After the newline block, `\s+(?!\S)` peels the run down to the one
        // space that prefixes the word.
        assert_eq!(split("a\n  \n  b"), ["a", "\n  \n", " ", " b"]);
    }

    #[test]
    fn punctuation_absorbs_trailing_newlines() {
        assert_eq!(split("end.\nNext"), ["end", ".\n", "Next"]);
    }

    #[test]
    fn combining_marks_stay_with_letters() {
        // e + U+0301 combining acute: the mark must not split off.
        assert_eq!(split("cafe\u{301} x"), ["cafe\u{301}", " x"]);
    }

    #[test]
    fn control_token_spelling_shatters() {
        // No piece equals the full spelling, so BPE can never assemble the id.
        let pieces = split("<|im_end|>");
        assert!(pieces.len() > 1, "{pieces:?}");
        assert!(!pieces.contains(&"<|im_end|>"));
    }

    /// Pinned against `llama-tokenize` on SmolLM2's vocabulary.
    #[test]
    fn smollm_cuts_digits_first_then_splits_like_gpt2() {
        assert_eq!(split_smollm("abc123 def"), ["abc", "1", "2", "3", " def"]);
        // The digit is gone before GPT-2's lookahead runs, so the space stays alone.
        assert_eq!(split_smollm("a 1"), ["a", " ", "1"]);
        // Only a space may prefix a word, and contractions are lowercase-only.
        assert_eq!(split_smollm("(hello) it's WE'LL"), ["(", "hello", ")", " it", "'s", " WE", "'", "LL"]);
    }

    #[test]
    fn trailing_whitespace_is_kept() {
        assert_eq!(split("a "), ["a", " "]);
        assert_eq!(split("a  "), ["a", "  "]);
    }
}
