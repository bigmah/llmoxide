//! The `qwen35` pre-tokenizer.
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
//! combining marks travel with the letters they modify.

use crate::unicode;

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
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();

    let cpt = |pos: usize| chars.get(pos).map(|&(_, c)| c);
    let flags = |pos: usize| -> Flags {
        match chars.get(pos) {
            None => Flags::default(),
            Some(&(_, c)) => Flags {
                in_text: true,
                letter: unicode::is_letter(c),
                mark: unicode::is_mark(c),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn split(s: &str) -> Vec<&str> {
        split_qwen35(s)
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

    #[test]
    fn trailing_whitespace_is_kept() {
        assert_eq!(split("a "), ["a", " "]);
        assert_eq!(split("a  "), ["a", "  "]);
    }
}
