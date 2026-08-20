//! Rank-ordered byte-pair encoding, in the two dialects our checkpoints use.
//!
//! **gemma4** ([`Style::MetaSpace`]), determined empirically from the file:
//!
//! * `tokenizer.ggml.scores` is uniformly `-1000`, so it carries no signal —
//!   this is rank-ordered BPE, not a unigram/SentencePiece model.
//! * Spaces are rewritten to `▁` (U+2581) before merging, SentencePiece-style,
//!   but `add_space_prefix` is false so no leading `▁` is inserted.
//! * There is no pre-tokenizer regex. Digits fall out as single tokens simply
//!   because no multi-digit ASCII merges exist in the table.
//! * Characters outside the vocabulary decompose into `<0xNN>` byte tokens.
//!
//! **qwen35** ([`Style::ByteLevel`], `tokenizer.ggml.model = "gpt2"`):
//!
//! * Text is first split by the `qwen35` pre-tokenizer ([`pre`]); merges never
//!   cross pieces, which is also what keeps control-token spellings inert.
//! * Each piece's *bytes* map to stand-in characters (GPT-2's byte encoder:
//!   space is `Ġ`, newline `Ċ`), so every input is coverable without an
//!   unknown token and the vocabulary needs no `<0xNN>` fallbacks.
//!
//! The merge loop itself is shared: both dialects are rank-ordered BPE.

use std::collections::HashMap;

use gguf::Gguf;

pub mod pre;
pub mod unicode;

/// SentencePiece meta-space; stands in for U+0020 inside tokens.
const META_SPACE: char = '\u{2581}';

/// How raw text becomes the symbols BPE merges over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    /// SentencePiece-surface BPE: spaces to `▁`, `<0xNN>` byte fallback.
    MetaSpace,
    /// GPT-2 byte-level BPE behind the qwen35 pre-tokenizer.
    ByteLevel,
}

/// GPT-2's byte-to-character bijection. Printable Latin-1 bytes stand for
/// themselves; the 68 others (controls, space, DEL..NBSP, soft hyphen) take
/// consecutive codepoints from U+0100, so every byte has a visible spelling.
pub fn byte_to_char(b: u8) -> char {
    match b {
        0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => b as char,
        0x00..=0x20 => char::from_u32(0x100 + b as u32).unwrap(),
        0x7F..=0xA0 => char::from_u32(0x121 + (b - 0x7F) as u32).unwrap(),
        0xAD => '\u{143}',
    }
}

/// Inverse of [`byte_to_char`]; `None` for characters outside the image.
pub fn char_to_byte(c: char) -> Option<u8> {
    match c as u32 {
        u @ (0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF) => Some(u as u8),
        u @ 0x100..=0x120 => Some((u - 0x100) as u8),
        u @ 0x121..=0x142 => Some(0x7F + (u - 0x121) as u8),
        0x143 => Some(0xAD),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Normal,
    Unknown,
    Control,
    Byte,
    Unused,
}

impl TokenKind {
    fn from_gguf(v: u32) -> Self {
        match v {
            2 => Self::Unknown,
            // 3 = control, 4 = user-defined; both are atomic markers we never
            // split and never emit as text.
            3 | 4 => Self::Control,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }

    pub fn is_control(self) -> bool {
        matches!(self, Self::Control | Self::Unknown)
    }
}

pub struct Tokenizer {
    tokens: Vec<String>,
    kinds: Vec<TokenKind>,
    ids: HashMap<String, u32>,
    /// `(left, right) -> (rank, merged)`. Resolving merges to ids up front turns
    /// the inner BPE loop into integer hashing instead of string concatenation.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// `<0x00>`..`<0xFF>` ids for byte fallback, if the vocab provides them.
    byte_tokens: Option<Box<[u32; 256]>>,
    pub style: Style,
    pub bos: u32,
    pub eos: u32,
    pub unk: u32,
    pub add_bos: bool,
    pub add_space_prefix: bool,
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Self> {
        // `model` names the algorithm family; `pre` names the exact splitter.
        // Only splitters we have validated against llama.cpp are accepted —
        // running a byte-level vocab through the wrong split does not fail, it
        // just tokenizes subtly differently everywhere.
        let style = match g.str("tokenizer.ggml.model").unwrap_or("llama") {
            "gpt2" => {
                let pre = g.str("tokenizer.ggml.pre").unwrap_or("");
                anyhow::ensure!(
                    pre == "qwen35",
                    "byte-level vocab with unsupported pre-tokenizer {pre:?} \
                     (only \"qwen35\" is implemented)"
                );
                Style::ByteLevel
            }
            _ => Style::MetaSpace,
        };

        let tokens: Vec<String> = g.string_array("tokenizer.ggml.tokens")?.to_vec();
        let kinds: Vec<TokenKind> = g
            .u32_array("tokenizer.ggml.token_type")
            .map(|v| v.into_iter().map(TokenKind::from_gguf).collect())
            .unwrap_or_else(|_| vec![TokenKind::Normal; tokens.len()]);
        anyhow::ensure!(
            kinds.len() == tokens.len(),
            "token_type/tokens length mismatch"
        );

        let mut ids = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            // First occurrence wins; duplicate spellings are pathological but
            // shouldn't make loading fail.
            ids.entry(t.clone()).or_insert(i as u32);
        }

        // Merges are stored as "left right", ordered best-rank-first.
        let raw_merges = g.string_array("tokenizer.ggml.merges").unwrap_or(&[]);
        let mut merges = HashMap::with_capacity(raw_merges.len());
        let mut merged = String::new();
        for (rank, m) in raw_merges.iter().enumerate() {
            let Some((l, r)) = m.split_once(' ') else {
                continue;
            };
            let (Some(&li), Some(&ri)) = (ids.get(l), ids.get(r)) else {
                continue;
            };
            merged.clear();
            merged.push_str(l);
            merged.push_str(r);
            let Some(&mi) = ids.get(&merged) else { continue };
            merges.entry((li, ri)).or_insert((rank as u32, mi));
        }

        let byte_tokens = (0..256u32)
            .map(|b| ids.get(&format!("<0x{b:02X}>")).copied())
            .collect::<Option<Vec<_>>>()
            .and_then(|v| <Box<[u32; 256]>>::try_from(v.into_boxed_slice()).ok());

        Ok(Self {
            tokens,
            kinds,
            ids,
            merges,
            byte_tokens,
            style,
            bos: g.u64("tokenizer.ggml.bos_token_id").unwrap_or(2) as u32,
            eos: g.u64("tokenizer.ggml.eos_token_id").unwrap_or(1) as u32,
            unk: g.u64("tokenizer.ggml.unknown_token_id").unwrap_or(3) as u32,
            // When the key is absent llama.cpp defaults by vocab family:
            // SentencePiece vocabs prepend BOS, byte-level (gpt2) ones don't.
            // Getting this wrong is silent and *almost* harmless — one stray
            // `<|endoftext|>` up front — until the model reads the whole
            // prompt as a document fragment and ends its turn early.
            add_bos: g
                .get("tokenizer.ggml.add_bos_token")
                .and_then(gguf::Value::as_bool)
                .unwrap_or(style == Style::MetaSpace),
            add_space_prefix: g
                .get("tokenizer.ggml.add_space_prefix")
                .and_then(gguf::Value::as_bool)
                .unwrap_or(false),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_text(&self, id: u32) -> &str {
        self.tokens.get(id as usize).map_or("", String::as_str)
    }

    pub fn kind(&self, id: u32) -> TokenKind {
        self.kinds
            .get(id as usize)
            .copied()
            .unwrap_or(TokenKind::Normal)
    }

    /// Look up a token by its exact vocabulary spelling, e.g. `"<|turn>"`.
    pub fn id_of(&self, text: &str) -> Option<u32> {
        self.ids.get(text).copied()
    }

    /// Encode text, optionally prefixing BOS.
    ///
    /// Control-token spellings appearing in the input are deliberately *not*
    /// interpreted — prompt assembly inserts those by id, so text arriving from
    /// a user or a tool result can never forge a turn boundary.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos && self.add_bos {
            out.push(self.bos);
        }
        self.encode_into(text, &mut out);
        out
    }

    pub fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        match self.style {
            Style::MetaSpace => self.encode_metaspace(text, out),
            // Merges never cross pre-tokenizer pieces, so each piece is an
            // independent BPE problem.
            Style::ByteLevel => {
                for piece in pre::split_qwen35(text) {
                    let syms = piece.bytes().map(|b| self.seed_sym(byte_to_char(b)));
                    self.merge_and_emit(syms.collect(), out);
                }
            }
        }
    }

    fn encode_metaspace(&self, text: &str, out: &mut Vec<u32>) {
        // Normalize to the SentencePiece meta-space representation.
        let mut norm = String::with_capacity(text.len() + META_SPACE.len_utf8());
        if self.add_space_prefix {
            norm.push(META_SPACE);
        }
        for c in text.chars() {
            norm.push(if c == ' ' { META_SPACE } else { c });
        }

        // Seed one symbol per character. A character absent from the vocab can
        // never participate in a merge, so it is parked for byte fallback.
        let syms: Vec<Sym> = norm.chars().map(|c| self.seed_sym(c)).collect();
        self.merge_and_emit(syms, out);
    }

    fn seed_sym(&self, c: char) -> Sym {
        let mut buf = [0u8; 4];
        let s = c.encode_utf8(&mut buf);
        Sym {
            id: self.ids.get(s).copied(),
            text: s.to_string(),
            prev: usize::MAX,
            next: usize::MAX,
            alive: true,
        }
    }

    fn merge_and_emit(&self, mut syms: Vec<Sym>, out: &mut Vec<u32>) {
        let n = syms.len();
        if n == 0 {
            return;
        }
        for i in 0..n {
            syms[i].prev = i.checked_sub(1).unwrap_or(usize::MAX);
            syms[i].next = if i + 1 == n { usize::MAX } else { i + 1 };
        }

        // Classic priority-queue BPE: always apply the globally best-ranked
        // adjacent pair, re-examining only the neighbours it creates.
        let mut heap = std::collections::BinaryHeap::new();
        for i in 0..n.saturating_sub(1) {
            self.push_bigram(&syms, i, i + 1, &mut heap);
        }

        while let Some(bg) = heap.pop() {
            let (l, r) = (bg.left, bg.right);
            // Skip stale entries: either side already consumed or re-merged.
            if !syms[l].alive || !syms[r].alive || syms[l].next != r {
                continue;
            }
            if syms[l].id != Some(bg.left_id) || syms[r].id != Some(bg.right_id) {
                continue;
            }

            let right_text = std::mem::take(&mut syms[r].text);
            syms[l].text.push_str(&right_text);
            syms[l].id = Some(bg.merged);
            syms[r].alive = false;

            let nxt = syms[r].next;
            syms[l].next = nxt;
            if nxt != usize::MAX {
                syms[nxt].prev = l;
                self.push_bigram(&syms, l, nxt, &mut heap);
            }
            let prv = syms[l].prev;
            if prv != usize::MAX {
                self.push_bigram(&syms, prv, l, &mut heap);
            }
        }

        let mut i = 0usize;
        while i != usize::MAX {
            let s = &syms[i];
            match s.id {
                Some(id) => out.push(id),
                // Byte-level seeds are always in the vocab (all 256 stand-in
                // characters exist), so fallback only arises for meta-space.
                None => match self.style {
                    Style::MetaSpace => self.push_byte_fallback(&s.text, out),
                    Style::ByteLevel => out.push(self.unk),
                },
            }
            i = s.next;
        }
    }

    fn push_bigram(
        &self,
        syms: &[Sym],
        l: usize,
        r: usize,
        heap: &mut std::collections::BinaryHeap<Bigram>,
    ) {
        let (Some(li), Some(ri)) = (syms[l].id, syms[r].id) else {
            return;
        };
        if let Some(&(rank, merged)) = self.merges.get(&(li, ri)) {
            heap.push(Bigram {
                rank,
                left: l,
                right: r,
                left_id: li,
                right_id: ri,
                merged,
            });
        }
    }

    fn push_byte_fallback(&self, text: &str, out: &mut Vec<u32>) {
        match &self.byte_tokens {
            Some(bt) => out.extend(text.bytes().map(|b| bt[b as usize])),
            None => out.push(self.unk),
        }
    }

    /// Decode ids to text, dropping control tokens.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut d = Decoder::new(self);
        let mut s = String::new();
        for &id in ids {
            s.push_str(&d.push(id));
        }
        s.push_str(&d.flush());
        s
    }
}

struct Sym {
    id: Option<u32>,
    text: String,
    prev: usize,
    next: usize,
    alive: bool,
}

/// Ordered so `BinaryHeap` (a max-heap) yields the *lowest* rank first, with
/// the left-most position breaking ties — matching reference BPE behaviour.
#[derive(PartialEq, Eq)]
struct Bigram {
    rank: u32,
    left: usize,
    right: usize,
    left_id: u32,
    right_id: u32,
    merged: u32,
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Incremental decoder for streaming.
///
/// Byte-fallback tokens can split a UTF-8 sequence across token boundaries, so
/// partial bytes are held back rather than emitted as replacement characters.
pub struct Decoder<'a> {
    tok: &'a Tokenizer,
    pending: Vec<u8>,
}

impl<'a> Decoder<'a> {
    pub fn new(tok: &'a Tokenizer) -> Self {
        Self {
            tok,
            pending: Vec::new(),
        }
    }

    /// Feed one token; returns whatever text is now complete.
    pub fn push(&mut self, id: u32) -> String {
        let kind = self.tok.kind(id);
        if kind.is_control() {
            return String::new();
        }
        let text = self.tok.token_text(id);

        // Byte-level spellings decode character-by-character to raw bytes;
        // partial UTF-8 sequences wait in `pending` like everything else.
        if self.tok.style == Style::ByteLevel {
            self.pending.extend(text.chars().filter_map(char_to_byte));
            return self.take_valid();
        }

        if kind == TokenKind::Byte {
            // Spelled "<0xNN>"; recover the raw byte.
            if let Some(b) = text
                .strip_prefix("<0x")
                .and_then(|s| s.strip_suffix('>'))
                .and_then(|s| u8::from_str_radix(s, 16).ok())
            {
                self.pending.push(b);
                return self.take_valid();
            }
        }

        // `replace` allocates only when a meta-space is actually present.
        self.pending
            .extend_from_slice(text.replace(META_SPACE, " ").as_bytes());
        self.take_valid()
    }

    /// Emit the longest valid UTF-8 prefix, keeping any partial tail buffered.
    fn take_valid(&mut self) -> String {
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                let good = e.valid_up_to();
                // `error_len == None` means the tail is a truncated sequence
                // that a later token may complete; anything else is genuinely
                // invalid and is flushed lossily rather than held forever.
                if good == 0 && e.error_len().is_none() {
                    return String::new();
                }
                if good == 0 {
                    let out = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    return out;
                }
                let out = String::from_utf8_lossy(&self.pending[..good]).into_owned();
                self.pending.drain(..good);
                out
            }
        }
    }

    /// Flush any trailing bytes at end of stream.
    pub fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}
