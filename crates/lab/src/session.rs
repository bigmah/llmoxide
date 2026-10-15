//! Everything the page asks of the model, without any of the page.
//!
//! Two independent conversations share one set of weights: the *inspector*,
//! which runs a single prompt with probes on and steps it a token at a time,
//! and the *chat*, which runs without probes and keeps its cache between turns.
//! Each has its own KV cache, so poking at one never disturbs the other.

use gguf::Gguf;
use model::sample::{Rng, Sampler, Sampling};
use tokenizer::Tokenizer;

use crate::engine::{Cache, Model, Probe};

/// The system turn SmolLM2's chat template inserts when there is none.
const DEFAULT_SYSTEM: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

pub struct Lab<'a> {
    pub model: Model<'a>,
    pub tok: Tokenizer,
    im_start: u32,
    im_end: u32,

    // -- inspector --------------------------------------------------------
    pub probe_cache: Cache,
    pub probe: Probe,
    /// Every token the inspector has run, prompt and generated.
    pub probe_ids: Vec<u32>,
    probe_logits: Vec<f32>,
    probe_rng: Rng,

    // -- chat ----------------------------------------------------------------
    pub chat_cache: Cache,
    /// The exact tokens resident in `chat_cache`, so a turn that only appends
    /// runs only what it appends.
    chat_ids: Vec<u32>,
    history: Vec<(Role, String)>,
    reply: Vec<u32>,
    emitted: usize,
    chat_logits: Vec<f32>,
    sampler: Sampler,
}

/// What one step of generation produced.
pub struct Step {
    pub id: u32,
    pub done: bool,
}

impl<'a> Lab<'a> {
    pub fn new(g: &'a Gguf, chat_ctx: usize, probe_ctx: usize) -> anyhow::Result<Self> {
        let model = Model::load(g)?;
        let tok = Tokenizer::from_gguf(g)?;
        let id = |s: &str| tok.id_of(s).ok_or_else(|| anyhow::anyhow!("vocab has no {s}"));
        let (im_start, im_end) = (id("<|im_start|>")?, id("<|im_end|>")?);
        // SmolLM2's card recommends 0.2 / 0.9. A small model repeats itself,
        // so a light repetition penalty on top.
        let sampling = Sampling {
            temperature: 0.2,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_window: 64,
            seed: 0x5eed,
        };
        Ok(Self {
            probe_cache: Cache::new(&model.cfg, probe_ctx),
            chat_cache: Cache::new(&model.cfg, chat_ctx),
            model,
            tok,
            im_start,
            im_end,
            probe: Probe::default(),
            probe_ids: Vec::new(),
            probe_logits: Vec::new(),
            probe_rng: Rng::new(7),
            chat_ids: Vec::new(),
            history: Vec::new(),
            reply: Vec::new(),
            emitted: 0,
            chat_logits: Vec::new(),
            sampler: Sampler::new(sampling),
        })
    }

    /// A token as text. Byte-level vocabularies spell bytes with stand-in
    /// characters (space is `Ġ`, newline `Ċ`); this undoes that.
    pub fn piece(&self, id: u32) -> String {
        if self.tok.kind(id).is_control() {
            return self.tok.token_text(id).to_string();
        }
        let raw = self.tok.token_text(id);
        let bytes: Option<Vec<u8>> = raw.chars().map(tokenizer::char_to_byte).collect();
        match bytes {
            Some(b) => String::from_utf8_lossy(&b).into_owned(),
            None => raw.to_string(),
        }
    }

    /// ChatML, the way SmolLM2's template renders it. Control tokens go in by
    /// id and text is tokenized between them, so nothing typed can forge one.
    fn render(&self, turns: &[(Role, &str)]) -> Vec<u32> {
        let mut ids = Vec::new();
        let turn = |ids: &mut Vec<u32>, role: &str, text: &str| {
            ids.push(self.im_start);
            self.tok.encode_into(&format!("{role}\n{text}"), ids);
            ids.push(self.im_end);
            self.tok.encode_into("\n", ids);
        };
        turn(&mut ids, "system", DEFAULT_SYSTEM);
        for (role, text) in turns {
            let role = match role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            turn(&mut ids, role, text);
        }
        ids.push(self.im_start);
        self.tok.encode_into("assistant\n", &mut ids);
        ids
    }

    // -- inspector --------------------------------------------------------

    /// Start over on `text`: wipe the inspector's cache and prefill the prompt
    /// with probes on. `chat` wraps it in the chat template first.
    pub fn inspect(&mut self, text: &str, chat: bool) -> anyhow::Result<()> {
        let ids = if chat {
            self.render(&[(Role::User, text)])
        } else {
            self.tok.encode(text, true)
        };
        anyhow::ensure!(!ids.is_empty(), "nothing to run");
        anyhow::ensure!(
            ids.len() < self.probe_cache.capacity,
            "{} tokens is more than the inspector's {}",
            ids.len(),
            self.probe_cache.capacity
        );
        self.probe_cache.wipe();
        self.probe_logits = self.model.forward(&mut self.probe_cache, &ids, Some(&mut self.probe));
        self.probe_ids = ids;
        Ok(())
    }

    /// The inspector's next-token distribution at `temperature`, top `k`.
    pub fn probe_top(&self, temperature: f32, k: usize) -> Vec<(u32, f32)> {
        top_k(&self.probe_logits, temperature, k)
    }

    /// Pick the inspector's next token (greedy at temperature 0) and run it.
    pub fn probe_step(&mut self, temperature: f32) -> anyhow::Result<Step> {
        anyhow::ensure!(!self.probe_logits.is_empty(), "nothing to continue");
        anyhow::ensure!(self.probe_cache.len < self.probe_cache.capacity, "the inspector's context is full");
        let id = if temperature <= 0.0 {
            model::sample::argmax(&self.probe_logits)
        } else {
            // A plain draw from the distribution the page is showing (its top
            // 50, which at any sane temperature is all of the mass).
            let probs = top_k(&self.probe_logits, temperature, 50);
            let mut r = self.probe_rng.next_f32() * probs.iter().map(|p| p.1).sum::<f32>();
            probs.iter().find(|(_, p)| { r -= p; r <= 0.0 }).map_or(probs[0].0, |p| p.0)
        };
        self.probe_logits = self.model.forward(&mut self.probe_cache, &[id], Some(&mut self.probe));
        self.probe_ids.push(id);
        Ok(Step { id, done: self.model.cfg.eog.contains(&id) })
    }

    /// Logit lens at every layer boundary of the inspector's last position.
    pub fn lens(&mut self, k: usize) -> Vec<Vec<(u32, f32)>> {
        let d = self.model.cfg.d_model;
        let rows: Vec<Vec<f32>> = self.probe.last.chunks_exact(d).map(<[f32]>::to_vec).collect();
        rows.iter().map(|r| top_k(&self.model.lens(r), 1.0, k)).collect()
    }

    /// The `k` tokens whose embeddings point most nearly the same way as
    /// `id`'s, by cosine similarity.
    pub fn neighbors(&self, id: u32, k: usize) -> Vec<(u32, f32)> {
        let e = &self.model.w.embed;
        let d = e.in_dim();
        let mut a = vec![0.0; d];
        let mut b = vec![0.0; d];
        e.dequant_row_into(id as usize, &mut a);
        let na = crate::engine::dot(&a, &a).sqrt();
        let mut sims: Vec<(u32, f32)> = (0..e.out_dim())
            .filter(|&j| j != id as usize && !self.tok.kind(j as u32).is_control())
            .map(|j| {
                e.dequant_row_into(j, &mut b);
                let nb = crate::engine::dot(&b, &b).sqrt();
                (j as u32, crate::engine::dot(&a, &b) / (na * nb).max(1e-12))
            })
            .collect();
        let k = k.min(sims.len());
        sims.select_nth_unstable_by(k - 1, |x, y| y.1.total_cmp(&x.1));
        sims.truncate(k);
        sims.sort_by(|x, y| y.1.total_cmp(&x.1));
        sims
    }

    pub fn embedding(&self, id: u32) -> Vec<f32> {
        let mut v = vec![0.0; self.model.cfg.d_model];
        self.model.w.embed.dequant_row_into(id as usize, &mut v);
        v
    }

    // -- chat ----------------------------------------------------------------

    /// Add a user turn and prefill whatever the cache does not already hold.
    /// Returns `(tokens run, tokens reused)`.
    pub fn chat_begin(&mut self, text: &str) -> anyhow::Result<(usize, usize)> {
        self.history.push((Role::User, text.to_string()));
        let turns: Vec<(Role, &str)> = self.history.iter().map(|(r, t)| (*r, t.as_str())).collect();
        let ids = self.render(&turns);
        if ids.len() + 16 > self.chat_cache.capacity {
            self.history.pop();
            anyhow::bail!("the conversation is full ({} tokens). Forget it to start over.", ids.len());
        }
        // Prefix reuse: keep what matches, rewind the rest.
        let keep = ids
            .iter()
            .zip(&self.chat_ids)
            .take_while(|(a, b)| a == b)
            .count()
            .min(ids.len() - 1);
        self.chat_cache.len = keep;
        self.chat_logits = self.model.forward(&mut self.chat_cache, &ids[keep..], None);
        self.chat_ids = ids;
        self.reply.clear();
        self.emitted = 0;
        Ok((self.chat_ids.len() - keep, keep))
    }

    /// Generate one token of the reply. Returns the new text (possibly empty,
    /// when a token ends partway through a UTF-8 character) and whether the
    /// reply is over.
    pub fn chat_next(&mut self) -> (String, bool) {
        let mut logits = std::mem::take(&mut self.chat_logits);
        let id = self.sampler.sample(&mut logits, &self.reply);
        let full = self.chat_cache.len + 1 >= self.chat_cache.capacity;
        if self.model.cfg.eog.contains(&id) || full {
            self.finish();
            return (String::new(), true);
        }
        self.reply.push(id);
        self.chat_logits = self.model.forward(&mut self.chat_cache, &[id], None);
        self.chat_ids.push(id);

        // Decode the whole reply and hand back what is new. Holding back an
        // incomplete character is the decoder's job; it only emits whole ones.
        let mut dec = tokenizer::Decoder::new(&self.tok);
        let text: String = self.reply.iter().map(|&i| dec.push(i)).collect();
        let new = text[self.emitted..].to_string();
        self.emitted = text.len();
        (new, false)
    }

    /// End the reply where it stands (the stop button, or the end of turn).
    pub fn finish(&mut self) {
        let text = self.tok.decode(&self.reply);
        self.history.push((Role::Assistant, text));
        // The cache holds the reply but not the `<|im_end|>\n` the template
        // will put after it; the next turn's prefix match rewinds to here.
        self.reply.clear();
    }

    pub fn chat_tokens(&self) -> usize {
        self.chat_cache.len
    }

    /// Forget the conversation: overwrite the cache, the token ids and the
    /// text. The strings are freed through the zeroing allocator, so the
    /// bytes they held are overwritten too, not just released.
    pub fn wipe(&mut self) {
        self.chat_cache.wipe();
        self.probe_cache.wipe();
        secret::zero_slice(&mut self.chat_ids);
        self.chat_ids.clear();
        secret::zero_slice(&mut self.probe_ids);
        self.probe_ids.clear();
        secret::zero_slice(&mut self.reply);
        self.reply.clear();
        self.history.clear();
        self.chat_logits.clear();
        self.probe_logits.clear();
        self.probe = Probe::default();
        self.emitted = 0;
    }
}

/// Softmax at `temperature` over the logits, then the `k` most likely.
pub fn top_k(logits: &[f32], temperature: f32, k: usize) -> Vec<(u32, f32)> {
    if logits.is_empty() || k == 0 {
        return Vec::new();
    }
    let t = if temperature <= 0.0 { 1e-3 } else { temperature };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|l| ((l - max) / t).exp()).sum();
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let k = k.min(idx.len());
    idx.select_nth_unstable_by(k - 1, |a, b| logits[*b as usize].total_cmp(&logits[*a as usize]));
    idx.truncate(k);
    idx.sort_by(|a, b| logits[*b as usize].total_cmp(&logits[*a as usize]));
    idx.into_iter()
        .map(|i| (i, ((logits[i as usize] - max) / t).exp() / sum))
        .collect()
}
