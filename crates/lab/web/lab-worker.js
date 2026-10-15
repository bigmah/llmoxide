// The lab engine, off the page's thread.
//
// The page posts {id, op, ...args}; every op answers with one {id, ok, ...}
// message. `id` is the message's, so no op may take an argument named `id`.
// Loading and chatting also post {type: 'progress' | 'chat'} along
// the way. Arrays of floats travel as transferred Float32Array buffers.

import init, { Engine } from './lab.js';

let engine = null;
let stopping = false;
const now = () => performance.now();
const yieldNow = () => new Promise(r => setTimeout(r, 0));

function forward(extra = {}) {
  const attn = engine.attention();
  const rms = engine.rms();
  return {
    shape: Array.from(engine.probe_shape()),
    attn, rms,
    top: JSON.parse(engine.top(extra.temp ?? 0.7, 12)),
    transfer: [attn.buffer, rms.buffer],
  };
}

async function fetchModel(url, id) {
  let cache = null;
  try { cache = await caches.open('llmoxide-lab-v1'); } catch (_) { /* not a secure context */ }
  const hit = cache && await cache.match(url);
  if (hit) return { bytes: new Uint8Array(await hit.arrayBuffer()), cached: true };

  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetching ${url}: ${resp.status}`);
  const total = Number(resp.headers.get('content-length')) || 0;
  const reader = resp.body.getReader();
  const parts = [];
  let loaded = 0, last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    if (now() - last > 100) { last = now(); self.postMessage({ type: 'progress', id, loaded, total }); }
  }
  const bytes = new Uint8Array(loaded);
  let off = 0;
  for (const p of parts) { bytes.set(p, off); off += p.length; }
  if (cache) { try { await cache.put(url, new Response(bytes)); } catch (_) { /* quota */ } }
  return { bytes, cached: false };
}

const ops = {
  async load({ url, chatCtx = 1024, probeCtx = 160 }, id) {
    const t0 = now();
    await init();
    const { bytes, cached } = await fetchModel(url, id);
    const t1 = now();
    engine = new Engine(bytes, chatCtx, probeCtx);
    return { config: JSON.parse(engine.config()), cached, fetchMs: t1 - t0, buildMs: now() - t1 };
  },

  tokenize({ text }) {
    const t0 = now();
    const tokens = JSON.parse(engine.tokenize(text));
    return { tokens, ms: now() - t0 };
  },

  embedding({ token }) {
    const vec = engine.embedding(token);
    return { vec, transfer: [vec.buffer] };
  },

  neighbors({ token, k = 10 }) {
    const t0 = now();
    const near = JSON.parse(engine.neighbors(token, k));
    return { near, ms: now() - t0 };
  },

  inspect({ text, chat, temp }) {
    const t0 = now();
    const tokens = JSON.parse(engine.inspect(text, chat));
    const ms = now() - t0;
    return { tokens, ms, ...forward({ temp }) };
  },

  step({ temp, pick }) {
    const t0 = now();
    const tok = JSON.parse(engine.step(pick));
    const ms = now() - t0;
    return { tok, ms, ...forward({ temp }) };
  },

  top({ temp }) {
    return { top: JSON.parse(engine.top(temp, 12)) };
  },

  lens({ k = 3 }) {
    const t0 = now();
    const rows = JSON.parse(engine.lens(k));
    return { rows, ms: now() - t0 };
  },

  async chat({ text }, id) {
    stopping = false;
    const t0 = now();
    const { ran, reused } = JSON.parse(engine.chat_begin(text));
    const prefillMs = now() - t0;
    self.postMessage({ type: 'prefill', id, ran, reused, ms: prefillMs });
    const t1 = now();
    let n = 0;
    for (;;) {
      if (stopping) { engine.chat_stop(); break; }
      const piece = engine.chat_next();
      if (piece === undefined) break;
      n++;
      self.postMessage({ type: 'chat', id, piece, n, ms: now() - t1 });
      // Let a 'stop' message in between tokens.
      await yieldNow();
    }
    return { ran, reused, prefillMs, n, decodeMs: now() - t1, stats: JSON.parse(engine.chat_stats()) };
  },

  stop() { stopping = true; return {}; },

  stats() { return { stats: JSON.parse(engine.chat_stats()) }; },

  wipe() { return JSON.parse(engine.wipe()); },
};

self.onmessage = async (e) => {
  const { id, op, ...args } = e.data;
  try {
    if (op !== 'load' && !engine) throw new Error('the model is not loaded');
    const r = (await ops[op](args, id)) || {};
    const transfer = r.transfer || [];
    delete r.transfer;
    self.postMessage({ id, ok: true, ...r }, transfer);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
