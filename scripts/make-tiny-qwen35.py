#!/usr/bin/env python3
"""Build a tiny random-weight qwen35 GGUF for graph validation.

Copies the real checkpoint's metadata (so the tokenizer and key layout are
authentic) but overrides every shape key downward, then emits random F32
weights: 8 main layers (full attention at 3 and 7) plus one NextN/MTP block.
Deterministic under seed 42.
"""
import struct, sys, math, random

SRC = "/Users/tonyradtke/dev/llmoxide/models/Qwen3.8-27B-Q6_K.gguf"
DST = sys.argv[1] if len(sys.argv) > 1 else "tiny-qwen35.gguf"

# --- tiny shape --------------------------------------------------------------
D = 64          # embedding_length
FFN = 128
N_HEADS, N_KV, HEAD_DIM, N_ROT = 4, 2, 16, 8
CONV_K, S_DIM, N_KH, N_VH = 4, 8, 2, 4
D_INNER = N_VH * S_DIM              # 32
CONV_DIM = 2 * N_KH * S_DIM + D_INNER  # 64
BLOCKS, NEXTN = 9, 1                # 8 main + 1 MTP
FULL_ATTN = {3, 7}                  # (i+1) % 4 == 0 among main layers
CTX = 4096

OVERRIDES = {
    "qwen35.block_count": ("u32", BLOCKS),
    "qwen35.context_length": ("u32", CTX),
    "qwen35.embedding_length": ("u32", D),
    "qwen35.feed_forward_length": ("u32", FFN),
    "qwen35.attention.head_count": ("u32", N_HEADS),
    "qwen35.attention.head_count_kv": ("u32", N_KV),
    "qwen35.attention.key_length": ("u32", HEAD_DIM),
    "qwen35.attention.value_length": ("u32", HEAD_DIM),
    "qwen35.rope.dimension_count": ("u32", N_ROT),
    "qwen35.rope.dimension_sections": ("arr_i32", [2, 1, 1, 0]),
    "qwen35.ssm.conv_kernel": ("u32", CONV_K),
    "qwen35.ssm.state_size": ("u32", S_DIM),
    "qwen35.ssm.group_count": ("u32", N_KH),
    "qwen35.ssm.time_step_rank": ("u32", N_VH),
    "qwen35.ssm.inner_size": ("u32", D_INNER),
    "qwen35.nextn_predict_layers": ("u32", NEXTN),
    "general.name": ("str", "tiny qwen35 validation model"),
}

# --- copy metadata with overrides -------------------------------------------
f = open(SRC, "rb")
def u32(): return struct.unpack("<I", f.read(4))[0]
def u64(): return struct.unpack("<Q", f.read(8))[0]
def s(): return f.read(u64()).decode("utf-8", errors="replace")
SC = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def raw_value(ty):
    start = f.tell()
    if ty == 8:
        f.seek(u64(), 1)
    elif ty == 9:
        ety, cnt = u32(), u64()
        if ety == 8:
            for _ in range(cnt): f.seek(u64(), 1)
        else:
            f.seek(cnt * SC[ety], 1)
    else:
        f.seek(SC[ty], 1)
    end = f.tell()
    f.seek(start)
    return f.read(end - start)

assert u32() == 0x46554747
version = u32()
_nt = u64()
n_kv = u64()

def enc_str(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b

kv_out = []
for _ in range(n_kv):
    key = s()
    ty = u32()
    raw = raw_value(ty)
    if key in OVERRIDES:
        kind, val = OVERRIDES[key]
        if kind == "u32":
            kv_out.append(enc_str(key) + struct.pack("<I", 4) + struct.pack("<I", val))
        elif kind == "str":
            kv_out.append(enc_str(key) + struct.pack("<I", 8) + enc_str(val))
        elif kind == "arr_i32":
            body = struct.pack("<IQ", 5, len(val)) + b"".join(struct.pack("<i", v) for v in val)
            kv_out.append(enc_str(key) + struct.pack("<I", 9) + body)
    else:
        kv_out.append(enc_str(key) + struct.pack("<I", ty) + raw)
f.close()

# --- tensors -----------------------------------------------------------------
rng = random.Random(42)

def mat(in_dim, out_dim, scale=0.08):
    return [rng.gauss(0.0, scale) for _ in range(in_dim * out_dim)]

def norm_w(n):
    return [1.0 + rng.gauss(0.0, 0.15) for _ in range(n)]

tensors = []  # (name, dims_in_ne_order, values)

def add(name, dims, vals):
    cnt = math.prod(dims)
    assert len(vals) == cnt, f"{name}: {len(vals)} != {cnt}"
    tensors.append((name, dims, vals))

VOCAB = 248320
add("token_embd.weight", [D, VOCAB], mat(D, VOCAB, 0.05))
add("output_norm.weight", [D], norm_w(D))

for i in range(BLOCKS):
    p = f"blk.{i}."
    is_mtp = i >= BLOCKS - NEXTN
    add(p + "attn_norm.weight", [D], norm_w(D))
    add(p + "post_attention_norm.weight", [D], norm_w(D))
    add(p + "ffn_gate.weight", [D, FFN], mat(D, FFN))
    add(p + "ffn_up.weight", [D, FFN], mat(D, FFN))
    add(p + "ffn_down.weight", [FFN, D], mat(FFN, D))
    if i in FULL_ATTN or is_mtp:
        add(p + "attn_q.weight", [D, N_HEADS * HEAD_DIM * 2], mat(D, N_HEADS * HEAD_DIM * 2))
        add(p + "attn_k.weight", [D, N_KV * HEAD_DIM], mat(D, N_KV * HEAD_DIM))
        add(p + "attn_v.weight", [D, N_KV * HEAD_DIM], mat(D, N_KV * HEAD_DIM))
        add(p + "attn_q_norm.weight", [HEAD_DIM], norm_w(HEAD_DIM))
        add(p + "attn_k_norm.weight", [HEAD_DIM], norm_w(HEAD_DIM))
        add(p + "attn_output.weight", [N_HEADS * HEAD_DIM, D], mat(N_HEADS * HEAD_DIM, D))
    else:
        add(p + "attn_qkv.weight", [D, CONV_DIM], mat(D, CONV_DIM))
        add(p + "attn_gate.weight", [D, D_INNER], mat(D, D_INNER))
        add(p + "ssm_conv1d.weight", [CONV_K, CONV_DIM], mat(CONV_K, CONV_DIM, 0.3))
        add(p + "ssm_dt.bias", [N_VH], [rng.gauss(0.0, 0.5) for _ in range(N_VH)])
        add(p + "ssm_a", [N_VH], [-(0.2 + 1.5 * rng.random()) for _ in range(N_VH)])
        add(p + "ssm_alpha.weight", [D, N_VH], mat(D, N_VH, 0.3))
        add(p + "ssm_beta.weight", [D, N_VH], mat(D, N_VH, 0.3))
        add(p + "ssm_norm.weight", [S_DIM], norm_w(S_DIM))
        add(p + "ssm_out.weight", [D_INNER, D], mat(D_INNER, D))
    if is_mtp:
        add(p + "nextn.eh_proj.weight", [2 * D, D], mat(2 * D, D))
        add(p + "nextn.enorm.weight", [D], norm_w(D))
        add(p + "nextn.hnorm.weight", [D], norm_w(D))
        add(p + "nextn.shared_head_norm.weight", [D], norm_w(D))

# --- write -------------------------------------------------------------------
ALIGN = 32
out = open(DST, "wb")
out.write(struct.pack("<IIQQ", 0x46554747, version, len(tensors), len(kv_out)))
for kv in kv_out:
    out.write(kv)

offset = 0
infos = []
for name, dims, vals in tensors:
    infos.append(offset)
    out.write(enc_str(name))
    out.write(struct.pack("<I", len(dims)))
    for d in dims:
        out.write(struct.pack("<Q", d))
    out.write(struct.pack("<I", 0))  # F32
    out.write(struct.pack("<Q", offset))
    nbytes = math.prod(dims) * 4
    offset = (offset + nbytes + ALIGN - 1) // ALIGN * ALIGN

pos = out.tell()
pad = (pos + ALIGN - 1) // ALIGN * ALIGN - pos
out.write(b"\x00" * pad)

for (name, dims, vals), want_off in zip(tensors, infos):
    base = out.tell()
    out.write(struct.pack(f"<{len(vals)}f", *vals))
    nbytes = math.prod(dims) * 4
    pad = (nbytes + ALIGN - 1) // ALIGN * ALIGN - nbytes
    out.write(b"\x00" * pad)

out.close()
import os
print(f"wrote {DST}: {os.path.getsize(DST)/2**20:.1f} MiB, {len(tensors)} tensors")
