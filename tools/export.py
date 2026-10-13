# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "safetensors", "numpy", "huggingface_hub", "tokenizers"]
# ///
"""Convert LFM2.5-230M from HF safetensors into the engine's .lfm file.

    uv run tools/export.py                       # Q8_0  -> out/lfm-q8.lfm  (~250 MB, for the device)
    uv run tools/export.py --quant bf16          # BF16  -> out/lfm-bf16.lfm (exact weights, host tests)
    uv run tools/export.py --quant q8,mlp=q4     # mixed -> out/lfm-q8-mlp-q4.lfm
    uv run tools/export.py --prompt "Hello"      # also rewrite out/prompt.txt

--quant: a default encoding, then group=encoding overrides. Groups: mlp (w1 w2
w3), conv (conv in/out proj), attn (q k v o), embed (embedding = LM head).
Encodings: bf16, q8, q4.

Copy out/lfm-q8.lfm to the SD card as /lfm.lfm and out/prompt.txt as /prompt.txt.

File layout (little-endian). Must match lfm/include/lfm/format.hpp:

    FileHeader                  fixed size, at offset 0
    tensors                     each aligned to 4096 bytes, in forward order
    vocab index                 uint32[vocab + 1]: byte offsets into vocab bytes
    vocab bytes                 raw UTF-8 / byte-level pieces, concatenated

Tensor encodings:
    F32     float32, row-major                         (norms, conv kernels)
    BF16    bfloat16, row-major                        (matrices, --dtype bf16)
    Q8_0    per row: cols/32 blocks of {f16 scale, int8[32]}, 34 bytes each;
            w = scale * q                              (matrices, q8)
    Q4_0    per row: cols/32 blocks of {f16 scale, uint8[16]}, 18 bytes each;
            byte j = element j (low nibble) | element j+16 (high nibble),
            w = scale * (nibble - 8)                   (matrices, q4)
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

MODEL = "LiquidAI/LFM2.5-230M"
ROOT = Path(__file__).resolve().parent.parent

MAGIC = b"LFM2ENG\0"
VERSION = 2
MAX_LAYERS = 16
ALIGN = 4096

DTYPE_F32, DTYPE_BF16, DTYPE_Q8_0, DTYPE_Q4_0 = 0, 1, 2, 3
DTYPES = {"bf16": DTYPE_BF16, "q8": DTYPE_Q8_0, "q4": DTYPE_Q4_0}
GROUPS = {"mlp": ["w1", "w2", "w3"], "conv": ["conv_in", "conv_out"], "attn": ["q", "k", "v", "o"]}
LAYER_CONV, LAYER_ATTN = 0, 1

# LayerOffsets: 14 x uint64, see format.hpp.
LAYER_FIELDS = ["operator_norm", "ffn_norm", "w1", "w2", "w3",
                "conv_in", "conv_kernel", "conv_out",
                "q", "k", "v", "o", "q_norm", "k_norm"]
# FileHeader before the layer table: magic, 12 x u32, 2 x f32, 16 x u32 layer types,
# 4 x u64 (lands on offset 128, so no padding in the C++ struct either).
HEADER_HEAD = struct.Struct("<8s12I2f16I4Q")
# Then LayerOffsets[16], then uint8 layer_dtype[16][16] (Dtype per LAYER_FIELDS index),
# then the LM head shortlist: u64 offset, u32 dtype, u32 candidates,
# u64 low-rank projection offset, u32 rank, u32 reserved.
HEADER_TAIL = struct.Struct("<QIIQII")
HEADER_SIZE = HEADER_HEAD.size + MAX_LAYERS * len(LAYER_FIELDS) * 8 + MAX_LAYERS * 16 + HEADER_TAIL.size


def bytes_to_unicode():
    """GPT-2 byte-level BPE: printable stand-in character for each byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def vocab_pieces(tok_json: dict, vocab_size: int) -> list[bytes]:
    """Raw bytes of every token id. Ids past the tokenizer's vocab are empty."""
    to_byte = {c: b for b, c in bytes_to_unicode().items()}
    pieces = [b""] * vocab_size
    for s, i in tok_json["model"]["vocab"].items():
        if i < vocab_size:
            pieces[i] = bytes(to_byte[c] for c in s) if all(c in to_byte for c in s) else s.encode()
    for a in tok_json.get("added_tokens", []):
        if a["id"] < vocab_size:
            pieces[a["id"]] = a["content"].encode()
    return pieces


def quantize_q8_0(w: np.ndarray) -> bytes:
    rows, cols = w.shape
    assert cols % 32 == 0, w.shape
    blocks = w.reshape(rows, cols // 32, 32).astype(np.float32)
    scale = (np.abs(blocks).max(axis=-1) / 127.0).astype(np.float16)
    s = scale.astype(np.float32)[..., None]
    q = np.where(s > 0, np.round(blocks / np.where(s > 0, s, 1)), 0).clip(-127, 127).astype(np.int8)
    out = np.empty((rows, cols // 32), dtype=[("d", "<f2"), ("q", "i1", 32)])
    out["d"] = scale
    out["q"] = q
    return out.tobytes()


def quantize_q4_0(w: np.ndarray) -> bytes:
    """Q4_0 with a per-block scale search.

    llama.cpp's Q4_0 maps the largest-magnitude value to -8 exactly. Here each
    block also tries slightly larger divisors (the extreme value then lands
    inside the range and the rest get finer steps) and keeps the scale with
    the smallest squared error. Same format, decoded identically.
    """
    rows, cols = w.shape
    assert cols % 32 == 0, w.shape
    blocks = w.reshape(rows, cols // 32, 32).astype(np.float32)
    idx = np.abs(blocks).argmax(axis=-1)[..., None]
    signed_max = np.take_along_axis(blocks, idx, axis=-1)[..., 0]

    def encode(scale16):
        s = scale16.astype(np.float32)[..., None]
        q = np.where(s != 0, np.round(blocks / np.where(s != 0, s, 1)), 0) + 8
        q = q.clip(0, 15)
        err = ((q - 8) * s - blocks) ** 2
        return q.astype(np.uint8), err.sum(axis=-1)

    best_scale = (signed_max / -8.0).astype(np.float16)
    best_q, best_err = encode(best_scale)
    for k in np.arange(-8.0, -5.4, 0.25)[1:]:
        for sign_max in (signed_max, -signed_max):  # extreme value on the +7 side too
            scale = (sign_max / k).astype(np.float16)
            q, err = encode(scale)
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_scale = np.where(better, scale, best_scale)
            best_q = np.where(better[..., None], q, best_q)

    out = np.empty((rows, cols // 32), dtype=[("d", "<f2"), ("q", "u1", 16)])
    out["d"] = best_scale
    out["q"] = best_q[..., :16] | (best_q[..., 16:] << 4)
    return out.tobytes()


def parse_quant(spec: str) -> dict[str, int]:
    """'q8,mlp=q4' -> {field or 'embed': dtype} for every matrix field."""
    parts = spec.split(",")
    default = DTYPES[parts[0]]
    out = {f: default for fields in GROUPS.values() for f in fields}
    out["embed"] = default
    for part in parts[1:]:
        group, enc = part.split("=")
        for f in GROUPS.get(group, [group]):
            out[f] = DTYPES[enc]
    return out


class Writer:
    def __init__(self, f):
        self.f = f

    def _align(self):
        pos = self.f.tell()
        pad = (-pos) % ALIGN
        self.f.write(b"\0" * pad)
        return pos + pad

    def f32(self, t: torch.Tensor) -> int:
        off = self._align()
        self.f.write(t.float().contiguous().numpy().astype("<f4").tobytes())
        return off

    def matrix(self, t: torch.Tensor, dtype: int) -> int:
        assert t.ndim == 2
        off = self._align()
        if dtype == DTYPE_BF16:
            self.f.write(t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().astype("<i2").tobytes())
        elif dtype == DTYPE_Q8_0:
            self.f.write(quantize_q8_0(t.float().numpy()))
        else:
            self.f.write(quantize_q4_0(t.float().numpy()))
        return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="q8", help="e.g. q8 | bf16 | q8,mlp=q4,embed=q4")
    ap.add_argument("--lm-head", choices=["none", "q4", "lowrank"], default="q4",
                    help="q4: also store a Q4 copy of the embedding to shortlist LM head candidates "
                         "(exact top-1, ~half the LM head reads); skipped if embed is already q4. "
                         "lowrank: shortlist with a rank-r table (out/lm_basis.npy, tools/lm_basis.py); "
                         "~17 MB instead of 38, the true token is in the shortlist ~99.8% of the time")
    ap.add_argument("--lm-head-candidates", type=int, default=None, help="default 32 (q4) / 256 (lowrank)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    src = Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors"]))
    cfg = json.loads((src / "config.json").read_text())
    tok_json = json.loads((src / "tokenizer.json").read_text())
    w = load_file(str(src / "model.safetensors"))

    dim, ff = cfg["hidden_size"], cfg.get("block_ff_dim", cfg["intermediate_size"])
    n_layers, n_heads, n_kv = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    vocab, conv_l = cfg["vocab_size"], cfg["conv_L_cache"]
    head_dim = dim // n_heads
    layer_types = [LAYER_ATTN if t == "full_attention" else LAYER_CONV for t in cfg["layer_types"]]
    ff = w["model.layers.0.feed_forward.w1.weight"].shape[0]  # actual, after any auto-adjust
    eps, theta = cfg["norm_eps"], cfg.get("rope_theta", cfg.get("rope_parameters", {}).get("rope_theta"))
    assert n_layers <= MAX_LAYERS and theta, (n_layers, theta)

    quant = parse_quant(args.quant)
    name = args.quant.replace(",", "-").replace("=", "-")
    use_lm_head = args.lm_head == "q4" and quant["embed"] != DTYPE_Q4_0
    use_lowrank = args.lm_head == "lowrank"
    candidates = args.lm_head_candidates or (256 if use_lowrank else 32)
    if use_lowrank:
        basis = torch.from_numpy(np.load(ROOT / "out" / "lm_basis.npy"))  # [dim, r]
        name += f"-head-lr{basis.shape[1]}"
    if use_lm_head:
        name += "-head-q4"
    out = args.out or ROOT / "out" / f"lfm-{name}.lfm"
    out.parent.mkdir(parents=True, exist_ok=True)

    layers = []
    with open(out, "wb") as f:
        f.write(b"\0" * HEADER_SIZE)
        wr = Writer(f)
        mat = lambda field, t: wr.matrix(t, quant[field])
        for i in range(n_layers):
            p = f"model.layers.{i}."
            o = dict.fromkeys(LAYER_FIELDS, 0)
            o["operator_norm"] = wr.f32(w[p + "operator_norm.weight"])
            if layer_types[i] == LAYER_CONV:
                o["conv_in"] = mat("conv_in", w[p + "conv.in_proj.weight"])
                o["conv_kernel"] = wr.f32(w[p + "conv.conv.weight"].reshape(dim, conv_l))
                o["conv_out"] = mat("conv_out", w[p + "conv.out_proj.weight"])
            else:
                a = p + "self_attn."
                o["q"] = mat("q", w[a + "q_proj.weight"])
                o["k"] = mat("k", w[a + "k_proj.weight"])
                o["v"] = mat("v", w[a + "v_proj.weight"])
                o["q_norm"] = wr.f32(w[a + "q_layernorm.weight"])
                o["k_norm"] = wr.f32(w[a + "k_layernorm.weight"])
                o["o"] = mat("o", w[a + "out_proj.weight"])
            o["ffn_norm"] = wr.f32(w[p + "ffn_norm.weight"])
            o["w1"] = mat("w1", w[p + "feed_forward.w1.weight"])
            o["w3"] = mat("w3", w[p + "feed_forward.w3.weight"])
            o["w2"] = mat("w2", w[p + "feed_forward.w2.weight"])
            layers.append(o)
            print(f"layer {i:2d} {'attn' if layer_types[i] else 'conv'}  @ {f.tell() / 1e6:7.1f} MB", flush=True)

        final_norm = wr.f32(w["model.embedding_norm.weight"])
        embed = wr.matrix(w["model.embed_tokens.weight"], quant["embed"])
        lm_head = wr.matrix(w["model.embed_tokens.weight"], DTYPE_Q4_0) if use_lm_head else 0
        lm_proj = lm_rank = 0
        if use_lowrank:
            # z = V^T h (rank floats), then approx logit[v] = (E V)[v] . z
            lm_rank = basis.shape[1]
            lm_proj = wr.matrix(basis.T.contiguous(), DTYPE_Q8_0)                          # [rank, dim]
            lm_head = wr.matrix(w["model.embed_tokens.weight"].float() @ basis, DTYPE_Q8_0)  # [vocab, rank]
            print(f"low-rank head    @ {f.tell() / 1e6:7.1f} MB (rank {lm_rank})", flush=True)
        print(f"embed            @ {f.tell() / 1e6:7.1f} MB", flush=True)

        pieces = vocab_pieces(tok_json, vocab)
        index = np.zeros(vocab + 1, dtype="<u4")
        index[1:] = np.cumsum([len(p) for p in pieces])
        vocab_index = wr._align()
        f.write(index.tobytes())
        vocab_bytes = f.tell()
        f.write(b"".join(pieces))
        total = f.tell()

        types = layer_types + [0] * (MAX_LAYERS - n_layers)
        head = HEADER_HEAD.pack(MAGIC, VERSION, quant["embed"], dim, ff, n_layers, n_heads, n_kv, head_dim,
                                vocab, conv_l, 0, 0, eps, theta, *types,
                                embed, final_norm, vocab_index, vocab_bytes)
        table = b"".join(struct.pack(f"<{len(LAYER_FIELDS)}Q", *(o[k] for k in LAYER_FIELDS)) for o in layers)
        table += b"\0" * (MAX_LAYERS - n_layers) * len(LAYER_FIELDS) * 8
        dtypes = b""
        for i in range(MAX_LAYERS):
            row = [quant.get(k, DTYPE_F32) if i < n_layers else 0 for k in LAYER_FIELDS]
            dtypes += bytes(row + [0] * (16 - len(row)))
        head_dtype = DTYPE_Q8_0 if use_lowrank else DTYPE_Q4_0 if use_lm_head else 0
        tail = HEADER_TAIL.pack(lm_head, head_dtype, candidates, lm_proj, lm_rank, 0)
        f.seek(0)
        f.write(head + table + dtypes + tail)
        assert f.tell() == HEADER_SIZE

    print(f"wrote {out} ({total / 1e6:.1f} MB, {args.quant})")

    ids = Tokenizer.from_file(str(src / "tokenizer.json")).encode(args.prompt).ids
    prompt_path = ROOT / "out" / "prompt.txt"
    prompt_path.write_text(" ".join(map(str, ids)) + "\n")
    print(f"wrote {prompt_path}: {args.prompt!r} -> {ids}")


if __name__ == "__main__":
    main()
