# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "safetensors", "numpy", "huggingface_hub", "tokenizers"]
# ///
"""Convert LFM2.5-230M from HF safetensors into the engine's .lfm file.

    uv run tools/export.py                       # Q8_0  -> out/lfm-q8.lfm  (~250 MB, for the device)
    uv run tools/export.py --dtype bf16          # BF16  -> out/lfm-bf16.lfm (exact weights, host tests)
    uv run tools/export.py --prompt "Hello"      # also rewrite out/prompt.txt

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
            w = scale * q                              (matrices, --dtype q8)
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
VERSION = 1
MAX_LAYERS = 16
ALIGN = 4096

DTYPE_F32, DTYPE_BF16, DTYPE_Q8_0 = 0, 1, 2
LAYER_CONV, LAYER_ATTN = 0, 1

# LayerOffsets: 14 x uint64, see format.hpp.
LAYER_FIELDS = ["operator_norm", "ffn_norm", "w1", "w2", "w3",
                "conv_in", "conv_kernel", "conv_out",
                "q", "k", "v", "o", "q_norm", "k_norm"]
# FileHeader before the layer table: magic, 12 x u32, 2 x f32, 16 x u32 layer types,
# 4 x u64 (lands on offset 128, so no padding in the C++ struct either).
HEADER_HEAD = struct.Struct("<8s12I2f16I4Q")
HEADER_SIZE = HEADER_HEAD.size + MAX_LAYERS * len(LAYER_FIELDS) * 8


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


class Writer:
    def __init__(self, f, matrix_dtype: int):
        self.f = f
        self.matrix_dtype = matrix_dtype

    def _align(self):
        pos = self.f.tell()
        pad = (-pos) % ALIGN
        self.f.write(b"\0" * pad)
        return pos + pad

    def f32(self, t: torch.Tensor) -> int:
        off = self._align()
        self.f.write(t.float().contiguous().numpy().astype("<f4").tobytes())
        return off

    def matrix(self, t: torch.Tensor) -> int:
        assert t.ndim == 2
        off = self._align()
        if self.matrix_dtype == DTYPE_BF16:
            self.f.write(t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().astype("<i2").tobytes())
        else:
            self.f.write(quantize_q8_0(t.float().numpy()))
        return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["q8", "bf16"], default="q8")
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

    matrix_dtype = DTYPE_BF16 if args.dtype == "bf16" else DTYPE_Q8_0
    out = args.out or ROOT / "out" / f"lfm-{args.dtype}.lfm"
    out.parent.mkdir(parents=True, exist_ok=True)

    layers = []
    with open(out, "wb") as f:
        f.write(b"\0" * HEADER_SIZE)
        wr = Writer(f, matrix_dtype)
        for i in range(n_layers):
            p = f"model.layers.{i}."
            o = dict.fromkeys(LAYER_FIELDS, 0)
            o["operator_norm"] = wr.f32(w[p + "operator_norm.weight"])
            if layer_types[i] == LAYER_CONV:
                o["conv_in"] = wr.matrix(w[p + "conv.in_proj.weight"])
                o["conv_kernel"] = wr.f32(w[p + "conv.conv.weight"].reshape(dim, conv_l))
                o["conv_out"] = wr.matrix(w[p + "conv.out_proj.weight"])
            else:
                a = p + "self_attn."
                o["q"] = wr.matrix(w[a + "q_proj.weight"])
                o["k"] = wr.matrix(w[a + "k_proj.weight"])
                o["v"] = wr.matrix(w[a + "v_proj.weight"])
                o["q_norm"] = wr.f32(w[a + "q_layernorm.weight"])
                o["k_norm"] = wr.f32(w[a + "k_layernorm.weight"])
                o["o"] = wr.matrix(w[a + "out_proj.weight"])
            o["ffn_norm"] = wr.f32(w[p + "ffn_norm.weight"])
            o["w1"] = wr.matrix(w[p + "feed_forward.w1.weight"])
            o["w3"] = wr.matrix(w[p + "feed_forward.w3.weight"])
            o["w2"] = wr.matrix(w[p + "feed_forward.w2.weight"])
            layers.append(o)
            print(f"layer {i:2d} {'attn' if layer_types[i] else 'conv'}  @ {f.tell() / 1e6:7.1f} MB", flush=True)

        final_norm = wr.f32(w["model.embedding_norm.weight"])
        embed = wr.matrix(w["model.embed_tokens.weight"])
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
        head = HEADER_HEAD.pack(MAGIC, VERSION, matrix_dtype, dim, ff, n_layers, n_heads, n_kv, head_dim,
                                vocab, conv_l, 0, 0, eps, theta, *types,
                                embed, final_norm, vocab_index, vocab_bytes)
        table = b"".join(struct.pack(f"<{len(LAYER_FIELDS)}Q", *(o[k] for k in LAYER_FIELDS)) for o in layers)
        table += b"\0" * (MAX_LAYERS - n_layers) * len(LAYER_FIELDS) * 8
        f.seek(0)
        f.write(head + table)

    print(f"wrote {out} ({total / 1e6:.1f} MB, {args.dtype})")

    ids = Tokenizer.from_file(str(src / "tokenizer.json")).encode(args.prompt).ids
    prompt_path = ROOT / "out" / "prompt.txt"
    prompt_path.write_text(" ".join(map(str, ids)) + "\n")
    print(f"wrote {prompt_path}: {args.prompt!r} -> {ids}")


if __name__ == "__main__":
    main()
