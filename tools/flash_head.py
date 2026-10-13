# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "safetensors", "numpy", "huggingface_hub", "tokenizers"]
# ///
"""LM head shortlist table for the ESP32's internal flash (lmhead partition).

Rank-128 projection (first columns of out/lm_basis.npy, tools/lm_basis.py)
at Q8, and the [vocab, 128] approximate-logit table at Q4: ~4.9 MB, read
through mmap instead of the SD card. Layout: lfm::FlashHeadHeader in
lfm/include/lfm/format.hpp. tools/sparsity_eval.py --flashhead measured the
true token inside the top 1024 for 99.3% of positions (top 2048: 100%).

    uv run tools/flash_head.py                      # -> out/lmhead.bin
    tools/flash_part.sh lmhead out/lmhead.bin
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import DTYPE_Q4_0, DTYPE_Q8_0, MODEL, quantize_q4_0, quantize_q8_0  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HEADER = struct.Struct("<8s7I4xQQQII")  # = FlashHeadHeader v2 (72 bytes)
LFM_HEAD = struct.Struct("<8s12I2f16I4Q")  # .lfm FileHeader start (export.py HEADER_HEAD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--candidates", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "lmhead.bin")
    ap.add_argument("--model", type=Path, default=ROOT / "out" / "lfm-q4-head-lr256.lfm",
                    help="the .lfm on the device: frequent tokens' rows are copied from it byte for byte")
    ap.add_argument("--max-bytes", type=int, default=0x5B0000, help="lmhead partition size")
    args = ap.parse_args()

    basis = torch.from_numpy(np.load(ROOT / "out" / "lm_basis.npy"))[:, : args.rank]  # [dim, rank]
    w = load_file(str(Path(snapshot_download(MODEL, allow_patterns=["*.safetensors"])) / "model.safetensors"))
    E = w["model.embed_tokens.weight"].float()
    proj = quantize_q8_0(basis.T.contiguous().numpy())          # [rank, dim]
    table = quantize_q4_0((E @ basis).numpy())                   # [vocab, rank]

    proj_off = 128
    table_off = proj_off + len(proj)
    table_off += (-table_off) % 64
    cache_off = table_off + len(table)
    cache_off += (-cache_off) % 64

    # Frequent tokens' exact rows: as many as fit, copied from the .lfm file.
    with open(args.model, "rb") as f:
        fields = LFM_HEAD.unpack(f.read(LFM_HEAD.size))
        embed_dtype, dim, vocab = fields[2], fields[3], fields[9]
        embed_off = fields[-4]
        row = {DTYPE_Q4_0: dim // 32 * 18, DTYPE_Q8_0: dim // 32 * 34}[embed_dtype]
        rows = min(vocab, (args.max_bytes - cache_off) // row)
        f.seek(embed_off)
        cache = f.read(rows * row)
    head = HEADER.pack(b"LFMHEAD\0", 2, E.shape[0], E.shape[1], args.rank, args.candidates,
                       DTYPE_Q8_0, DTYPE_Q4_0, proj_off, table_off, cache_off, rows, embed_dtype)
    blob = head + b"\0" * (proj_off - len(head)) + proj
    blob += b"\0" * (table_off - len(blob)) + table
    blob += b"\0" * (cache_off - len(blob)) + cache
    print(f"cached exact rows of tokens 0..{rows - 1}")
    args.out.write_bytes(blob)
    print(f"wrote {args.out}: rank {args.rank}, {len(blob) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
