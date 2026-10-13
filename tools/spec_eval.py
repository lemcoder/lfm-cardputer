# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.57", "datasets", "numpy", "tokenizers", "huggingface_hub"]
# ///
"""How well would cheap drafts predict LFM2.5-230M's greedy tokens?

Speculative decoding with greedy verification is lossless: the model checks
k drafted tokens in one pass over its weights (the same reads as one token)
and keeps the drafts that match its own greedy choice. What it buys depends
only on how often the draft is right. This script measures that.

    uv run tools/spec_eval.py                  # wikitext-103 n-gram table + prompt lookup
    uv run tools/spec_eval.py --gen-prompts 40 --gen-tokens 160
    uv run tools/spec_eval.py --export out/ngram.bin    # also write the device table (lfm/draft.hpp)

Drafts:
  ngram   trigram (a, b) -> most frequent next token, bigram fallback, built
          from a corpus and cut to a flash-sized budget (--budget-mb)
  lookup  prompt lookup: find the last 2 tokens earlier in the context and
          propose what followed them there

Simulation per greedy sequence S (exactly what the device would get): at
position i draft d1..dk from S[:i] (each draft extends the context with the
previous drafts), accept the longest prefix with d_j == S[i+j-1], advance by
accepted + 1 (the model's own token). Reported: tokens per pass.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

MODEL = "LiquidAI/LFM2.5-230M"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "out" / "spec"
TRI_ENTRY_BYTES = 8  # on-device: 4-byte key fingerprint + 2-byte token + padding


def tokenize_corpus(tok: Tokenizer, split: str, limit_mb: int) -> np.ndarray:
    path = CACHE / f"wikitext103-{split}-{limit_mb}mb.npy"
    if path.exists():
        return np.load(path)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    texts, size = [], 0
    for t in ds["text"]:
        if t.strip():
            texts.append(t)
            size += len(t)
            if size > limit_mb * 1_000_000:
                break
    ids = []
    for i in range(0, len(texts), 10_000):
        for enc in tok.encode_batch(texts[i:i + 10_000], add_special_tokens=False):
            ids.extend(enc.ids)
    arr = np.asarray(ids, dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def top_next(keys: np.ndarray, nxt: np.ndarray):
    """For each distinct key: most frequent next token and its count."""
    pair = (keys.astype(np.uint64) << np.uint64(17)) | nxt.astype(np.uint64)
    uniq, counts = np.unique(pair, return_counts=True)
    k = (uniq >> np.uint64(17)).astype(np.uint64)
    v = (uniq & np.uint64((1 << 17) - 1)).astype(np.int32)
    order = np.lexsort((-counts, k))  # by key, then count desc
    k, v, counts = k[order], v[order], counts[order]
    first = np.ones(len(k), dtype=bool)
    first[1:] = k[1:] != k[:-1]
    return k[first], v[first], counts[first]


def build_ngram(corpus: np.ndarray, budget_mb: float):
    c = corpus.astype(np.int64)
    # bigram: full table, 65536 x 2 bytes
    bk, bv, _ = top_next(c[:-1], c[1:])
    bigram = np.full(65536, -1, dtype=np.int32)
    bigram[bk.astype(np.int64)] = bv
    # trigram: keep the contexts with the most top-1 hits that fit the budget
    tk, tv, tc = top_next(c[:-2] * 65536 + c[1:-1], c[2:])
    budget = int((budget_mb * 1e6 - 65536 * 2) / TRI_ENTRY_BYTES)
    keep = np.argsort(-tc)[:budget]
    trigram = dict(zip(tk[keep].tolist(), tv[keep].tolist()))
    return bigram, trigram


def write_table(path: Path, bigram: np.ndarray, trigram: dict, load: float = 0.6) -> None:
    """Binary table read by lfm::Drafter (layout in lfm/include/lfm/draft.hpp)."""
    slots = 1
    while slots * load < len(trigram):
        slots *= 2
    bits = slots.bit_length() - 1
    keys = np.zeros(slots, dtype="<u4")
    nexts = np.full(slots, 0xFFFF, dtype="<u2")
    for key, nxt in trigram.items():
        a, b = divmod(int(key), 65536)
        k32 = (a << 16) | b
        s = ((k32 * 0x9E3779B1) & 0xFFFFFFFF) >> (32 - bits)
        while nexts[s] != 0xFFFF:
            s = (s + 1) & (slots - 1)
        keys[s], nexts[s] = k32, nxt
    entries = np.zeros(slots, dtype=[("key", "<u4"), ("next", "<u2"), ("pad", "<u2")])
    entries["key"], entries["next"] = keys, nexts
    bi = np.where(bigram >= 0, bigram, 0xFFFF).astype("<u2")
    header = b"LFMNGRM\0" + np.array([1, 65536, slots, 0], dtype="<u4").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + bi.tobytes() + entries.tobytes())
    print(f"wrote {path}: {len(trigram)} trigrams in {slots} slots, {path.stat().st_size / 1e6:.2f} MB")


def draft_ngram(ctx: list[int], k: int, bigram, trigram) -> list[int]:
    out, c = [], list(ctx[-2:])
    for _ in range(k):
        d = trigram.get(c[-2] * 65536 + c[-1], -1) if len(c) >= 2 else -1
        if d < 0:
            d = int(bigram[c[-1]])
        if d < 0:
            break
        out.append(d)
        c.append(d)
    return out


def draft_lookup(ctx: list[int], k: int, n: int = 2) -> list[int]:
    """Most recent earlier occurrence of the last n tokens; propose what followed."""
    if len(ctx) <= n:
        return []
    tail = ctx[-n:]
    for start in range(len(ctx) - n - 1, -1, -1):
        if ctx[start:start + n] == tail:
            return ctx[start + n:start + n + k]
    return []


def simulate(seqs, prompt_lens, draft_fn, k: int):
    passes = tokens = 0
    for s, p in zip(seqs, prompt_lens):
        i = p
        while i < len(s):
            drafts = draft_fn(s[:i], k)
            acc = 0
            while acc < len(drafts) and i + acc < len(s) and drafts[acc] == s[i + acc]:
                acc += 1
            step = min(acc + 1, len(s) - i)
            tokens += step
            passes += 1
            i += step
    return tokens / passes


def greedy_sequences(n_prompts: int, n_tokens: int, tok: Tokenizer):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    paras = [t for t in ds["text"] if len(t) > 400 and not t.strip().startswith("=")]
    step = max(1, len(paras) // n_prompts)
    prompts = [[1] + tok.encode(paras[i * step], add_special_tokens=False).ids[:32] for i in range(n_prompts)]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()
    seqs, plens = [], []
    t0 = time.time()
    for ids in prompts:
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=n_tokens, do_sample=False, eos_token_id=7, pad_token_id=0)
        seqs.append(out[0].tolist())
        plens.append(len(ids))
    print(f"generated {sum(len(s) - p for s, p in zip(seqs, plens))} greedy tokens in {time.time() - t0:.0f} s on {device}")
    return seqs, plens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-mb", type=int, default=600, help="wikitext-103 train text to count")
    ap.add_argument("--budget-mb", type=float, nargs="+", default=[1.0, 2.5, 5.5])
    ap.add_argument("--gen-prompts", type=int, default=24)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--export", type=Path, help="write the n-gram table for the device here")
    ap.add_argument("--export-mb", type=float, default=1.0, help="trigram budget for --export")
    args = ap.parse_args()

    tok = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json"))
    t0 = time.time()
    corpus = tokenize_corpus(tok, "train", args.corpus_mb)
    print(f"corpus: {len(corpus) / 1e6:.1f}M tokens ({time.time() - t0:.0f} s)")

    if args.export:
        bigram, trigram = build_ngram(corpus, args.export_mb)
        write_table(args.export, bigram, trigram)
        if args.gen_prompts == 0:
            return

    seqs, plens = greedy_sequences(args.gen_prompts, args.gen_tokens, tok)
    sample = tok.decode(seqs[0][plens[0]:plens[0] + 60])
    print(f"sample continuation: {sample!r}\n")

    print(f"{'draft':<26} {'k=1':>6} {'k=2':>6} {'k=3':>6}   (tokens per pass; 1.00 = no gain)")
    look = lambda ctx, k: draft_lookup(ctx, k)
    print(f"{'prompt lookup':<26} " + " ".join(f"{simulate(seqs, plens, look, k):6.2f}" for k in (1, 2, 3)))
    for mb in args.budget_mb:
        t0 = time.time()
        bigram, trigram = build_ngram(corpus, mb)
        ng = lambda ctx, k: draft_ngram(ctx, k, bigram, trigram)

        def both(ctx, k):  # lookup when it has something, else n-gram
            return draft_lookup(ctx, k) or ng(ctx, k)

        row = " ".join(f"{simulate(seqs, plens, ng, k):6.2f}" for k in (1, 2, 3))
        row2 = " ".join(f"{simulate(seqs, plens, both, k):6.2f}" for k in (1, 2, 3))
        print(f"{f'ngram {mb} MB ({len(trigram) / 1e3:.0f}K tri)':<26} {row}")
        print(f"{f'lookup, else ngram {mb} MB':<26} {row2}   [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
