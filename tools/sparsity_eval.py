# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.57", "numpy", "huggingface_hub", "tokenizers"]
# ///
"""Simulate two ways to read fewer weight bytes per token, before building them.

A. Clustered LM head. The LM head scans all 65536 embedding rows per token
   (~38 MB at Q4, 29% of a token's reads). Instead: k-means the rows into K
   clusters stored contiguously, score the K centroids, read only the top C
   clusters. Reported: how often the true greedy token is inside them, and
   the MB read (centroids + chosen clusters, Q4 rows of 576 bytes).

B. MLP block sparsity. SD reads punish scattered small reads, so neurons are
   skipped in blocks of 32 (one block = a contiguous run of transposed rows):
     w2:     blocks of act = silu(w1 x) * (w3 x) with the smallest norms
     w1/w3:  blocks of the MLP input x with the smallest norms
   Reported: perplexity and top-1 agreement vs the dense model, and the
   fraction of MLP bytes skipped.

    uv run tools/sparsity_eval.py                    # uses out/eval.txt (tools/eval_text.py)
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

MODEL = "LiquidAI/LFM2.5-230M"
ROOT = Path(__file__).resolve().parent.parent
Q4_ROW = 576        # bytes of one 1024-wide Q4_0 row
BLOCK = 32          # neurons per skippable block


def load(device):
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()
    ids = [int(t) for t in (ROOT / "out" / "eval.txt").read_text().split()]
    return model, torch.tensor([ids], device=device)


# ------------------------------------------------------------------ A. LM head

def kmeans(x: torch.Tensor, k: int, iters: int = 12, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(x.shape[0], generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        # assign by Euclidean distance; argmin ||x - c||^2 = argmax (x.c - |c|^2 / 2)
        assign = torch.cat([(xb @ c.T - 0.5 * (c * c).sum(1)).argmax(1) for xb in x.split(8192)])
        sums = torch.zeros_like(c).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k).clamp(min=1).unsqueeze(1).float()
        c = sums / counts
    return c, assign


def lm_head_eval(model, ids, device):
    with torch.no_grad():
        out = model.model(ids)
        h = out.last_hidden_state[0]                       # [T, 1024], after the final norm
        E = model.lm_head.weight if hasattr(model, "lm_head") else model.model.embed_tokens.weight
        logits = h @ E.T
        ref = model(ids).logits[0]
        assert torch.equal(logits.argmax(1), ref.argmax(1)), "hidden state isn't the LM head input"
        truth = logits.argmax(1)

    full_mb = E.shape[0] * Q4_ROW / 1e6
    print(f"\nA. clustered LM head ({h.shape[0]} positions; full scan = {full_mb:.1f} MB)")
    print(f"{'K':>6} {'C':>4} {'hit':>7} {'MB mean':>8} {'MB p95':>7}")
    for k in (256, 512, 1024):
        t0 = time.time()
        c, assign = kmeans(E, k)
        sizes = torch.bincount(assign, minlength=k)
        cl_scores = h @ c.T                                # [T, K]
        true_cl = assign[truth]                            # cluster holding the right token
        order = cl_scores.argsort(1, descending=True)
        rank = (order == true_cl.unsqueeze(1)).float().argmax(1)  # 0 = best-scored cluster
        for cc in (1, 2, 4, 8, 16, 32):
            hit = (rank < cc).float().mean().item()
            rows = sizes[order[:, :cc]].sum(1).float()
            mb = (k * Q4_ROW + rows * Q4_ROW) / 1e6
            print(f"{k:6d} {cc:4d} {hit:7.1%} {mb.mean().item():8.2f} {mb.quantile(0.95).item():7.2f}")
        print(f"       (k-means {time.time() - t0:.0f} s, cluster sizes {sizes.min().item()}-{sizes.max().item()})")


# ------------------------------------------------------------------ B. MLP sparsity

def keep_blocks(t: torch.Tensor, keep: float) -> torch.Tensor:
    """Zero all but the `keep` fraction of 32-wide blocks with the largest L2 norm, per position."""
    if keep >= 1.0:
        return t
    *lead, n = t.shape
    b = t.reshape(*lead, n // BLOCK, BLOCK)
    norms = b.norm(dim=-1)
    k = max(1, math.ceil(keep * (n // BLOCK)))
    thresh = norms.topk(k, dim=-1).values[..., -1:]
    return (b * (norms >= thresh).unsqueeze(-1)).reshape(*lead, n)


def patch_mlps(model, keep_in: float, keep_act: float):
    for layer in model.model.layers:
        ff = layer.feed_forward

        def fwd(x, ff=ff):
            xin = keep_blocks(x, keep_in)
            act = F.silu(ff.w1(xin)) * ff.w3(xin)
            return ff.w2(keep_blocks(act, keep_act))

        ff.forward = fwd


def unpatch_mlps(model):
    for layer in model.model.layers:
        layer.feed_forward.__dict__.pop("forward", None)


def mlp_eval(model, ids):
    with torch.no_grad():
        dense = model(ids).logits[0, :-1].float()
    target = ids[0, 1:]
    dense_top = dense.argmax(1)
    ppl0 = math.exp(F.cross_entropy(dense, target).item())

    # Per-token MLP bytes: w1, w3 (input-sparse) and w2 (act-sparse) are equal size.
    print(f"\nB. MLP block sparsity (blocks of {BLOCK}); dense ppl {ppl0:.3f}")
    print(f"{'keep x':>7} {'keep act':>9} {'ppl':>8} {'top1=dense':>11} {'MLP bytes':>10} {'token MB':>9}")
    configs = [(1, 1), (1, .8), (1, .6), (1, .5), (1, .4), (.8, 1), (.6, 1), (.8, .6), (.7, .5), (.6, .5), (.5, .4)]
    for keep_in, keep_act in configs:
        patch_mlps(model, keep_in, keep_act)
        with torch.no_grad():
            lg = model(ids).logits[0, :-1].float()
        unpatch_mlps(model)
        ppl = math.exp(F.cross_entropy(lg, target).item())
        agree = (lg.argmax(1) == dense_top).float().mean().item()
        mlp_frac = (2 * keep_in + keep_act) / 3
        token_mb = 130 - 62 * (1 - mlp_frac)
        print(f"{keep_in:7.0%} {keep_act:9.0%} {ppl:8.3f} {agree:11.1%} {mlp_frac:10.0%} {token_mb:9.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-head", action="store_true")
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, ids = load(device)
    if not args.skip_head:
        lm_head_eval(model, ids, device)
    if not args.skip_mlp:
        mlp_eval(model, ids)


if __name__ == "__main__":
    main()
