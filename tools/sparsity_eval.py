# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.57", "numpy", "huggingface_hub", "tokenizers", "datasets"]
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


# ------------------------------------------------------------------ C. low-rank shortlist

def lowrank_eval(model, ids, device):
    """Shortlist LM head candidates with a rank-r copy of the embedding.

    approx score = (E V) (V^T h): a [vocab, r] table (int8 + per-row scale on
    the device) scanned instead of [vocab, 1024]; the top S candidates are
    then re-scored exactly. Reported: how often the true greedy token is in
    the shortlist, and MB read per token (table + S Q4 rows).
    """
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download

    with torch.no_grad():
        E = model.model.embed_tokens.weight                       # [V, 1024]
        h = model.model(ids).last_hidden_state[0]                 # eval positions
        truth = (h @ E.T).argmax(1)

        # Fit hidden-state statistics on different text (wikitext validation).
        tok = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json"))
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
        texts = [t for t in ds["text"] if len(t) > 300][:60]
        hs = []
        for t in texts:
            x = torch.tensor([[1] + tok.encode(t, add_special_tokens=False).ids[:256]], device=device)
            hs.append(model.model(x).last_hidden_state[0])
        H = torch.cat(hs)                                         # [N, 1024]
        print(f"\nC. low-rank shortlist ({h.shape[0]} eval positions; basis fit on {H.shape[0]} other positions)")

        # Bases: right singular vectors of E, or of E weighted by hidden-state covariance.
        _, _, VE = torch.linalg.svd(E.cpu(), full_matrices=False)
        VE = VE.T.to(device)                                      # [1024, 1024], columns
        C = (H.T @ H / H.shape[0]).cpu()
        L = torch.linalg.cholesky(C + 1e-4 * torch.eye(C.shape[0]))
        _, _, W = torch.linalg.svd((E.cpu() @ L), full_matrices=False)  # directions mattering for typical h
        VH = torch.linalg.solve(L.T, W.T).to(device)              # [1024, 1024]
        VH = VH / VH.norm(dim=0, keepdim=True)
        Q, _ = torch.linalg.qr(VH)

        print(f"{'basis':>6} {'r':>4} {'S':>5} {'hit':>7} {'MB':>6}")
        for name, V in (("E-svd", VE), ("h-wtd", Q)):
            for r in (64, 128, 256):
                Vr = V[:, :r]
                T = E @ Vr                                        # [V, r]
                # int8 per row, as stored on the device
                sc = T.abs().amax(1, keepdim=True) / 127
                Tq = (T / sc).round().clamp(-127, 127) * sc
                approx = (h @ Vr) @ Tq.T                          # [P, V]
                order = approx.argsort(1, descending=True)
                rank = (order == truth.unsqueeze(1)).float().argmax(1)
                for S in (32, 64, 128, 256):
                    hit = (rank < S).float().mean().item()
                    mb = (E.shape[0] * (r + 4) + S * Q4_ROW) / 1e6
                    print(f"{name:>6} {r:4d} {S:5d} {hit:7.1%} {mb:6.2f}")


# ------------------------------------------------------------------ D. flash-sized shortlist

def flash_head_eval(model, ids, device):
    """Smaller low-rank tables (to fit the internal flash) with longer shortlists."""
    V = torch.from_numpy(np.load(ROOT / "out" / "lm_basis.npy")).to(device)   # [1024, 256], ordered
    with torch.no_grad():
        E = model.model.embed_tokens.weight
        h = model.model(ids).last_hidden_state[0]
        truth = (h @ E.T).argmax(1)

        def q8(t):
            sc = t.abs().amax(1, keepdim=True) / 127
            return (t / sc).round().clamp(-127, 127) * sc

        def q4(t):  # Q4_0 blocks of 32
            b = t.reshape(t.shape[0], -1, 32)
            sc = b.abs().amax(-1, keepdim=True) / 7.5
            return ((b / sc).round().clamp(-8, 7) * sc).reshape(t.shape)

        print(f"\nD. flash-sized low-rank shortlist ({h.shape[0]} positions)")
        print(f"{'rank':>5} {'quant':>5} {'MB':>6} {'S':>5} {'hit':>7}")
        for r in (64, 96, 128):
            Vr = V[:, :r]
            T = E @ Vr
            for name, fn, bpe in (("q8", q8, 1 + 2 / 32), ("q4", q4, 0.5 + 2 / 32)):
                approx = (h @ Vr) @ fn(T).T
                rank = (approx.argsort(1, descending=True) == truth.unsqueeze(1)).float().argmax(1)
                mb = E.shape[0] * r * bpe / 1e6
                for S in (256, 512, 1024, 2048):
                    print(f"{r:5d} {name:>5} {mb:6.2f} {S:5d} {(rank < S).float().mean().item():7.1%}")


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
    ap.add_argument("--lowrank", action="store_true", help="only part C")
    ap.add_argument("--flashhead", action="store_true", help="only part D")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, ids = load(device)
    if args.lowrank:
        lowrank_eval(model, ids, device)
        return
    if args.flashhead:
        flash_head_eval(model, ids, device)
        return
    if not args.skip_head:
        lm_head_eval(model, ids, device)
    if not args.skip_mlp:
        mlp_eval(model, ids)


if __name__ == "__main__":
    main()
