# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.57", "numpy", "datasets", "huggingface_hub", "tokenizers", "safetensors"]
# ///
"""How much does 2-bit cost? Quality of 2-bit layer weights vs today's Q4_0.

Formats (along each row, as the engine stores them):
  q4      Q4_0, blocks of 32, f16 scale (4.5 bits/weight): the device today
  q2b64   blocks of 64, f16 scale + f16 min, codes 0..3, w = scale*q + min (2.5 bits/weight)
  q2b32   same with blocks of 32 (3.0 bits/weight)
Quantizers: rtn (per-block min/max search), gptq (column-by-column error
compensation with the inputs' Hessian from calibration text).

Metrics: perplexity on WikiText-2 test windows, and next-token agreement with
the Q4 model on its own greedy chat replies (what speculative decoding and
the user see). Embedding stays Q4 in every variant.

    uv run tools/quant2_study.py [--variants q4 q2b64-rtn q2b64-gptq q2b64-gptq-mlp ...]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import draft_heads as dh  # noqa: E402
import export  # noqa: E402

DEVICE = dh.DEVICE
MLP = ("feed_forward.w1", "feed_forward.w2", "feed_forward.w3")


# ---------------------------------------------------------------- formats

def q4_roundtrip(w: torch.Tensor) -> torch.Tensor:
    a = w.detach().cpu().float().numpy()
    return torch.from_numpy(dh.dequant_q4_0(export.quantize_q4_0(a), *a.shape))


def q2_params(blocks: torch.Tensor, iters: int = 12, levels: int = 4):
    """Per-block (scale, min) for 4 levels minimising squared error.
    blocks [..., B]. Starts from min/max, then alternates assignment and a
    least-squares refit of scale and min."""
    lo = blocks.min(-1, keepdim=True).values
    hi = blocks.max(-1, keepdim=True).values
    top = levels - 1
    scale = ((hi - lo) / top).clamp_min(1e-12)
    mn = lo
    for _ in range(iters):
        q = ((blocks - mn) / scale).round().clamp(0, top)
        # least squares for w ~ scale*q + mn
        n = blocks.shape[-1]
        sq, sw = q.sum(-1, keepdim=True), blocks.sum(-1, keepdim=True)
        sqq, sqw = (q * q).sum(-1, keepdim=True), (q * blocks).sum(-1, keepdim=True)
        det = n * sqq - sq * sq
        ok = det.abs() > 1e-12
        new_scale = torch.where(ok, (n * sqw - sq * sw) / torch.where(ok, det, torch.ones_like(det)), scale)
        new_mn = torch.where(ok, (sw - new_scale * sq) / n, mn)
        scale, mn = new_scale.clamp_min(1e-12), new_mn
    # f16 storage
    return scale.half().float(), mn.half().float()


def q2_rtn(w: torch.Tensor, block: int, levels: int = 4) -> torch.Tensor:
    rows, cols = w.shape
    b = w.float().reshape(rows, cols // block, block)
    scale, mn = q2_params(b, levels=levels)
    q = ((b - mn) / scale).round().clamp(0, levels - 1)
    return (q * scale + mn).reshape(rows, cols)


def q2_gptq(w: torch.Tensor, H: torch.Tensor, block: int, damp: float = 0.01, levels: int = 4) -> torch.Tensor:
    """GPTQ (Frantar et al.) with per-block (scale, min) fixed at the start of
    each block of columns, from the error-updated weights."""
    W = w.float().clone()  # CPU: the Cholesky needs float64 (not on MPS)
    rows, cols = W.shape
    H = H.double()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    Q = torch.zeros_like(W)
    for c0 in range(0, cols, block):
        c1 = c0 + block
        W1 = W[:, c0:c1].clone()
        scale, mn = q2_params(W1[:, None, :], levels=levels)
        scale, mn = scale[:, 0], mn[:, 0]
        Err = torch.zeros_like(W1)
        Hinv1 = Hinv[c0:c1, c0:c1]
        for i in range(block):
            wcol = W1[:, i]
            q = ((wcol[:, None] - mn) / scale).round().clamp(0, levels - 1)
            qw = (q * scale + mn)[:, 0]
            Q[:, c0 + i] = qw
            err = (wcol - qw) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err[:, i] = err
        W[:, c1:] -= Err @ Hinv[c0:c1, c1:]
    return Q


# ---------------------------------------------------------------- data

def wiki_windows(tok, n_windows: int, length: int):
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok.encode(text, add_special_tokens=False).ids
    return [[1] + ids[i * length:(i + 1) * length - 1] for i in range(n_windows)]


def calib_sequences(n: int):
    seqs = []
    for path in ("selfgen_chat.npy", "selfgen.npy", "selfgen_alpaca.npy"):
        seqs += dh.split_seqs(np.load(ROOT / "out" / "spec" / path))[: n // 3]
    return [s[:512].tolist() for s in seqs]


@torch.no_grad()
def hessians(model, seqs):
    """H = sum x x^T of every Linear's input over the calibration text."""
    H, hooks = {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".layers." in name:
            def hook(m, inp, out, name=name):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                H[name] = H.get(name, 0) + x.T @ x
            hooks.append(mod.register_forward_hook(hook))
    for s in seqs:
        model(torch.tensor([s], device=DEVICE))
    for h in hooks:
        h.remove()
    return {k: v.cpu() for k, v in H.items()}


@torch.no_grad()
def evaluate(model, windows, chat):
    nll = n = 0.0
    for w in windows:
        x = torch.tensor([w], device=DEVICE)
        logits = model(x).logits[0, :-1].float()
        nll += torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum").item()
        n += len(w) - 1
    agree = tot = 0
    for s, p, ref in chat:
        pred = model(torch.tensor([s], device=DEVICE)).logits[0].argmax(-1).cpu()
        agree += (pred[p - 1:-1] == ref).sum().item()
        tot += len(ref)
    return float(np.exp(nll / n)), agree / tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+",
                    default=["bf16", "q4", "q2b64-rtn", "q2b64-gptq", "q2b32-gptq", "q2b64-gptq-mlp"])
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--calib", type=int, default=96)
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(export.MODEL, "tokenizer.json"))
    base = AutoModelForCausalLM.from_pretrained(export.MODEL, dtype=torch.float32).to(DEVICE).eval()
    orig = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    q4 = torch.load(dh.OUT / "q4_state.pt")  # Q4_0 round trip of every matrix (draft_heads.py)
    windows = wiki_windows(tok, args.windows, 512)

    # Reference greedy chat replies of the Q4 model (what the device says today).
    base.load_state_dict({**orig, **q4})
    chat = []
    for p in dh.CHAT_PROMPTS:
        ids = dh.chat_ids(tok, p)
        out = base.generate(torch.tensor([ids], device=DEVICE), max_new_tokens=96, do_sample=False,
                            eos_token_id=7, pad_token_id=0)[0]
        # Reference: the Q4 model's own argmax in a full forward pass (same
        # code path as the variants; generate()'s cached path differs slightly).
        ref = base(out[None]).logits[0].argmax(-1).cpu()[len(ids) - 1:-1]
        chat.append((out.tolist(), len(ids), ref))

    need_h = any("gptq" in v for v in args.variants)
    if need_h:
        base.load_state_dict(orig)
        t0 = time.time()
        H = hessians(base, calib_sequences(args.calib))
        print(f"Hessians for {len(H)} matrices in {time.time() - t0:.0f} s", flush=True)

    layer_mats = [k for k, v in orig.items() if v.ndim == 2 and ".layers." in k and v.shape[1] % 64 == 0]
    for variant in args.variants:
        t0 = time.time()
        state = dict(orig)
        if variant != "bf16":
            state.update(q4)  # embedding and anything not replaced below: Q4
        if variant.startswith(("q2", "q3")):
            fmt, method = variant.split("-")[0], variant.split("-")[1]
            block = int(fmt[3:])
            levels = 1 << int(fmt[1])
            only_mlp = variant.endswith("-mlp")
            for k in layer_mats:
                if only_mlp and not any(m in k for m in MLP):
                    continue
                w = orig[k]
                state[k] = (q2_rtn(w, block, levels) if method == "rtn"
                            else q2_gptq(w, H[k[: -len(".weight")]], block, levels=levels))
        base.load_state_dict(state)
        ppl, agree = evaluate(base, windows, chat)
        bits = {"bf16": 16, "q4": 4.5}.get(variant)
        if bits is None:
            per = int(variant[1]) + 32 / int(variant.split("-")[0][3:])
            bits = per if not variant.endswith("-mlp") else None
        print(f"{variant:16s} ppl {ppl:7.2f}   agrees with Q4 on chat {100 * agree:5.1f}%   "
              f"{'' if bits is None else f'{bits:.2f} bits/w'}   ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
