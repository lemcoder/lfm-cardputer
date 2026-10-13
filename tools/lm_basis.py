# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.57", "numpy", "huggingface_hub", "tokenizers", "datasets"]
# ///
"""Basis for the low-rank LM head shortlist (export.py --lm-head lowrank).

logit[v] = E[v] . h. With an orthonormal basis V (1024 x r): approx logit =
(E V)[v] . (V^T h). V is chosen so the approximation is good for the hidden
states the model actually produces: the top right singular vectors of E L,
where L L^T is the covariance of h (fit on wikitext validation text, not on
the eval text). tools/sparsity_eval.py --lowrank measured this: rank 256,
shortlist 256 -> the true greedy token is in the shortlist 99.8% of the time.

    uv run tools/lm_basis.py            # -> out/lm_basis.npy  (float32 [1024, 256], columns)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

MODEL = "LiquidAI/LFM2.5-230M"
ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--texts", type=int, default=60)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "lm_basis.npy")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()
    tok = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json"))
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t) > 300][: args.texts]

    with torch.no_grad():
        hs = []
        for t in texts:
            x = torch.tensor([[1] + tok.encode(t, add_special_tokens=False).ids[:256]], device=device)
            hs.append(model.model(x).last_hidden_state[0])
        H = torch.cat(hs).cpu()
        E = model.model.embed_tokens.weight.cpu()

    C = H.T @ H / H.shape[0]
    L = torch.linalg.cholesky(C + 1e-4 * torch.eye(C.shape[0]))
    _, _, W = torch.linalg.svd(E @ L, full_matrices=False)
    V = torch.linalg.solve(L.T, W.T)            # columns: directions that matter for typical h
    V = V / V.norm(dim=0, keepdim=True)
    Q, _ = torch.linalg.qr(V[:, : args.rank])   # orthonormal
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, Q.numpy().astype(np.float32))
    print(f"wrote {args.out}: [{Q.shape[0]}, {Q.shape[1]}] from {H.shape[0]} hidden states")


if __name__ == "__main__":
    main()
