# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.57", "numpy", "datasets", "huggingface_hub", "tokenizers"]
# ///
"""Corpus of the model's own greedy text, for the speculative n-gram table.

Drafts should predict what *this model* writes greedily, so the table is
built from its greedy continuations of many clean prompts (WikiText
paragraph openings with the dataset's tokenization artifacts undone:
"@,@" -> ",", " ," -> ",", ...).

    uv run tools/selfgen_corpus.py --prompts 4000 --tokens 256   # -> out/spec/selfgen.npy
"""

import argparse
import re
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


def detok(t: str) -> str:
    """Undo WikiText's tokenization."""
    t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    t = re.sub(r" ([,.;:!?)\]'])", r"\1", t)
    t = re.sub(r"([(\[]) ", r"\1", t)
    t = t.replace(" 's", "'s").replace(" n't", "n't").replace('" ', '"')
    return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=4000)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--prompt-tokens", type=int, default=24)
    args = ap.parse_args()

    tok = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json"))
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    paras = []
    for t in ds["text"]:
        if len(t) > 200 and not t.strip().startswith("="):
            paras.append(detok(t))
            if len(paras) >= args.prompts * 3:
                break
    step = len(paras) // args.prompts
    prompts = [[1] + tok.encode(paras[i * step], add_special_tokens=False).ids[: args.prompt_tokens]
               for i in range(args.prompts)]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16 if device == "mps" else torch.float32)
    model = model.to(device).eval()
    out, t0, n = [], time.time(), 0
    for i in range(0, len(prompts), args.batch):
        batch = prompts[i:i + args.batch]
        L = max(len(p) for p in batch)
        ids = torch.tensor([[0] * (L - len(p)) + p for p in batch], device=device)   # left-pad
        mask = (ids != 0).long()
        with torch.no_grad():
            gen = model.generate(ids, attention_mask=mask, max_new_tokens=args.tokens, do_sample=False,
                                 eos_token_id=7, pad_token_id=0)
        for row, p in zip(gen.tolist(), batch):
            seq = [x for x in row if x != 0]
            if 7 in seq[len(p):]:
                seq = seq[: len(p) + seq[len(p):].index(7) + 1]
            out.extend(seq + [-1])  # -1 separates sequences
            n += len(seq) - len(p)
        print(f"{i + len(batch)}/{len(prompts)} prompts, {n} generated tokens, {n / (time.time() - t0):.0f} tok/s", flush=True)

    path = ROOT / "out" / "spec" / "selfgen.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(out, dtype=np.int32))
    print(f"wrote {path}: {n} generated tokens")


if __name__ == "__main__":
    main()
