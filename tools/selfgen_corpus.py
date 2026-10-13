# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.57", "numpy", "datasets", "huggingface_hub", "tokenizers"]
# ///
"""Corpus of the model's own greedy text, to train the draft head on
(tools/draft_heads.py).

Drafts should predict what *this model* writes greedily, so the head learns
from its greedy continuations of many clean prompts (WikiText
paragraph openings with the dataset's tokenization artifacts undone:
"@,@" -> ",", " ," -> ",", ...).

    uv run tools/selfgen_corpus.py --prompts 4000 --tokens 256   # -> out/spec/selfgen.npy
    uv run tools/selfgen_corpus.py --source alpaca --q4 --out out/spec/selfgen_alpaca.npy

--source chat: one ChatML user turn per Alpaca instruction; the answer runs
to <|im_end|> (what the on-device chat sees).
--source alpaca: prompts from the Alpaca instruction set in several shapes
(the opening of an answer, "Q: ...\nA:", the bare instruction), for lists,
code, letters and Q&A next to WikiText's prose. --q4: generate with the
Q4_0 round-tripped model (what the device runs; tools/draft_heads.py).
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


def alpaca_prompts(tok: Tokenizer, n: int) -> list[list[int]]:
    """Plain-text prompts in three shapes, cycling: the first 6-20 tokens of an
    answer, "Q: <instruction>\nA:", and "<instruction>\n"."""
    ds = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=0)
    rng = np.random.default_rng(0)
    out = []
    for i, ex in enumerate(ds):
        if len(out) >= n:
            break
        instr = ex["instruction"].strip()
        if ex["input"].strip():
            instr += "\n" + ex["input"].strip()
        kind = i % 3
        if kind == 0:
            ids = tok.encode(ex["output"], add_special_tokens=False).ids
            if len(ids) < 24:
                continue
            ids = ids[: int(rng.integers(6, 21))]
        elif kind == 1:
            ids = tok.encode(f"Q: {instr}\nA:", add_special_tokens=False).ids
        else:
            ids = tok.encode(instr + "\n", add_special_tokens=False).ids
        if len(ids) > 64:
            continue
        out.append([1] + ids)
    return out


def chat_prompts(tok: Tokenizer, n: int) -> list[list[int]]:
    """One user turn in the ChatML template the device chat uses (instructions
    from the other end of the shuffled Alpaca set than alpaca_prompts)."""
    ds = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=0)
    out = []
    for ex in ds.select(range(len(ds) - 1, -1, -1)):
        if len(out) >= n:
            break
        text = ex["instruction"].strip() + ("\n" + ex["input"].strip() if ex["input"].strip() else "")
        ids = tok.encode(f"<|startoftext|><|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n",
                         add_special_tokens=False).ids
        if len(ids) <= 96:
            out.append(ids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=4000)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--prompt-tokens", type=int, default=24)
    ap.add_argument("--source", choices=["wiki", "alpaca", "chat"], default="wiki")
    ap.add_argument("--q4", action="store_true", help="generate with the Q4_0 round-tripped model")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "spec" / "selfgen.npy")
    args = ap.parse_args()

    tok = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json"))
    if args.source == "wiki":
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
    elif args.source == "alpaca":
        prompts = alpaca_prompts(tok, args.prompts)
    else:
        prompts = chat_prompts(tok, args.prompts)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if args.q4:
        import draft_heads
        model = draft_heads.q4_model()
    else:
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

    path = args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(out, dtype=np.int32))
    print(f"wrote {path}: {n} generated tokens")


if __name__ == "__main__":
    main()
