# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.57", "numpy", "datasets", "huggingface_hub", "tokenizers", "safetensors"]
# ///
"""Draft heads for speculative decoding (Medusa / Hydra style), trained on
the model's own greedy text.

After a pass the engine has h, the final (normed) hidden state at the
position that produced the newest token `fed`. A draft head guesses the
tokens after `fed` from h alone, so drafts cost no extra pass over the
weights. Two variants:

  medusa     head_k(h) predicts token k+1 after h's position, one head per k
  recurrent  one shared step (h, embedding of the last token) -> h', applied
             again for the next draft with the token it just proposed

Draft logits use the model's own (tied) LM head.

    uv run tools/draft_heads.py collect          # hidden states of out/spec/selfgen.npy, Q4 model
    uv run tools/draft_heads.py train --variant recurrent
    uv run tools/draft_heads.py eval             # acceptance on clean prompts vs the n-gram drafter

Hidden states come from the model with Q4_0-dequantized matrices (what the
device runs), not the BF16 original.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import export  # noqa: E402

OUT = ROOT / "out" / "heads"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def dequant_q4_0(raw: bytes, rows: int, cols: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=[("d", "<f2"), ("q", "u1", 16)]).reshape(rows, cols // 32)
    lo = (b["q"] & 15).astype(np.int8) - 8
    hi = (b["q"] >> 4).astype(np.int8) - 8
    q = np.concatenate([lo, hi], axis=-1).astype(np.float32)
    return (q * b["d"].astype(np.float32)[..., None]).reshape(rows, cols)


def q4_model():
    """The HF model with every matrix replaced by its Q4_0 round trip."""
    cache = OUT / "q4_state.pt"
    model = AutoModelForCausalLM.from_pretrained(export.MODEL, dtype=torch.float32)
    if cache.exists():
        model.load_state_dict(torch.load(cache), strict=False)
    else:
        sd = {}
        t0 = time.time()
        for name, p in model.named_parameters():
            if p.ndim == 2 and p.shape[1] % 32 == 0:
                w = p.detach().numpy()
                sd[name] = torch.from_numpy(dequant_q4_0(export.quantize_q4_0(w), *w.shape))
        print(f"quantized {len(sd)} matrices in {time.time() - t0:.0f} s")
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save(sd, cache)
        model.load_state_dict(sd, strict=False)
    return model.to(DEVICE).eval()


def split_seqs(corpus: np.ndarray) -> list[np.ndarray]:
    cut = np.flatnonzero(corpus == -1)
    out, start = [], 0
    for c in cut:
        if c - start >= 4:
            out.append(corpus[start:c])
        start = c + 1
    return out


@torch.no_grad()
def hidden_and_argmax(model, seq: list[int] | np.ndarray):
    ids = torch.tensor(np.asarray(seq)[None], device=DEVICE)
    h = model.model(input_ids=ids).last_hidden_state[0]  # [T, dim], after the final norm
    E = model.lm_head.weight
    pred = torch.cat([(h[i:i + 512] @ E.T).argmax(-1) for i in range(0, len(h), 512)])
    return h, pred


def collect(args):
    seqs = [s for path in args.corpus for s in split_seqs(np.load(path))]
    rng = np.random.default_rng(0)
    seqs = [seqs[i] for i in rng.permutation(len(seqs))]  # mix sources: the held-out tail isn't one source
    model = q4_model()
    n = sum(len(s) for s in seqs)
    dim = model.config.hidden_size
    H = np.lib.format.open_memmap(OUT / "h.npy", mode="w+", dtype=np.float16, shape=(n, dim))
    tok = np.empty(n, np.int32)    # input token at each position
    pred = np.empty(n, np.int32)   # Q4 model's greedy next token there
    seg = np.empty(n, np.int32)    # sequence index (positions don't chain across sequences)
    t0, at = time.time(), 0
    for si, s in enumerate(seqs):
        h, p = hidden_and_argmax(model, s)
        H[at:at + len(s)] = h.half().cpu().numpy()
        tok[at:at + len(s)] = s
        pred[at:at + len(s)] = p.cpu().numpy()
        seg[at:at + len(s)] = si
        at += len(s)
        if si % 200 == 0:
            print(f"{si}/{len(seqs)} sequences, {at} positions, {time.time() - t0:.0f} s", flush=True)
    H.flush()
    np.savez(OUT / "meta.npz", tok=tok, pred=pred, seg=seg)
    agree = np.mean(pred[:-1][seg[:-1] == seg[1:]] == tok[1:][seg[:-1] == seg[1:]])
    print(f"wrote {n} positions; Q4 greedy agrees with the corpus (BF16 greedy) on {100 * agree:.1f}%")


# ---------------------------------------------------------------- heads

class Medusa(nn.Module):
    def __init__(self, dim: int, rank: int, depth: int):
        super().__init__()
        self.depth = depth
        self.down = nn.ModuleList(nn.Linear(dim, rank, bias=False) for _ in range(depth))
        self.up = nn.ModuleList(nn.Linear(rank, dim, bias=False) for _ in range(depth))
        for u in self.up:
            nn.init.zeros_(u.weight)

    def forward(self, h, toks, E, norm_w):
        """h [B, dim]; toks [B, depth] (unused). Returns per-depth hidden [depth][B, dim]."""
        return [h + self.up[k](F.silu(self.down[k](h))) for k in range(len(self.down))]


class Recurrent(nn.Module):
    """h_{k+1} = h_k + up(silu(down([h_k, norm(emb(tok_k))])))."""

    def __init__(self, dim: int, rank: int, depth: int):
        super().__init__()
        self.depth = depth
        self.down = nn.Linear(2 * dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def step(self, h, tok, E):
        e = E[tok]
        e = e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + 1e-5)
        return h + self.up(F.silu(self.down(torch.cat([h, e], -1))))

    def forward(self, h, toks, E, norm_w):
        """toks[:, k]: the token fed at draft step k (teacher forcing in training)."""
        out = []
        for k in range(self.depth):
            h = self.step(h, toks[:, k], E)
            out.append(h)
        return out


def make_head(variant: str, dim: int, rank: int, depth: int):
    return (Medusa if variant == "medusa" else Recurrent)(dim, rank, depth)


def head_logits(hk, E, norm_w):
    hn = hk * torch.rsqrt(hk.pow(2).mean(-1, keepdim=True) + 1e-5) * norm_w
    return hn @ E.T


def training_rows(meta, depth: int, holdout: float):
    """Positions j where the chain h_j -> fed -> draft_1 .. draft_depth exists.

    fed = pred[j] (the model's next token), draft_k target = pred[j+k], valid
    while the corpus continues the way the Q4 model would (tok[j+k] == pred[j+k-1]).
    """
    tok, pred, seg = meta["tok"], meta["pred"], meta["seg"]
    n = len(tok)
    idx = np.arange(n - depth)
    ok = np.ones(len(idx), bool)
    for k in range(1, depth + 1):
        ok &= seg[idx + k] == seg[idx]
        ok &= tok[idx + k] == pred[idx + k - 1]
    idx = idx[ok]
    feed = np.stack([pred[idx + k] for k in range(depth)], 1)          # fed, draft_1, ...
    target = np.stack([pred[idx + k + 1] for k in range(depth)], 1)    # draft_1, draft_2, ...
    cut = int(len(idx) * (1 - holdout))
    return (idx[:cut], feed[:cut], target[:cut]), (idx[cut:], feed[cut:], target[cut:])


def train(args):
    H = np.load(OUT / "h.npy", mmap_mode="r")
    meta = np.load(OUT / "meta.npz")
    tr, va = training_rows(meta, args.depth, 0.03)
    print(f"{len(tr[0])} training rows, {len(va[0])} held out")
    model = AutoModelForCausalLM.from_pretrained(export.MODEL, dtype=torch.float32)
    q4 = torch.load(OUT / "q4_state.pt")
    E = q4["model.embed_tokens.weight"].to(DEVICE)
    norm_w = model.model.embedding_norm.weight.detach().to(DEVICE)
    dim = E.shape[1]
    del model

    head = make_head(args.variant, dim, args.rank, args.depth).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    steps = args.epochs * len(tr[0]) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    t0, step = time.time(), 0
    for epoch in range(args.epochs):
        perm = rng.permutation(len(tr[0]))
        for b in range(0, len(perm) - args.batch + 1, args.batch):
            sel = np.sort(perm[b:b + args.batch])
            h = torch.from_numpy(np.asarray(H[tr[0][sel]], np.float32)).to(DEVICE)
            feed = torch.from_numpy(tr[1][sel]).long().to(DEVICE)
            tgt = torch.from_numpy(tr[2][sel]).long().to(DEVICE)
            outs = head(h, feed, E, norm_w)
            loss = sum(F.cross_entropy(head_logits(o, E, norm_w), tgt[:, k]) for k, o in enumerate(outs))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item() / args.depth:.3f} {time.time() - t0:.0f} s",
                      flush=True)
        acc = evaluate_rows(head, H, va, E, norm_w)
        print(f"epoch {epoch}: held-out top-1 per depth (drafting on its own guesses): "
              + ", ".join(f"{100 * a:.1f}%" for a in acc), flush=True)
    path = OUT / f"{args.variant}_r{args.rank}_d{args.depth}.pt"
    torch.save({"variant": args.variant, "rank": args.rank, "depth": args.depth, "state": head.state_dict()}, path)
    print(f"wrote {path}")


@torch.no_grad()
def draft_chain(head, h, fed, E, norm_w, depth):
    """Free-running drafts: each step feeds the previous guess. h [B, dim], fed [B]."""
    if isinstance(head, Recurrent):
        out, tok = [], fed
        for _ in range(depth):
            h = head.step(h, tok, E)
            tok = head_logits(h, E, norm_w).argmax(-1)
            out.append(tok)
        return torch.stack(out, 1)
    outs = head(h, None, E, norm_w)
    return torch.stack([head_logits(o, E, norm_w).argmax(-1) for o in outs[:depth]], 1)


@torch.no_grad()
def evaluate_rows(head, H, rows, E, norm_w, limit=20000):
    idx, feed, tgt = (r[:limit] for r in rows)
    hits = []
    for b in range(0, len(idx), 1024):
        h = torch.from_numpy(np.asarray(H[idx[b:b + 1024]], np.float32)).to(DEVICE)
        d = draft_chain(head, h, torch.from_numpy(feed[b:b + 1024, 0]).long().to(DEVICE), E, norm_w, tgt.shape[1])
        hits.append((d.cpu().numpy() == tgt[b:b + 1024]))
    hits = np.concatenate(hits)
    return [np.mean(np.all(hits[:, :k + 1], axis=1)) for k in range(tgt.shape[1])]


def load_head(path: Path, dim: int):
    ck = torch.load(path)
    head = make_head(ck["variant"], dim, ck["rank"], ck["depth"]).to(DEVICE)
    head.load_state_dict(ck["state"])
    return head.eval()


CHAT_PROMPTS = [
    "Hello", "Hi! Who are you?", "What is the capital of France?", "Tell me a joke.",
    "Write a haiku about the sea.", "How do I boil an egg?", "What is 17 times 23?",
    "Explain what a neural network is in simple terms.", "Give me three tips for better sleep.",
    "Translate 'good morning' into Spanish and German.", "What's the difference between weather and climate?",
    "Write a Python function that reverses a string.", "Summarize the plot of Romeo and Juliet.",
    "Can you recommend a good book?", "Why is the sky blue?", "List the planets of the solar system.",
    "How are you today?", "What should I cook for dinner tonight?", "Write a short email asking for a day off.",
    "What is the meaning of life?",
]


def chat_ids(tok, text: str) -> list[int]:
    return tok.encode(f"<|startoftext|><|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n",
                      add_special_tokens=False).ids


def evaluate(args):
    import spec_eval
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(export.MODEL, "tokenizer.json"))
    model = q4_model()
    E = model.lm_head.weight.detach()
    norm_w = model.model.embedding_norm.weight.detach()

    sets = {
        "chat": [chat_ids(tok, p) for p in CHAT_PROMPTS],
        "clean": [[1] + tok.encode(p, add_special_tokens=False).ids for p in spec_eval.CLEAN_PROMPTS],
    }
    corpus = np.load(ROOT / "out" / "spec" / "selfgen.npy")
    bigram, trigram = spec_eval.build_ngram(corpus, 1.0)  # the table flashed on the device
    heads = {p.stem: load_head(p, E.shape[1]) for p in sorted(OUT.glob("*_r*_d*.pt"))}

    for set_name, prompts in sets.items():
        seqs, plens, hs = [], [], []
        for ids in prompts:
            with torch.no_grad():
                out = model.generate(torch.tensor([ids], device=DEVICE), max_new_tokens=args.tokens,
                                     do_sample=False, eos_token_id=7, pad_token_id=0)
            seq = out[0].tolist()
            seqs.append(seq)
            plens.append(len(ids))
            hs.append(hidden_and_argmax(model, seq)[0])
        print(f"\n== {set_name}: {sum(len(q) - p for q, p in zip(seqs, plens))} greedy tokens, {len(seqs)} prompts")

        cur = [0]  # index of the sequence being simulated

        def ngram_fn(ctx, k):
            d = spec_eval.draft_lookup(ctx, k)
            return d if d else spec_eval.draft_ngram(ctx, k, bigram, trigram)

        def head_fn(head, lookup_first):
            def fn(ctx, k):
                if lookup_first:
                    d = spec_eval.draft_lookup(ctx, k)
                    if d:
                        return d
                h = hs[cur[0]][len(ctx) - 2][None]  # produced fed = ctx[-1]
                fed = torch.tensor([ctx[-1]], device=DEVICE)
                return draft_chain(head, h, fed, E, norm_w, min(k, head.depth))[0].tolist()
            return fn

        drafters = {"ngram (current)": (ngram_fn, 0.0)}
        for name, head in heads.items():
            drafters[name] = (head_fn(head, False), 0.5)
            drafters["lookup + " + name] = (head_fn(head, True), 0.5)
        for name, (fn, extra) in drafters.items():
            for k in (1, 2, 3):
                passes = tokens = drafted = 0
                for s_i, (q, p) in enumerate(zip(seqs, plens)):
                    cur[0] = s_i
                    st = {}
                    spec_eval.simulate([q], [p], fn, k, st)
                    passes += st["passes"]
                    tokens += st["tokens"]
                    drafted += st["drafted"]
                # Device: ~11.0 s per pass + ~0.35 s per draft position verified;
                # heads add ~0.5 s per pass (their weights + a draft LM-head scan each).
                s_tok = ((11.0 + extra) * passes + 0.35 * drafted) / tokens
                print(f"{name:34s} k={k}: {tokens / passes:.2f} tokens/pass, est {s_tok:.2f} s/token", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["collect", "train", "eval"])
    ap.add_argument("--variant", choices=["medusa", "recurrent"], default="recurrent")
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tokens", type=int, default=128, help="eval: greedy tokens per clean prompt")
    ap.add_argument("--corpus", type=Path, nargs="+", default=[ROOT / "out" / "spec" / "selfgen.npy"],
                    help="collect: self-generated corpora (-1 separates sequences)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"collect": collect, "train": train, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
