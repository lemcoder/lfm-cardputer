# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.57", "numpy", "datasets", "huggingface_hub", "tokenizers", "safetensors"]
# ///
"""Tree drafts vs chains, simulated with the trained draft head.

A pass verifies the newest token plus a tree of drafts (up to 6 more
positions at batch 7). A node's children are the head's top-k guesses after
it; the pass keeps the longest root-to-leaf path that matches the model's
greedy text, plus the model's own next token.

    uv run tools/tree_sim.py [--head out/heads/recurrent_r512_d3.pt]

Cost model (device, s): pass 9.6 + 0.2 per verified position (LM head scan +
re-score) + 0.15 per head step (one step serves all its children).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import draft_heads as dh  # noqa: E402
import export  # noqa: E402

# Tree shapes: per node (parent index, rank among the parent's guesses);
# index 0 is the root (the fed token). Siblings come from one head step.
SHAPES = {
    "chain2": [(0, 0), (1, 0)],
    "chain3": [(0, 0), (1, 0), (2, 0)],
    "chain4": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "chain6": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
    "2x1": [(0, 0), (0, 1), (1, 0), (2, 0)],                     # top-2 first, one more under each
    "2x2": [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0), (4, 0)],     # top-2 first, chain of 2 under each
    "3+3": [(0, 0), (0, 1), (0, 2), (1, 0), (4, 0), (5, 0)],     # top-3 first, chain of 3 under the best
    "2+2+": [(0, 0), (0, 1), (1, 0), (1, 1), (3, 0), (5, 0)],    # top-2 at depth 1 and 2 on the best path
    "3x1+": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (4, 0)],
}


@torch.no_grad()
def build_tree(head, h0, fed, E, norm_w, shape, vocab=None):
    """Tokens of every node (root = fed). One head step per node with children."""
    n = len(shape) + 1
    tok = [fed] + [None] * len(shape)
    hid = [h0] + [None] * len(shape)
    steps = 0
    topk_cache = {}
    for i, (parent, rank) in enumerate(shape, start=1):
        if parent not in topk_cache:
            h = head.step(hid[parent][None], torch.tensor([tok[parent]], device=dh.DEVICE), E)
            logits = dh.head_logits(h, E, norm_w)[0]
            topk_cache[parent] = (h[0], torch.topk(logits, 3).indices.tolist())
            steps += 1
        h_after, cands = topk_cache[parent]
        tok[i] = cands[rank]
        hid[i] = h_after  # the state that produced this guess, fed with the guess next
    children = {}
    for i, (parent, _) in enumerate(shape, start=1):
        children.setdefault(parent, []).append(i)
    return tok, children, steps, n


@torch.no_grad()
def dynamic_tree(head, h0, fed, E, norm_w, m=6, depth=4, k=2, c=3, conf="exact", topn=16):
    """EAGLE-2: expand the k best nodes per layer (value = product of the
    head's confidences along the path), then keep the m best of all nodes.
    conf "exact": softmax over the vocab; "top16": over the top topn logits
    only (what the device can compute). Returns (tokens, children, expansions
    per layer, positions)."""
    nodes = [{"tok": fed, "parent": -1, "v": 1.0, "d": 0, "h": h0}]
    frontier, layer_sizes = [0], []
    for _ in range(depth):
        hs = torch.stack([nodes[i]["h"] for i in frontier])
        toks = torch.tensor([nodes[i]["tok"] for i in frontier], device=dh.DEVICE)
        h_new = head.step(hs, toks, E)
        logits = dh.head_logits(h_new, E, norm_w)
        if conf == "exact":
            probs = torch.softmax(logits.float(), -1)
            pv, pi = torch.topk(probs, c, -1)
        else:
            lv, li = torch.topk(logits.float(), topn, -1)
            pr = torch.softmax(lv, -1)
            pv, pi = pr[:, :c], li[:, :c]
        layer_sizes.append(len(frontier))
        new = []
        for row, f in enumerate(frontier):
            for j in range(c):
                nodes.append({"tok": int(pi[row, j]), "parent": f, "v": nodes[f]["v"] * float(pv[row, j]),
                              "d": nodes[f]["d"] + 1, "h": h_new[row]})
                new.append(len(nodes) - 1)
        frontier = sorted(new, key=lambda i: -nodes[i]["v"])[:k]
    keep = sorted(range(1, len(nodes)), key=lambda i: (-nodes[i]["v"], nodes[i]["d"]))[:m]
    keep = sorted(keep, key=lambda i: (nodes[i]["d"], -nodes[i]["v"]))  # parents before children
    index = {0: 0}
    for i in keep:
        index[i] = len(index)
    toks = [fed] + [nodes[i]["tok"] for i in keep]
    children = {}
    for i in keep:
        p = nodes[i]["parent"]
        if p in index:
            children.setdefault(index[p], []).append(index[i])
    return toks, children, layer_sizes, len(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, default=dh.OUT / "recurrent_r512_d3.pt")
    ap.add_argument("--tokens", type=int, default=128)
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(export.MODEL, "tokenizer.json"))
    model = dh.q4_model()
    E = model.lm_head.weight.detach()
    norm_w = model.model.embedding_norm.weight.detach()
    head = dh.load_head(args.head, E.shape[1], "q4")
    sets = {
        "chat": [dh.chat_ids(tok, p) for p in dh.CHAT_PROMPTS],
        "clean": [[1] + tok.encode(p, add_special_tokens=False).ids for p in dh.CLEAN_PROMPTS],
    }
    for set_name, prompts in sets.items():
        seqs = []
        for ids in prompts:
            out = model.generate(torch.tensor([ids], device=dh.DEVICE), max_new_tokens=args.tokens, do_sample=False,
                                 eos_token_id=7, pad_token_id=0)
            s = out[0].tolist()
            seqs.append((s, len(ids), dh.hidden_and_argmax(model, s)[0]))
        print(f"\n== {set_name}: {sum(len(s) - p for s, p, _ in seqs)} tokens")
        for name, shape in SHAPES.items():
            passes = tokens = cost = 0.0
            for s, p, hs in seqs:
                i = p  # next position to produce; s[i-1] is fed, hs[i-2] produced it
                while i < len(s):
                    fed = s[i - 1]
                    drafts = dh.draft_lookup(s[:i], 6)
                    if drafts:  # prompt lookup first, as on the device: a chain
                        acc = 0
                        while acc < len(drafts) and i + acc < len(s) and drafts[acc] == s[i + acc]:
                            acc += 1
                        n_pos, steps = 1 + len(drafts), 0
                    else:
                        toks, children, steps, n_pos = build_tree(head, hs[i - 2], fed, E, norm_w, shape)
                        node, acc = 0, 0
                        while True:
                            nxt = [c for c in children.get(node, []) if i + acc < len(s) and toks[c] == s[i + acc]]
                            if not nxt:
                                break
                            node = nxt[0]
                            acc += 1
                    step = min(acc + 1, len(s) - i)
                    tokens += step
                    passes += 1
                    cost += 9.6 + 0.2 * n_pos + 0.15 * steps
                    i += step
            print(f"{name:7s}: {tokens / passes:.2f} tokens/pass, est {cost / tokens:.2f} s/token")
        # EAGLE-2 style dynamic trees (budget m drafts, depth, expand k per layer, c guesses per node).
        for conf in ("exact", "top16"):
            for depth, k, c in ((3, 2, 3), (4, 2, 3), (5, 2, 2), (4, 3, 3), (6, 1, 3), (5, 3, 2)):
                passes = tokens = cost = 0.0
                for s_, p, hs in seqs:
                    i = p
                    while i < len(s_):
                        fed = s_[i - 1]
                        drafts = dh.draft_lookup(s_[:i], 6)
                        if drafts:
                            acc = 0
                            while acc < len(drafts) and i + acc < len(s_) and drafts[acc] == s_[i + acc]:
                                acc += 1
                            n_pos, step_cost = 1 + len(drafts), 0.0
                        else:
                            toks, children, layers, n_pos = dynamic_tree(head, hs[i - 2], fed, E, norm_w, 6, depth,
                                                                         k, c, conf)
                            node, acc = 0, 0
                            while True:
                                nxt = [ch for ch in children.get(node, []) if i + acc < len(s_) and toks[ch] == s_[i + acc]]
                                if not nxt:
                                    break
                                node = nxt[0]
                                acc += 1
                            # one flash scan per layer, shared by its nodes
                            step_cost = sum(0.15 + 0.03 * (n - 1) for n in layers)
                        step = min(acc + 1, len(s_) - i)
                        tokens += step
                        passes += 1
                        cost += 9.6 + 0.2 * n_pos + step_cost
                        i += step
                name = f"dyn {conf} d{depth} k{k} c{c}"
                print(f"{name:22s}: {tokens / passes:.2f} tokens/pass, est {cost / tokens:.2f} s/token", flush=True)


if __name__ == "__main__":
    main()
