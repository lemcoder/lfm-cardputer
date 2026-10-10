# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.57", "safetensors", "huggingface_hub"]
# ///
"""Dump reference ("golden") activations from the HF LFM2.5-230M model.

The C++ engine in lfm/ is checked against these, tensor by tensor.

    uv run tools/golden.py                       # default prompt -> golden/default/
    uv run tools/golden.py --prompt "Hi" --name hi

Output (golden/<name>/):
    manifest.json       prompt, token ids, config, weight file path + tensor list,
                        and every dumped tensor: name -> {file, shape}
    *.f32               raw little-endian float32, row-major, no header
    tokens.i32          prompt token ids (int32)
    generated.i32       greedy continuation token ids (int32)

Every activation has a leading position axis T: row t is what the model
computes for prompt token t. The model is causal, so row t equals what a
one-token-at-a-time decoder produces at step t -- which is how lfm/ runs.

Tensor names (L = layer index, T = prompt length, H = 16 heads, KV = 8):
    embed                         [T, 1024]   embedding lookup
    L{L}.input                    [T, 1024]   layer input (residual stream)
    L{L}.operator_norm            [T, 1024]
  conv layers:
    L{L}.conv.in_proj             [T, 3072]   = [B | C | x] along the last axis
    L{L}.conv.Bx                  [T, 1024]   B * x
    L{L}.conv.conv                [T, 1024]   causal depthwise conv(Bx), kernel 3
    L{L}.conv.Cy                  [T, 1024]   C * conv  (input to out_proj)
    L{L}.conv.out                 [T, 1024]   out_proj
  attention layers:
    L{L}.attn.q / .k / .v         [T, H|KV, 64]  projections
    L{L}.attn.q_norm / .k_norm    [T, H|KV, 64]  per-head RMSNorm
    L{L}.attn.q_rope / .k_rope    [T, H|KV, 64]  after RoPE (rotate_half form)
    L{L}.attn.ctx                 [T, 1024]   softmax(qk^T/8) v, heads concatenated
    L{L}.attn.out                 [T, 1024]   out_proj
  every layer:
    L{L}.resid_mid                [T, 1024]   input + operator output
    L{L}.ffn_norm                 [T, 1024]
    L{L}.ffn.w1 / .w3             [T, 2560]
    L{L}.ffn.act                  [T, 2560]   silu(w1) * w3
    L{L}.ffn.out                  [T, 1024]   w2(act)
    L{L}.output                   [T, 1024]   resid_mid + ffn.out
    final_norm                    [T, 1024]   embedding_norm (final RMSNorm)
    logits                        [T, 65536]  final_norm @ embed^T (tied head)
    rope.cos / rope.sin           [T, 64]
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM2.5-230M"
DEFAULT_PROMPT = "The capital of France is"
ROOT = Path(__file__).resolve().parent.parent


class Dumper:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.tensors: dict[str, dict] = {}

    def save(self, name: str, t: torch.Tensor) -> None:
        t = t.detach().to(torch.float32).contiguous()
        if t.dim() >= 2 and t.shape[0] == 1:
            t = t[0]  # drop batch axis
        file = f"{name}.f32"
        (self.out_dir / file).write_bytes(t.numpy().astype("<f4").tobytes())
        self.tensors[name] = {"file": file, "shape": list(t.shape)}


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--name", default="default")
    ap.add_argument("--generate", type=int, default=16, help="greedy tokens to generate")
    args = ap.parse_args()

    out_dir = ROOT / "golden" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.f32"):
        f.unlink()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # fp32 everywhere so the reference isn't polluted by bf16 rounding.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    cfg = model.config
    inner = model.model
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    ids = tok(args.prompt, return_tensors="pt").input_ids
    if ids[0, 0].item() != cfg.bos_token_id:
        ids = torch.cat([torch.tensor([[cfg.bos_token_id]]), ids], dim=1)
    T = ids.shape[1]

    d = Dumper(out_dir)
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def keep(name):
        def hook(_mod, _inp, out):
            captured[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    def keep_input(name):
        def pre_hook(_mod, inp):
            captured[name] = inp[0].detach()
        return pre_hook

    hooks.append(inner.embed_tokens.register_forward_hook(keep("embed")))
    hooks.append(inner.rotary_emb.register_forward_hook(
        lambda _m, _i, out: captured.update({"rope.cos": out[0].detach(), "rope.sin": out[1].detach()})))
    for L, layer in enumerate(inner.layers):
        hooks.append(layer.register_forward_pre_hook(keep_input(f"L{L}.input")))
        hooks.append(layer.register_forward_hook(keep(f"L{L}.output")))
        hooks.append(layer.operator_norm.register_forward_hook(keep(f"L{L}.operator_norm")))
        hooks.append(layer.ffn_norm.register_forward_hook(keep(f"L{L}.ffn_norm")))
        ff = layer.feed_forward
        hooks.append(ff.w1.register_forward_hook(keep(f"L{L}.ffn.w1")))
        hooks.append(ff.w3.register_forward_hook(keep(f"L{L}.ffn.w3")))
        hooks.append(ff.w2.register_forward_pre_hook(keep_input(f"L{L}.ffn.act")))
        hooks.append(ff.register_forward_hook(keep(f"L{L}.ffn.out")))
        if layer.is_attention_layer:
            a = layer.self_attn
            hooks.append(a.q_proj.register_forward_hook(keep(f"L{L}.attn.q")))
            hooks.append(a.k_proj.register_forward_hook(keep(f"L{L}.attn.k")))
            hooks.append(a.v_proj.register_forward_hook(keep(f"L{L}.attn.v")))
            hooks.append(a.q_layernorm.register_forward_hook(keep(f"L{L}.attn.q_norm")))
            hooks.append(a.k_layernorm.register_forward_hook(keep(f"L{L}.attn.k_norm")))
            hooks.append(a.out_proj.register_forward_pre_hook(keep_input(f"L{L}.attn.ctx")))
            hooks.append(a.register_forward_hook(keep(f"L{L}.attn.out")))
        else:
            c = layer.conv
            hooks.append(c.in_proj.register_forward_hook(keep(f"L{L}.conv.in_proj")))
            hooks.append(c.out_proj.register_forward_pre_hook(keep_input(f"L{L}.conv.Cy")))
            hooks.append(c.register_forward_hook(keep(f"L{L}.conv.out")))
    hooks.append(inner.embedding_norm.register_forward_hook(keep("final_norm")))

    with torch.no_grad():
        logits = model(ids, use_cache=False).logits
    for h in hooks:
        h.remove()
    captured["logits"] = logits

    # --- Derived tensors the HF code computes inline (no module to hook). ---
    # Recomputing them here also double-checks our understanding of the math:
    # each must reproduce the next hooked tensor.
    checks: list[tuple[str, float]] = []
    cos, sin = captured["rope.cos"][0], captured["rope.sin"][0]  # [T, 64]

    def rotate_half(x):
        x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    with torch.no_grad():
        for L, layer in enumerate(inner.layers):
            p = f"L{L}"
            if layer.is_attention_layer:
                for n, heads in (("q", cfg.num_attention_heads), ("k", cfg.num_key_value_heads)):
                    captured[f"{p}.attn.{n}"] = captured[f"{p}.attn.{n}"].view(1, T, heads, head_dim)
                    x = captured[f"{p}.attn.{n}_norm"][0]  # [T, heads, 64]
                    captured[f"{p}.attn.{n}_rope"] = x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]
                captured[f"{p}.attn.v"] = captured[f"{p}.attn.v"].view(1, T, cfg.num_key_value_heads, head_dim)

                # Plain causal GQA attention, then compare with HF's ctx.
                q = captured[f"{p}.attn.q_rope"].transpose(0, 1)                  # [H, T, 64]
                k = captured[f"{p}.attn.k_rope"].transpose(0, 1)                  # [KV, T, 64]
                v = captured[f"{p}.attn.v"][0].transpose(0, 1)                    # [KV, T, 64]
                group = cfg.num_attention_heads // cfg.num_key_value_heads
                k, v = k.repeat_interleave(group, 0), v.repeat_interleave(group, 0)
                scores = q @ k.transpose(-1, -2) / head_dim ** 0.5
                scores = scores.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
                ctx = (scores.softmax(-1) @ v).transpose(0, 1).reshape(T, -1)
                checks.append((f"{p}.attn.ctx", max_diff(ctx, captured[f"{p}.attn.ctx"][0])))
            else:
                Bv, Cv, xv = captured[f"{p}.conv.in_proj"][0].chunk(3, dim=-1)    # [T, 1024] each
                Bx = Bv * xv
                w = layer.conv.conv.weight[:, 0, :]                               # [1024, 3]
                # out[t] = w[:,0]*Bx[t-2] + w[:,1]*Bx[t-1] + w[:,2]*Bx[t]  (zeros before t=0)
                padded = F.pad(Bx.T, (2, 0))                                      # [1024, T+2]
                conv = sum(w[:, j:j + 1] * padded[:, j:j + T] for j in range(3)).T
                captured[f"{p}.conv.Bx"] = Bx
                captured[f"{p}.conv.conv"] = conv
                checks.append((f"{p}.conv.Cy", max_diff(Cv * conv, captured[f"{p}.conv.Cy"][0])))
            op_out = captured[f"{p}.attn.out" if layer.is_attention_layer else f"{p}.conv.out"]
            captured[f"{p}.resid_mid"] = captured[f"{p}.input"] + op_out
            checks.append((f"{p}.output", max_diff(captured[f"{p}.resid_mid"] + captured[f"{p}.ffn.out"],
                                                   captured[f"{p}.output"])))
        checks.append(("logits", max_diff(captured["final_norm"] @ inner.embed_tokens.weight.T, logits)))

    for name in sorted(captured, key=lambda n: (n.split(".")[0], n)):
        d.save(name, captured[name])

    # --- Greedy continuation, for an end-to-end check. ---
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=args.generate, do_sample=False)[0, T:]
    (out_dir / "tokens.i32").write_bytes(ids[0].numpy().astype("<i4").tobytes())
    (out_dir / "generated.i32").write_bytes(gen.numpy().astype("<i4").tobytes())

    weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
    manifest = {
        "model": MODEL_ID,
        "weights": weights_path,
        "prompt": args.prompt,
        "tokens": ids[0].tolist(),
        "token_pieces": [tok.decode([i]) for i in ids[0].tolist()],
        "generated": gen.tolist(),
        "generated_text": tok.decode(gen),
        "top5_last": [[tok.decode([i]), round(v, 4)] for v, i in
                      zip(*[x.tolist() for x in logits[0, -1].topk(5)])],
        "config": {k: getattr(cfg, k) for k in (
            "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "vocab_size", "conv_L_cache", "norm_eps", "layer_types")},
        "rope_theta": cfg.rope_parameters["rope_theta"] if hasattr(cfg, "rope_parameters") else cfg.rope_theta,
        "self_checks_max_abs_diff": dict(checks),
        "tensors": d.tensors,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    worst = max(checks, key=lambda c: c[1])
    print(f"prompt:    {args.prompt!r} -> {T} tokens {ids[0].tolist()}")
    print(f"pieces:    {manifest['token_pieces']}")
    print(f"top5 next: {manifest['top5_last']}")
    print(f"greedy:    {manifest['generated_text']!r}")
    print(f"weights:   {weights_path}")
    print(f"dumped {len(d.tensors)} tensors to {out_dir}")
    print(f"self-check worst max|diff|: {worst[0]} = {worst[1]:.2e}")
    if worst[1] > 1e-3:
        raise SystemExit("self-check failed: reconstructed math does not match HF")


if __name__ == "__main__":
    main()
