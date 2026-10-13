# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers", "huggingface_hub", "datasets"]
# ///
"""Compare the on-device tokenizer (host build: host/build/lfm_tok) with the
HF tokenizer on WikiText, Alpaca and hand-written edge cases.

    uv run tools/export_tokenizer.py && cmake --build host/build --target lfm_tok
    uv run tools/check_tokenizer.py [--n 3000]
"""

import argparse
import subprocess
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent

EDGE = [
    "Hello", "Hello, world!", "  two  spaces", "trailing   ", "\n\nnew\n\n\nlines\n", "tab\tseparated\tvalues",
    "I'm, you're, they've, we'll, he'd, it's, CAN'T", "3.14159 and 1234567 and 2024-10-13", "x=y+z*(a-b)/c;",
    "def f(x):\n    return x * 2\n", "e-mail: me@example.com", "$5 $$ ... !!!", "\r\n windows \r\n", "a\n \n b",
    " leading", "'quoted' \"double\"", "naïve café résumé", "日本語のテキスト", "emoji 🙂 here", "", " ", "\n",
    "ALL CAPS SHOUTING", "camelCaseWord snake_case_word", "100% sure, 2nd place, 3rd", "    \n    indented",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="samples per dataset")
    args = ap.parse_args()
    tok = Tokenizer.from_file(hf_hub_download("LiquidAI/LFM2.5-230M", "tokenizer.json"))

    samples = list(EDGE)
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    samples += [t for t in wiki["text"] if t.strip()][: args.n]
    alpaca = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=1)
    for ex in alpaca.select(range(args.n)):
        samples.append(f"Q: {ex['instruction']}\nA: {ex['output']}")

    proc = subprocess.run([str(ROOT / "host/build/lfm_tok"), str(ROOT / "out/tok.bin")],
                          input="".join(s.encode().hex() + "\n" for s in samples), capture_output=True, text=True,
                          check=True)
    got = proc.stdout.split("\n")
    bad = ascii_bad = 0
    ascii_total = 0
    shown = 0
    for s, line in zip(samples, got):
        want = tok.encode(s, add_special_tokens=False).ids
        mine = [int(x) for x in line.split()] if line != "ERROR" else None
        is_ascii = s.isascii()
        ascii_total += is_ascii
        if mine != want:
            bad += 1
            ascii_bad += is_ascii
            if shown < 8 and is_ascii:
                shown += 1
                if mine is None:
                    print(f"ERROR on {s[:60]!r}")
                    continue
                d = next((i for i, (a, b) in enumerate(zip(want, mine)) if a != b), min(len(want), len(mine)))
                ctx = tok.decode(want[max(0, d - 3):d + 3])
                print(f"MISMATCH at token {d} of {len(want)}, near {ctx!r}\n"
                      f"  hf  {[tok.id_to_token(i) for i in want[d:d + 5]]}\n"
                      f"  dev {[tok.id_to_token(i) for i in mine[d:d + 5]]}")
    print(f"{len(samples) - bad}/{len(samples)} identical; ASCII-only samples: {ascii_total - ascii_bad}/{ascii_total}")


if __name__ == "__main__":
    main()
