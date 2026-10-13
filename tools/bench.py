# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial", "tokenizers", "huggingface_hub"]
# ///
"""One device benchmark run: upload prompt + knobs, wait for the RESULT line.

    uv run tools/bench.py --label baseline --prompt "The capital of France is"
    uv run tools/bench.py --label spec-off --cfg spec=0,gen=24,ignore_eos=1 --prompt "..."

Knobs (firmware RunConfig, /sdcard/cfg.txt): spec, max_draft, max_batch,
io_kb, gen, ignore_eos, progress, runs. Results are appended to
out/bench_log.jsonl. The model file on the card is left alone.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import upload  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "bench"


def parse_result(line: str) -> dict:
    fields = {}
    for part in line.split("RESULT", 1)[1].split():
        k, v = part.split("=")
        fields[k] = float(v) if "." in v else int(v)
    return fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--cfg", default="", help="comma-separated key=value knobs")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--results", type=int, default=0, help="RESULT lines to wait for (default: the runs knob, or 1)")
    ap.add_argument("--port")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(hf_hub_download("LiquidAI/LFM2.5-230M", "tokenizer.json"))
    ids = tok.encode(args.prompt).ids
    (OUT / "prompt.txt").write_text(" ".join(map(str, ids)) + "\n")
    knobs = dict(kv.split("=") for kv in args.cfg.split(",") if kv)
    (OUT / "cfg.txt").write_text("".join(f"{k}={v}\n" for k, v in knobs.items()))

    port = args.port or upload.find_port()
    dev = upload.connect(port)
    upload.send_file(dev, OUT / "prompt.txt", "prompt.txt")
    upload.send_file(dev, OUT / "cfg.txt", "cfg.txt")
    dev.send_line("LFMUP END")
    dev.wait_for(("UP:BYE",), 5)

    want = args.results or int(knobs.get("runs", 1))
    t0 = time.time()
    lines = []
    while time.time() - t0 < args.timeout:
        for line in dev.lines():
            if " lfm:" in line or "E (" in line:
                lines.append(line)
                print(line, flush=True)
            if "RESULT" in line:
                res = parse_result(line)
                res.update(label=args.label, cfg=knobs, prompt=args.prompt, time=time.strftime("%H:%M:%S"))
                with open(ROOT / "out" / "bench_log.jsonl", "a") as f:
                    f.write(json.dumps(res) + "\n")
                print(json.dumps(res))
                want -= 1
                if want == 0:
                    return
            if "Guru" in line or "panic" in line or "fatal" in line:
                sys.exit(f"device error: {line}")
    sys.exit("timeout waiting for RESULT")


if __name__ == "__main__":
    main()
