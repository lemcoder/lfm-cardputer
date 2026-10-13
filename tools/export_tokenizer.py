# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub"]
# ///
"""Write the BPE merge table for the on-device tokenizer (lfm/tokenizer.hpp).

    uv run tools/export_tokenizer.py          # -> out/tok.bin
    uv run tools/upload.py out/tok.bin:tok.bin

Layout (little-endian):
    char     magic[8]        "LFMTOK\\0\\0"
    uint32   version         1
    uint32   n_merges
    uint16   byte_id[256]    token id of each single byte
    Merge    merges[n]       sorted by key, 8 bytes each:
        uint32 key           (left id << 16) | right id
        uint16 result        id of the merged token
        uint16 rank          merge priority, lower first
    uint32   n_added         non-special added tokens ("python", "<think>", ...):
    Added    added[n]        matched literally in the text before the split
        uint16 id, uint16 len, char text[len]
"""

import json
import struct
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
MODEL = "LiquidAI/LFM2.5-230M"


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def main():
    tok = json.loads(Path(hf_hub_download(MODEL, "tokenizer.json")).read_text())
    vocab: dict[str, int] = tok["model"]["vocab"]
    b2u = bytes_to_unicode()
    byte_id = [vocab[b2u[b]] for b in range(256)]

    merges = []
    for rank, m in enumerate(tok["model"]["merges"]):
        left, right = m if isinstance(m, list) else m.split(" ", 1)
        a, b, r = vocab[left], vocab[right], vocab[left + right]
        assert max(a, b, r) < 1 << 16 and rank < 1 << 16
        merges.append(((a << 16) | b, r, rank))
    merges.sort()
    keys = [m[0] for m in merges]
    assert len(set(keys)) == len(keys), "duplicate merge pair"

    out = ROOT / "out" / "tok.bin"
    with open(out, "wb") as f:
        f.write(b"LFMTOK\0\0" + struct.pack("<II", 1, len(merges)))
        f.write(struct.pack("<256H", *byte_id))
        for key, r, rank in merges:
            f.write(struct.pack("<IHH", key, r, rank))
        added = [a for a in tok["added_tokens"] if not a["special"]]
        f.write(struct.pack("<I", len(added)))
        for a in added:
            text = a["content"].encode()
            f.write(struct.pack("<HH", a["id"], len(text)) + text)
    print(f"wrote {out}: {len(merges)} merges, {len(added)} added tokens, {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
