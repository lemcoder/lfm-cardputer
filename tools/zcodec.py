# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "safetensors", "huggingface_hub", "tokenizers"]
# ///
"""Entropy coding of Q4_0 matrices, the encoder side of lfm/zcodec.hpp.

    uv run tools/zcodec.py test          # out/ztest.bin: tables + real chunks, for host / device tests
    uv run tools/zcodec.py pack          # out/lfm-q4-head-lr256.lfm -> out/lfm.lfz (its layer matrices)

The .lfz file (lfm/zfile.hpp) holds only the compressed Q4_0 layer
matrices; everything else is still read from the .lfm. Each matrix is cut
into chunks of CHUNK decoded bytes from its start, the engine's read block,
so every block the engine streams is one chunk.

Byte roles by position in the 18-byte Q4_0 block ("phase", counted from the
matrix start): 0 = scale low byte (raw), 1 = scale high byte (table S),
2..17 = two 4-bit values (table Q). Huffman codes, length-limited to 10
bits, bits LSB first; the decoder indexes a 2048-entry table with the next
11 bits.
"""

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAX_BITS = 10


def code_lengths(freq: np.ndarray, max_bits: int = MAX_BITS) -> np.ndarray:
    """Optimal length-limited prefix code lengths (package-merge). Symbols
    with zero frequency get count 1 so every byte stays encodable."""
    f = np.maximum(freq.astype(np.float64), 1.0)
    n = len(f)
    leaves = sorted((f[i], (i,)) for i in range(n))
    packages = list(leaves)
    for _ in range(max_bits - 1):
        merged = [(packages[i][0] + packages[i + 1][0], packages[i][1] + packages[i + 1][1])
                  for i in range(0, len(packages) - 1, 2)]
        packages = sorted(leaves + merged, key=lambda t: t[0])
    lengths = np.zeros(n, np.int64)
    for _, syms in packages[: 2 * n - 2]:
        for s in syms:
            lengths[s] += 1
    return lengths


def canonical_codes(lengths: np.ndarray) -> np.ndarray:
    """Canonical codes, bit-reversed so the first bit sent is bit 0."""
    order = sorted(range(len(lengths)), key=lambda s: (lengths[s], s))
    codes = np.zeros(len(lengths), np.int64)
    code, prev = 0, 0
    for s in order:
        code <<= lengths[s] - prev
        prev = lengths[s]
        codes[s] = int(format(code, f"0{lengths[s]}b")[::-1], 2)
        code += 1
    return codes


def decode_table(lengths: np.ndarray, codes: np.ndarray) -> np.ndarray:
    table = np.zeros(1 << MAX_BITS, np.uint16)
    for s in range(len(lengths)):
        step = 1 << lengths[s]
        table[codes[s]::step] = s | (lengths[s] << 8)
    return table


class Codec:
    def __init__(self, freq_q: np.ndarray, freq_s: np.ndarray):
        self.lq, self.ls = code_lengths(freq_q), code_lengths(freq_s)
        self.cq, self.cs = canonical_codes(self.lq), canonical_codes(self.ls)

    def tables(self) -> bytes:
        return decode_table(self.lq, self.cq).tobytes() + decode_table(self.ls, self.cs).tobytes()

    def encode(self, data: bytes, phase: int) -> bytes:
        """One chunk: data starts at matrix byte offset ≡ phase (mod 18)."""
        a = np.frombuffer(data, np.uint8).astype(np.int64)
        ph = (np.arange(len(a)) + phase) % 18
        lens = np.where(ph == 0, 8, np.where(ph == 1, self.ls[a], self.lq[a]))
        codes = np.where(ph == 0, a, np.where(ph == 1, self.cs[a], self.cq[a]))
        # Pack LSB first: bit position of each code's start.
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        total = int(lens.sum())
        out = np.zeros((total + 7) // 8 + 4, np.uint8)
        for bit in range(int(lens.max())):
            sel = lens > bit
            pos = starts[sel] + bit
            vals = (codes[sel] >> bit) & 1
            np.add.at(out, pos // 8, (vals << (pos % 8)).astype(np.uint8))
        return out[: (total + 7) // 8].tobytes()


def inplace_slack(codec, data: bytes, phase: int, comp_len: int, chunk: int) -> int:
    """Extra buffer bytes the device needs to decode this chunk in place.

    The compressed bytes sit at the end of a (chunk + slack) buffer and the
    output grows from its start; the decoder loads input 2 bytes at a time,
    so output byte t may overwrite input only below the next unread word.
    """
    a = np.frombuffer(data, np.uint8).astype(np.int64)
    ph = (np.arange(len(a)) + phase) % 18
    lens = np.where(ph == 0, 8, np.where(ph == 1, codec.ls[a], codec.lq[a]))
    bits_before = np.concatenate([[0], np.cumsum(lens)[:-1]])  # bits consumed before output byte t
    loaded = (bits_before // 16) * 2  # bytes certainly read into the bit buffer by then
    # write t at position t must be < input start + loaded  =>  t < chunk + slack - comp_len + loaded
    need = (np.arange(len(a)) + 1) - (chunk - comp_len) - loaded
    return max(0, int(need.max()))


Q4_ROW = lambda cols: cols // 32 * 18  # noqa: E731
LFM_HEAD = struct.Struct("<8s12I2f16I4Q")
FIELDS = ["operator_norm", "ffn_norm", "w1", "w2", "w3", "conv_in", "conv_kernel", "conv_out", "q", "k", "v", "o",
          "q_norm", "k_norm"]
MAX_LAYERS = 16


def lfm_matrices(path: Path):
    """(file offset, rows, cols, stream starts in bytes) of every Q4_0 layer matrix."""
    with open(path, "rb") as f:
        head = f.read(LFM_HEAD.size)
        fields = LFM_HEAD.unpack(head)
        dim, ff, n_layers, heads, kv, hd = fields[3], fields[4], fields[5], fields[6], fields[7], fields[8]
        offs = struct.unpack(f"<{MAX_LAYERS * 14}Q", f.read(MAX_LAYERS * 14 * 8))
        dtypes = f.read(MAX_LAYERS * 16)
    shapes = {"w1": (ff, dim), "w2": (dim, ff), "w3": (ff, dim), "conv_in": (3 * dim, dim), "conv_out": (dim, dim),
              "q": (heads * hd, dim), "k": (kv * hd, dim), "v": (kv * hd, dim), "o": (dim, heads * hd)}
    out = []
    for layer in range(n_layers):
        for fi, name in enumerate(FIELDS):
            off = offs[layer * 14 + fi]
            if name not in shapes or off == 0 or dtypes[layer * 16 + fi] != 3:  # Q4_0
                continue
            rows, cols = shapes[name]
            rb = Q4_ROW(cols)
            starts = [0, 2 * dim * rb] if name == "conv_in" else [0]  # the engine's streams (conv_op)
            out.append((off, rows, cols, starts))
    return out


def pack(lfm: Path, out: Path, chunk: int = 32768, slack: int = 64):
    mats = lfm_matrices(lfm)
    data = lfm.read_bytes()
    raws = [data[off: off + rows * Q4_ROW(cols)] for off, rows, cols, _ in mats]
    codec = Codec(*frequencies(raws))
    tensors, chunks, payload = [], [], bytearray()
    raw_total = comp_total = n_raw = 0
    slacks = []
    for (off, rows, cols, starts), raw in zip(mats, raws):
        assert all(st % chunk == 0 for st in starts), "a stream doesn't start on a chunk boundary"
        first = len(chunks)
        for c0 in range(0, len(raw), chunk):
            piece = raw[c0: c0 + chunk]
            comp = codec.encode(piece, c0 % 18)
            comp += bytes((-len(comp)) % 4)
            need = inplace_slack(codec, piece, c0 % 18, len(comp), len(piece))
            slacks.append(need)
            if len(comp) >= len(piece) or need > slack:  # store raw: decode = copy
                comp, flag = piece + bytes((-len(piece)) % 4), 1 << 31
                n_raw += 1
            else:
                flag = 0
            chunks.append((len(payload), len(comp) | flag))
            payload += comp
            raw_total += len(piece)
            comp_total += len(comp)
        tensors.append((off, len(raw), first, len(chunks) - first))
    tensors.sort()  # by .lfm offset: the device looks them up by binary search
    head = struct.pack("<8s6I", b"LFMZ\0\0\0\0", 1, chunk, slack, len(tensors), len(chunks), 0)
    tables = codec.tables()
    tens = b"".join(struct.pack("<QIHH", *t) for t in tensors)
    data_off = len(head) + len(tables) + len(tens) + 8 * len(chunks)
    data_off += (-data_off) % 512  # payload starts on a sector
    chk = b"".join(struct.pack("<II", data_off + o, l) for o, l in chunks)
    with open(out, "wb") as f:
        f.write(head + tables + tens + chk)
        f.write(bytes(data_off - f.tell()))
        f.write(payload)
    sl = np.array(slacks)
    print(f"wrote {out}: {len(tensors)} matrices, {len(chunks)} chunks ({n_raw} raw), "
          f"{raw_total / 1e6:.1f} MB -> {comp_total / 1e6:.1f} MB ({100 * (1 - comp_total / raw_total):.1f}% smaller); "
          f"in-place slack needed: max {sl.max()} B, 99% {int(np.percentile(sl, 99))} B")


def layer_matrices():
    """The device's Q4_0 layer matrices, as bytes (same quantizer as export.py)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import export
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    src = Path(snapshot_download(export.MODEL, allow_patterns=["*.safetensors"]))
    w = load_file(str(src / "model.safetensors"))
    for name, t in w.items():
        if t.ndim == 2 and "layers" in name and t.shape[1] % 32 == 0:
            yield name, export.quantize_q4_0(t.float().numpy())


def frequencies(mats):
    fq = np.zeros(256, np.int64)
    fs = np.zeros(256, np.int64)
    for raw in mats:
        b = np.frombuffer(raw, dtype=[("lo", "u1"), ("hi", "u1"), ("q", "u1", 16)])
        fq += np.bincount(b["q"].ravel(), minlength=256)
        fs += np.bincount(b["hi"], minlength=256)
    return fq, fs


def test_file(chunk: int = 32768, n_chunks: int = 64):
    mats = [raw for _, raw in layer_matrices()]
    codec = Codec(*frequencies(mats))
    chunks, raw_total, comp_total = [], 0, 0
    # Chunks from several matrices, at block-unaligned offsets as the engine reads them.
    for i in range(n_chunks):
        raw = mats[(i * 7) % len(mats)]
        off = (i * 3 * chunk) % max(1, len(raw) - chunk)
        off -= off % 4096  # tensor-relative block starts are multiples of the read block; keep it simple
        data = raw[off: off + chunk]
        comp = codec.encode(data, off % 18)
        chunks.append((data, off % 18, comp))
        raw_total += len(data)
        comp_total += len(comp)
    out = ROOT / "out" / "ztest.bin"
    with open(out, "wb") as f:
        f.write(b"LFMZTST\0" + struct.pack("<II", len(chunks), chunk))
        f.write(codec.tables())
        for data, phase, comp in chunks:
            f.write(struct.pack("<III", len(data), phase, len(comp)))
            f.write(data)
            f.write(comp + bytes((-len(comp)) % 4))
    print(f"wrote {out}: {len(chunks)} chunks, {raw_total / 1e6:.2f} MB -> {comp_total / 1e6:.2f} MB "
          f"({100 * (1 - comp_total / raw_total):.1f}% smaller); max code lengths Q {codec.lq.max()} S {codec.ls.max()}")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        test_file()
    elif sys.argv[1:2] == ["pack"]:
        pack(ROOT / "out" / "lfm-q4-head-lr256.lfm", ROOT / "out" / "lfm.lfz")
    else:
        sys.exit(__doc__)
