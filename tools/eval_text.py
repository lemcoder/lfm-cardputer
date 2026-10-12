# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers", "huggingface_hub"]
# ///
"""Tokenize a mixed evaluation text into out/eval.txt for host/eval.cpp.

    uv run tools/eval_text.py            # built-in text (~600 tokens)
    uv run tools/eval_text.py file.txt   # your own
"""

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent

TEXT = """The Industrial Revolution began in Great Britain in the second half of the eighteenth century and \
spread to continental Europe and North America over the following decades. Mechanised spinning and weaving \
transformed the textile industry, while improvements to the steam engine by James Watt made it practical to \
power factories, mines and, later, railways. Cities grew rapidly as workers moved from the countryside, and \
living conditions in the new industrial towns were often crowded and unhealthy.

Photosynthesis is the process by which green plants, algae and some bacteria convert light energy into \
chemical energy. In the chloroplasts, chlorophyll absorbs sunlight, which drives the splitting of water \
molecules and releases oxygen as a by-product. The energy captured is stored in molecules of glucose, built \
from carbon dioxide taken from the air.

To make a simple tomato sauce, warm two tablespoons of olive oil in a pan over medium heat. Add a finely \
chopped onion and cook it slowly until soft, about ten minutes. Stir in two cloves of garlic, then a tin of \
chopped tomatoes and a pinch of salt. Let it simmer for twenty minutes, stirring now and then, and finish \
with a handful of fresh basil.

def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print([fibonacci(i) for i in range(10)])

Maria looked out of the train window as the fields gave way to the outskirts of the city. She had not been \
back for eleven years, and she wondered whether the bakery on the corner of her old street was still open. \
When the train finally stopped, she picked up her suitcase, took a deep breath and stepped onto the platform.

The Pacific Ocean is the largest and deepest of the world's oceans, covering about a third of the surface of \
the Earth. Its name was given by the explorer Ferdinand Magellan, who found its waters calm after a stormy \
passage through the strait at the southern tip of South America.
"""


def main():
    text = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else TEXT
    tok = Tokenizer.from_file(hf_hub_download("LiquidAI/LFM2.5-230M", "tokenizer.json"))
    ids = [1] + tok.encode(text, add_special_tokens=False).ids
    out = ROOT / "out" / "eval.txt"
    out.write_text(" ".join(map(str, ids)) + "\n")
    print(f"wrote {out}: {len(ids)} tokens")


if __name__ == "__main__":
    main()
