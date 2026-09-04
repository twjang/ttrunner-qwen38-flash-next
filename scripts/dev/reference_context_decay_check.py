"""The CPU control for `context_decay_check.py`.

    uv run python scripts/dev/reference_context_decay_check.py [positions] [block]

The device's next-token accuracy falls from ~80 % over the first 128 positions to
0 % past 224. Before any of that is called a defect, the float32 reference has to
be run on the *same text* and shown not to do it -- otherwise the finding is
about the passage, not the implementation.

One forward, scored per block of positions, so the two are directly comparable.
"""
import sys

import torch

from _device_model import GGUF_DIR, tokenizer

from twtest.gguf.reader import GGUFModel
from twtest.reference.cache import HybridCache
from twtest.reference.config import Qwen4ExpConfig
from twtest.reference.model import Qwen4ExpModel
from twtest.reference.weights import WeightStore

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BLOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 32
torch.set_num_threads(8)

# the same passage `context_decay_check.py` uses, verbatim
PARA = (
    "The Rosetta Stone is a granodiorite stele inscribed with three versions of "
    "a decree issued in Memphis in 196 BC. The top and middle texts are in "
    "Ancient Egyptian, using hieroglyphic and Demotic scripts, while the bottom "
    "is in Ancient Greek. Because the decree has only minor differences between "
    "the three versions, the stone proved to be the key to deciphering Egyptian "
    "hieroglyphs, a writing system that had been unreadable for centuries. "
    "The stone was found in 1799 by French soldiers rebuilding a fort near the "
    "town of Rashid in the Nile Delta, and passed to British hands under the "
    "Capitulation of Alexandria two years later. Thomas Young established that "
    "the cartouches spelled a royal name phonetically, and Champollion showed "
    "that the hieroglyphic script recorded sounds as well as meanings, reading "
    "the name of Ramesses in 1822. "
)
EXTRA = (
    "Printing with movable type reached Europe in the fifteenth century, and the "
    "presses that followed changed what a book cost and therefore who could own "
    "one. Early editions imitated manuscript hands closely, because readers "
    "judged a page by the standards they already had. Roman types displaced "
    "blackletter in southern Europe within two generations, while northern "
    "printers kept the older forms for vernacular texts long after scholarly "
    "works had moved. Paper supply, not type, set the ceiling on an edition's "
    "size, and mills clustered where clean water and rags were both available. "
)
TEXT = PARA + EXTRA + PARA + EXTRA

gguf = GGUFModel.from_dir(GGUF_DIR)
cfg = Qwen4ExpConfig.from_gguf(gguf.metadata)
tok = tokenizer(cfg)
ids = tok.encode(TEXT)[: LIMIT + 1]
print(f"RESULT scoring {len(ids) - 1} positions of the device harness's text", flush=True)

model = Qwen4ExpModel(cfg, WeightStore(gguf, cache_bytes=2 << 30, row_cache_bytes=8 << 30))
h = model.forward(torch.tensor([ids]), HybridCache(cfg.num_layers))

hits = tot = 0
lo = 0
for i in range(len(ids) - 1):
    lg = model.logits(h[0, i]).float()
    hits += int(int(lg.argmax()) == ids[i + 1])
    tot += 1
    if tot == BLOCK or i == len(ids) - 2:
        print(f"RESULT positions {lo:4d}-{i:4d}: top-1 {hits:3d}/{tot:3d} = "
              f"{100 * hits / tot:5.1f}%", flush=True)
        hits = tot = 0
        lo = i + 1
