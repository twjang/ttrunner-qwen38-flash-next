"""Is the CPU reference a working language model? CPU only, one forward.

    uv run python scripts/dev/reference_quality.py [n_tokens]     (default 48)

Greedy text can look fine while the model is badly degraded -- a damaged model
still produces grammatical, repetitive English. Next-token accuracy and NLL on
real text do not have that failure mode: a working 8B model predicts the actual
next token of encyclopedic prose a large fraction of the time, and a broken one
does not, whatever its samples look like.

Use this to A/B a change to the *model* (as opposed to its precision), where
"the greedy output still reads fine" is not evidence either way.
"""
import sys

import torch

from _device_model import GGUF_DIR, tokenizer

from ttrunner_qwen38_flash_next.gguf.reader import GGUFModel
from ttrunner_qwen38_flash_next.reference.cache import HybridCache
from ttrunner_qwen38_flash_next.reference.config import Qwen4ExpConfig
from ttrunner_qwen38_flash_next.reference.model import Qwen4ExpModel
from ttrunner_qwen38_flash_next.reference.weights import WeightStore

N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
torch.set_num_threads(8)

TEXT = (
    "The Rosetta Stone is a granodiorite stele inscribed with three versions of "
    "a decree issued in Memphis in 196 BC. The top and middle texts are in "
    "Ancient Egyptian, using hieroglyphic and Demotic scripts, while the bottom "
    "is in Ancient Greek. Because the decree has only minor differences between "
    "the three versions, the stone proved to be the key to deciphering Egyptian "
    "hieroglyphs, a writing system that had been unreadable for centuries."
)

gguf = GGUFModel.from_dir(GGUF_DIR)
cfg = Qwen4ExpConfig.from_gguf(gguf.metadata)
tok = tokenizer(cfg)
ids = tok.encode(TEXT)[:N]
model = Qwen4ExpModel(cfg, WeightStore(gguf, cache_bytes=2 << 30, row_cache_bytes=8 << 30))

h = model.forward(torch.tensor([ids]), HybridCache(cfg.num_layers))
hits, nll, top5 = 0, [], 0
for i in range(len(ids) - 1):
    lg = model.logits(h[0, i]).float()
    target = ids[i + 1]
    hits += int(lg.argmax()) == target
    top5 += target in lg.topk(5).indices.tolist()
    nll.append(float(-torch.log_softmax(lg, dim=-1)[target]))
srt = sorted(nll)
n = len(nll)
print(f"RESULT next-token top-1 {hits}/{n} = {100 * hits / n:.1f}%   "
      f"top-5 {top5}/{n} = {100 * top5 / n:.1f}%", flush=True)
print(f"RESULT NLL mean {sum(nll) / n:.3f}  median {srt[n // 2]:.3f}  "
      f"perplexity(mean) {pow(2.718281828, sum(nll) / n):.1f}", flush=True)
print("RESULT a working 8B model on prose like this predicts the next token "
      "roughly half the time, with median NLL near 1", flush=True)
