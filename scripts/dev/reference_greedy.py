"""Greedy text from the float32 CPU reference. The convention arbiter.

    uv run python scripts/dev/reference_greedy.py [n_tokens]      (default 12)

README records what llama.cpp produces for this prompt on this GGUF:

    'The capital of France is Paris. The capital of Germany is Berlin. The capital of'

and the reference reproduced it. So this prints what the reference says now,
which is how a change to the model -- the DeltaNet head expansion, say -- is
judged: coherent and matching means the convention is right, degenerate means
it is not. CPU only.
"""
import sys

import torch

from _device_model import GGUF_DIR, tokenizer

from twtest.gguf.reader import GGUFModel
from twtest.reference.cache import HybridCache
from twtest.reference.config import Qwen4ExpConfig
from twtest.reference.model import Qwen4ExpModel
from twtest.reference.weights import WeightStore

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
torch.set_num_threads(8)

gguf = GGUFModel.from_dir(GGUF_DIR)
cfg = Qwen4ExpConfig.from_gguf(gguf.metadata)
model = Qwen4ExpModel(cfg, WeightStore(gguf, cache_bytes=2 << 30, row_cache_bytes=8 << 30))
tok = tokenizer(cfg)

PROMPT = "The capital of France is"
ids = tok.encode(PROMPT)
cache = HybridCache(cfg.num_layers)
out = []
h = model.forward(torch.tensor([ids]), cache)
for i in range(N):
    t = int(model.logits(h[0, -1]).argmax())
    out.append(t)
    print(f"RESULT token {i + 1}/{N} -> {t} {tok.decode(out)!r}", flush=True)
    h = model.forward(torch.tensor([[t]]), cache)
print(f"RESULT prompt {PROMPT!r}", flush=True)
print(f"RESULT continuation {tok.decode(out)!r}", flush=True)
