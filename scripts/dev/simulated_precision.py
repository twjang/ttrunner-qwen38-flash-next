"""How much of the device's gap is the weights, and how much is everything else?

    uv run python scripts/dev/simulated_precision.py [total]      (default 48)

The device agrees with the float32 oracle on 40 % of greedy tokens. Every block
is at its dtype floor and nothing structural is left, so the gap is precision --
but "precision" is two different things with two different fixes:

  * the weights, quantised a second time by the converter (the checkpoint is
    already UD-IQ4_XS and the experts become bfloat4_b on top of that), and
  * everything else -- bf16 activations, block-float accumulation inside the
    matmuls, and any op that rounds differently from torch.

`tt/blockfloat.py` reproduces ttnn's bfloat8_b/bfloat4_b elementwise, so the
first can be measured on its own: run the reference in float32 but hand it
weights round-tripped through each tensor's device dtype. Whatever agreement
that loses is the weights' share; whatever the device loses beyond it is not.

CPU only -- no device, so it can share the machine with a device job.
"""
import sys

import numpy as np
import torch

from _device_model import GGUF_DIR, tokenizer

from twtest.gguf.reader import GGUFModel
from twtest.reference.cache import HybridCache
from twtest.reference.config import Qwen4ExpConfig
from twtest.reference.model import Qwen4ExpModel
from twtest.reference.weights import WeightStore
from twtest.tt.blockfloat import round_trip
from twtest.tt.plan import Residency, plan_for

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 48
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
ids = tok.encode(TEXT)[:TOTAL]

seen: dict[str, str] = {}
skipped: dict[str, str] = {}


def quantise(name: str, flat: np.ndarray) -> np.ndarray:
    """Round-trip a tensor through the dtype the converter gives it on device."""
    try:
        plan = plan_for(name)
    except Exception:
        return flat
    if plan.residency is not Residency.DEVICE:
        return flat                       # stays on the host, dequantised
    try:
        out = round_trip(flat, plan.dtype)
    except Exception as exc:              # a size that is not a whole number of blocks
        skipped[name] = f"{plan.dtype}: {exc}"
        return flat
    seen[name] = plan.dtype
    return out


def predictions(store: WeightStore):
    model = Qwen4ExpModel(cfg, store)
    h = model.forward(torch.tensor([ids]), HybridCache(cfg.num_layers))
    return [int(model.logits(h[0, i]).float().argmax()) for i in range(len(ids))], h


exact_store = WeightStore(gguf, cache_bytes=2 << 30, row_cache_bytes=8 << 30)
exact, h_exact = predictions(exact_store)
print(f"RESULT exact reference ready, {len(ids)} positions", flush=True)

quant_store = WeightStore(
    gguf, cache_bytes=2 << 30, row_cache_bytes=8 << 30, quant_sim=quantise
)
quant, h_quant = predictions(quant_store)
print(f"RESULT simulated-device-weights reference ready "
      f"({len(seen)} tensors round-tripped, {len(skipped)} skipped)", flush=True)
for name, why in list(skipped.items())[:5]:
    print(f"RESULT   skipped {name}: {why}", flush=True)
by_dtype: dict[str, int] = {}
for d in seen.values():
    by_dtype[d] = by_dtype.get(d, 0) + 1
print(f"RESULT   dtypes {by_dtype}", flush=True)

idx = list(range(1, len(ids)))
n = sum(exact[i] == quant[i] for i in idx)
print(f"RESULT weights-only agreement {n}/{len(idx)} = {100 * n / len(idx):.1f}%", flush=True)

d = (h_quant - h_exact).abs().max().item()
scale = h_exact.abs().max().item()
print(f"RESULT final hidden rel {100 * d / scale:.2f}%", flush=True)
print("RESULT compare against the device's 40.4% (scripts/dev/three_way_agreement.py)", flush=True)
