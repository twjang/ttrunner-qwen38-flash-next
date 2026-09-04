"""Does the *reference* agree with itself when a prompt is consumed at once?

    uv run python scripts/dev/reference_prefill_vs_decode.py [len]   (default 4)

The device has two paths for a prompt and they disagree. So does the CPU
reference: `chunk_gated_delta_rule` for a whole sequence, `recurrent_gated_delta_rule`
one token at a time. They are the same function mathematically, so in float32
they should agree to nothing, and the device is then expected to reproduce that
agreement to within its own precision.

If instead the reference disagrees with itself by a visible margin, the two
device paths cannot be held to a tighter standard than that, and the acceptance
criterion for prefill has to become a token-agreement rate rather than a
per-layer distance.

Runs on the CPU only -- no device, so it can share the machine with a device job.
"""
import sys

import torch

from _device_model import GGUF_DIR

from ttrunner_qwen38_flash_next.gguf.reader import GGUFModel
from ttrunner_qwen38_flash_next.reference.cache import HybridCache
from ttrunner_qwen38_flash_next.reference.config import Qwen4ExpConfig
from ttrunner_qwen38_flash_next.reference.model import Qwen4ExpModel
from ttrunner_qwen38_flash_next.reference.weights import WeightStore

L = int(sys.argv[1]) if len(sys.argv) > 1 else 4
torch.set_num_threads(8)

gguf = GGUFModel.from_dir(GGUF_DIR)
cfg = Qwen4ExpConfig.from_gguf(gguf.metadata)
store = WeightStore(gguf, cache_bytes=2 << 30, row_cache_bytes=8 << 30)
model = Qwen4ExpModel(cfg, store)

prompt = [1000 + ((i * 37) % 5000) for i in range(L)]

seq_h: dict[int, torch.Tensor] = {}
model.probe = lambda layer, hidden: seq_h.__setitem__(layer, hidden[0].clone())
h_seq = model.forward(torch.tensor([prompt]), HybridCache(cfg.num_layers))
print("RESULT sequential forward done", flush=True)

dec_h: dict[int, list[torch.Tensor]] = {}
model.probe = lambda layer, hidden: dec_h.setdefault(layer, []).append(hidden[0, 0].clone())
cache = HybridCache(cfg.num_layers)
for i, t in enumerate(prompt):
    h_dec = model.forward(torch.tensor([[t]]), cache)
    print(f"  token {i + 1}/{L}", flush=True)
model.probe = None

for layer in range(cfg.num_layers):
    a = seq_h[layer]
    b = torch.stack(dec_h[layer])
    scale = max(a.abs().max().item(), 1e-6)
    print(
        f"RESULT layer {layer:2d} rel {100 * (a - b).abs().max().item() / scale:8.4f}%  scale {scale:.3f}",
        flush=True,
    )

t_seq = int(model.logits(h_seq[0, -1]).argmax())
t_dec = int(model.logits(h_dec[0, -1]).argmax())
print(f"RESULT next token  whole-prompt {t_seq}  token-by-token {t_dec}  "
      f"{'MATCH' if t_seq == t_dec else 'DIFFER'}", flush=True)
