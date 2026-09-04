"""Do the device branches track the reference *across steps*, not just at one?

    uv run python scripts/dev/branch_sequence_check.py [layer ...] [--steps N]
                                                        (default 0 3, 8 steps)

`layer_decode_bisect.py` drives one token into an empty state, so it says
nothing about anything carried between steps: the convolution rings, the
DeltaNet recurrent matrix, the K/V cache, rope at a non-zero position, or the
PLE n-gram history. Those are exactly what a decode path lives on, and a fault
there is invisible at position 0.

This feeds the same sequence of inputs to a device branch and to the reference's
own branch -- the reference carrying its state in a HybridCache, the device in a
LayerState -- and reports the distance at every step. A number that is flat is
the precision floor; one that climbs with the step index is a state bug.
"""
import sys

import torch
import ttnn

from _device_model import host_row, open_model

from ttrunner_qwen38_flash_next.reference.cache import HybridCache
from ttrunner_qwen38_flash_next.reference.model import Qwen4ExpModel

args = [a for a in sys.argv[1:] if not a.startswith("--")]
STEPS = 8
for i, a in enumerate(sys.argv):
    if a == "--steps":
        STEPS = int(sys.argv[i + 1])
LAYERS = [int(x) for x in args] or [0, 3]

mesh, cfg, m = open_model()
ref = Qwen4ExpModel(cfg, m.host)

torch.manual_seed(0)
inputs = [torch.randn(1, 1, 1, cfg.hidden_size) * 0.5 for _ in range(STEPS)]

for LAYER in LAYERS:
    kind = "QSA" if cfg.is_full_attention(LAYER) else "DeltaNet"
    print(f"RESULT ---- layer {LAYER} ({kind}) ----", flush=True)
    cache = HybridCache(cfg.num_layers)
    state = m.new_state(batch=1)
    st = state[LAYER]

    for t, x in enumerate(inputs):
        host_in = x.reshape(1, 1, cfg.hidden_size).float()
        dev_in = m.to_dev(x)
        if cfg.is_full_attention(LAYER):
            positions = cache.extend_positions(torch.tensor([[t]]))
            cos, sin = ref.rotary(positions)
            want = ref._full_attention(host_in, LAYER, cos, sin, cache, t)
            got = m._attention_step(dev_in, LAYER, st, t)
        else:
            want = ref._linear_attention(host_in, LAYER, cache)
            got = m._linear_attention_step(dev_in, LAYER, st)
        w = want.reshape(-1).float()
        g = host_row(mesh, got).reshape(-1).float()
        scale = max(w.abs().max().item(), 1e-9)
        print(f"RESULT step {t:2d} rel {100 * (g - w).abs().max().item() / scale:8.2f}%  "
              f"scale {scale:.4f}", flush=True)
    del state

ttnn.close_mesh_device(mesh)
