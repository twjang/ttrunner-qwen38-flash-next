"""Per-layer bisection of `prefill` against `step` on one short prompt.

    uv run python scripts/dev/prefill_bisect.py [len]        (default 4)

Uses `TTModel.probe`, which both paths call after every layer. Prints, per
layer, the max |prefill - step| at each prompt position next to the tensor's
scale. Read it top-down: the first layer whose error jumps is the culprit and
everything below it is propagation. This is what found the missing all-reduce
and the layer-less conv-tap cache key.
"""
import sys

import torch
import ttnn

from _device_model import host_row, open_model, synthetic_prompt

L = int(sys.argv[1]) if len(sys.argv) > 1 else 4
mesh, cfg, m = open_model()
P = synthetic_prompt(L)

step_h: dict[int, list[torch.Tensor]] = {}
m.probe = lambda layer, hidden: step_h.setdefault(layer, []).append(host_row(mesh, hidden)[0, 0, 0])
sts = m.new_state(batch=1)
for t in P:
    m.step([t], sts)
del sts

pre_h: dict[int, torch.Tensor] = {}
m.probe = lambda layer, hidden: pre_h.__setitem__(layer, host_row(mesh, hidden)[0, 0])
stp = m.new_state(batch=1)
m.prefill(P, stp, moe_chunk=8)
m.probe = None

for layer in range(cfg.num_layers):
    s = torch.stack(step_h[layer])
    p = pre_h[layer][:L]
    per_pos = (s - p).abs().amax(dim=-1)
    kind = "QSA" if cfg.is_full_attention(layer) else "DN "
    ple = "Y" if layer in m._ple_layers else "n"
    print(
        f"RESULT layer {layer:2d} {kind} ple={ple} maxdiff per pos "
        f"{[round(x, 3) for x in per_pos.tolist()]}  scale {s.abs().max().item():.2f}", flush=True,
    )
ttnn.close_mesh_device(mesh)
