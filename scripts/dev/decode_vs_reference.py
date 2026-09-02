"""Per-layer diff of the device `step` path against the CPU reference oracle.

    uv run python scripts/dev/decode_vs_reference.py [len]      (default 4)

This is the discriminating experiment for the prefill question. `prefill_bisect`
tells us how far prefill is from decode, but not whether that distance is a bug
or just bf16: the decode path is verified token-for-token, yet its per-layer
hidden state still drifts from float32 by *something*. This script measures that
something, and prefill's distance from the same oracle, in one run:

    step vs ref      the bf16 noise floor -- the best any device path can do
    prefill vs ref   what prefill actually achieves
    prefill vs step  what `prefill_bisect.py` reports, repeated for alignment

If prefill-vs-ref tracks step-vs-ref layer for layer, prefill is at the floor and
the remaining token disagreement is sampling-boundary noise, not a defect. If it
is materially worse from layer 3 onward, the bug is in `_attention_chunk`.

The reference is minutes per token, so keep the length small. Errors are
reported relative to the reference's own scale at that layer, because the
hyper-connection stream grows by ~2 orders of magnitude across the 48 layers and
absolute numbers are unreadable.
"""
import sys

import torch
import ttnn

from _device_model import host_row, open_model, synthetic_prompt

from twtest.reference.cache import HybridCache
from twtest.reference.model import Qwen4ExpModel

L = int(sys.argv[1]) if len(sys.argv) > 1 else 4
mesh, cfg, m = open_model()
P = synthetic_prompt(L)

# -- device: one token at a time, exactly as the engine decodes --------------
step_h: dict[int, list[torch.Tensor]] = {}
m.probe = lambda layer, hidden: step_h.setdefault(layer, []).append(host_row(mesh, hidden)[0, 0, 0])
sts = m.new_state(batch=1)
for t in P:
    m.step([t], sts)
del sts

# -- device: the chunked prefill path ---------------------------------------
pre_h: dict[int, torch.Tensor] = {}
m.probe = lambda layer, hidden: pre_h.__setitem__(layer, host_row(mesh, hidden)[0, 0])
stp = m.new_state(batch=1)
m.prefill(P, stp, moe_chunk=8)
m.probe = None
del stp

# -- host: the float32 oracle, also one token at a time ---------------------
ref = Qwen4ExpModel(cfg, m.host)
ref_h: dict[int, list[torch.Tensor]] = {}
ref.probe = lambda layer, hidden: ref_h.setdefault(layer, []).append(hidden[0, 0].clone())
cache = HybridCache(cfg.num_layers)
for i, t in enumerate(P):
    print(f"  reference token {i + 1}/{L}", flush=True)
    ref.forward(torch.tensor([[t]]), cache)

print(f"RESULT header layer kind ple  step_vs_ref  prefill_vs_ref  prefill_vs_step  scale", flush=True)
for layer in range(cfg.num_layers):
    r = torch.stack(ref_h[layer])
    s = torch.stack(step_h[layer])
    p = pre_h[layer][:L]
    scale = r.abs().max().item()

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return (a - b).abs().max().item() / max(scale, 1e-6)

    kind = "QSA" if cfg.is_full_attention(layer) else "DN "
    ple = "Y" if layer in m._ple_layers else "n"
    print(
        f"RESULT layer {layer:2d} {kind} ple={ple} "
        f"{rel(s, r) * 100:9.2f}% {rel(p, r) * 100:13.2f}% {rel(p, s) * 100:14.2f}% "
        f"{scale:10.2f}",
        flush=True,
    )
ttnn.close_mesh_device(mesh)
