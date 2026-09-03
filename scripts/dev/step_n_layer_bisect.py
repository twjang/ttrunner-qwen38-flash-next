"""Which block breaks `step_n` past one tile of rows?

    uv run python scripts/dev/step_n_layer_bisect.py [k]              (default 33)

`step_n` reproduces k sequential steps exactly to k=16 and is 35.68 % out on the
hidden from k=33, the same figure at 33, 48 and 64 (`step_n_check.py`). This
walks the layers with `TTModel.probe` and reports the first one to diverge, so
the search starts somewhere specific: layer 0 is DeltaNet, layer 3 the first
sparse-attention layer, and every 4th after that.

Compares the *last* row, which is the one that has seen every token, against the
last of k sequential steps.
"""
import sys

import torch
import ttnn

from _device_model import open_model, synthetic_prompt

K = int(sys.argv[1]) if len(sys.argv) > 1 else 33
PRE = 8

mesh, cfg, m = open_model(max_seq_len=512)
prompt = synthetic_prompt(PRE + K)
pre, draft = prompt[:PRE], prompt[PRE : PRE + K]


def row(t, idx):
    x = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1].float()
    return x.reshape(-1, x.shape[-1])[idx].reshape(-1)


seq: dict[int, torch.Tensor] = {}
st = m.new_state(batch=1)
for t in pre:
    m.step([t], st)
for i, t in enumerate(draft):
    last = i == len(draft) - 1
    m.probe = (lambda layer, h: seq.__setitem__(layer, row(h, -1))) if last else None
    m.step([t], st)
m.probe = None
del st

got: dict[int, torch.Tensor] = {}
st = m.new_state(batch=1)
for t in pre:
    m.step([t], st)
m.probe = lambda layer, h: got.__setitem__(layer, row(h, K - 1))
m.step_n(draft, st)
m.probe = None

print(f"RESULT k={K}, comparing the last row against the last of {K} steps", flush=True)
print(f"RESULT {'layer':>6} {'kind':>10} {'rel %':>10}", flush=True)
first = None
for layer in sorted(seq):
    a, b = got.get(layer), seq[layer]
    if a is None or a.shape != b.shape:
        print(f"RESULT {layer:6d}  shape {tuple(a.shape) if a is not None else None} "
              f"vs {tuple(b.shape)}", flush=True)
        continue
    rel = float((a - b).abs().max() / b.abs().max().clamp(min=1e-6)) * 100
    kind = "QSA" if layer in getattr(m, "_attn_layers", set()) or (layer + 1) % 4 == 0 else "DeltaNet"
    if rel > 1.0 and first is None:
        first = layer
    if layer < 12 or rel > 1.0:
        print(f"RESULT {layer:6d} {kind:>10} {rel:9.3f}%", flush=True)
print(f"RESULT first layer over 1 %: {first}", flush=True)
ttnn.close_mesh_device(mesh)
