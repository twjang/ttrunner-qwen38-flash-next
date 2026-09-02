"""Is the state `prefill` leaves behind the state `step` would have left?

    uv run python scripts/dev/prefill_state_check.py [len]        (default 8)

`prefill_check.py` compares the hidden a prompt produces; this compares what
the prompt leaves in the slot, which is what every generated token afterwards
depends on. A ring in the wrong order or a recurrent state one token behind
shows up here immediately and nowhere else -- the returned hidden can be right
while generation still walks off.

Per layer: the DeltaNet recurrent matrix, the convolution ring (and its step
counter), the PLE ring, and the K/V cache over the positions written.
"""
import sys

import torch
import ttnn

from _device_model import host_row, open_model, synthetic_prompt

L = int(sys.argv[1]) if len(sys.argv) > 1 else 8
mesh, cfg, m = open_model()
P = synthetic_prompt(L)

stp = m.new_state(batch=1)
m.prefill(P, stp)
sts = m.new_state(batch=1)
for t in P:
    m.step([t], sts)


def cmp(tag, a, b):
    if a is None and b is None:
        return
    if (a is None) != (b is None):
        print(f"RESULT {tag:34s} ONE SIDE MISSING (prefill={a is not None} step={b is not None})", flush=True)
        return
    ta, tb = host_row(mesh, a), host_row(mesh, b)
    if ta.shape != tb.shape:
        print(f"RESULT {tag:34s} SHAPE {tuple(ta.shape)} vs {tuple(tb.shape)}", flush=True)
        return
    scale = max(tb.abs().max().item(), 1e-6)
    print(f"RESULT {tag:34s} rel {100 * (ta - tb).abs().max().item() / scale:8.2f}%  scale {scale:.3f}", flush=True)


print(f"RESULT positions prefill={stp.positions} step={sts.positions}", flush=True)
print(f"RESULT history len prefill={len(stp.histories[0])} step={len(sts.histories[0])}", flush=True)
for layer in range(cfg.num_layers):
    a, b = stp[layer], sts[layer]
    if a.conv_step != b.conv_step:
        print(f"RESULT layer {layer:2d} conv_step {a.conv_step} vs {b.conv_step}", flush=True)
    if a.ple_step != b.ple_step:
        print(f"RESULT layer {layer:2d} ple_step {a.ple_step} vs {b.ple_step}", flush=True)
    cmp(f"layer {layer:2d} recurrent", a.recurrent, b.recurrent)
    if a.conv is not None and b.conv is not None:
        for i, (x, y) in enumerate(zip(a.conv, b.conv)):
            cmp(f"layer {layer:2d} conv[{i}]", x, y)
    if a.ple_conv is not None and b.ple_conv is not None:
        for i, (x, y) in enumerate(zip(a.ple_conv, b.ple_conv)):
            cmp(f"layer {layer:2d} ple_conv[{i}]", x, y)
    if a.keys is not None and b.keys is not None:
        hd, n_kv = cfg.head_dim, cfg.num_kv_heads
        ka = ttnn.slice(a.keys, (0, 0, 0, 0), (1, n_kv, L, hd))
        kb = ttnn.slice(b.keys, (0, 0, 0, 0), (1, n_kv, L, hd))
        cmp(f"layer {layer:2d} keys[:{L}]", ka, kb)
        va = ttnn.slice(a.values, (0, 0, 0, 0), (1, n_kv, L, hd))
        vb = ttnn.slice(b.values, (0, 0, 0, 0), (1, n_kv, L, hd))
        cmp(f"layer {layer:2d} values[:{L}]", va, vb)

ttnn.close_mesh_device(mesh)
