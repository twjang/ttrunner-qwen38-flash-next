"""Is the residual gap discrete routing, or continuous precision?

    uv run python scripts/dev/routing_overlap.py [layer ...]      (default 0 3 24 47)

Every branch of every layer now sits within a few percent of the float32
reference, and the MoE is the widest of them. Two very different things could
explain that: the experts' own bfloat4_b error, or the router picking a
different top-10 out of 512 because a ~1 % perturbation of its input crossed a
near-tie. The first is smooth and bounded; the second is discrete and can move a
token on its own.

This runs the router both ways on the same input and reports how many of the ten
selected experts agree, and how close the boundary was.
"""
import sys

import torch
import torch.nn.functional as F
import ttnn

from _device_model import host_row, open_model

LAYERS = [int(x) for x in sys.argv[1:]] or [0, 3, 24, 47]
mesh, cfg, m = open_model()
torch.manual_seed(0)

TRIALS = 8
for LAYER in LAYERS:
    overlaps, gaps = [], []
    for _ in range(TRIALS):
        x = torch.randn(1, 1, 1, cfg.hidden_size) * 0.5
        want = F.linear(x.reshape(1, cfg.hidden_size).float(),
                        m.host.get(f"blk.{LAYER}.ffn_gate_inp.weight").float())
        got = host_row(mesh, ttnn.linear(m.to_dev(x), m.w.blk(LAYER, "ffn_gate_inp.weight"))
                       ).reshape(1, -1)[:, : cfg.num_experts].float()
        k = cfg.num_experts_per_tok
        a = set(want[0].topk(k).indices.tolist())
        b = set(got[0, : want.shape[-1]].topk(k).indices.tolist())
        overlaps.append(len(a & b))
        srt = want[0].sort(descending=True).values
        gaps.append(float(srt[k - 1] - srt[k]))       # margin at the cutoff
    print(
        f"RESULT layer {LAYER:2d} experts agreeing {sum(overlaps) / TRIALS:.1f}/{cfg.num_experts_per_tok}"
        f"   cutoff margin mean {sum(gaps) / TRIALS:.4f}",
        flush=True,
    )
ttnn.close_mesh_device(mesh)
