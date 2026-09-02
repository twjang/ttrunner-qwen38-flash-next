"""Is `shared_expert` per-token exact however many rows travel together?

    uv run python scripts/dev/shared_expert_rows_check.py [layer]     (default 0)

It has no routing -- four dense linears, a silu and a sigmoid gate -- so it
should be, and `prefill` could then compute it once per layer instead of once
per MoE sub-chunk. But `moe_block` is *also* meant to be per-token and stops
being so past one tile of rows (`moe_rows_check.py`, handoff 5.8), so this asks
rather than assumes: same question, same method, one row at a time as the
reference.
"""
import sys

import torch
import ttnn

from _device_model import open_model

from twtest.tt import moe

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ROWS = 128

mesh, cfg, m = open_model(max_seq_len=512)
torch.manual_seed(0)
host_x = (torch.randn(1, 1, ROWS, cfg.hidden_size) * 0.05).bfloat16()
args = [m.w.blk(LAYER, n) for n in (
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
    "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight",
)]


def run(rows, start=0):
    x = ttnn.from_torch(
        host_x[:, :, start : start + rows].contiguous(), dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=m.replicate,
    )
    out = moe.shared_expert(x, *args)
    return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1].float()


print(f"RESULT building the per-row reference ({ROWS} single-row calls)", flush=True)
ref = torch.cat([run(1, start=i) for i in range(ROWS)], dim=-2)
print(f"RESULT {'group':>6} {'worst row':>10} {'rows over 1%':>13}", flush=True)
for g in (1, 8, 16, 32, 64, 128):
    got = torch.cat(
        [run(min(g, ROWS - lo), start=lo)[:, :, : min(g, ROWS - lo)] for lo in range(0, ROWS, g)],
        dim=-2,
    )
    scale = ref.abs().max().clamp(min=1e-9)
    per_row = (got[0, 0] - ref[0, 0]).abs().amax(dim=-1) / scale * 100.0
    bad = int((per_row > 1.0).sum())
    print(f"RESULT {g:6d} {float(per_row.max()):9.3f}% {bad:9d}/{ROWS}"
          f"{'' if bad == 0 else '   <-- NOT per-token'}", flush=True)
ttnn.close_mesh_device(mesh)
