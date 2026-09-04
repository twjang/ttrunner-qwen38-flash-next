"""Does a row's MoE output depend on how many rows travel with it?

    uv run python scripts/dev/moe_rows_check.py [layer]              (default 0)

It must not. The MoE is per-token: each row picks its own experts and is
weighted by its own probabilities, so `moe_block` on 8 rows and on 128 rows must
agree on the 8 rows they share. That is an exact invariant needing no reference
implementation, which makes it the right tool for the `moe_chunk` cliff --
prefill output is bit-identical for row-groups of 8, 16 and 32 and degrades at
64 and 128 (`TTModel.prefill`, handoff 5.8), and 32 is exactly one tile.

Reports, per group size, how far the shared rows have moved from the 1-row
answer, and does the same for the intermediates so the first one to move names
the op.
"""
import sys

import torch
import ttnn

from _device_model import open_model

from ttrunner_qwen38_flash_next.tt import moe

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SIZES = [1, 8, 16, 32, 64, 128]
MAX = max(SIZES)

mesh, cfg, m = open_model(max_seq_len=512)
torch.manual_seed(0)

# Activations of a realistic scale; the invariant holds for any input.
host_x = (torch.randn(1, 1, MAX, cfg.hidden_size) * 0.05).bfloat16()
router_w = m.w.blk(LAYER, "ffn_gate_inp.weight")
if m.fuse_expert_gate_up:
    gate_w, up_w = m.w.fused_gate_up(LAYER), None
else:
    gate_w = m.w.blk(LAYER, "ffn_gate_exps.weight")
    up_w = m.w.blk(LAYER, "ffn_up_exps.weight")
down_w = m.w.blk(LAYER, "ffn_down_exps.weight")


def run(rows: int, start: int = 0):
    x = ttnn.from_torch(
        host_x[:, :, start : start + rows].contiguous(), dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=m.replicate,
    )
    logits = ttnn.linear(x, router_w, compute_kernel_config=moe.HIFI4)
    probs = ttnn.softmax(logits, dim=-1, compute_kernel_config=moe.HIFI4)
    values, _ = ttnn.topk(probs, k=cfg.num_experts_per_tok, dim=-1, largest=True, sorted=True)
    v = list(values.shape)
    threshold = ttnn.slice(
        values, (0, 0, 0, cfg.num_experts_per_tok - 1), (v[0], v[1], v[2], cfg.num_experts_per_tok)
    )
    keep = ttnn.ge(probs, threshold, dtype=ttnn.bfloat16)
    kept = ttnn.multiply(probs, keep)
    weights = ttnn.divide(kept, ttnn.sum(kept, dim=-1, keepdim=True))
    sparsity = ttnn.max(keep, dim=-2, keepdim=True)
    out = moe.moe_block(
        x, router_w, gate_w, up_w, down_w,
        cfg.num_experts_per_tok, cfg.num_experts, cfg.hidden_size, cfg.expert_intermediate,
    )

    def down(t):
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1].float()

    keep_h = down(keep)
    return {
        "keep": keep_h, "weights": down(weights),
        "experts_selected": float(keep_h.sum()),
        # the device's union, and the union the same mask implies on the host --
        # `ttnn.max` over the row axis has to cross tiles once rows > 32
        "union": int(down(sparsity).sum()),
        "union_host": int(keep_h.amax(dim=-2).sum()),
        "out": down(out),
    }


# The ground truth for per-token semantics: every row computed on its own, so
# no grouping can have influenced it. `moe_chunk` is only allowed to change how
# rows are grouped, so any group size that disagrees with this is wrong.
ROWS = 64
print(f"RESULT building the per-row reference ({ROWS} single-row calls)", flush=True)
ref = torch.cat([run(1, start=i)["out"][:, :, :1] for i in range(ROWS)], dim=-2)

print(f"RESULT layer {LAYER}  hidden {cfg.hidden_size}  experts {cfg.num_experts} "
      f"top_k {cfg.num_experts_per_tok}", flush=True)
print(f"RESULT grouping {ROWS} rows; each group size must reproduce the per-row answer",
      flush=True)
print(f"RESULT {'moe_chunk':>10} {'union(last grp)':>16} {'worst row':>10} "
      f"{'rows over 1%':>13}", flush=True)
for g in (1, 8, 16, 32, 64):
    pieces, union, union_host = [], 0, 0
    for lo in range(0, ROWS, g):
        r = run(min(g, ROWS - lo), start=lo)
        pieces.append(r["out"][:, :, : min(g, ROWS - lo)])
        union, union_host = r["union"], r["union_host"]
    got = torch.cat(pieces, dim=-2)
    scale = ref.abs().max().clamp(min=1e-9)
    per_row = (got[0, 0] - ref[0, 0]).abs().amax(dim=-1) / scale * 100.0
    bad = int((per_row > 1.0).sum())
    print(f"RESULT {g:10d} {union:16d} {float(per_row.max()):9.3f}% "
          f"{bad:9d}/{ROWS}   union host {union_host:4d} dev {union:4d}"
          f"{'   <-- MISMATCH' if union != union_host else ''}", flush=True)

print("\nRESULT any group size with rows over 1% is not computing per-token MoE",
      flush=True)
ttnn.close_mesh_device(mesh)
