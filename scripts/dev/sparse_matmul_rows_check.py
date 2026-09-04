"""Is `ttnn.sparse_matmul` row-count independent at production dims?

    uv run python scripts/dev/sparse_matmul_rows_check.py [layer]     (default 0)

`moe_block` stops being per-token past one tile of rows (handoff 5.8): groups of
1, 8, 16 and 32 reproduce a per-row MoE exactly, 64 gets 33 of 64 rows wrong.
Routing is not the cause and neither is the broadcast -- `ttnn.repeat` to
[1, 512, M, 2560] is exact at every row count. That leaves the sparse matmul,
which was exact in isolation at E=64, K=256, N=128, roughly a hundredth of the
real shape.

So: real expert weights, a realistic mask, and one question -- does a row's
output depend on how many rows travel with it? It must not; the matmul is per
row.
"""
import sys

import torch
import ttnn

from _device_model import open_model

from ttrunner_qwen38_flash_next.tt import moe

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ROWS = 64

mesh, cfg, m = open_model(max_seq_len=512)
torch.manual_seed(0)
E, K = cfg.num_experts, cfg.hidden_size
w = m.w.fused_gate_up(LAYER) if m.fuse_expert_gate_up else m.w.blk(LAYER, "ffn_gate_exps.weight")
N = w.shape[-1]
host_x = (torch.randn(1, 1, ROWS, K) * 0.05).bfloat16()

# a realistic union: about half the experts, as 64 rows of top-10 routing give
torch.manual_seed(1)
mask = torch.zeros(1, 1, 1, E)
mask[0, 0, 0, torch.randperm(E)[:255]] = 1.0
sparsity = ttnn.from_torch(mask.contiguous(), dtype=ttnn.bfloat16,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=m.replicate)
PICKS = [int(i) for i in torch.nonzero(mask[0, 0, 0])[:4, 0]]


def run(rows, start=0):
    x = ttnn.from_torch(host_x[:, :, start:start + rows].contiguous(), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=m.replicate)
    out = ttnn.sparse_matmul(
        ttnn.repeat(x, (1, E, 1, 1)), w, sparsity=sparsity, nnz=None,
        is_input_a_sparse=True, is_input_b_sparse=True,
        program_config=moe.sparse_program_config(rows, K, N),
        compute_kernel_config=moe.HIFI4,
    )
    keep = [ttnn.to_torch(ttnn.slice(out, (0, e, 0, 0), (1, e + 1, rows, N)),
                          mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1].float()
            for e in PICKS]
    ttnn.deallocate(out)
    return torch.cat(keep, dim=1)          # [1, len(PICKS), rows, N]


print(f"RESULT layer {LAYER}  experts {E}  K {K}  N {N}  union {int(mask.sum())}", flush=True)
print(f"RESULT experts sampled: {PICKS}", flush=True)
ref = torch.cat([run(1, i) for i in range(ROWS)], dim=-2)
print(f"RESULT {'group':>6} {'worst row':>11} {'rows over 1%':>14}", flush=True)
for g in (1, 8, 16, 32, 64):
    got = torch.cat([run(min(g, ROWS - lo), lo)[:, :, : min(g, ROWS - lo)]
                     for lo in range(0, ROWS, g)], dim=-2)
    scale = ref.abs().max().clamp(min=1e-9)
    per_row = (got - ref).abs().amax(dim=(1, 3))[0] / scale * 100.0
    bad = int((per_row > 1.0).sum())
    print(f"RESULT {g:6d} {float(per_row.max()):10.3f}% {bad:9d}/{ROWS}"
          f"{'' if bad == 0 else '   <-- row-count dependent'}", flush=True)
ttnn.close_mesh_device(mesh)
