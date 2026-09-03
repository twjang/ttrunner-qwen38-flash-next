"""One row tile is this hardware's reproducibility boundary. Demonstrated on a linear.

    uv run python scripts/dev/row_tile_boundary_check.py

A plain `ttnn.linear` returns a given row *identically* for any row count that
fits in one 32-row tile, and differently once it does not -- by one bf16 ulp,
the same amount at 33, 48, 64 and 128. Nothing about the MoE is involved.

This is the single fact behind three findings that were recorded separately:

  * `moe_chunk` cannot exceed 32 without changing the answer (handoff 5.8)
  * batch 64 decodes differently from batch 1 (README, 5.8)
  * `step_n` cannot reproduce k sequential steps past k=32 (5.5)

All three compare a row computed among <= 32 rows against the same row computed
among more, and the comparison cannot come out equal. It is not fixable at the
model level: splitting `expert_ffn` into 32-row groups so `per_core_M` stays 1
was tried and changes nothing, because the ops *before* the experts -- the
router's linear among them -- have already diverged.

Distinct from the `sparse_matmul` defect in 5.8, which was a genuine bug (rows
past the first tile silently dropped when K spans more than one block) and is
fixed. That one was corruption; this one is arithmetic.
"""
import torch, ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
K, N = 2560, 512
torch.manual_seed(0)
w = ttnn.from_torch((torch.randn(1, 1, K, N) * 0.02).bfloat16().contiguous(),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
rows = (torch.randn(1, 1, 128, K) * 0.05).bfloat16()
HIFI4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                         fp32_dest_acc_en=True, packer_l1_acc=True)


def run(m):
    x = ttnn.from_torch(rows[:, :, :m].contiguous(), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh)
    out = ttnn.linear(x, w, compute_kernel_config=HIFI4)
    return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1].float()


ref = run(1)[0, 0, 0]
print(f"PROBE {'rows':>6} {'row 0 vs m=1':>14}", flush=True)
for m in (1, 8, 16, 32, 33, 48, 64, 128):
    got = run(m)[0, 0, 0]
    same = torch.equal(got, ref)
    err = float((got - ref).abs().max())
    print(f"PROBE {m:6d} {'identical' if same else f'differs {err:.3g}':>14}", flush=True)
ttnn.close_mesh_device(mesh)
