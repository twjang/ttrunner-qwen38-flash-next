"""Does sharding the experts across devices give the same answer?

    uv run python scripts/dev/expert_shard_equivalence_check.py

Today the expert stacks are split on the *intermediate* axis: every device holds
all 512 experts at a quarter width. That makes `sparse_matmul` write
[1, 512, M, K] per device -- 84 MB a layer to carry ~59 KB of selected expert
output, which invariant 27 shows is where decode's time goes.

Sharding on the *expert* axis instead gives each device 128 whole experts, so the
output is a quarter the size. The arithmetic is the same either way: a token's
answer is the weighted sum over its experts, and it does not matter which device
computed which term as long as they are summed. The all-reduce already there does
that summing.

The piece that was unproven is the routing mask. Every device computes the same
512 logits, but each needs only its own 128 columns -- and `ttnn.mesh_partition`
is exactly that (it is the inverse of all_gather; dev0 gets 0-127, dev1 128-255,
verified).

This builds both paths on the same random weights and compares. Synthetic
weights on purpose: it tests the *scheme*, and needs no cache conversion to do it.
"""
import time

import torch
import ttnn

from ttrunner_qwen38_flash_next.tt.moe import expert_ffn, sparse_program_config, HIFI4

E, M, K, N = 512, 32, 2560, 640          # N is the full intermediate width
NDEV = 4
TOPK = 10

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)
rep = ttnn.ReplicateTensorToMesh(mesh)
comp = ttnn.ConcatMeshToTensor(mesh, dim=0)

x_t = torch.randn(1, 1, M, K) * 0.1
gate_t = torch.randn(1, E, K, N) * 0.05      # [1, E, K, N]
down_t = torch.randn(1, E, N, K) * 0.05      # [1, E, N, K]
# a routing mask: the same experts chosen on every device
keep_t = torch.zeros(1, 1, M, E)
for r in range(M):
    keep_t[0, 0, r, torch.randperm(E)[:TOPK]] = 1.0
w_t = keep_t / keep_t.sum(-1, keepdim=True)

x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                    mesh_mapper=rep)
weights = ttnn.from_torch(w_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                          mesh_mapper=rep)
keep = ttnn.from_torch(keep_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                       mesh_mapper=rep)


def combine(per_expert, w, n_exp):
    gpe = ttnn.permute(w, (0, 3, 2, 1))
    return ttnn.sum(ttnn.multiply(per_expert, gpe), dim=1, keepdim=True)


def run(shard_experts: bool):
    if shard_experts:
        gw = ttnn.from_torch(gate_t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                             device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
        dw = ttnn.from_torch(down_t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                             device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
        e_local = E // NDEV
        w_local = ttnn.mesh_partition(weights, dim=-1)
        k_local = ttnn.mesh_partition(keep, dim=-1)
    else:
        # today: all experts, intermediate split four ways
        gw = ttnn.from_torch(gate_t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                             device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
        dw = ttnn.from_torch(down_t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                             device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-2))
        e_local, w_local, k_local = E, weights, keep

    def once():
        sparsity = ttnn.to_layout(
            ttnn.typecast(ttnn.max(k_local, dim=-2, keepdim=True), ttnn.bfloat16),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        per = expert_ffn(x, gw, gw, dw, sparsity, None, e_local, K, N)
        # the same collective the model uses after its MoE
        return ttnn.all_reduce(combine(per, w_local, e_local),
                               cluster_axis=1, topology=ttnn.Topology.Linear)

    return gw, dw, once


results = {}
for label, shard in (("intermediate (today)", False), ("expert axis (new)", True)):
    gw, dw, once = run(shard)
    try:
        out = ttnn.to_torch(once(), mesh_composer=comp)[:1].double()
    except Exception as exc:                                    # noqa: BLE001
        print(f"RESULT {label}: failed -- {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:130]}", flush=True)
        continue
    for _ in range(3):
        once()
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(9):
        t0 = time.perf_counter()
        once()
        ttnn.synchronize_device(mesh)
        ts.append(1000 * (time.perf_counter() - t0))
    ts.sort()
    results[label] = (out, ts[len(ts) // 2])
    print(f"RESULT {label:22s} {ts[len(ts) // 2]:7.3f} ms", flush=True)

if len(results) == 2:
    a, ta = results["intermediate (today)"]
    b, tb = results["expert axis (new)"]
    d = (a - b).abs()
    scale = a.abs().max().item() or 1.0
    print(f"RESULT max diff {d.max().item():.3e}  relative {d.max().item() / scale:.3e}  "
          f"speedup {ta / tb:5.2f}x", flush=True)

    # Which of the two is closer to the truth? They sum in different orders --
    # today four partial sums over quarter-width intermediates, the new one whole
    # experts at full width -- so they cannot agree exactly, and "differs" says
    # nothing on its own (invariant 20). float64 from the same operands does.
    xq = x_t.to(torch.bfloat16).double()[0, 0]
    exact = torch.zeros(M, K, dtype=torch.float64)
    gq = gate_t.to(torch.bfloat16).double()[0]
    dq = down_t.to(torch.bfloat16).double()[0]
    for e in range(E):
        rows = keep_t[0, 0, :, e].bool()
        if not rows.any():
            continue
        h = torch.nn.functional.silu(xq[rows] @ gq[e]) * (xq[rows] @ gq[e])
        contrib = h @ dq[e]
        exact[rows] += contrib * w_t[0, 0, rows, e][:, None].double()
    for label, (out, _) in results.items():
        err = (out.reshape(M, K) - exact).abs()
        print(f"RESULT {label:22s} vs float64: max {err.max().item():.4e}  "
              f"mean {err.mean().item():.4e}", flush=True)

ttnn.close_mesh_device(mesh)
