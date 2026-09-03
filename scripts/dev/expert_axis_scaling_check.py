"""Would sharding experts across devices pay, and by how much?

    uv run python scripts/dev/expert_axis_scaling_check.py

Ablation puts 51.7 ms of the 204 ms traced step in the expert down projection and
26.1 in gate|up. Both write [1, E, M, K] with E = 512, and at M = 1 TILE_LAYOUT
pads the row axis to 32, so the down output alone is 84 MB a layer to carry
2.6 MB. `output_tile` cannot shrink it (rejected by the op) and the consumer side
is already fixed (`_combine`), so the only remaining lever is a smaller E.

E is 512 on *every* device because the expert weights are EXPERT_COLUMN-sharded:
each card holds all 512 experts at a quarter of the intermediate width. Sharding
on the expert axis instead would give each card 128 whole experts -- same bytes of
weights, a quarter of the output tensor, and one full-width contraction per
expert rather than four partial sums to add up.

That is a cache rebuild plus a routing change, so measure the payoff first. If
the two matmuls scale with E, the saving is most of three quarters of 77.8 ms; if
they are flat, the refactor buys nothing and this file is why it was not done.
`topk` is held fixed so only E moves.
"""
import time

import torch
import ttnn

from twtest.tt.moe import sparse_program_config, HIFI4

M, N, K = 1, 160, 2560
TOPK = 12

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)
rep = ttnn.ReplicateTensorToMesh(mesh)
tile = dict(layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)


def timeit(fn):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(15):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh)
        ts.append(1000 * (time.perf_counter() - t0))
    ts.sort()
    return ts[len(ts) // 2]


print(f"RESULT M={M} K={K} topk={TOPK} (topk fixed)", flush=True)


def build(E, n_local, topk_local):
    x = ttnn.from_torch(torch.randn(1, 1, M, K) * 0.1, dtype=ttnn.bfloat16, **tile)
    gu = ttnn.from_torch(torch.randn(1, E, K, 2 * n_local) * 0.05, dtype=ttnn.bfloat16, **tile)
    h = ttnn.from_torch(torch.randn(1, E, M, n_local) * 0.1, dtype=ttnn.bfloat16, **tile)
    dw = ttnn.from_torch(torch.randn(1, E, n_local, K) * 0.05, dtype=ttnn.bfloat16, **tile)
    keep = torch.zeros(1, 1, 1, E)
    keep[..., torch.randperm(E)[:max(topk_local, 1)]] = 1.0
    sp = ttnn.to_layout(ttnn.from_torch(keep, dtype=ttnn.bfloat16, **tile),
                        ttnn.ROW_MAJOR_LAYOUT)

    def gate_up():
        return ttnn.sparse_matmul(
            x, gu, sparsity=sp, nnz=None, is_input_a_sparse=False, is_input_b_sparse=True,
            program_config=sparse_program_config(M, K, 2 * n_local),
            compute_kernel_config=HIFI4)

    def down():
        return ttnn.sparse_matmul(
            h, dw, sparsity=sp, nnz=None, is_input_a_sparse=True, is_input_b_sparse=True,
            program_config=sparse_program_config(M, n_local, K),
            compute_kernel_config=HIFI4)

    return gate_up, down, (x, gu, h, dw, sp)


# The two real configurations, not an E sweep at fixed width. Sharding experts
# instead of the intermediate keeps the bytes of weights and the FLOPs per device
# identical -- a quarter of the experts, each at full width -- and only the output
# tensor changes: down goes [1, 512, M, 2560] -> [1, 128, M, 2560], a quarter,
# while gate|up stays the same size (E falls 4x, the output width rises 4x).
for label, E, n_local, tk in (
    ("now:    intermediate-sharded, E=512 x N=160", 512, 160, TOPK),
    ("would be: expert-sharded,     E=128 x N=640", 128, 640, max(TOPK // 4, 1)),
):
    gate_up, down, bufs = build(E, n_local, tk)
    g, d = timeit(gate_up), timeit(down)
    print(f"RESULT {label}   gate|up {g:7.3f} ms   down {d:7.3f} ms   "
          f"sum {g + d:7.3f} ms", flush=True)
    for t in bufs:
        ttnn.deallocate(t)

ttnn.close_mesh_device(mesh)
