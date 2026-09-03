"""Is a prefill chunk's expert FFN cheaper in 64-row groups?

    uv run python scripts/dev/expert_split_check.py

`expert_ffn` is exactly per-token -- a row's result does not depend on which
rows it is grouped with, once `sparse_program_config` gives it a single K block
above one row tile (handoff 5.8). So the group width is free to choose on speed
alone, which is not true of routing.

That matters because the two sparse_matmul modes cross over at 64 rows: the
broadcast mode (no per-expert copy of the input) wins 1.8-2.35x up to M=64 and
loses at M=128. A 128-row prefill chunk currently takes the losing side. Running
it as two 64-row groups would put both on the winning side, at the cost of one
extra set of dispatches per layer on a path that is dispatch-bound.

Measures the whole FFN -- gate|up, slices, silu, multiply, down -- both ways, and
checks the split really is bit-identical rather than trusting the claim.
"""
import time

import torch
import ttnn

from twtest.tt.moe import sparse_program_config, HIFI4

E, K, N = 512, 2560, 160
TOPK = 64          # a 128-row chunk selects a wide union
MS = (128, 256)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)
rep = ttnn.ReplicateTensorToMesh(mesh)
tile = dict(layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

# bfloat4_b, as `plan.py` stores the expert weights. With bfloat16 here the
# replicated gate|up alone is 838 MB and M=128 dies in program.cpp:1555.
gu = ttnn.from_torch(torch.randn(1, E, K, 2 * N) * 0.05, dtype=ttnn.bfloat4_b, **tile)
dw = ttnn.from_torch(torch.randn(1, E, N, K) * 0.05, dtype=ttnn.bfloat4_b, **tile)
keep = torch.zeros(1, 1, 1, E)
keep[..., torch.randperm(E)[:TOPK]] = 1.0
sp = ttnn.to_layout(ttnn.from_torch(keep, dtype=ttnn.bfloat16, **tile), ttnn.ROW_MAJOR_LAYOUT)


def ffn(x, broadcast):
    m = x.shape[-2]
    if broadcast:
        both = ttnn.reshape(ttnn.sparse_matmul(
            x, gu, sparsity=sp, nnz=None, is_input_a_sparse=False, is_input_b_sparse=True,
            program_config=sparse_program_config(m, K, 2 * N),
            compute_kernel_config=HIFI4), (1, E, m, 2 * N))
    else:
        both = ttnn.sparse_matmul(
            ttnn.repeat(x, (1, E, 1, 1)), gu, sparsity=sp, nnz=None,
            is_input_a_sparse=True, is_input_b_sparse=True,
            program_config=sparse_program_config(m, K, 2 * N),
            compute_kernel_config=HIFI4)
    gate = ttnn.slice(both, (0, 0, 0, 0), (1, E, m, N))
    up = ttnn.slice(both, (0, 0, 0, N), (1, E, m, 2 * N))
    hidden = ttnn.multiply(ttnn.silu(gate), up)
    return ttnn.sparse_matmul(
        hidden, dw, sparsity=sp, nnz=None, is_input_a_sparse=True, is_input_b_sparse=True,
        program_config=sparse_program_config(m, N, K), compute_kernel_config=HIFI4)


def timeit(fn):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(9):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh)
        ts.append(1000 * (time.perf_counter() - t0))
    ts.sort()
    return ts[len(ts) // 2]


comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
for M in MS:
    x = ttnn.from_torch(torch.randn(1, 1, M, K) * 0.1, dtype=ttnn.bfloat16, **tile)

    def whole():
        return ffn(x, broadcast=False)

    def split(width=64):
        outs = []
        for lo in range(0, M, width):
            xs = ttnn.slice(x, (0, 0, lo, 0), (1, 1, lo + width, K))
            outs.append(ffn(xs, broadcast=True))
        return ttnn.concat(outs, dim=2)

    a = ttnn.to_torch(whole(), mesh_composer=comp)[:1]
    b = ttnn.to_torch(split(), mesh_composer=comp)[:1]
    same = torch.equal(a, b)
    d = (a - b).abs()
    w, s = timeit(whole), timeit(lambda: split())
    print(f"RESULT M={M:4d}  whole {w:7.3f} ms   64-row split {s:7.3f} ms   "
          f"{w / s:4.2f}x   bit-identical {'YES' if same else 'NO'} "
          f"(max diff {d.max().item():.3e})", flush=True)
    ttnn.deallocate(x)

ttnn.close_mesh_device(mesh)
