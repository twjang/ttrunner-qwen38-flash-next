"""Can the expert FFN drop its `repeat`, and is the result bit-identical?

    uv run python scripts/dev/sparse_broadcast_check.py [E]      (default 512)

Sweeps M, because the answer is not the same at both ends. Decode is M = 1,
where the copy is 32 rows of padding per real row; a prefill chunk is M = 128,
where it is not padding at all and the broadcast mode has to fan the same rows
out to every expert instead. The crossover decides which form each caller gets.

`expert_ffn` feeds `sparse_matmul` in its (a sparse, b sparse) mode, which wants
`input_tensor_a` already laid out per expert -- hence
`ttnn.repeat(x, (1, E, 1, 1))`. At decode M is 1, and TILE_LAYOUT pads the row
axis to 32, so that repeat materialises [1, 512, 32, 2560] of which 1 row in 32
is real: ~84 MB a layer, 48 layers a token, to hold 512 copies of one row.

The op's own modes table offers (a dense, b sparse): `input_tensor_a` [A, B, M, K]
against [1, E, K, N] with sparsity [A, B, 1, E], giving [A, B, 1, E, M, N]. With
A = B = 1 that is exactly this computation with no copy at all.

Same arithmetic per expert, so it should be bit-identical -- "should" being the
word that has been wrong often enough on this project to be worth a check.
Compares every element, not row 0 (013's lesson), and times both.
"""
import time

import torch
import ttnn

from _device_model import GGUF_DIR  # noqa: F401  (keeps env contract in one place)
from twtest.tt.moe import sparse_program_config, HIFI4

import sys
E = int(sys.argv[1]) if len(sys.argv) > 1 else 512
K, N = 2560, 320
MS = (1, 8, 32, 64, 128)
TOPK = 12

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)

w_t = torch.randn(1, E, K, N, dtype=torch.float32) * 0.05
keep = torch.zeros(1, 1, 1, E)
keep[..., torch.randperm(E)[:TOPK]] = 1.0

rep = ttnn.ReplicateTensorToMesh(mesh)
tile = dict(layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
w = ttnn.from_torch(w_t, dtype=ttnn.bfloat16, **tile)
sparsity = ttnn.to_layout(
    ttnn.from_torch(keep, dtype=ttnn.bfloat16, **tile), ttnn.ROW_MAJOR_LAYOUT)

print(f"RESULT E={E} K={K} N={N} topk={TOPK}", flush=True)
for M in MS:
    x_t = torch.randn(1, 1, M, K, dtype=torch.float32) * 0.1
    x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, **tile)
    pc = sparse_program_config(M, K, N)

    def with_repeat():
        b = ttnn.repeat(x, (1, E, 1, 1))
        return ttnn.sparse_matmul(
            b, w, sparsity=sparsity, nnz=None, is_input_a_sparse=True,
            is_input_b_sparse=True, program_config=pc, compute_kernel_config=HIFI4)

    def without_repeat():
        out = ttnn.sparse_matmul(
            x, w, sparsity=sparsity, nnz=None, is_input_a_sparse=False,
            is_input_b_sparse=True, program_config=pc, compute_kernel_config=HIFI4)
        return ttnn.reshape(out, (1, E, M, N))

    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    a = ttnn.to_torch(with_repeat(), mesh_composer=comp)[:1]
    b = ttnn.to_torch(without_repeat(), mesh_composer=comp)[:1]
    same = torch.equal(a, b)

    times = {}
    for label, fn in (("repeat", with_repeat), ("broadcast", without_repeat)):
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
        times[label] = ts[len(ts) // 2]
    r, bc = times["repeat"], times["broadcast"]
    print(f"RESULT M={M:4d}  repeat {r:7.3f} ms   broadcast {bc:7.3f} ms   "
          f"{r / bc:5.2f}x   identical {'YES' if same else 'NO'}", flush=True)

ttnn.close_mesh_device(mesh)
