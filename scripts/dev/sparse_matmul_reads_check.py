"""Does `sparse_matmul` actually avoid reading the experts it skips?

    uv run python scripts/dev/sparse_matmul_reads_check.py

The arithmetic that prompts this: 4x p150a have aggregate DRAM bandwidth in the
same class as a top-end consumer GPU, and the weights a decode step genuinely
needs are ~1 GB per device -- the dense projections plus the ~11.5 experts of 512
that routing selects. At the 273-393 GB/s these boards demonstrably stream
(`gemv_bandwidth_check.py`), that is a couple of milliseconds. The step takes 173.

So either the bandwidth is not being used, or something is being read that does
not need to be. `expert_axis_scaling_check.py` already hinted at the second:
dropping E from 512 to 128 made the down projection 2.8x faster, which is
scaling with the *total* expert count rather than the selected one.

This isolates it. One weight, one shape, only the number of non-zeros in the
sparsity mask changes:

* time proportional to nnz -> the op reads only what it needs, and decode's cost
  is somewhere else entirely.
* time flat in nnz -> it streams all 512 experts every token, and that single
  fact accounts for the gap.
"""
import time

import torch
import ttnn

from ttrunner_qwen38_flash_next.tt.moe import sparse_program_config, HIFI4

E, M, N, K = 512, 32, 160, 2560          # the down projection, per device
NNZS = (1, 2, 8, 32, 128, 512)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)
rep = ttnn.ReplicateTensorToMesh(mesh)
tile = dict(layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

h = ttnn.from_torch(torch.randn(1, E, M, N) * 0.1, dtype=ttnn.bfloat16, **tile)
dw = ttnn.from_torch(torch.randn(1, E, N, K) * 0.05, dtype=ttnn.bfloat4_b, **tile)
w_mb = E * N * K * 0.5625 / 1e6
pc = sparse_program_config(M, N, K)
print(f"RESULT down projection: E={E} N={N} K={K}, all experts = {w_mb:.1f} MB/device",
      flush=True)
print("RESULT  nnz    time    if it read only nnz    实 GB/s (all E)", flush=True)

base = None
for nnz in NNZS:
    keep = torch.zeros(1, 1, 1, E)
    keep[..., :nnz] = 1.0
    sp = ttnn.to_layout(ttnn.from_torch(keep, dtype=ttnn.bfloat16, **tile),
                        ttnn.ROW_MAJOR_LAYOUT)

    def fn():
        return ttnn.sparse_matmul(h, dw, sparsity=sp, nnz=None, is_input_a_sparse=True,
                                  is_input_b_sparse=True, program_config=pc,
                                  compute_kernel_config=HIFI4)

    for _ in range(3):
        fn()
    ttnn.synchronize_device(mesh)
    best = None
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(20):
            fn()
        ttnn.synchronize_device(mesh)
        dt = (time.perf_counter() - t0) / 20
        best = dt if best is None else min(best, dt)
    base = base or best
    ideal = w_mb * nnz / E
    print(f"RESULT {nnz:5d}  {1000 * best:7.3f} ms   would be {ideal:7.2f} MB   "
          f"{w_mb / 1e3 / best:7.1f}   {best / base:5.2f}x vs nnz=1", flush=True)
    ttnn.deallocate(sp)

ttnn.close_mesh_device(mesh)
