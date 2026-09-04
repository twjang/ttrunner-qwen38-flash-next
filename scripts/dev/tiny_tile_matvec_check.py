"""Is there a matrix-vector path that skips the 32x row padding? No.

    uv run python scripts/dev/tiny_tile_matvec_check.py

ttnn has no `matvec` or `gemv`. What it has instead is a configurable tile:
`ttnn.Tile([1, 32])` and friends construct, and `linear`/`matmul` take an
`output_tile`. At decode every activation is one row and TILE_LAYOUT rounds it to
32, so a 1x32 tile looks like the obvious way to stop paying for 31 rows of
nothing.

It is not, for two independent reasons, and this file records both so the idea is
not retried:

1. **Below 16 rows it silently returns the wrong answer.** With a tiny
   activation tile and a normal 32x32 weight the op runs and produces no error:
   16x32 matches the exact product as well as 32x32 does (2.97e-03), and 8x32 and
   1x32 are off by ~0.9 -- not rounding, wrong. Every combination that matches the
   weight's tile to the activation's is rejected outright, and `sparse_matmul`
   refuses a tiny `output_tile` at all (`tiny_tile_check.py`).

2. **It would not help.** A dense matvec at M=1 is bound by reading the weight,
   not by the padded rows: [2560, 3072] of weights against one row of activation.
   Measured, every tile from 1x32 to 32x32 lands within noise of each other
   (0.130-0.138 ms), including the 16x32 that is correct.

Where the padding *did* cost was the MoE, because there it multiplies: E=512
slots x 32 padded rows x 2560 made an 84 MB tensor carrying 2.6 MB (`022`). That
is the one place a tiny tile would have paid, and it is the one place the op
refuses it.
"""
import time

import torch
import ttnn

from ttrunner_qwen38_flash_next.tt.ops import HIFI4

K, N = 2560, 3072          # the QSA q|gate projection, per device
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)
rep = ttnn.ReplicateTensorToMesh(mesh)
comp = ttnn.ConcatMeshToTensor(mesh, dim=0)

xt = torch.randn(1, 1, 1, K) * 0.1
wt = torch.randn(1, 1, K, N) * 0.05
exact = (xt.to(torch.bfloat16).double()[0, 0] @ wt.to(torch.bfloat16).double()[0, 0]).reshape(-1)
w32 = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                      mesh_mapper=rep)


def timeit(fn):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(30):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh)
        ts.append(1000 * (time.perf_counter() - t0))
    ts.sort()
    return ts[len(ts) // 2]


print(f"RESULT no matvec/gemv in ttnn; testing tiles on a {K}x{N} projection at M=1",
      flush=True)
for tile in ([32, 32], [16, 32], [8, 32], [1, 32]):
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                        mesh_mapper=rep, tile=ttnn.Tile(tile))
    fn = lambda x=x, t=tile: ttnn.linear(x, w32, compute_kernel_config=HIFI4,
                                         output_tile=ttnn.Tile(t))
    got = ttnn.to_torch(fn(), mesh_composer=comp)[:1].double().reshape(-1)[:N]
    err = (got - exact).abs().max().item()
    verdict = "ok" if err < 0.05 else "*** WRONG ***"
    print(f"RESULT tile={str(tile):9s} {timeit(fn):7.3f} ms   max err vs exact "
          f"{err:.4e}  {verdict}", flush=True)
    ttnn.deallocate(x)

print("RESULT --- matching the weight's tile to the activation's ---", flush=True)
for spec in ([1, 32], [8, 32]):
    try:
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=rep, tile=ttnn.Tile(spec))
        w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=rep, tile=ttnn.Tile(spec))
        ttnn.linear(x, w, compute_kernel_config=HIFI4, output_tile=ttnn.Tile(spec))
        print(f"RESULT weight tile={spec}: accepted", flush=True)
    except Exception as exc:                                      # noqa: BLE001
        print(f"RESULT weight tile={spec}: rejected -- {type(exc).__name__}", flush=True)

ttnn.close_mesh_device(mesh)
