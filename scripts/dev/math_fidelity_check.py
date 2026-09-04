"""How much does HiFi4 cost on a 4-bit weight, and does it buy anything?

    uv run python scripts/dev/math_fidelity_check.py

The unpack of a bfloat4_b weight already happens in SRAM: DRAM -> L1 over the
NoC, then the Tensix unpacker feeds the compute registers from L1. There is
nowhere else to do it. What *is* a choice is how many math passes those unpacked
values get -- `MathFidelity.LoFi` is one pass, `HiFi4` is four -- and this model
uses HiFi4 everywhere "for consistency with every other matmul".

That is worth questioning for a weight with a 4-bit mantissa: the extra passes
exist to preserve bits the operand does not have. If LoFi gives the same answer
on a bfloat4_b weight, HiFi4 is paying up to 4x the math time for nothing.

Error is measured against the *dequantised* weight, not the original float, so
what is compared is the fidelity's contribution and not bfloat4_b's.
"""
import time

import torch
import ttnn

import os
K, M = 2560, 32
N = int(os.environ.get('N', '3072'))   # 3072 is floor-bound; use 32768 to reach bandwidth
LEVELS = [("LoFi", ttnn.MathFidelity.LoFi), ("HiFi2", ttnn.MathFidelity.HiFi2),
          ("HiFi3", ttnn.MathFidelity.HiFi3), ("HiFi4", ttnn.MathFidelity.HiFi4)]

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)
rep = ttnn.ReplicateTensorToMesh(mesh)
comp = ttnn.ConcatMeshToTensor(mesh, dim=0)

xt = torch.randn(1, 1, M, K) * 0.1
wt = torch.randn(1, 1, K, N) * 0.05
x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=mesh, mesh_mapper=rep)

for wname, wdtype in (("bfloat4_b", ttnn.bfloat4_b), ("bfloat8_b", ttnn.bfloat8_b),
                      ("bfloat16 ", ttnn.bfloat16)):
    w = ttnn.from_torch(wt, dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                        mesh_mapper=rep)
    # the exact product of what the device actually holds
    wq = ttnn.to_torch(w, mesh_composer=comp)[:1].double()[0, 0]
    exact = xt.to(torch.bfloat16).double()[0, 0] @ wq

    base = None
    for label, fid in LEVELS:
        kern = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fid, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        fn = lambda k=kern: ttnn.linear(x, w, compute_kernel_config=k)
        for _ in range(3):
            fn()
        ttnn.synchronize_device(mesh)
        best = None
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(30):
                fn()
            ttnn.synchronize_device(mesh)
            dt = (time.perf_counter() - t0) / 30
            best = dt if best is None else min(best, dt)
        got = ttnn.to_torch(fn(), mesh_composer=comp)[:1].double()[0, 0]
        err = (got - exact).abs().max().item()
        base = base or best
        print(f"RESULT {wname} {label:6s} {1000 * best:7.3f} ms   "
              f"{base / best:5.2f}x vs LoFi-ref   max err vs dequantised weight {err:.3e}",
              flush=True)
    ttnn.deallocate(w)

ttnn.close_mesh_device(mesh)
