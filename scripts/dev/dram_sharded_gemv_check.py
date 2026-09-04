"""Does the DRAM-sharded matmul beat the interleaved one for a decode GEMV?

    uv run python scripts/dev/dram_sharded_gemv_check.py

`022` measured that a dense projection at M=1 is bound by reading its weight, and
concluded the 32x row padding was irrelevant. That is about the *bytes*; it says
nothing about how well those bytes are fetched.
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` -- "a specialized config
for very narrow tensors stored in DRAM" -- exists precisely for this shape: the
weight is width-sharded across the 8 DRAM banks, the activation is held in L1,
and each core reduces its own slice.

Compares against the interleaved default on the projections decode actually runs,
for both correctness (float64) and time.
"""
import time

import torch
import ttnn

from ttrunner_qwen38_flash_next.tt.ops import HIFI4

SHAPES = [("qsa q|gate", 2560, 3072), ("qsa out", 1536, 2560), ("deltanet qkv", 2560, 2048)]
BANKS = 8
TILE = 32
import os
WDTYPE = ttnn.bfloat4_b if os.environ.get('W4', '1') == '1' else ttnn.bfloat16

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
torch.manual_seed(0)
rep = ttnn.ReplicateTensorToMesh(mesh)
comp = ttnn.ConcatMeshToTensor(mesh, dim=0)


def grid(n):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))})


BATCH = 60


def timeit(fn):
    """Throughput, not latency: enqueue a run of calls and synchronise once.

    Synchronising after every call measures dispatch, which is ~57-90 us on this
    build (invariant 23) and swamps a 137 us GEMV -- so a per-call timing cannot
    see a bandwidth difference at all. Inside a trace the dispatch is gone and
    the device is the bottleneck, which is what this approximates.
    """
    for _ in range(3):
        fn()
    ttnn.synchronize_device(mesh)
    best = None
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(BATCH):
            fn()
        ttnn.synchronize_device(mesh)
        dt = 1000 * (time.perf_counter() - t0) / BATCH
        best = dt if best is None else min(best, dt)
    return best


for label, K, N in SHAPES:
    xt = torch.randn(1, 1, TILE, K) * 0.1          # one real row, padded as decode is
    wt = torch.randn(1, 1, K, N) * 0.05
    exact = (xt.to(torch.bfloat16).double()[0, 0, :1] @ wt.to(torch.bfloat16).double()[0, 0])

    x_dram = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=mesh, mesh_mapper=rep)
    # bfloat4_b, as `plan.py` stores these weights. The dtype is the whole
    # question here: DRAM sharding is about fetching weight bytes well, and
    # bfloat4_b is a quarter of the bytes bfloat16 would be.
    w_dram = ttnn.from_torch(wt, dtype=WDTYPE, layout=ttnn.TILE_LAYOUT,
                             device=mesh, mesh_mapper=rep)
    base_fn = lambda: ttnn.linear(x_dram, w_dram, compute_kernel_config=HIFI4)
    base_ms = timeit(base_fn)
    base_out = ttnn.to_torch(base_fn(), mesh_composer=comp)[:1].double().reshape(TILE, N)[0]
    print(f"RESULT {label:13s} interleaved   {base_ms:7.3f} ms   "
          f"(vs float64, unquantised: {(base_out - exact).abs().max().item():.3e})", flush=True)

    # weight width-sharded across the DRAM banks; activation resident in L1
    try:
        w_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid(BANKS), [K, N // BANKS], ttnn.ShardOrientation.ROW_MAJOR),
        )
        w_sh = ttnn.to_memory_config(w_dram, w_mem)
        x_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(grid(BANKS), [TILE, K // BANKS], ttnn.ShardOrientation.ROW_MAJOR),
        )
        x_sh = ttnn.to_memory_config(x_dram, x_mem)
    except Exception as exc:                                       # noqa: BLE001
        print(f"RESULT {label:13s} sharding rejected -- {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:110]}", flush=True)
        continue

    for in0_block_w in (K // TILE // BANKS, 1):
        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w, per_core_M=1, per_core_N=N // TILE // BANKS,
            fused_activation=None,
        )
        out_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(grid(BANKS), [TILE, N // BANKS], ttnn.ShardOrientation.ROW_MAJOR),
        )
        try:
            fn = lambda: ttnn.linear(x_sh, w_sh, program_config=pc,
                                     memory_config=out_mem, compute_kernel_config=HIFI4)
            got = ttnn.to_torch(fn(), mesh_composer=comp)[:1].double().reshape(TILE, N)[0]
            # against the interleaved answer, not the float64 one: with
            # bfloat4_b weights both paths are ~0.1 off an unquantised reference
            # and that says nothing about either. Agreeing with each other does.
            drift = (got - base_out).abs().max().item()
            ms = timeit(fn)
            print(f"RESULT {label:13s} dram-sharded  {ms:7.3f} ms   "
                  f"vs interleaved {drift:.3e}  in0_block_w={in0_block_w}  "
                  f"{base_ms / ms:4.2f}x  {'same answer' if drift == 0 else 'DIFFERS'}",
                  flush=True)
        except Exception as exc:                                   # noqa: BLE001
            print(f"RESULT {label:13s} dram-sharded in0_block_w={in0_block_w} rejected -- "
                  f"{type(exc).__name__}: {str(exc).splitlines()[0][:110]}", flush=True)

ttnn.close_mesh_device(mesh)
