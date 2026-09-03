"""A self-contained attempt at this bug that does NOT reproduce it.

    python minimal_attempt_does_not_reproduce.py [ops] [rows_a] [rows_b]
                                                   default: 64 32 64

Kept because a negative control is worth as much as the reproducer: it says
what a minimal case fails to capture, so nobody repeats the search.

It builds two traces holding the same chain of ops at two *different* row
counts -- which is the shape difference `repro.py` shows to be the trigger at
model scale -- captures both, and replays A, B, A. It completes every time.

Axes tried, all clean:

    ops per trace     64, 128, 1024
    rows              32 vs 64 (one row tile against two)
    `all_reduce`      on and off (REPRO_NO_COLLECTIVE=1)
    in-place write    on and off (REPRO_NO_INPLACE=1), into a buffer outliving
                      the trace, as the real graph does for its K/V cache

So a chain of one op kind, at up to 1024 programs, with a collective and a
persistent in-place write, is not enough. What the failing traces have that this
does not: ~5000 programs each, many op kinds (sparse_matmul over 512 experts,
sdpa_decode, topk, rms_norm, softmax, concat, permute), height-sharded L1
buffers, and tens of megabytes of recorded commands. Narrowing further from
here means starting at `repro.py` and removing, not starting here and adding.
"""
import os
import sys
import threading
import time

import ttnn

OPS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
COLLECTIVE = os.environ.get("REPRO_NO_COLLECTIVE") is None
INPLACE = os.environ.get("REPRO_NO_INPLACE") is None
ROWS_A = int(sys.argv[2]) if len(sys.argv) > 2 else 32
ROWS_B = int(sys.argv[3]) if len(sys.argv) > 3 else 64
K = 256
BUDGET = 180.0
KERNEL = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


def chain(x, w, n, store=None):
    """n blocks of matmul + all_reduce, mirroring one layer of the real graph.

    The collective is here because it is the piece the failing model traces have
    that a chain of local ops does not: every layer ends in an `all_reduce` over
    the four devices, and cross-device bookkeeping is the plausible place for a
    trace's recorded state to be invalidated by another trace.
    """
    for _ in range(n):
        x = ttnn.matmul(x, w, compute_kernel_config=KERNEL)
        if COLLECTIVE:
            x = ttnn.all_reduce(x, cluster_axis=1, topology=ttnn.Topology.Linear)
        if store is not None:
            # In-place write into a buffer that outlives the trace, as the real
            # graph does for its K/V cache and its convolution rings.
            ttnn.copy(x, store)
    return x


def build(mesh, rows):
    """Warm the graph, then capture it. Returns (trace_id, input_buffer)."""
    w = ttnn.from_torch_like if False else None  # noqa: F841  (kept import-free)
    weight = ttnn.ones((1, 1, K, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    held = ttnn.ones((1, 1, rows, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    store = ttnn.zeros((1, 1, rows, K), dtype=ttnn.bfloat16,
                       layout=ttnn.TILE_LAYOUT, device=mesh) if INPLACE else None
    chain(held, weight, OPS, store)          # warm: a capture cannot load binaries
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    chain(held, weight, OPS, store)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    return tid


def main(out):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 << 20)
    try:
        print(f"[repro] {OPS} matmuls per trace, {ROWS_A} rows vs {ROWS_B} rows", flush=True)
        a = build(mesh, ROWS_A)
        print("[repro] captured A", flush=True)
        b = build(mesh, ROWS_B)
        print("[repro] captured B", flush=True)

        ttnn.execute_trace(mesh, a, cq_id=0, blocking=True)
        print("[repro] replayed A", flush=True)
        ttnn.execute_trace(mesh, b, cq_id=0, blocking=True)   # <-- hangs here
        print("[repro] replayed B", flush=True)
        ttnn.execute_trace(mesh, a, cq_id=0, blocking=True)
        print("[repro] replayed A again -- NO HANG", flush=True)
        out["ok"] = True
    except Exception as exc:
        out["error"] = " ".join(str(exc).split())[:300]
        print(f"[repro] FAILED {out['error']}", flush=True)
    finally:
        out["done"] = True


res: dict = {}
threading.Thread(target=main, args=(res,), daemon=True).start()
deadline = time.time() + BUDGET
while time.time() < deadline and not res.get("done"):
    time.sleep(0.5)
if not res.get("done"):
    print(f"[repro] HUNG -- no progress in {BUDGET:.0f}s. The last line printed says "
          "which replay blocked. The board needs `tt-smi -r all`.", flush=True)
else:
    print(f"[repro] finished: {res}", flush=True)
