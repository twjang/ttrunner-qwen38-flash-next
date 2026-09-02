"""NEGATIVE control: two tiny traces alternate fine, whatever their program counts.

    uv run python scripts/dev/repro_trace_program_count.py

Self-contained -- no model, no weights, a few seconds. It was written to be the
filing-ready reproduction for the hang in 5.6 and it **does not reproduce it**,
which is worth as much: whatever the defect is, small graphs do not trigger it,
so any upstream report has to carry the model-scale harness
(`spec_capture_ladder.py`) rather than this.

It also falsifies the mechanism 5.6 briefly claimed. Reading
tt_metal/impl/trace/dispatch.cpp suggested a program-count mismatch --
`record_begin` -> `reset_host_dispatch_state_for_trace` zeroes the host
launch-message write pointer for capture, and
`FDMeshCommandQueue::enqueue_trace` -> `update_worker_state_post_trace_execution`
then *sets* it to the executed trace's own count -- so two traces of different
sizes would leave host and worker pointers disagreeing. At model scale that
predicted correctly (two step_n captures at k=2 and k=4 hang; the same graph
captured twice does not), but the control was confounded: two captures of one
graph share their program count *and* their kernel binaries. Here, 2 programs
against 5 alternates cleanly, so program count alone is not it.

A watchdog bounds the run so a hang reports rather than wedging the session.
"""
import threading
import time

import ttnn

BUDGET = 120.0
SHAPE = (1, 1, 32, 32)


def build(mesh, n_ops):
    """Capture a trace of `n_ops` elementwise programs over a fixed buffer."""
    src = ttnn.zeros(SHAPE, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    for _ in range(n_ops):                       # warm: allocate + JIT before capture
        src = ttnn.add(src, src)
    held = ttnn.zeros(SHAPE, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = held
    for _ in range(n_ops):
        out = ttnn.add(out, out)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    return tid


def main(out):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=64 << 20)
    try:
        for label, (a_ops, b_ops) in (
            ("same program count (2 and 2)", (2, 2)),
            ("different program counts (2 and 5)", (2, 5)),
        ):
            print(f"RESULT --- {label} ---", flush=True)
            a, b = build(mesh, a_ops), build(mesh, b_ops)
            ttnn.execute_trace(mesh, a, cq_id=0, blocking=True)
            print("RESULT   replayed A", flush=True)
            ttnn.execute_trace(mesh, b, cq_id=0, blocking=True)   # <-- hangs when sizes differ
            print("RESULT   replayed B", flush=True)
            ttnn.execute_trace(mesh, a, cq_id=0, blocking=True)
            print("RESULT   replayed A again -- OK", flush=True)
            ttnn.release_trace(mesh, a)
            ttnn.release_trace(mesh, b)
        out["ok"] = True
    except Exception as exc:
        out["error"] = " ".join(str(exc).split())[:200]
        print(f"RESULT FAILED {out['error']}", flush=True)
    finally:
        out["done"] = True


res: dict = {}
th = threading.Thread(target=main, args=(res,), daemon=True)
th.start()
deadline = time.time() + BUDGET
while time.time() < deadline and not res.get("done"):
    time.sleep(0.5)
if not res.get("done"):
    print(f"RESULT HUNG -- no progress in {BUDGET:.0f}s. The last line printed says which "
          "replay blocked; the board needs `tt-smi -r all`.", flush=True)
else:
    print(f"RESULT completed: {res}", flush=True)
