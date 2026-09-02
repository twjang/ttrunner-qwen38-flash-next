"""Which step of the engine's setup hangs the capture? One rung per run.

    uv run python scripts/dev/spec_capture_ladder.py <stage> [k]      (k = 2)

`traced_step_n_check.py` performs the same two captures on the same state and
never hangs; `TTEngine(speculate=k)` does, and the boards then need `tt-smi -r`.
Ten causes are excluded in `docs/iterations/016`, so this walks the remaining
difference one element at a time rather than proposing an eleventh.

The engine's `_device_loop` sets up in this order:

    1  state = new_state(batch=1)
    2  model.step([0], state)                 warm, so layer state exists
    3  snap = model.snapshot(state)           ~200 tensors        <- not in the harness
    4  TracedDecoder(model, state)            capture #1
    5  TracedStepN(model, state, k)           capture #2

Stages:
    baseline   1 2 4 5        -- what the harness already does; must pass
    snapshot   1 2 3 4 5      -- adds only the snapshot allocation
    thread     snapshot, on a worker thread
    no_replay  snapshot, but never replay capture #1 before taking capture #2
    both_replay  capture both, replay the decoder, then replay step_n  -- HANGS
    stepn_only   capture both, replay *only* step_n: is the trigger the second
                 live trace, or the interleaving?
    two_stepn    two step_n captures (k=2 and k=4), replayed alternately. Is the
                 defect specific to decoder-vs-step_n, or general to any two
                 traces? If general, the traced chunked prefill in 5.2 is
                 blocked by the same thing, since it would alternate a prefill
                 replay with the decoder's.
    two_same     two captures of the *same* graph (step_n at the same k), so the
                 two traces have identical program counts, replayed alternately.
                 Tests the mechanism the tt-metal source implies: `enqueue_trace`
                 ends in `update_worker_state_post_trace_execution`, which *sets*
                 the host launch-message wptr to that trace's program count,
                 while `record_begin` reset it to 0 for the capture. Two traces
                 with different counts therefore leave host and worker pointers
                 desynchronised -- a hang. Equal counts should not.
    verify_cq    the whole point: with both traces live and the decoder replayed
                 in between, does the step_n replay produce the *right tokens*,
                 and how long does it take? "It did not hang" is not a fix -- a
                 trace that silently does nothing also does not hang, and the
                 engine's verify came back in 9.6 ms where a standalone replay
                 costs 255, accepting 0 of every 5 drafted tokens.

`both_replay` is the one the stack dump pointed at: the engine hangs inside
`ttnn.execute_trace` for the step_n graph (traced.py:130), not during capture,
and `traced_step_n_check.py` never replays step_n while the decoder's trace is
also live -- it releases both first, then replays step_n alone on a fresh state.

`no_replay` is the difference that survived the rungs above. Every harness that
works -- `traced_step_n_check.py`, `traced_step_n_thread.py` -- replays the
decoder several times between the two captures. `_device_loop` constructs
`TracedDecoder`, calls `reset()`, and goes straight to `TracedStepN` without
ever replaying it.

Evidence pointing here: the hung run's last line is metal's own
"Allocating device buffers is unsafe due to the existence of an active trace",
and the process then spins rather than blocking. The snapshot is the one large
allocation the engine makes that the harness does not, and it competes with a
second trace region for the ~7 GB left beside 24.94 GB of weights.

Bounded on purpose: a watchdog reports rather than hanging the session. If it
does hang, the boards need `tt-smi -r all` before the next run.
"""
import os
import sys
import threading
import time

import ttnn

from _device_model import open_model, synthetic_prompt

from twtest.tt.traced import TracedDecoder, TracedStepN

STAGE = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
BUDGET = 420.0
STAGES = ("baseline", "snapshot", "thread", "no_replay", "both_replay", "stepn_only", "verify_cq", "two_stepn", "two_same")
if STAGE not in STAGES:
    raise SystemExit(f"stage must be one of {STAGES}, got {STAGE}")

# The engine that hangs runs at 2048, the harness that works at 512 -- and the
# hung run's last line is an allocation warning, so headroom is a variable, not
# a detail. 24.94 GB of weights leaves ~7 GB, and two trace regions plus a
# 4x-larger K/V compete for it.
SEQ = int(os.environ.get("TWTEST_MAX_SEQ", "512"))
# TWTEST_TRACE_REGION_MB mirrors what `TTEngine` passes and no harness ever has:
# the engine opens its mesh with `trace_region_size=(len(widths) + 1) * 128 MB`
# -- 256 MB at speculate=2 -- while `open_model` passes nothing and takes ttnn's
# default. Unset means the default, i.e. what the working harnesses do.
REGION = os.environ.get("TWTEST_TRACE_REGION_MB")
region = int(REGION) * (1 << 20) if REGION else None
print(f"RESULT stage {STAGE} k={K} max_seq_len {SEQ} "
      f"trace_region {REGION + ' MB' if REGION else 'default'}", flush=True)
# TWTEST_STEPN_CQ=1 puts the step_n capture on its own command queue.
STEPN_CQ = int(os.environ.get("TWTEST_STEPN_CQ", "0"))
mesh, cfg, m = open_model(
    max_seq_len=SEQ, trace_region_bytes=region,
    num_command_queues=2 if STEPN_CQ else None,
)
if STEPN_CQ:
    print(f"RESULT step_n capture on cq {STEPN_CQ}", flush=True)
prompt = synthetic_prompt(8 + K)
done = threading.Event()
out: dict = {}


def mem(tag):
    """Best-effort free-DRAM reading; the API name has moved around."""
    for attr in ("get_memory_view", "dram_memory_view"):
        try:
            view = getattr(mesh, attr)(ttnn.BufferType.DRAM)
            print(f"RESULT mem[{tag}] {view}", flush=True)
            return
        except Exception:
            continue
    print(f"RESULT mem[{tag}] unavailable", flush=True)


def rewind(st):
    """Undo everything the warmups and the capture consumed."""
    for layer in st.layers:
        for name in ("recurrent", "conv", "ple_conv", "keys", "values"):
            buf = getattr(layer, name)
            if buf is None:
                continue
            for entry in (buf if isinstance(buf, list) else [buf]):
                ttnn.copy(ttnn.zeros(list(entry.shape), dtype=entry.dtype,
                                     layout=entry.layout, device=mesh), entry)
        layer.conv_step = 0
        layer.ple_step = 0
    st.positions = [0]
    st.histories = [[]]


def verify_cq():
    """Correctness and cost of a cq-`STEPN_CQ` step_n replay, decoder trace live."""
    draft = prompt[8 : 8 + K]

    st = m.new_state(batch=1)
    for t in prompt[:8]:
        m.step([t], st)
    want = m.greedy_tokens(m.step_n(draft, st))[:K]
    print(f"RESULT eager step_n tokens {want}", flush=True)
    del st

    st = m.new_state(batch=1)
    print("RESULT capturing the decoder", flush=True)
    dec = TracedDecoder(m, st)
    dec.reset()
    print(f"RESULT capturing step_n on cq {STEPN_CQ}", flush=True)
    tr = TracedStepN(m, st, K, cq_id=STEPN_CQ)
    print("RESULT both captured; rewinding", flush=True)
    rewind(st)
    dec.reset()
    rewind(st)
    # warm through the *decoder's trace*, so both traces have run this round --
    # the interleaving that hangs on one queue. TWTEST_NO_DEC_REPLAY=1 warms
    # eagerly instead: the control that says whether this harness can produce a
    # MATCH at all, since the interleaved flow either hangs (cq 0) or comes back
    # wrong (cq 1) and neither proves the fixture is sound.
    if os.environ.get("TWTEST_NO_DEC_REPLAY"):
        print("RESULT warming eagerly (no decoder replay) -- control", flush=True)
        for t in prompt[:8]:
            m.step([t], st)
    else:
        for t in prompt[:8]:
            dec.step([t])
    print("RESULT replaying step_n", flush=True)
    t0 = time.perf_counter()
    got = m.greedy_tokens(tr.step_n(draft))[:K]
    ttnn.synchronize_device(mesh)
    ms = 1000 * (time.perf_counter() - t0)
    print(f"RESULT traced step_n tokens {got}   {ms:.1f} ms", flush=True)
    print(f"RESULT {'MATCH' if got == want else 'DIFFER'} "
          f"-- a replay that does nothing would be fast and wrong", flush=True)
    out["match"] = got == want
    out["ms"] = ms
    tr.release()
    dec.release()


def two_stepn():
    """Two distinct step_n graphs, replayed alternately. No decoder involved."""
    toks = synthetic_prompt(32)          # long enough for 8 warm + 2 + 4 + 2
    st = m.new_state(batch=1)
    for t in toks[:8]:
        m.step([t], st)
    print("RESULT capturing step_n k=2", flush=True)
    a = TracedStepN(m, st, 2)
    print("RESULT capturing step_n k=4", flush=True)
    b = TracedStepN(m, st, 4)
    print("RESULT both captured; replaying k=2", flush=True)
    a.step_n(toks[8:10])
    if os.environ.get("TWTEST_EAGER_BETWEEN"):
        # The mechanism says the *host* launch-message pointer is left at the
        # executed trace's program count while the next trace was recorded
        # against zero. Ordinary (non-trace) dispatch maintains those pointers
        # through the normal path, so a real program in between might resync
        # them. If it does, both 5.2's traced prefill and 5.6 are unblocked.
        print("RESULT running an eager program between the replays", flush=True)
        z = ttnn.zeros((1, 1, 32, 32), dtype=ttnn.bfloat16,
                       layout=ttnn.TILE_LAYOUT, device=mesh)
        ttnn.add(z, z)
        ttnn.synchronize_device(mesh)
        print("RESULT eager program done", flush=True)
    print("RESULT k=2 replayed; replaying k=4  <-- the alternation", flush=True)
    b.step_n(toks[10:14])
    print("RESULT k=4 replayed; replaying k=2 again", flush=True)
    a.step_n(toks[14:16])
    print("RESULT alternated cleanly -- the defect is NOT general", flush=True)
    out["general"] = False
    a.release()
    b.release()


def two_same():
    """Two captures of one graph: distinct traces, identical program counts."""
    toks = synthetic_prompt(32)
    st = m.new_state(batch=1)
    for t in toks[:8]:
        m.step([t], st)
    print(f"RESULT capturing step_n k={K} (first)", flush=True)
    a = TracedStepN(m, st, K)
    print(f"RESULT capturing step_n k={K} (second, same graph)", flush=True)
    b = TracedStepN(m, st, K)
    print("RESULT both captured; replaying the first", flush=True)
    a.step_n(toks[8 : 8 + K])
    print("RESULT replaying the second  <-- the alternation, equal program counts",
          flush=True)
    b.step_n(toks[8 + K : 8 + 2 * K])
    print("RESULT back to the first", flush=True)
    a.step_n(toks[8 + 2 * K : 8 + 3 * K])
    print("RESULT alternated cleanly -- equal program counts do NOT hang, so the "
          "defect is the count mismatch", flush=True)
    a.release()
    b.release()


def work():
    try:
        if STAGE == "verify_cq":
            verify_cq()
            out["ok"] = True
            return
        if STAGE == "two_stepn":
            two_stepn()
            out["ok"] = True
            return
        if STAGE == "two_same":
            two_same()
            out["ok"] = True
            return
        st = m.new_state(batch=1)
        print(f"RESULT [{STAGE}] warming", flush=True)
        for t in prompt[:8]:
            m.step([t], st)
        mem("after warmup")

        if STAGE in ("snapshot", "thread"):
            print("RESULT allocating the snapshot buffers", flush=True)
            t0 = time.perf_counter()
            snap = m.snapshot(st)
            n = sum(1 for _ in (snap if isinstance(snap, (list, tuple)) else [snap]))
            print(f"RESULT snapshot allocated ({n} top-level) in "
                  f"{1000 * (time.perf_counter() - t0):.0f} ms", flush=True)
            out["snap"] = snap
            mem("after snapshot")

        print("RESULT capturing the decoder", flush=True)
        dec = TracedDecoder(m, st)
        dec.reset()
        print("RESULT decoder captured", flush=True)
        if STAGE == "no_replay":
            print("RESULT skipping the decoder replay (what the engine does)", flush=True)
        else:
            for t in prompt[:8]:
                dec.step([t])
        mem("after decoder capture")

        print(f"RESULT capturing step_n k={K}  <-- the engine hangs here", flush=True)
        tr = TracedStepN(m, st, K, cq_id=STEPN_CQ)
        print("RESULT step_n captured", flush=True)
        mem("after step_n capture")

        if STAGE == "stepn_only":
            print("RESULT replaying step_n only (decoder captured but never replayed)",
                  flush=True)
            tr.step_n(prompt[8 : 8 + K])
            print("RESULT step_n replayed", flush=True)

        if STAGE == "both_replay":
            # Replay each trace with the other still live -- the combination the
            # working harness skips and the engine relies on.
            print("RESULT replaying the decoder (both traces live)", flush=True)
            dec.step([1000])
            print("RESULT decoder replayed", flush=True)
            # TWTEST_SYNC_BETWEEN=1: does a device sync between the two replays
            # clear it? Both `execute_trace` calls are already blocking=True on
            # cq 0, so if this helps, "blocking" is not settling what the next
            # trace needs settled.
            if os.environ.get("TWTEST_SYNC_BETWEEN"):
                print("RESULT synchronizing between the replays", flush=True)
                ttnn.synchronize_device(mesh)
            print("RESULT replaying step_n (both traces live)  <-- the stack dump's line",
                  flush=True)
            tr.step_n(prompt[8 : 8 + K])
            print("RESULT step_n replayed", flush=True)

        tr.release()
        dec.release()
        out["ok"] = True
    except Exception as exc:
        out["error"] = " ".join(str(exc).split())[:300]
        print(f"RESULT FAILED {out['error']}", flush=True)
    finally:
        done.set()


if STAGE == "thread":
    threading.Thread(target=work, name="device", daemon=True).start()
    finished = done.wait(BUDGET)
else:
    threading.Thread(target=work, name="device", daemon=True).start()
    finished = done.wait(BUDGET)

if not finished:
    print(f"RESULT HUNG in stage '{STAGE}' -- nothing completed in {BUDGET:.0f}s", flush=True)
    print("RESULT this rung reproduces the engine's hang; boards need tt-smi -r all", flush=True)
else:
    print(f"RESULT stage '{STAGE}' completed: {'ok' if out.get('ok') else out}", flush=True)
    ttnn.close_mesh_device(mesh)
