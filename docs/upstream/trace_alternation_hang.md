# Alternating replays of two captured traces hangs the device

**Status:** ready to file. Not filed — that is the maintainer's call, and nothing
here has been sent anywhere.

**Affects:** `pjrt-plugin-tt` 1.4.0 (bundled tt-metal), 4× Blackhole p150a,
`FabricConfig::FABRIC_1D`, `MeshShape(1, 4)`.

**Impact for us:** blocks two features outright — a traced speculative verifier
alongside a traced decode step, and a traced chunked prefill alongside a traced
decode step. Both are the same pattern: two captures, replayed in alternation.
See `docs/HANDOFF.md` §5.2 and §5.6.

## Summary

With two traces captured on the same mesh device, replaying **trace A, then
trace B** hangs inside `ttnn.execute_trace` (blocking=True, cq 0). The call never
returns; the process spins and the boards need `tt-smi -r all` afterwards. There
is no error and no log line.

The following all work, which is what makes it specific:

| | |
|---|---|
| replay one trace repeatedly, any number of times | works |
| capture A and B, then replay **only** B | works |
| capture A and B, replay A then B | **hangs** |
| capture the *same graph* twice as A and B, replay A, B, A | works |
| A and B the same graph, B with one extra op (kernel already present) | works |
| A and B the same graph, B with one extra op introducing a **new** kernel | works |
| A and B the same graph, B with **50** extra ops appended | works |
| two *small* traces, any program counts, alternated | works |
| replay A many times, **release both traces**, capture C, replay C | **hangs** |

| `step_n` at k=2 and k=**3** — adjacent widths | **hangs** |

The controls draw a sharp line, and it is not about how *different* the graphs
are or by how much. Trace B may be trace A **plus appended operations** — one or
fifty, including ones that introduce a kernel A never uses — and the pair
alternates fine. What hangs is the two traces holding **differently-shaped
versions of the same operations**, and k=2 against k=3 is enough of a difference
to do it: `k` is unrolled into the recurrence and convolution, so changing it
re-shapes every layer rather than appending to it.

Put the other way: **B ⊇ A is safe; B and A disagreeing about a shape is not.**

## Reproduction

Model-scale, which is the only scale that reproduces it. From this repository:

```
# hangs, ~7 min to reach the hang, then the watchdog reports
PYTHONPATH=src:scripts/dev TWTEST_MAX_SEQ=2048 TWTEST_TRACE_REGION_MB=384 \
  uv run python scripts/dev/spec_capture_ladder.py two_stepn 2

# the same two captures, replaying only the second: clean
PYTHONPATH=src:scripts/dev TWTEST_MAX_SEQ=2048 TWTEST_TRACE_REGION_MB=384 \
  uv run python scripts/dev/spec_capture_ladder.py stepn_only 2

# two captures of one graph, alternated A/B/A: clean
PYTHONPATH=src:scripts/dev TWTEST_MAX_SEQ=2048 TWTEST_TRACE_REGION_MB=384 \
  uv run python scripts/dev/spec_capture_ladder.py two_same 2

# same graph, second capture one op longer: clean (rules out program count)
PYTHONPATH=src:scripts/dev TWTEST_MAX_SEQ=2048 TWTEST_TRACE_REGION_MB=384 \
  uv run python scripts/dev/spec_capture_ladder.py plus_one 2

# ... and with that op introducing a new kernel binary: also clean
PYTHONPATH=src:scripts/dev TWTEST_MAX_SEQ=2048 TWTEST_TRACE_REGION_MB=384 \
  TWTEST_NEW_KERNEL=1 uv run python scripts/dev/spec_capture_ladder.py plus_one 2

# ... and with fifty appended ops, so the counts differ a lot: also clean
PYTHONPATH=src:scripts/dev TWTEST_MAX_SEQ=2048 TWTEST_TRACE_REGION_MB=384 \
  TWTEST_EXTRA_OPS=50 uv run python scripts/dev/spec_capture_ladder.py plus_one 2

# adjacent widths, k=2 against k=3: hangs, so magnitude is not the variable
PYTHONPATH=src:scripts/dev TWTEST_MAX_SEQ=2048 TWTEST_TRACE_REGION_MB=384 \
  TWTEST_SECOND_K=3 uv run python scripts/dev/spec_capture_ladder.py two_stepn 2
```

`two_stepn` captures two `step_n` graphs (k=2 and k=4) over the same model state,
replays k=2 (fine), then k=4 (hangs). No engine, no asyncio, no worker-thread
machinery is involved; a watchdog bounds the run so it reports rather than
wedging the session.

Where it stops, from `faulthandler.dump_traceback_later`:

```
  twtest/tt/traced.py:130  in step_n          ttnn.execute_trace(...)
  twtest/tt/engine.py:565  in speculate_round
  twtest/tt/engine.py:691  in _device_loop
```

**A minimal standalone script does *not* reproduce it**:
`scripts/dev/repro_trace_program_count.py` alternates two trivial traces (2 and
5 elementwise programs) cleanly. Whatever the trigger is, it needs scale — the
model graphs are thousands of ops with hundreds of distinct kernels.

## Excluded by experiment

* **A device sync between replays.** Both `execute_trace` calls already pass
  `blocking=True`, and `ttnn.synchronize_device` with no `cq_id` waits on every
  queue. Inserting one changes nothing.
* **A second command queue.** Capturing the second trace on cq 1 stops the hang
  and is worse: the replay does not execute, it merely stops blocking, returning
  the previous contents of its output buffer (`[201058, 0]` where the eager path
  returns `[75, 220]`, in 12 ms against 265).
* **An ordinary program between the replays**, on the theory that non-trace
  dispatch would resync whatever the trace path leaves stale. Still hangs.
* **`MeshDevice.reset_sub_device_stall_group()` between the replays**, since
  `enqueue_trace` takes per-sub-device ownership and updates worker state indexed
  by sub-device. Still hangs.
* **A much larger `trace_region_size`** — 1 GB, against the 256–384 MB the two
  traces need — on the theory that their buffers were colliding in an undersized
  region. Still hangs, so it is not region pressure.
* **`ttnn.release_trace` on the first trace before capturing and replaying the
  second.** Still hangs, which is the strongest single hint we have: whatever
  `enqueue_trace` leaves behind is not undone by releasing the trace that left
  it. `scripts/dev/traced_step_n_check.py` is the case that shows this — it
  replays a decode trace many times, releases it and the step_n capture, then
  captures a fresh step_n and replays that. It reaches its k-loop and never
  returns, at 40 minutes.
* **Capture order and allocation.** Capturing both before replaying either,
  allocating all buffers before any capture, capturing on a worker thread,
  `max_seq_len` 512 vs 2048, and an explicit `trace_region_size` (default vs
  256 MB vs 384 MB) each make no difference.

## A suspect, offered as a suspect

In `tt_metal/impl/trace/dispatch.cpp`:

* `record_begin` → `reset_host_dispatch_state_for_trace` zeroes the host
  launch-message write pointer before capture, commenting that "every time trace
  runs on device, it will ensure that the workers reset their rptr to be in sync
  with device".
* `FDMeshCommandQueue::enqueue_trace` → `update_worker_state_post_trace_execution`
  then **sets** that pointer to the executed trace's own program count
  (`set_mcast_wptr(desc.num_traced_programs_needing_go_signal_multicast)`).

Read together, each trace is recorded against a zeroed pointer and leaves the
pointer at its own count, so alternating two traces of different sizes would
leave host and workers disagreeing and the dispatcher waiting on a go signal
that never matches — a hang, no error, in the right place.

**This is ruled out.** The prediction it makes — equal program counts should
alternate fine — held at model scale (`two_same`), but that control captures the
*same graph* twice and so shares program count **and** kernel binaries. Two
further controls separate them, and both pass:

* `plus_one`: A and B are the same graph, B with one extra `ttnn.add` recorded
  inside the capture. Program counts differ by one, binaries identical.
  Alternates cleanly.
* `plus_one` with `TWTEST_EXTRA_OPS=50`: fifty appended ops rather than one, so
  the program counts differ substantially while B still contains A's programs
  unchanged. Alternates cleanly, ruling out the *size* of the difference as well
  as its existence.
* `plus_one` with `TWTEST_NEW_KERNEL=1`: the extra op is `ttnn.atan`, which the
  model never uses, so trace B references a kernel binary trace A does not.
  (It is warmed before capture, since a capture cannot load a new binary.)
  Alternates cleanly.

So neither program count nor a single distinct binary is the trigger, and nor is
the *size* of the difference: k=2 against k=3 hangs. The standalone script
alternating 2- and 5-program traces agrees that count is not it.

What separates the passing controls from the failing ones is that in every
passing case trace B contains trace A's programs unchanged and merely appends to
them, while in every failing case the two traces hold differently-shaped
versions of the same operations.

## What we do not know

Why *re-shaping* an operation across two traces differs from *appending* one.
Program count is out, a differing kernel binary is out, and magnitude is out
(k=2 against k=3 hangs). Per-program config-buffer state is the candidate that
fits the shape of the evidence and that we did not test:
`update_worker_state_post_trace_execution` calls
`config_buffer_mgr[index].mark_completely_full(...)` and carries a
`TODO(jbauman): Reuse old state from the trace`. Total trace-buffer footprint and
the number of distinct programs resident across both traces are also untested.

We also do not know why nothing small reproduces it.
