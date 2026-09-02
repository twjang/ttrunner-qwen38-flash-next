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
| two *small* traces, any program counts, alternated | works |

The rows below the hang are the controls, and together they say the two graphs
have to differ *substantially*: a one-program difference does not do it, and
neither does a distinct kernel binary referenced by only one of them.

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
* `plus_one` with `TWTEST_NEW_KERNEL=1`: the extra op is `ttnn.atan`, which the
  model never uses, so trace B references a kernel binary trace A does not.
  (It is warmed before capture, since a capture cannot load a new binary.)
  Alternates cleanly.

So neither program count nor a single distinct binary is the trigger. The
standalone script alternating 2- and 5-program traces agrees.

## What we do not know

What distinguishes "two graphs that differ a lot" from the marginal differences
above. Program count is out, and so is one differing binary. Candidates we did
not test: total trace-buffer footprint, the number of *distinct* programs
resident across both traces, and per-program config-buffer state
(`update_worker_state_post_trace_execution` also calls
`config_buffer_mgr[index].mark_completely_full(...)`, carrying a
`TODO(jbauman): Reuse old state from the trace`).

The two graphs that do hang, `step_n` at k=2 and k=4, differ structurally
throughout — k is unrolled into the recurrence and convolution, so tensor shapes
and matmul program configs differ at every layer, not just the op count.
