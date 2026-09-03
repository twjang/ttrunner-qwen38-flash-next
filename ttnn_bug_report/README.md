# Alternating replays of two captured traces hangs the device

**Environment.** `pjrt-plugin-tt` 1.4.0 (bundled tt-metal), 4× Blackhole p150a,
`FabricConfig::FABRIC_1D`, `MeshShape(1, 4)`, one command queue.

## Symptom

With two traces captured on the same mesh device, replaying **trace A, then
trace B** never returns from `ttnn.execute_trace` (`blocking=True`, cq 0). The
process spins on one thread — 13:49 of CPU in 12:25 of wall clock — and the
boards afterwards need `tt-smi -r all`. There is no exception and no log line.

Stack, from `faulthandler.dump_traceback_later`:

```
  ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
```

## Reproducing

```
TT_GGUF_DIR=/path/to/gguf TT_CACHE_DIR=/path/to/cache python repro.py 2 4
```

Captures the same 48-layer decode graph twice at two different row counts, then
replays A, B, A. A watchdog reports after 420 s rather than hanging the caller.
Expect a few minutes of kernel compilation before the hang. Output:

```
[repro] capturing A (k=2)
[repro] capturing B (k=4)
[repro] replaying A
[repro] replaying B   <-- hangs here
[repro] HUNG -- no progress in 420s.
```

It needs the model. `minimal_attempt_does_not_reproduce.py` is a self-contained
version that does **not** reproduce, kept so the same ground is not searched
twice; its docstring lists the axes tried.

## What separates the failing case from the passing ones

| | |
|---|---|
| replay one trace repeatedly, any number of times | works |
| capture A and B, replay **only** B | works |
| two captures of the **identical** graph, replayed A, B, A | works |
| B is A **plus one appended op** | works |
| B is A plus one appended op introducing a **new kernel** | works |
| B is A **plus fifty appended ops** | works |
| A and B the same graph at **k=2 and k=4** (shapes differ) | **hangs** |
| the same at **k=2 and k=3** — adjacent | **hangs** |

So it is not the number of programs, not a differing kernel binary, and not the
size of the difference. The line is:

> **B ⊇ A is safe. B and A disagreeing about a shape is not.**

Changing `k` unrolls differently through the recurrence and convolution, so
every layer's tensors and matmul program configs change, rather than anything
being appended.

## Workarounds ruled out by experiment

1. `ttnn.synchronize_device` between the replays — no effect. Both
   `execute_trace` calls already pass `blocking=True`, and with no `cq_id` the
   sync waits on every queue.
2. **A second command queue** — stops the hang and is worse: the replay does not
   execute, it merely stops blocking, returning the previous contents of its
   output buffer (`[201058, 0]` where the eager path returns `[75, 220]`, in
   12 ms against 265).
3. An ordinary non-trace program between the replays, on the theory that normal
   dispatch would resync what the trace path leaves stale — still hangs.
4. `MeshDevice.reset_sub_device_stall_group()` between the replays, since
   `enqueue_trace` takes per-sub-device ownership — still hangs.
5. A **1 GB** `trace_region_size` against the 256–384 MB the two traces need, in
   case they were colliding in an undersized region — still hangs.
6. `ttnn.release_trace` on the first trace before capturing and replaying the
   second — still hangs. Whatever `enqueue_trace` leaves behind is not undone by
   releasing the trace that left it.

## A suspect, offered as a suspect

In `tt_metal/impl/trace/dispatch.cpp`:

* `record_begin` → `reset_host_dispatch_state_for_trace` zeroes the host
  launch-message write pointer before capture, commenting that "every time trace
  runs on device, it will ensure that the workers reset their rptr to be in sync
  with device".
* `FDMeshCommandQueue::enqueue_trace` →
  `update_worker_state_post_trace_execution` then **sets** that pointer to the
  executed trace's own program count, and calls
  `config_buffer_mgr[index].mark_completely_full(...)` with a
  `TODO(jbauman): Reuse old state from the trace`.

We could not confirm this and think it is at best incomplete: the program-count
reading it implies is contradicted by the table above, where a fifty-program
difference is harmless. The per-program **config-buffer** half fits better,
since that is state a program's *shape* would change, but it cannot be varied
from Python.

## What we do not know

Why re-shaping an operation across two traces differs from appending one, and
why nothing small reproduces it. The distinguishing experiment we could not run
needs an instrumented build.
