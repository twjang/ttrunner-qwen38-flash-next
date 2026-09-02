# 018 — The capture was never the problem

For four board resets the record said "capturing `step_n` inside the engine hangs
the device". Ten causes had been excluded by experiment, all of them things the
standalone harnesses do differently, and the list was getting longer rather than
shorter. The handoff's instruction was right — bisect from the harness that works
toward the engine — and following it turned up six more exclusions and no answer.
What ended it was asking the failing process where it was.

## Observation 1 — six more rungs, all green

`spec_capture_ladder.py` rebuilds `_device_loop`'s setup one element at a time.
Each of these passed, i.e. captured both traces without hanging:

| rung | element added |
|---|---|
| `baseline` | what `traced_step_n_check.py` already does |
| `snapshot` | the ~200-tensor snapshot allocation the harnesses never make |
| `thread` | the same, on a worker thread |
| 2048 | the engine's `max_seq_len`, four times the K/V |
| 256 MB | the engine's explicit `trace_region_size`, which no harness passes |
| `no_replay` | capturing #2 without ever replaying #1, as the engine does |

Also excluded: two engines in one process — the hang reproduces with the
speculative engine as the only one, via `TWTEST_SPEC_ONLY=2`.

Sixteen exclusions and no cause is a sign the question is wrong, not that the
list is incomplete.

## Observation 2 — the failing side knew all along

`_device_loop` printed nothing during setup, so a hung run's log ended at
metal's own "Allocating device buffers is unsafe due to the existence of an
active trace" and left everything after it to inference. Two cheap instruments
settled in one run each what the ladder had not in six:

* a `mark()` line per setup step — and **the setup completes**. Both captures
  succeed, the decoder resets, the state rewinds, and it enters the serve loop.
* `faulthandler.dump_traceback_later` — and the device thread is in

      traced.py:130   in step_n        ttnn.execute_trace(...)
      engine.py:565   in speculate_round
      engine.py:691   in _device_loop

So the hang is the **first replay of the verifier**, not the capture of it. The
refusal in `TTEngine` had been describing the wrong operation for four resets.

## Observation 3 — it is the interleaving, and there is no queue that fixes it

With the right question, the ladder answered immediately. `traced_step_n_check.py`
never replays `step_n` while the decoder's trace is live: it measures the decoder
with both captured, releases both, and then replays `step_n` alone on a fresh
state. That is the one combination a speculating engine cannot avoid — plain
rounds replay the decoder, drafted rounds replay `step_n`.

| configuration | outcome |
|---|---|
| cq 0, replay `step_n` alone (decoder captured, never replayed) | correct, 265 ms |
| cq 0, replay decoder, then `step_n` | **hangs** in `ttnn.execute_trace` |
| cq 1, replay decoder, then `step_n` | returns in **12 ms** with **wrong tokens** |

Two things that look like fixes and are not:

* **A device sync between the replays.** Both `execute_trace` calls already pass
  `blocking=True`, and `synchronize_device` with no `cq_id` waits on every queue.
  Inserting one changes nothing.
* **A second command queue.** It stops the hang, which is why it was briefly
  committed as the fix. Then the engine's own report gave it away: verify came
  back in 9.6 ms where a standalone replay costs 265, and it accepted **0 of
  every 5** drafted tokens. Checked directly, the cq-1 replay returns
  `[201058, 0]` where the eager `step_n` returns `[75, 220]` — it is not
  executing, merely not blocking. Trading a hang for silent corruption is a
  worse trade, so it is reverted.

The control that makes those readings trustworthy: the same harness on cq 0
*without* the decoder replay returns `[75, 220]` in 265.0 ms. The fixture is
sound; the interleaving is not.

## Observation 4 — it is not about *which* two traces

Found later, while starting 5.2's step 5. A traced chunked prefill would
alternate with the traced decode step exactly as a traced verifier does, so the
question "is this defect specific to decoder-vs-step_n?" stopped being academic.

It is not. Two `step_n` captures at k=2 and k=4, replayed alternately with no
decoder anywhere in the process, hang identically
(`spec_capture_ladder.py two_stepn`). **Any two traces replayed alternately hang
this build.**

Which retroactively explains a symptom recorded in 5.2 long before anyone knew
the cause: capturing the prefill graph made the trace replay "come back as token
0 repeated". That is a trace that is not executing and not blocking either,
returning whatever the buffers last held -- the same signature the
second-command-queue experiment produced above. Two items that looked like
separate mysteries are one defect, and one upstream fix closes both.

## Observation 5 — a mechanism from the source, and how it failed

Reading the vendor's code beat guessing at it, which the earlier rungs had been
doing. `FDMeshCommandQueue::enqueue_trace` finishes with
`trace_dispatch::update_worker_state_post_trace_execution`, and that function
**sets** the host launch-message write pointer to the executed trace's own
program count rather than advancing it:

    worker_launch_message_buffer_state[index].set_mcast_wptr(
        desc.num_traced_programs_needing_go_signal_multicast);

Capture does the mirror: `record_begin` calls
`reset_host_dispatch_state_for_trace`, which zeroes the same pointer, commenting
that "every time trace runs on device, it will ensure that the workers reset
their rptr to be in sync with device".

So each trace is recorded against a zeroed pointer and leaves it at its own
count. Alternate two traces of different sizes and the host's pointer and the
workers' read pointer disagree; the dispatcher waits on a go signal that never
matches. It explains the hang exactly, in the right place, with no error.

**And it is not established.** The prediction -- equal program counts should
alternate happily -- came back true at model scale: `spec_capture_ladder.py
two_same` captures one graph twice and replays A, B, A cleanly where `two_stepn`
at k=2 and k=4 hangs. That was written up as confirmation. It is not:
two captures of the *same* graph share their program count **and** their kernel
binaries, so the control cannot separate those. Building the tidy standalone
reproduction settled it the other way --
`scripts/dev/repro_trace_program_count.py` alternates tiny traces of 2 and 5
programs without trouble.

So program count alone is not the trigger; binary residency is the obvious
untested alternative; and the defect needs scale, since nothing small
reproduces it at all.

## What is actually established

* Replaying one trace repeatedly is fine, however many times.
* Replaying a second trace *alone*, after capturing both, is fine.
* Alternating two **distinct model-scale** traces hangs in `ttnn.execute_trace`.
* Alternating two captures of the **same** model-scale graph does not.
* Tiny traces do not reproduce any of it.

Three workarounds are excluded: a device sync, a second command queue (which
trades the hang for a silently wrong replay), and an intervening eager program.

## The second lesson, which is the same as the first

Observation 3 says "it stopped failing" is not "it works". This section is the
same error one level up: a mechanism that explains the symptom, plus one
confirming test, was written up as established -- and the test had a confound
sitting in plain view. A prediction is only worth what its control isolates.

## Where it leaves 5.6## Where it leaves 5.6

Still refused, but for a stated and reproducible reason instead of a mystery, and
the reproduction is sixty lines with no engine, no asyncio and no admission loop
in it — small enough to hand upstream, which is the next step. Everything else
the scheme needs remains verified on its own: `step_n` reproduces k sequential
steps, `TracedStepN` replays at 255 ms for k=2 against 472 for two traced steps,
`snapshot`/`restore` roll a discarded draft back identically, the drafter and
accept rule are unit-tested, and offline pricing says 1.70–2.14x on prompts that
quote their context.

Kept from the investigation: the `mark()` progress lines (on only while
speculating), `TWTEST_ALLOW_SPECULATION=1` so the real engine can be driven
rather than reconstructed, `TWTEST_STACK_DUMP` in `speculation_check.py`, and
`spec_capture_ladder.py` itself.

## The lesson, which is not "bisect"

Bisecting was the right instruction and it was being followed; it produced six
correct exclusions and no answer, because every rung tested the setup and the
setup was never broken. The cheaper move — three print statements and a stack
dump — was available the whole time and would have redirected the search on the
first run. When a bisection keeps coming back green, stop adding rungs and make
the failure describe itself.

An earlier form of the same error is in `014`: reasoning from a plausible model
of the defect instead of measuring it. And the near-miss here is worth recording
too — the queue change was committed as a fix on the evidence that the hang
stopped, before anything checked that the replay still produced the right
tokens. "It no longer fails" is not "it works"; a no-op passes the first test
and fails the only one that matters.
