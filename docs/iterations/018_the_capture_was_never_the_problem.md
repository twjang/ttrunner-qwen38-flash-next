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

## Where it leaves 5.6

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
