# 016 — Speculation, and the multi-row step under it

Single-user latency is op-count bound: a step issues 6355 device ops and the
model only produces one token for them. Speculation is the way out — verify
several candidate tokens in one step — and this iteration built the machinery
and then measured what it is actually worth, which turned out to depend on three
things nobody had priced.

## Observation 1 — the chunked path cannot be the verifier

The obvious verifier already existed: `prefill` consumes k tokens with the right
causality and is verified. Timed against feeding the same tokens one at a time:

| k | chunked pass | k eager steps | speedup |
|---|---|---|---|
| 1 | 1310 ms | 508 ms | 0.39x |
| 2 | 1360 | 1017 | 0.75x |
| 4 | 1461 | 2034 | 1.39x |
| 8 | 1440 | 4068 | 2.83x |

The cost is almost independent of k, because `deltanet.prepare()` round-trips
~30 MB a layer to the host whatever k is. So it only beats *eager* decode past
k≈3 — and a traced step is 236 ms, so four of them are 944 ms and the chunked
pass never beats them at all. Being host-bound it also cannot be captured.

**Remedy.** `step_n`: k tokens of one sequence on the batch axis. A step is flat
in batch up to 64 rows, so everything per-token costs what one row costs; only
the DeltaNet convolution (whose window is the three rows before it) and the
recurrence (whose state is the previous row's) are unrolled. It reproduces k
sequential steps **exactly** -- 0.00 % on the hidden at every position.

## Observation 2 — and it has to be traced, or it loses to what it replaces

| k | step_n eager | k eager steps | step_n traced | k traced steps | vs traced |
|---|---|---|---|---|---|
| 2 | 672.3 ms | 1035.5 | **255.0 ms** | 471.8 | 1.85x |
| 4 | 864.4 | 2071.0 | **276.6** | 943.6 | 3.41x |
| 8 | 1233.1 | 4142.1 | **323.9** | 1887.2 | 5.83x |

Eager `step_n` at k=4 costs 864 ms against 2071 for four eager steps, which
looks like a 2.4x win and is not one: four *traced* steps are 944 ms. Captured,
the same k=4 is 276.6 ms and the win is real.

Two things the capture needed. `_attention_step_n` was rewritten from the chunk
path's single masked `scaled_dot_product_attention` -- whose K/V slice grows with
position, so a capture is only valid inside one tile -- to k `sdpa_decode` calls
at `cur_pos = start + i`, which take the position as a tensor. And the bound
input names carry k, because sharing them handed a k=4 host tensor to a k=2
buffer.

## Observation 3 — a recurrence cannot be truncated

A transformer rolls back a rejected draft by reading less of its K/V cache. This
model cannot: the DeltaNet recurrent state and the conv rings have absorbed all k
tokens. So `snapshot`/`restore` copy everything that accumulates unconditionally
-- ~85 MB -- and deliberately not the K/V or the indexer's block cache, which
attention never reads past `cur_pos`.

Verified by running a draft that is thrown away and restoring: the continuation
is identical token for token.

## Observation 4 — the acceptance rate is the whole story, and it is measurable early

Speculation's value is the acceptance rate, and that needed no implementation at
all: generate greedily once, then run the drafter offline against the token
stream the model actually produced. Greedy acceptance means those counts are the
real ones.

| workload | drafter fires | accepted | vs eager | vs traced |
|---|---|---|---|---|
| open prose | 3-9 % | 8-33 % | 0.92-0.99x | 0.95-1.00x |
| copy-heavy (RAG, quoting) | 36-67 % | 52-87 % | 1.28-1.53x | **1.70-2.14x** |

Prompt-lookup drafting is a bet on repetition. On open prose the last n-gram
rarely recurs with a predictable continuation, so the drafter almost never fires
and the scheme is neutral; on a prompt whose answer quotes its context it fires
on half the positions and pays 2x.

Two corrections came out of writing this down. Charging `T(k)` for every round
rather than only the drafted ones understated the scheme badly. And the drafter
gave up on the first match too close to the end instead of continuing further
back, which silently cost drafts -- the same bug was in the engine's copy.

## Observation 5 — one capture, because two corrupt each other

A partial acceptance wants to replay its prefix with `step_n(j+1)`, which means a
capture per width. Capturing the second one warns:

```
Allocating device buffers is unsafe due to the existence of an active trace.
These buffers may be corrupted once ...
```

and the run wedged the cards hard enough to need `tt-smi -r`. A ladder of widths
is not worth that, so there is exactly one capture.

At k=2 it costs nothing: the draft is a single token, acceptance is
all-or-nothing, and the only replay width is 1 -- an ordinary traced step, which
already exists. Larger k replays a partial prefix with single steps instead,
which is slower.

A capture also wants far more trace region than the 128 MB default -- four asked
for over 253 MB -- and the region is fixed when the mesh opens, so it is sized
from k there.

## Observation 6 — greedy acceptance is not exact here

The claim that greedy acceptance makes speculation identical to token-by-token
decoding is standard, and it is wrong on this stack. Two separate single-engine
runs, one speculating and one not, diverge at token **29** on the copy-heavy
prompt and **39** on the open-prose one.

Greedy acceptance is exact only if the verifier and the stepper agree bit for
bit. They do not: `step_n` computes row i's logits inside a k-row batch where
`step` computes them in a 1-row batch, and bf16 rounding differs in the last
bits. `step_n_check.py` reports 0.00 % on the hidden state, but that is a
*rounded* maximum, and argmax amplifies whatever is left.

Every emitted token is still the argmax of the verifier's own logits, so what
comes out is *a* greedy decode -- just not the same one. On a model where two
correct implementations already disagree on half their greedy tokens (`014`),
that is a caveat rather than a defect. But it is not what "exact" promises, so
the flag is off by default and the constructor says so.

This is also the more useful lesson than any of the timings: the claim was
checked only because it was cheap to check, and it failed. The check needed two
separate processes, because two engines in one still hang (Observation 7).

## Observation 7 — the hang was two engines, not speculation

Capturing the `step_n` graph inside the engine hung it, and the boards came back
only after `tt-smi -r` -- five times over the session. The cause turned out not
to be speculation at all: a **second `TTEngine` constructed in the same process
after closing the first** hangs on its own capture. With one engine per process
speculation runs to completion.

The warning that accompanies it is a red herring. It appears exactly once in
every *successful* capture too; the hanging runs simply emit it twice, which is
what pointed at the two-engine sequence. Counting the warnings in the logs was
free and decided in one command what six device experiments had not.

Part of it is a real engine bug, now fixed: `TTEngine.close` closed the mesh
without releasing its captured traces, so the devices were left with a trace
registered. That alone does not account for the hang -- it still reproduces with
the fix -- but closing a mesh under a live trace is wrong regardless, and any
program that builds two engines was hitting it.

Standalone the same capture is fine: `scripts/dev/traced_step_n_check.py` builds
a `TracedStepN`, replays it, and measures 255 ms at k=2 with the tokens matching
eager. Eight candidate causes were tested; the eliminations are the useful part,
because each cost a device cycle:

| candidate | test | result |
|---|---|---|
| The post-capture state rewind allocates a zeros buffer per ring | the standalone harness does exactly this | works — **not it** |
| Two captures live at once (decoder + step_n) | engine forces `use_trace=False` when speculating | still hangs — **not it** |
| A ladder of captures, several live traces | reduced to one capture | still hangs — **not it** |
| Capturing off the main thread | `traced_step_n_thread.py` captures and replays on a worker thread | 253.7 ms, works — **not it** |
| The enlarged trace region (384 MB) starving DRAM | left at the 128 MB default | still hangs — **not it** |
| The single-token step graph allocating *after* the capture, on the engine's first eager step | `TracedStepN` now warms `model.step` and the LM head before capturing | still hangs — **not it** |
| Captured traces leaking past `close` | `TTEngine.close` now releases them | still hangs — **not it**, but a real bug |
| A second live capture slowing the first one's replay | timed a traced step with a `step_n` capture also live | 236.1 vs 236.3 ms — **not it** |
| **Two engines in one process** | ran a single engine | **completes — this was it** |

Two of those fixes were kept regardless of the verdict. Warming the single-token
step before capture is right because a caller that speculates still takes
ordinary steps, and letting them allocate the 48-layer graph after a capture is
the hazard this module's docstring opens with. Releasing traces in `close` is
right because closing a mesh under a live trace is simply wrong.

What is still open is *why* a second engine hangs. It is an engine-lifecycle
question rather than a speculation one, and it deserves its own investigation:
anything that opens and closes a mesh twice in one process is affected.

## Observation 8 — and it is not yet faster

With one engine per process, both traces live, and the snapshot buffers reused,
generation-only rates against a 240.0 / 240.4 ms baseline:

| `speculate` | drafted per verify | copy-heavy | open prose |
|---|---|---|---|
| 2 | 1 | 414.6 ms/tok | 492.3 ms/tok |
| 9 | 8 | **253.9 ms/tok** | 507.9 ms/tok |

`speculate` counts the tokens a verify *feeds*, so it drafts `speculate - 1`.
Reading it as the drafted count put the engine in its worst configuration: at
`speculate=2` the draft is a single token, one following token is easy to find,
so the drafter fires on nearly every round and every failure pays a verify plus
a replay. The offline pricing's k is the drafted count, so its k=8 is
`speculate=9`.

Even at `speculate=9` the best case is 0.95x, and open prose is 2.1x *worse*
than baseline while drafting less often than copy-heavy -- which is backwards
from any drafting-cost model and means the overhead is not in the drafting. That
is unexplained, and it is where the next attempt should start: instrument the
engine to count drafted versus plain rounds and time each, rather than inferring
from end-to-end rates.

## Where it leaves us

Everything the scheme needs is built and verified on its own:

| piece | state |
|---|---|
| `TTModel.step_n` | reproduces k sequential steps **exactly** (0.00 %) |
| `TracedStepN` | 255 ms at k=2 against 472 for two traced steps; tokens match eager |
| `snapshot` / `restore` | a discarded draft rolls back token-for-identically |
| `prompt_lookup_draft`, `accepted_prefix` | unit-tested, including the search bug |
| offline pricing | 1.70-2.14x on prompts that quote their context |
| end to end in the engine | runs, one engine per process; best 0.95x, not yet faster |

Two things stand between that and a feature worth switching on, and neither is
about drafting:

1. **Where the per-round overhead goes.** Instrument drafted versus plain rounds
   and time each. The end-to-end rates say the cost is not in the drafter, and
   inferring further from them is guessing.
2. **Why a second engine in one process hangs.** An engine-lifecycle bug that
   happens to block this, and blocks anything else that opens a mesh twice.

Then, in order: carry the QSA selection through `step_n` (`_attention_step_n`
reads per row, so there is room for a per-row mask), and only then MTP, whose
value is precisely that it drafts on *every* position rather than only where the
context repeats -- the limitation Observation 4 measures.

Two process notes, since between them they cost five board resets. `kill -9` on
a process that is mid-capture leaves the cards needing `tt-smi -r`; check `ps`
and let device jobs finish. And `pgrep -f <pattern>` matches the shell running
it, so it kills the caller -- match on the executable instead:
`ps -eo pid,comm,args | awk '$2 ~ /^python/ && /script_name/'`.
