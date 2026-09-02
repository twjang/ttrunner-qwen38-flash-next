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

## Observation 6 — and the engine integration hangs the device

`TTEngine(speculate=k)` raises. Not as a performance caveat: capturing the
`step_n` graph inside the engine's device thread hangs it, and the boards come
back only after `tt-smi -r`. That happened three times -- with the decoder's
trace live and with it off -- always preceded by the warning in Observation 5.

Standalone the same capture is fine: `scripts/dev/traced_step_n_check.py` builds
a `TracedStepN`, replays it, and measures 255 ms at k=2, with the tokens matching
eager. So it is something about capturing inside the engine, not about the
capture itself. The obvious suspects, none yet confirmed:

* the state rewind after the capture allocates a zeros buffer per ring, which is
  allocation-after-capture -- though `TracedDecoder.reset` does exactly the same
  and works;
* two captures live at once (decoder + step_n), which the warning names -- but
  the hang reproduced with `use_trace=False`, so that is not sufficient;
* something in the device thread specifically, since the standalone harness
  captures on the main thread.

A flag that bricks the accelerators is worse than no flag, so the constructor
refuses with a pointer here rather than trying.

## Where it leaves us

Everything the scheme needs is built and verified on its own:

| piece | state |
|---|---|
| `TTModel.step_n` | reproduces k sequential steps **exactly** (0.00 %) |
| `TracedStepN` | 255 ms at k=2 against 472 for two traced steps; tokens match eager |
| `snapshot` / `restore` | a discarded draft rolls back token-for-identically |
| `prompt_lookup_draft`, `accepted_prefix` | unit-tested, including the search bug |
| offline pricing | 1.70-2.14x on prompts that quote their context |

What is missing is one thing: a capture that survives inside the engine's device
thread. After that, in order -- make allocating during a second capture safe
(which unlocks the width ladder and larger k), carry the QSA selection through
`step_n`, and only then look at MTP, whose value is precisely that it drafts on
*every* position rather than only where the context repeats, which is the
limitation Observation 4 measures.

One process note, since it cost three board resets: `kill -9` on a process that
is mid-capture leaves the cards needing `tt-smi -r`. Check `pgrep` and let device
jobs finish.
