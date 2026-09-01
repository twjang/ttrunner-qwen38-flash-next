# 011 — Continuous batching, and the assumption that had always held

Date: 2026-09-01

---

## Where the step time actually goes

Re-profiled at B=64 after the ring buffers landed. The distribution has flattened:
no single section dominates the way PLE (51 ms in one layer) once did.

| section | per call | calls | total | share |
|---|---|---|---|---|
| deltanet | 7.768 ms | 36 | 280 ms | 30 % |
| moe_routed | 5.858 ms | 48 | 281 ms | 30 % |
| hc_gate | 1.400 ms | 96 | 134 ms | 14 % |
| reinject | 0.566 ms | 96 | 54 ms | 6 % |
| shared_expert | 0.871 ms | 48 | 42 ms | 4 % |
| qsa_attention | 3.409 ms | 12 | 41 ms | 4 % |
| all_reduce | 0.429 ms | 48 | 21 ms | 2 % |
| ple | 7.056 ms | 1 | 7 ms | 1 % |

---

## Observation 1 — two harnesses disagreed by 26 %, and the optimistic one was wrong

`prof_step.py` reported **107.62 tok/s** at B=64; `t_batchall.py` reported
**85.27** for the same configuration. Both were "measurements", which is exactly
why the disagreement mattered more than either number.

The difference was warmup: the profiler used 2 warmup iterations and 3 samples,
so JIT'd kernels and allocator warm-up landed inside the measured window.

A careful run — 5 warmup steps, 25 samples, median with p10/p90:

```
   B  median ms      p10      p90     tok/s   ms/tok
   1      518.3    511.2    529.8      1.93   518.27
  16      549.0    544.7    562.1     29.14    34.32
  32      550.5    547.4    553.3     58.13    17.20
  64      742.8    739.6    757.2     86.16    11.61
```

Reproduced independently: 86.16 and 86.24 on separate runs. **86.16 tok/s** is
the honest figure. `benchmark_step` defaults were raised to 5/25 and the reason
recorded in the docstring, because a 26 % optimistic error is worse than no
measurement — it gets quoted.

The shape matters as much as the number: B=1 (518 ms) and B=32 (550 ms) cost
nearly the same, so the step is fixed-cost dominated up to B=32, and B=64 buys
2× the work for 1.35× the time.

---

## Observation 2 — the batch ceiling is a tile cliff, not a limit to nudge

B=72 fails:

```
circular buffers on core range [0-0 - 5-0] grow to 1913600 B, max L1 1572864 B
```

1913600 × (64/96) = 1275733, which fits. Batch pads to multiples of 32, so
B=65..96 all allocate as 96 and all overflow by ~22 %. B=64 is not *near* a
ceiling — it is the last valid step before the next tile boundary. Probing 66,
70 or 80 is wasted effort.

---

## Observation 3 — a rejected optimization became correct when the code changed

Trace capture had been measured earlier as anti-composing with batch: 1.80× at
B=1 but **0.97×** at B=16, and it was shelved on that evidence. Re-measured
against the current op mix (ring buffers, fused delta rule, reinject fix):

| batch | eager | traced | gain |
|---|---|---|---|
| 16 | 28.53 tok/s | 43.44 tok/s | **1.52×** |
| 32 | 55.98 tok/s | 76.05 tok/s | **1.36×** |
| 48 | 65.27 tok/s | 66.66 tok/s | 1.02× |
| 64 | 85.98 tok/s | 86.78 tok/s | 1.01× |

The remedy is a policy, not a mode: **trace below B=48, eager at or above it.**
Trace pays only where dispatch overhead is exposed; by B=48 each op carries
enough device work to hide it.

The lesson is about the shelf, not the trace. A measurement is only valid for
the code it was taken against, and "we tried that, it didn't help" silently
expires. Had that note been trusted, the largest remaining win would have been
left on the floor.

---

## Observation 4 — op count is a poor proxy, and so is my intuition about layout

`gated_residual_mix` is 14 % of the step, so it got a micro-profile:

```
grouped_rms_norm           0.441 ms
reshape+sum+scale (mean)   0.367 ms
sigmoid+multiply           0.357 ms
linear down (10240->320)   0.272 ms
linear inject (10240->4)   0.153 ms
linear up  (320->10240)    0.115 ms
```

The mean costs 3× the 320→10240 matmul. The theory: it reshapes to
`[1, M, 4, 2560]` and sums over a size-4 axis that TILE layout pads to 32, so it
sums 8× padded zeros. The fix — four contiguous slices and three adds, avoiding
the padded axis — was **2.4× slower** (1.065 ms vs 0.442), at both M=1 and M=64.

Slicing is data movement; the padding costs less than materialising four
tensors. The single `sum` stays. That is the fourth layout intuition this
session to lose to a measurement, after the `repeat_interleave` reinject.

---

## Observation 5 — a 2560×12 matmul costs the same as a 2560×2560

Stage-profiling DeltaNet's four input projections at B=64:

```
linear attn_qkv  (2560x2560)   0.130 ms
linear attn_gate (2560x1536)   0.130 ms
linear ssm_alpha (2560x12)     0.130 ms
linear ssm_beta  (2560x12)     0.130 ms
FUSED qkv|gate   (2560x4096)   0.141 ms
```

There is a ~0.13 ms floor per op regardless of shape. Fusing `attn_qkv|attn_gate`
(both bfloat8_b) is legal and saves 0.120 ms/call — **4.3 ms/step, 0.45 %** — and
was rejected as not worth the sharding complexity. Fusing in `ssm_alpha/beta` is
blocked anyway: they are float32 per the residency plan, and `concat` refuses
mixed dtypes.

That floor is largely per-op host sync, not a cost trace can remove at large
batch — which is consistent with Observation 3, where trace stops helping by
B=48.

---

## Observation 6 — the engine never used the batched path at all

Every number above was unreachable through the API. The device loop was:

```python
for seq in list(active):
    hidden = self.model.step(seq.next_token, seq.state)   # one sequence at a time
```

Each sequence had its own `TTState` and took a full device step, so N concurrent
requests cost N steps. The whole optimization campaign was invisible to anyone
using the server. The module docstring had flagged this as known-remaining work.

**Remedy.** All sequences share one `TTState` of `max_concurrency` slots and
advance in lockstep. `TTModel.reset_slot` clears a freed slot so the batch never
drains between requests:

- The K/V cache is deliberately **not** cleared — attention reads only up to
  `cur_pos`, so a previous occupant's entries beyond the new position are
  unreachable, and zeroing 48 layers per admission would cost more than it saves.
- What must be cleared is what accumulates unconditionally: the DeltaNet
  recurrent state and both convolution rings.
- Every write is in place (`output_tensor=`). Rebinding to fresh tensors would
  change addresses and silently invalidate a captured trace.
- `chunked_prefill` now **raises** instead of being ignored: `TTModel.prefill` is
  single-sequence and cannot be expressed in a lockstep batch. Silently giving a
  caller something other than what they asked for is worse than failing.

---

## Observation 7 — the bug that had been latent since batching was written

`reset_slot` passed its direct test: a sequence in a reused slot produced tokens
identical to the same sequence in a fresh slot. The third check did not — after
resetting slot 0, **slot 2's logits moved by 5.625**, though slot 2's history was
unchanged.

Diagnosis, one hypothesis at a time rather than by inspection:

1. **Is the mask zeroing the wrong elements?** No. Both broadcast forms
   (`[1,B,1,1]` onto `[1,B,C,1]`, `[B*nv,1,1,1]` onto `[B*nv,1,D,D]`) zero
   exactly the intended slot, max diff **0.0**.
2. **Is there cross-slot coupling** — does the MoE's union-of-experts formulation
   let one slot's content perturb another? No. Same slot-2 history with entirely
   different slot-0 tokens: max diff **0.000000**, and the model is deterministic.
3. **Position.** The only thing left. After a reset, slot 0 sits at position 0
   while slot 2 is at 5.

The cause, in `_attention_step`:

```python
ttnn.update_cache(st.keys, k_upd, positions[0])   # one index, whole batch
...
q, st.keys, st.values, is_causal=True, cur_pos=positions,   # reads per-sequence
```

Cache **writes** used slot 0's position for every sequence while **reads** used
per-sequence positions. That is correct only while the batch advances in lockstep
from a common start — true of every benchmark ever run here, and false the moment
a slot is refilled mid-flight.

**Remedy.** The engine builds `TTModel(traceable_kv=True)`, whose
`paged_update_cache` takes the index as a per-sequence tensor. The single-index
path now refuses divergent positions with a message naming the fix, rather than
corrupting silently.

**Result.** Neighbour drift **5.625 → 0.000000**; slot reuse still bit-exact.

### The near-miss worth recording

Without the neighbour check, this would have shipped looking perfect: 94 tests
passing, single-sequence output bit-exact, throughput 45× better. It would have
returned text conditioned on corrupted attention history — only under
concurrency, only after the first slot recycled, and never in a way that crashed.

The generalisable form: **an invariant that has always held is not a verified
invariant.** Lockstep positions were never a design decision, just a property of
how every test happened to drive the model. The first feature to violate it found
a bug that had been latent since batching was written.

`tests/test_continuous_batching.py` locks the contract (guard present, engine on
the per-sequence path). The device-side proof needs four Blackhole cards and
lives here.

---

## Cumulative

| | before | after | gain |
|---|---|---|---|
| weight load | 2373 s | 6.0 s | **395×** |
| decode throughput | 0.75 tok/s | **86.16 tok/s** (B=64) | **115×** |
| B=1 traced latency | 1338 ms | 267 ms | 5.0× |
| B=16 throughput (traced) | — | 43.44 tok/s | 1.52× over eager |

Tests: 94 passing.

---

## Observation 8 — the LM head gather, and an exact shortcut

With the cache path fixed, `logits()` stood out: **119.6 ms at B=32**, 17 % of the
step, gathering `[B, 248320]` float32 off four devices every token when greedy
decoding needs one integer per sequence.

The LM head is column-sharded, so device *d* owns vocabulary columns
`[d*W, (d+1)*W)`. A per-device `(max, argmax)` therefore composes into the global
argmax. This is exact **only** because the split is even — 248320/4 = 62080,
itself a whole number of 32-wide tiles. With an uneven vocabulary the last shard
is zero-padded and a padded column at 0.0 would beat genuinely negative logits,
so `greedy_tokens` returns `None` on that case and the caller falls back.

```
B= 8 MATCH  logits  35.3 ms -> greedy 11.6 ms  (3.0x)
B=32 MATCH  logits 193.7 ms -> greedy 37.9 ms  (5.1x)
B=64 MATCH  logits 232.8 ms -> greedy 71.8 ms  (3.2x)
```

Taken only when every slot sampling this step is greedy; one `temperature > 0`
slot forces the full gather, which then serves the greedy slots too.

Also measured: `traceable_kv=True` costs **nothing** (561.0 ms vs 569.5 at B=32).
The correctness fix in Observation 7 was free.

---

## Observation 9 — a performance win that had silently broken correctness

Trace capture was about to be wired into the engine on the strength of
Observation 3's numbers. `TracedDecoder.reset()` crashed first: the ring-buffer
optimization changed `conv`/`ple_conv` from tensors to *lists*, and `reset()`
still called `.shape` on them.

That crash pointed at something much worse. The rings advance by

```python
ring[pos] = col          # rebinds a Python list slot
st.ple_step += 1         # host-side counter
```

Neither is a device op. **Trace capture records device ops only**, so a replayed
step never rotates the ring and reads a convolution history frozen at capture
time. Verified rather than argued:

```
eager : [198, 96304, 16, 4960, 198, 97033, 97901, 98142]
traced: [91, 11, 198, 13962, 220, 6458, 220, 6458]
MISMATCH — first divergence at output token 0
```

Every trace number previously recorded — 1.52× at B=16, 1.36× at B=32, the 5.0×
B=1 latency in the README — was a speedup **on a wrong computation**. The trace
benchmarks timed the step and never checked what it computed, so a correctness
regression hid inside a performance win for as long as it took to try to use it.

**Remedy.** `TTModel.trace_safe_rings` keeps the read indices fixed and shifts the
contents with `ttnn.copy`, which capture does record. `TracedDecoder` sets it
before the warmup step, so capture and replay agree on how the rings advance. The
eager path keeps the free rebinding.

**Result** — correctness and timing measured in the same run, so this cannot
recur silently:

```
B= 1 MATCH  eager 521.4 ms -> traced 254.4 ms (2.05x)
B=16 MATCH  eager 560.2 ms -> traced 363.6 ms (1.54x)   44.01 tok/s
B=32 MATCH  eager 563.5 ms -> traced 416.6 ms (1.35x)   76.82 tok/s
```

The speedups survive the fix — the copies sit inside the trace, where dispatch is
free. The engine now captures a trace below 48 slots and falls back to eager if
capture fails.

### What generalises

A benchmark that measures only time will certify a wrong answer. Both real bugs
this session were found by a *correctness* check that a performance change had no
obvious reason to affect — the neighbour-drift check in Observation 7, and
comparing traced tokens against eager here. Optimization work should pair every
timing harness with an output comparison, because the failure mode is not a
crash: it is a fast, plausible, wrong answer.

---

## Cumulative (revised)

| | before | after | gain |
|---|---|---|---|
| weight load | 2373 s | 6.0 s | **395×** |
| decode, batch 64 (eager) | 0.75 tok/s | **86.16 tok/s** | **115×** |
| decode, batch 32 (traced) | — | **76.82 tok/s** | 1.35× over eager |
| decode, batch 1 (traced) | 1338 ms | **254.4 ms** | 5.3× |
| server, 32 concurrent | ~1.9 tok/s (serial) | **35.96 tok/s** | ~19× |

Tests: 98 passing.

---

## Observation 10 — the same trace, correct through the model and wrong through the engine

With the rings fixed, trace was wired into the engine: **9.95 → 18.21 tok/s at 8
concurrent (1.83×)**, single request 1.34 → 2.45. Then the output was checked.

```
trace-on   'The capital of France is' -> '!!!!!!!"'
trace-off  'The capital of France is' -> ' Paris.\n\nThe French city in Europe'
```

A clean 1.83× while emitting `'!!!!'`. Twice in one session a trace change
produced a convincing speedup on a wrong answer — the second time in the exact
harness built to catch the first.

Eight hypotheses, each given a standalone reproduction. **All passed**, so all are
excluded:

| hypothesis | result |
|---|---|
| `reset_slot`'s eager ops between replays | matches eager |
| repeated post-capture allocation | matches |
| `max_seq_len` 256 vs 4096 | matches |
| 128 MB vs 200 MB trace region | matches |
| distinct tokens per slot | matches |
| filler token 0 in idle slots | matches |
| live slot index 0 vs 7 | matches |
| capture + replay on a worker thread, mesh opened on main | matches |

Instrumenting the engine showed the hidden state is not zero (absmax 119, 30, 75)
— the graph runs and computes *something*, just not the right thing. Notably
`reset_slot` cannot be the cause: with a single request the state has just been
`reset()` to zeros, so resetting its slot is semantically a no-op, and the output
is still wrong.

**Remedy.** `use_trace` defaults to **False** in `TTEngine`. The exclusion list
lives in the code next to the flag so the next attempt starts from experiment
nine, not one.

**Result.** The served path is the eager one: 35.96 tok/s at 32 concurrent, with
output verified against the reference (`' Paris.'`, `' blue, the ocean is blue'`).
The trace speedups remain real and verified *through `TTModel`* — they are simply
not reachable through the engine yet.

This is the honest end state rather than the flattering one. The temptation was
to keep the 1.83 × and the headline that goes with it; a speedup whose output is
`'!!!!'` is worth nothing, and reporting it as progress would have been a lie
that took one prompt to discover.

---

## Open

- The engine/trace interaction above. Needs a failing reproduction outside
  `TTEngine`; every isolated one built so far passes.
- Prompt tokens cost one device step each (12 output tokens take 17 steps).
  `TTModel.prefill` is ~10× faster but single-sequence, so it cannot run inside a
  lockstep batch; a batched chunked prefill is the next real throughput item.
- `deltanet` (280 ms) and `moe_routed` (281 ms) remain co-dominant at 30 % each.
  Both are doing genuine matmul work; the structural waste is gone, and the four
  input projections fuse for only 0.45 %.

---

## Observation 11 — the MoE's cost is a floor, not the waste

The MoE is 30 % of the step, and the obvious target was its *union waste*: the
broadcast formulation computes |union of selected experts| × M rows, and the
union grows with M. Measured, per layer, per device:

| M | time | union | waste | issued |
|---|---|---|---|---|
| 1 | 4.497 ms | 10/512 | 1.0× | 0.0 TFLOP/s |
| 8 | 5.499 ms | 62 | 6.2× | 0.9 |
| 16 | 6.384 ms | 96 | 9.6× | 2.4 |
| 32 | 7.873 ms | 189 | 18.9× | 7.6 |
| 64 | 13.382 ms | 233 | 23.3× | 11.0 |

Two corrections fall out. The union at M=64 is 233, not the 512 the docstring
assumed, so the waste is 23× rather than 51×. And **at M=1, where the waste is
exactly 1.0×, the block still costs 4.497 ms** — most of the in-model 5.858 ms.
The cost is a per-call floor, not the work discarded.

That explains the earlier "chunking is monotonically worse" result rather than
merely restating it: splitting M=64 into eight groups of 8 removes waste but pays
the floor eight times (8 × 5.499 = 44 ms against 13.4 ms).

**So the lever is issuing fewer sparse_matmuls.** `gate_w` and `up_w` are both
bfloat4_b, both EXPERT_COLUMN-sharded and identically shaped, so they fuse into
one call:

```
M=  1 separate 0.961 ms  fused 0.782 ms  1.23x
M= 32 separate 4.419 ms  fused 2.467 ms  1.79x
M= 64 separate 7.749 ms  fused 4.210 ms  1.84x
```

### The trap in building the fused tensor

`ttnn.concat` on two bfloat4_b tensors **requantises** them — 0.0547 max error
against the originals — and the model's output changed: tokens diverged at the
second position, with the fused run emitting a stop token early. End-to-end it
looked like a clean win (85.79 → 97.14 tok/s, 1.13×) while generating different
text, and that speedup is itself suspect: corrupted weights change the routing,
so the union, so the cost.

Dequantising to float and quantising the concatenation **once** is exact
(measured 0.0 against both halves), because the per-device width 160 is a whole
number of 16-element blocks, so the shared exponents line up.
`scripts/fuse_expert_gate_up.py` builds the fused weights that way from the
existing cache — no GGUF re-read — and leaves the originals in place.

Third time this session that a speedup had to be checked for what it computed,
and the third time it mattered.

### Result, with exactly-built weights

`scripts/fuse_expert_gate_up.py` wrote all 192 shards (48 layers × 4 devices,
~40 GB, ~1 h). Same prompt, 25 samples after 5 warmup, one mode per process --
both weight sets resident at once does not fit:

```
[split] tokens [11, 427, 378, 490, 17045, 291]
[split] B=1 533.7 ms   B=32 554.3 ms (57.73 tok/s)   B=64 743.1 ms (86.13 tok/s)
[fused] tokens [11, 427, 378, 490, 17045, 291]
[fused] B=1 573.5 ms   B=32 576.9 ms (55.46 tok/s)   B=64 664.6 ms (96.30 tok/s)
```

**Tokens identical**, confirming the block-alignment argument. But the win is
batch-dependent, and the isolated benchmark could not have shown it: fused is
1.12× at 64 and **0.94× at 1 and 32**. In isolation the fused pair was faster
everywhere (1.23× at M=1); in the model the wider program config costs more than
the dispatch it saves until there are enough rows to pay for it. Measuring an op
on its own says what the op costs, not what the model costs.

The weight set is fixed at construction — both resident does not fit — so
`TTEngine` selects it by slot count (≥64), and falls back silently when the
prebuilt tensors are absent.

**Decode throughput 0.75 → 96.30 tok/s: 128×.**

---

## Observation 12 — thirteen experiments on the engine/trace bug, and what they rule out

The traced decoder is correct through `TTModel` and corrupt through `TTEngine`.
Rather than keep guessing, the shadow experiment turned it into a bisection: patch
`TracedDecoder.step` to also run an eager step, and the engine produced
`' Paris.'` instead of `'!!!!'`. A bug that *disappears* when unrelated work is
added is a state or ordering problem, not a logic error.

Bisecting what the shadow did:

| interleaved after each replay | result |
|---|---|
| nothing | garbage |
| `synchronize_device` | garbage |
| read back the output | garbage |
| a tiny eager op | garbage |
| a large allocate-and-free | garbage |
| `execute_trace(blocking=True)` | garbage |
| **a full eager `model.step`** (before *or* after) | **correct** |

And differential instrumentation on the inputs: every bound buffer the trace
reads — `embed`, `ngram`, `rope_cos`, `rope_sin`, `cur_pos` — is **bit-identical**
to what an eager step writes, maxdiff 0.0 on all five, on every step. (The first
run showed `ngram` differing by 0.05; that was the harness calling `_fill_inputs`
before the token was appended to the history, not a real difference. Worth
stating: my own instrumentation lied to me first.)

So it is **not** synchronisation, **not** the inputs, **not** allocation size, and
not any of the ten structural candidates. Something a full eager step
re-establishes in device state is missing on replay. The workaround costs an
entire eager step, which is precisely what trace exists to avoid, so it stays a
diagnostic rather than a fix, and `use_trace` stays False.

`execute_trace` is now `blocking=True` regardless: it did not fix this, but the
caller reads the output immediately and correctness should not rest on timing.
It costs 1-4 %:

```
B= 1 MATCH  eager 515.7 ms -> traced 255.3 ms (2.02x)
B=16 MATCH  eager 552.7 ms -> traced 364.3 ms (1.52x)
B=32 MATCH  eager 538.7 ms -> traced 417.8 ms (1.29x)
```

### A hazard the fused experts introduced

The fused `ffn_gateup_exps` tensors live in the same manifest as the halves they
replace, so any harness doing `for n in w.entries: w.get(n)` now allocates both
sets and dies with *"Not enough space to allocate 235929600 B"*. `TTWeights`
preload now skips whichever set will not be read. Adding a second spelling of the
same weights to a shared manifest is a footgun; the alternative -- a separate
cache directory -- would have avoided it.

---

## Observation 13 — the conv was re-slicing constant weights every token

With the fused experts in, the profile shifted: `moe_routed` 34.8 %, `deltanet`
29.7 %, `hc_gate` 13.0 %. DeltaNet had never been broken down internally, and its
four input projections are only 6.7 % of it, so the other 93 % was unmeasured:

| stage | per call | share of deltanet |
|---|---|---|
| delta_rule | 3.843 ms | 37.2 % |
| conv (ring) | 2.065 ms | 20.0 % |
| l2norm ×2 | 1.402 ms | 13.6 % |
| rest | 3.011 ms | 29.2 % |

The conv held plain waste that two earlier passes over this code missed:

```python
for tap in range(k):
    w_tap = ttnn.slice(weight, (0, 0, 0, tap), (1, 1, channels, tap + 1))
```

The weights are constant, yet every token re-sliced them: **4 taps × 36 DeltaNet
layers + 4 for the PLE = 148 redundant ops per step**. They are now sliced once on
first use and cached on the model.

### The correction it forced

The measured effect, two runs per mode, 25 samples each:

| batch | split | fused |
|---|---|---|
| 1 | 522.4 / 520.3 ms | 504.8 / 523.0 ms |
| 32 | 57.57 / 57.97 tok/s | 59.51 / 58.49 tok/s |
| 64 | 86.66 / 86.60 tok/s | **97.47 / 97.36 tok/s** |

Observation 11 reported the fused experts as **0.94× at batch 1 and 32**, and the
engine gated fusion on slot count because of it. That is wrong. The regression
came from a single run taken immediately after writing 40 GB of new weight files,
reading them cold — 6.1 s weight load against 5.3 s now. With the conv taps
hoisted and the cache warm, fusion is neutral at batch 1 and ~2 % better at 32.
The slot-count gate is gone; fusion is on whenever the weights exist.

Two lessons, both about my own measurements rather than the hardware. A single
run is not a measurement when the run had a cold cache — the repeat is what
distinguishes a property from an accident. And a decision built on one number
(the ≥64 gate) inherits that number's error, so it has to be revisited when the
number is.

**Decode throughput 0.75 → 97.4 tok/s: 130×.**

---

## Observation 14 — two more candidates, both closed by measurement

With DeltaNet decomposed, two structural ideas looked promising. Neither
survived, and both cost one benchmark instead of a refactor.

**The conv's layout round trip.** Every call converts `[1,1,B,C]` to `[1,B,C,1]`
and back — 4 data-movement ops × 36 layers. The column layout dates from the
old concat-based window; with a ring, each entry is already a per-sequence
column, so it looked like leftover staleness of the same kind as the conv taps.

```
permute+transpose  ->[1,B,C,1]         0.149 ms
transpose+permute  back                0.131 ms   (0.280 ms/call = 10.1 ms/step)
tap multiply [1,B,C,1] x [1,1,C,1]     0.242 ms
tap multiply [1,1,B,C] x [1,1,1,C]     0.356 ms   <- 47 % slower
```

There are four tap multiplies per call, so the flat layout adds 0.456 ms to save
0.280 ms: a net loss. The round trip buys a faster inner loop, and it stays.

**The head-tiling repeat.** K/Q have 4 local heads to V's 12, so q and k are
`repeat`ed threefold — measured at **0.781 ms/call, 28.1 ms/step (4.3 %)**, the
largest single item in DeltaNet's remaining bucket. The materialisation is
avoidable in principle: leave q/k at `[B*n_k, 1, 1, Dk]` and broadcast against a
state viewed as `[B*n_k, reps, Dk, Dv]`. `ttnn.matmul` refuses:

```
TT_FATAL matmul_device_operation.cpp:306: a_shape[i] == b_shape[i] || (a_shape[i] == ...
```

The alternative — packing `reps` into the output dimension as
`[B*n_k, 1, Dk, reps*Dv]` — needs the recurrent state laid out with `reps`
adjacent to `Dv`, where the head tiling currently makes `reps` the *slowest*
axis. That is a state-layout redesign plus compensating permutes that could
consume the 4.3 %, so it belongs with the MoE grouped-GEMM as a project rather
than a tweak.

### Where the decode step now stands

`moe_routed` 34.8 %, `deltanet` 29.7 %, `hc_gate` 13.0 %. Within DeltaNet:
`delta_rule` 37 % (already fused into one stacked matmul), the conv 20 % (taps
now hoisted, layout measured optimal), `l2norm` ×2 13.6 %, head tiling 4.3 % of
the step. Each remaining item is either at the floor of what the available ops
express, or needs a custom kernel.
