# 012 — Single-user latency, and the full 262144-token context

Date: 2026-09-01

Goal: assume the server has one user. Minimise latency for batch < 4, and make
the K/V cache reach the model's full supported context.

---

## Observation 1 — long context costs memory, not time

The model advertises 262144 tokens; the engine was defaulting to 4096. Two
questions decide whether the full context is usable: does it fit, and what does
it cost per token.

It fits at one sequence. The cache is `(batch, 2 kv-heads, seq, 256)` in bf16 for
each of k and v, on 12 sparse-attention layers, replicated per device:

```
2 x 1 x 2 x 262144 x 256 x 2 bytes x 12 layers = 6.44 GB per device
```

against roughly 8 GB left after 24.94 GB of weights. And the per-token cost is
**flat in position** — one step, timed at a position set directly rather than
generated to:

```
pos=4      495.9 ms      pos=1024    496.5 ms
pos=64     500.6 ms      pos=4096    495.7 ms
pos=256    495.7 ms      pos=16384   504.0 ms
pos=512    496.0 ms      pos=65536   501.2 ms
```

Flat because the step is dominated by the MoE and DeltaNet, which do not see the
context at all, and because the sparse-attention budget is fixed at 2048 no
matter how long the history is. The 36 DeltaNet layers carry a constant-size
recurrent state, so only 12 of 48 layers hold anything that grows.

The first attempt at this measurement tried to *generate* to position 65536,
which at ~0.5 s/token is nine hours. Setting `state.positions` directly is
equivalent for timing, since attention reads up to `cur_pos` and the cache
contents do not affect the cost.

**Remedy.** `max_seq_len` defaults to the model's `context_length`, and slots and
context are checked against the DRAM budget at construction:

```
K/V cache for 2 slots at 262144 tokens is 12.9 GB per device, over the ~8 GB left
after weights. Lower max_concurrency or max_seq_len: the two trade directly, and
one slot at the full 262144 context fits.
```

`max_concurrency` now defaults to 1. For a single user that is the right trade:
every extra slot costs 6.4 GB of context reach.

---

## Observation 2 — the trace bug was allocation ordering, and the LM head was the culprit

Trace capture is worth 2.02x at batch 1, which for a single user is the entire
latency story. It had been off by default because a captured decoder was correct
through `TTModel` and produced garbage through `TTEngine` -- ' Paris.' came back
as '!!!!' -- after thirteen experiments failed to isolate it.

The break came from asking a different question. Every previous "fix" interleaved
a full eager step before *every* replay, which costs exactly what trace saves. So:
is it per-replay, or one-time?

```
one eager step, on the first replay only   ->  CORRECT
```

One-time. That reframed it from a per-step interaction to an initialisation
problem, and the remaining bisection was quick:

| after capture | result |
|---|---|
| construct a second TTState (allocates nothing -- lazy) | garbage |
| one full eager `model.step` | correct |

So it is eager *execution* that matters, not merely holding memory. Two attempts
to put that step inside `TracedDecoder.__init__` still failed, which ruled out
"any eager step, anywhere" and left the difference that mattered: the step that
worked ran on a **freshly allocated** state, i.e. it allocated buffers *after*
`end_trace_capture`.

**Root cause.** A trace records a graph bound to fixed addresses, so everything
the steady-state loop allocates must exist before capture. The warmup step covers
the decode graph — but not the LM head. `output.weight` is loaded lazily, and
`logits`/`greedy_tokens` allocate their own intermediates, and in the engine all
of that first happens *after* the capture region, landing on memory the recorded
graph depends on.

**Remedy.** `TracedDecoder` runs `logits` and `greedy_tokens` on the warmup
output before `begin_trace_capture`.

This explains every earlier null result at once. The standalone harnesses all
preloaded the full weight set (including `output.weight`) and called the LM head
before capturing, so none of them could reproduce it — the ten "excluded"
hypotheses were excluded against a setup that had already applied the fix by
accident. A tiny op and an 8 MB allocate-and-free did not help because neither
creates the LM-head buffers; a full eager step did, by allocating enough to move
things.

The lesson is about the shape of the search. Thirteen experiments asked "what is
different about the engine?" and each answered "not this". The one that worked
asked "is this per-step or one-time?" — a question about the *bug's* structure
rather than the environment's, and it cut the space in half at a stroke.

---

## Result

Single user, default configuration (1 slot, 262144 context, traced, fused
experts), verified against the reference output:

```
slots=1  seq=262144  trace=ON  fused=True
text ' Paris.\n\nThe French city in Europe'
16 tokens in 4.82 s -> 301.4 ms/token amortised, 229 ms/step
```

229 ms/step against 496 ms eager: **2.16x**. The amortised per-token figure is
higher because a 5-token prompt costs 5 device steps before the first output
token — prompts are fed one token per step, and batched chunked prefill is the
open item there.

---

## Observation 3 — chunked prefill: 11.3x, and still wrong

The full context is only useful if a long prompt can be loaded into it. Prompts
are fed one token per device step, so at 229 ms/step a 100k-token prompt is over
six hours before the first output token. A 262144-token context that takes hours
to fill is a number, not a capability.

`TTModel.prefill` exists for this and was stale in three ways, all now fixed:

* `_causal_conv_chunk` and `_ple_chunk` held their windows as single
  `[1,1,C,state_len]` tensors; the decode path had moved to *rings* of single
  columns. They now consume and produce rings — and the order matters, because
  the two decode modes index it oppositely: `trace_safe_rings` reads
  `state[age-1]` (newest first) while the rotating path at step 0 reads
  `state[0]` as the oldest. Handing back the wrong order is silent, the conv
  simply runs with its taps permuted.
* `_linear_attention_chunk` used the *global* head counts from before the
  DeltaNet was head-sharded, and prepared its inputs from `from_dev`, which
  returns device 0's copy. For a head-sharded tensor that is device 0's heads, so
  all four devices would have run the recurrence on the same quarter of them. It
  now gathers, prepares each device's own heads, and shards the eight prepared
  tensors back on dim 0.
* the prefill MoE named `ffn_gate_exps`/`ffn_up_exps` while the model runs fused,
  which loads the split halves *lazily on top of* the fused tensor — another
  ~11 GB per device, and an out-of-memory at the first MoE layer rather than a
  slow path. It now selects the same weights decode does.

The payoff is real:

```
prefill     128 tok in  5.77 s (  45.0 ms/tok)
step path   128 tok in 65.15 s ( 509.0 ms/tok)     11.3x
```

**And the tokens do not match.** Not drift over a long prompt — it differs from
the first chunk:

```
len=  4  prefill->154171  step->  2880   hidden maxdiff 28.9
len= 16  prefill->   695  step-> 14274   hidden maxdiff 45.1
len= 64  prefill->   351  step->    71   hidden maxdiff 26.2
```

Narrowed: after 128 tokens the layer-0 recurrent state is within 0.06 of the
sequential path — plausibly just precision, since the chunked op computes the
same recurrence a different way — while the final hidden is off by 45, so the
error amplifies through the stack rather than starting large. `prepare()` does
apply the q/k l2-norm and the q scale, so that is not the missing piece.

One measurement of mine was itself wrong along the way and is worth recording:
the first layer-by-layer diff reported the conv window off by 11.7, which is an
artifact. The decode ring is *rotating*, so after 128 steps it sits at phase
`128 % 3 = 2` while prefill leaves it at phase 0; comparing index to index
compares different ages. The `recurrent` column was the honest signal.

So prefill stays off. `TTEngine` refuses `chunked_prefill`, and
`tests/test_prefill_contract.py` locks the three shape contracts so the next
attempt starts from a path that runs and is merely wrong, rather than one that
dies in `concat`.

---

## Observation 4 — where the single-user step actually goes, and what is left

Every earlier profile was taken at batch 64. At batch 1 the distribution is
similar, with the hyper-connection gate taking a larger share (instrumented, so
the total is inflated ~40 % by per-section syncs; the shares are the signal):

| section | per call | calls | share |
|---|---|---|---|
| deltanet | 5.576 ms | 36 | 28.6 % |
| moe_routed | 4.166 ms | 48 | 28.5 % |
| hc_gate | 1.261 ms | 97 | 17.4 % |
| reinject | 0.445 ms | 96 | 6.1 % |
| shared_expert | 0.737 ms | 48 | 5.0 % |
| qsa_attention | 2.664 ms | 12 | 4.6 % |
| all_reduce | 0.285 ms | 84 | 3.4 % |
| ple | 3.842 ms | 1 | 0.5 % |

Two things follow.

**The engine is at device speed.** A traced step measured standalone is 255 ms;
through the engine it is 229 ms per step (4.81 s for 21 steps). There is no host
bubble left to overlap — tokenizer, sampling and the queue cost nothing
measurable against the step.

**What remains is not dispatch.** Trace already removes the per-op host cost, so
the 229 ms is kernel execution on very small tensors. At batch 1 the MoE selects
exactly `top_k` = 10 of 512 experts, so there is no union waste at all, and the
hyper-connection gate runs 97 times per token at ~11 ops each — roughly a
thousand kernel launches whose cost is a per-launch minimum rather than
arithmetic. A rough bandwidth bound says the step reads on the order of 1-2 GB
per device per token, which at DRAM speed would be single-digit milliseconds, so
this is 20-40x off a memory-bound floor.

That gap is real headroom, but closing it means fewer, larger kernels: a
gather-based grouped GEMM for the MoE, and a fused hyper-connection gate. Both
are custom-kernel projects rather than a rearrangement of the ops ttnn offers,
and the measurements above are what a successor would start from.

---

## Observation 5 — the floor is op count, and a fifth of the ops are layout

The claim that "what remains needs fewer, larger kernels" is checkable. Counting
every device op in one batch-1 step:

```
6355 device ops per step
eager  494.1 ms  ->  77.7 us/op
traced 229.0 ms  ->  36.0 us/op
```

Trace removes ~42 us/op of host dispatch and leaves ~36 us of device time per op,
on tensors that at batch 1 are a single row. Step time is op count times a
per-kernel floor, not arithmetic — which is why the earlier bandwidth estimate
(single-digit milliseconds of weight traffic) is 20-40x below what the step
costs.

What the ops are:

| op | count | | op | count |
|---|---|---|---|---|
| multiply | 1334 | | permute | 317 |
| **reshape** | **914** | | sum | 302 |
| linear | 760 | | silu | 230 |
| add | 461 | | rms_norm | 160 |
| slice | 444 | | | |
| sigmoid | 326 | | | |

**914 reshapes and 317 permutes — 1231 ops, 19 % of the step — are layout, not
arithmetic.** And they are not free:

```
reshape [1,1,1,2560] -> [1,1,2560]    (rank only, no tile change)    99.6 us
reshape [1,1,1,2560] -> [1,1,20,128]  (tile change)                 132.1 us
reshape [1,1,1,10240] -> [1,1,4,2560] (the hyper-connection mean)   132.9 us
permute [1,1,4,128]  -> [1,4,1,128]                                 164.4 us
multiply [1,1,1,2560]                 (reference: real arithmetic)  277.3 us
```

(Sync-inclusive, so the absolute numbers are inflated; the ratio is the signal.)
A layout op costs roughly half an arithmetic op of the same size, and a reshape
that only changes rank costs nearly as much as one that re-tiles — ttnn's reshape
has a fixed cost regardless of whether data has to move. So they cannot be made
cheaper, only removed.

The largest sources are the DeltaNet head tiling (~10 reshapes per layer x 36)
and the hyper-connection gate's stream mean (2 per call x 97). Removing them
means choosing tensor layouts so intermediate shapes already line up — a refactor
across the model rather than a local change, and one where a mistake is the
quiet kind: this session has already had four bugs that were invisible at batch 1
or in a timing-only harness.

That is the honest end state for the single-user path with the ops ttnn offers:

* **2.16x delivered** (496 -> 229 ms/step), verified token-for-token.
* the remaining 20-40x to a memory-bound floor is 6355 ops x ~36 us, and closing
  it means fewer ops: a gather-based grouped GEMM for the MoE, a fused
  hyper-connection gate, and a layout pass to delete the 1231 shape ops.
