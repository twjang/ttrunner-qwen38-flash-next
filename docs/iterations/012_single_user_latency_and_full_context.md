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
