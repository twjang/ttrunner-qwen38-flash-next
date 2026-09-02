# 013 — The two paths disagreed because the model was wrong

Prefill had been chasing the decode path for two iterations, on the assumption
that decode was correct and prefill had a bug. It turned out that prefill had
two bugs, decode had one, the CPU reference had one, and the last of those was
in the *model*, shared by every path — so "does prefill match decode" was never
a question that could be answered by comparing them to each other.

## Observation 1 — a per-layer distance cannot tell noise from a bug

`prefill_bisect.py` reports, per layer, how far prefill's hidden state is from
decode's. After 012's fixes that read 1 % at layer 0 growing to ~30 % at layer
47, with no jump anywhere: the shape of accumulation, not of a defect.

The obvious control — how far is *decode* from the float32 reference? — came
back at ~50 % at every layer including layer 0, flat. Flat from layer 0 is not
accumulation either, and the decode path was supposed to be verified.

The metric was the problem. The tensor being compared is the hyper-connection
stream: four redundant copies of the residual that the output mixer averages
down to one. Individual streams can wander a long way while the mixed result —
the only thing the LM head sees — barely moves. Every comparison after this one
uses either the mixed hidden or the token.

**Remedy.** Judge a path by the token it emits, against the float32 reference,
teacher-forced so disagreements are counted independently rather than
compounding after the first one (`three_way_agreement.py`). And when a block is
suspect, drive that block alone from a known input rather than reading it
through 48 layers of accumulation (`attn_block_check.py`,
`deltanet_block_check.py`, `branch_sequence_check.py`).

## Observation 2 — an additive mask pads with zeros

Driven alone, `_attention_chunk` disagreed with `_attention_step` by 92.7 % at
position 0 of an 8-token chunk, decaying monotonically to 29 % by position 7.

Position 0 with an empty cache is the simplest case attention has: one key, so
the softmax is 1 and the block collapses to `W_out @ (sigmoid(gate) * v)`, with
rope the identity. Computed in float32 from the GGUF weights, the decode path
was 1.07 % away and the chunked path 96.5 %.

Tracing the block's intermediates put every one of them at the bf16 floor up to
the `scaled_dot_product_attention` call, which returned something 99.7 % wrong.
The K/V slice and the mask are both TILE_LAYOUT, so a key length that is not a
multiple of 32 is padded to one — and an additive mask pads with **zeros**,
which means *attend to me*. The softmax was spreading over up to 31 all-zero key
positions. The decay across positions was dilution: as real positions arrive,
the padding is a smaller share of the row.

**Remedy.** Slice the cache to a whole tile and run the causal condition over
that width. Every query position is `< total` by construction, so the same
comparison that enforces causality also masks the pad. The chunked block then
matched the decode block bit-for-bit over 8 positions.

## Observation 3 — a ring is not a list

`prefill_state_check.py` compares what a prompt *leaves behind*, which is what
every token generated afterwards depends on and which no output comparison
touches. After an 8-token prefill the DeltaNet ring counters read 0 where decode
leaves 8, and the PLE ring's oldest column was 100 % wrong.

Neither decode mode lays a ring out oldest-at-index-0: `trace_safe_rings` keeps
the newest at index 0 and shifts on every step, the rotating path leaves the
oldest at `step % len(ring)` and moves the index instead. Both chunk paths
assumed oldest-first and neither advanced the counter.

**Remedy.** `_ring_oldest_first` / `_ring_from_oldest_first`, used by both chunk
paths, so a prompt consumed in chunks leaves exactly what the same prompt
consumed token by token leaves — ring and counter. The PLE ring drops from
100 % to ~1 %.

## Observation 4 — the oracle disagreed with itself

With prefill and decode both cleaned up they still emitted different tokens, so
the question became which of them the *reference* agreed with. It agreed with
neither, because the reference disagreed with itself: the same prompt through
one forward and fed token by token diverged at layer 3 — the first sparse
attention layer — 0.0000 % through layers 0-2 and 38 % by layer 47, with a
different next token.

`_indexer_mask` took "the trailing partial block is always visible" from the
global `kv_len`. A query at position p may only use blocks ending at or before
p, so everything after the last such block has to come from the tail, and the
tail is a property of p. From `kv_len` it belonged to the last query; and with
`kv_len` an exact multiple of the block size there was no tail at all, so
`causal & qsa` was empty — a softmax over an all-masked row.

**Remedy.** `tail_start = ((query_pos + 1) // ratio) * ratio`, per query. The two
reference paths now agree to 1e-4 % at every layer and emit the same token,
which is what makes the reference usable as an oracle at all. Locked by
`tests/test_indexer_mask.py`, which needs neither a checkpoint nor a device.

## Observation 5 — V heads are tiled over K heads, and the shard layout was wrong

> **Corrected in `014`.** This section originally concluded that V heads are
> *grouped* and that the fix was `repeat_interleave`. That was wrong: it
> followed upstream's HF code without measuring, and it made both engines
> worse. The observation below is the part that held — the DeltaNet branch was
> 41 % out and the expansion was the cause — with the conclusion replaced.

With a trustworthy oracle the decode path could be measured, and it agreed with
it on 25 % of tokens of real text. Bisecting layer 0 — sub-block by sub-block
against the reference's own methods — put all of it in one place: the
hyper-connection mix before the branch at 1 %, the weights at their dtype floor,
and the DeltaNet branch **41 %** out. Tracing that block's intermediates
narrowed it to `q` and `k`, the two tensors expanded from K heads to V heads.
`v`, which is not expanded, was at 1.4 %.

`README` recorded, as a property of this checkpoint, that "DeltaNet V heads are
stored tiled, not grouped — Q/K expand with `repeat`, not `repeat_interleave`."
Upstream does the opposite (`modeling_qwen4_exp.py:594`), and that looked
decisive. It was not: the GGUF converter permutes the head order, so the two are
different models here, and the README was right. What settled it was
`reference_quality.py` — next-token accuracy on real text, 80.9 % tiled against
12.8 % grouped — a measurement that did not exist yet when this section was
written. See `014`.

The part of the diagnosis that survives: tiling **cannot** be served from the
device's shard layout. Device d holds k-heads `[4d, 4d+4)` and v-heads
`[12d, 12d+12)`; tiling sends v-head `12d+i` to k-head `(12d+i) % 16`, so
device 0's twelve v heads need k-heads 0-11, which live on three devices. The
device had been tiling its own four, a third pairing agreeing with neither
convention. Fixing it is a sharding problem, not an expansion one, and `014`
does that.


## Where it leaves us

> **Superseded by `014`.** The numbers below were measured with the wrong head
> pairing in both engines. They are kept because the *method* they introduced —
> judge a path against the float32 oracle, not against the other device path —
> is what made 014 possible, and because the conclusion they led to ("nothing
> structural is left, it is all precision") was itself wrong and worth not
> repeating.

Teacher-forced against the float32 oracle over 47 positions of real text:

| path | overall | oracle confident (margin > 2) | oracle close |
|---|---|---|---|
| decode | 19/47 = 40.4 % | 7/11 = 64 % | 12/36 = 33 % |
| prefill (16 prefilled) | 9/33 = 27.3 % | 2/6 = 33 % | 7/27 = 26 % |

The reading at the time was that nothing structural was left to find — every
branch of every layer kind was at its dtype floor, the weights round-tripped at
theirs, state carried between steps did not drift, and the router swapped about
one expert in fifty. All of that was true and none of it was the problem. What
the harnesses could not see was that both engines agreed on a wrong model, which
is exactly the failure mode "compare against the oracle" was supposed to catch —
and would have, had the oracle not been given the same wrong pairing in the same
change.

## What to distrust in the earlier log

* "verified token-for-token against the CPU reference" (README, 010, 012) meant
  one short prompt.
* Prefill's remaining distance from decode was never evidence about prefill on
  its own — decode was the larger error for most of it.
* This iteration's own Observation 5 conclusion, and the table above. See `014`.
