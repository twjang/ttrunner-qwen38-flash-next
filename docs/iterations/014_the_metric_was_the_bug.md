# 014 — The metric was the bug

`013` fixed four real defects and then drew a wrong conclusion from them: that
nothing structural was left and the remaining gap was precision. It was not. The
model was still wrong, in a way none of that iteration's harnesses could see,
because every one of them measured a *relative* quantity — this path against
that path, this dtype against that dtype — and both paths had been given the
same wrong head pairing in the same change.

The fix was to measure something absolute.

## Observation 1 — agreement with the oracle cannot fail loudly enough

`013` ended with the decode path agreeing with the float32 reference on 40 % of
greedy tokens and no explanation. So this iteration priced the precision policy,
simulating the device's exact block-float numerics inside the CPU reference
(`simulated_precision.py`, using the existing `quant_sim` hook — which turned
out not to be applied on `get_rows`' per-row path, the one the MoE experts come
down, so every earlier simulation had left the only bfloat4_b tensors exact).

Agreement with the exact reference, 47 positions:

| weights | agreement | final hidden |
|---|---|---|
| dense quantised, experts exact | 46.8 % | 16.00 % |
| experts quantised, dense exact | 34.0 % | 20.91 % |
| everything at its device dtype | 29.8 % | 17.03 % |
| …with experts raised to bfloat8_b | 29.8 % | 17.27 % |
| …with dense raised to bfloat16 | 29.8 % | 17.89 % |
| everything at bfloat16 | 38.3 % | 14.92 % |
| the actual device | 40.4 % | — |

Every configuration lands in a 30-47 % band whose standard error at n = 47 is
about 7 points, and raising the experts from 4 bits to 8 changes nothing at all.
That is not what a precision gradient looks like. It is what a **chaotic** system
looks like: 512 experts with top-10 routing, so any perturbation flips a
selection somewhere, and a flipped selection can change a token. Agreement
between two implementations of the same weights is therefore bounded well below
1 whatever their precision, and cannot distinguish a small numerical difference
from a wrong model.

**Remedy.** Measure the model against the *text*, not against another
implementation: next-token top-1 accuracy and NLL on real prose
(`reference_quality.py`, `device_quality.py`). Absolute, no reference forward
needed, and nothing to hide behind.

## Observation 2 — the oracle was not a language model

The first thing the new metric said was that the CPU reference predicted the
actual next token **12.8 %** of the time, with median NLL 9.7 and perplexity
4445. A working 8B model on encyclopedic prose does far better than that.

It had passed every check the project had. It reproduced llama.cpp's greedy
output in `004`. It generated fluent text on demand. That is the trap: a model
with a wrong head pairing still produces grammatical, locally-plausible English,
because most of the machinery is intact and greedy decoding is self-consistent.
Fluency is not evidence.

The tokenizer round-tripped the text exactly, so the targets were not the
problem — the model was.

## Observation 3 — V heads are tiled, and upstream does not settle it

`013` had changed the DeltaNet Q/K expansion from `repeat` to
`repeat_interleave`, on the strength of upstream's HF code
(`modeling_qwen4_exp.py:594`) and a structural argument that tiling cannot be
head-sharded. The README's third checkpoint quirk said the opposite, and was
overruled.

The new metric settles it in one run each:

| expansion | next-token top-1 | top-5 | mean NLL | perplexity |
|---|---|---|---|---|
| tiled (`repeat`) | **80.9 %** | 97.9 % | 0.703 | 2.0 |
| grouped (`repeat_interleave`) | 12.8 % | — | 8.4 | 4445 |

The GGUF converter permutes the head order, so upstream's expansion and this
checkpoint's are not the same operation. The README was right, and reading
upstream instead of measuring cost a day.

**Remedy.** Tiling restored in the reference, with the measurement recorded next
to it so the next reader does not re-litigate it from upstream's source.

## Observation 4 — the device could not express the pairing at all

The structural argument in `013` was the one true thing in that section: tiling
*cannot* be served from the device's shard layout. Device d holds k-heads
`[4d, 4d+4)` and v-heads `[12d, 12d+12)`, and tiling sends v-head `12d+i` to
k-head `(12d+i) % 16` — device 0's twelve v heads need k-heads 0-11, which live
on three devices. Tiling the four local heads, which is what the device did, is
a third pairing matching neither convention.

So the device's DeltaNet had been wrong since head-sharding was introduced, and
no expression in `_linear_attention_step` could fix it. The data has to move.

**Remedy.** All-gather the sixteen K/Q heads and let each device select its
twelve with a fixed 0/1 matrix — one matrix for all 36 layers, built once, each
device's own column block sharded to it. The chunked path needed nothing new:
its host-side `prepare()` already had every device's channels in hand and simply
indexes them.

| | top-1 | top-5 | mean NLL | perplexity |
|---|---|---|---|---|
| device before | 25.5 % | 51.1 % | 5.424 | 227 |
| device after | **83.0 %** | 97.9 % | 0.682 | 1.98 |
| float32 reference | 80.9 % | 97.9 % | 0.703 | 2.00 |

`'The capital of France is'` now continues `' Paris. The capital of Germany is
Berlin'` — the output `004` validated against llama.cpp — on both paths.

Cost: 239.1 ms traced against 236.3, **+1.2 %**, for 36 all-gathers and 72 small
matmuls. Fixing the shard layout at conversion time instead would cost nothing
at runtime and about 200 MB per device: give device d the twelve k/q heads its v
heads need rather than a contiguous four, and the expansion disappears
altogether. That is a change to `split_qkv_channels` and to everything keyed to
v-head order.

## Observation 5 — chunked prefill was ready as soon as the model was

With the pairing fixed, prefill needed no further numerical work. Judged the
same absolute way — prefill 32 tokens, then score the rest against the text:

| | top-1 | mean NLL | perplexity |
|---|---|---|---|
| prefilled | 14/16 = 87.5 % | 0.335 | 1.40 |
| stepped (same positions) | 13/15 = 86.7 % | 0.364 | 1.44 |

Indistinguishable. Three things then stood between it and the engine:

* the chunk paths rebound their conv and PLE rings to fresh tensors, and a trace
  replays against the addresses it recorded — they now copy in place;
* `prefill` assumed the slot was at position 0, so it could not follow a reused
  prefix — it now resumes from `state.positions[0]` and refuses a start that is
  not tile-aligned, because `fill_cache` asserts one;
* the engine has to leave the last prompt token to the decode loop, because the
  step that consumes it produces the first output logits.

**Not** solved: chunked prefill and a captured trace still cannot coexist.
Prefill runs eagerly and allocates gigabytes of temporaries per call, and after
a few of them the trace replay returned token 0 repeatedly. Eager prefill is
correct — warm and cold turns agree token for token — so the engine turns the
trace off when prefill is enabled and says so. A 2000-token prompt with 200
output tokens is ~194 s eager-and-prefilled against ~1047 s traced-and-stepped;
short prompts invert it.

## Observation 6 — a slot already holds the conversation

Independent of all the above: a finished sequence leaves its whole conversation
in the slot — recurrent state, conv rings, K/V. The next turn of the same chat
has that as an exact prefix, so it only has to feed what it added. A slot is
reusable only while its recorded prefix is exactly what its state consumed; an
idle engine does not step at all, so with one slot a follow-up turn always hits,
while a step taken for another slot feeds this one a filler token and
invalidates it.

Measured through the engine over three chat turns plus a deliberate miss: reuse
counts exact, the unrelated prompt correctly missing, warm output identical to
cold, and time to first token 2.07 s against 7.49 s.

## Where it leaves us

The engine is, for the first time, a working implementation of this model: 83 %
next-token accuracy against the reference's 80.9 %, on both the decode and the
chunked-prefill path, at 239 ms a token traced.

What this iteration should be remembered for is not the fix but the metric. Four
of the five defects in `013` and `014` survived because every check was relative
— one implementation against another, one dtype against another, a greedy sample
against a human's sense of fluency. The one check that would have caught all of
them costs a single forward pass and no oracle: *does the model predict the text?*
