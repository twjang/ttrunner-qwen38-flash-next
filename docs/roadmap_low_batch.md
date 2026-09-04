# Roadmap: chunked prefill and the batch < 4 regime

> **Superseded in part by `docs/iterations/013` and `014` (2026-09-02).** Part
> A's premise — that prefill is the thing that is wrong — did not survive
> measurement: the decode path had a bug of its own, the CPU reference that both
> were judged against had another, and the DeltaNet head pairing was wrong in
> every path. With those fixed the device predicts real text as well as the
> float32 reference (83.0 % next-token top-1 against 80.9 %), chunked prefill
> matches the decode path and is enabled for one-slot engines, and prefix reuse
> across chat turns is in. A1-A4 are therefore done or moot, and A5 (the QSA
> indexer) is done too -- `docs/iterations/015`. Part B still stands as written;
> its speculation items are only *exact* if the verifier is the model you meant
> to run, which as of 014 it finally is. `docs/HANDOFF.md` §5 carries the
> current ordering.

Date: 2026-09-02. Status of the code this is written against: branch
`feat/ttnn-inference-stack`, single user, 1 slot, 262144-token context, traced
decode at **236 ms/step** (486 ms eager). Everything below is a proposal with the
measurement it rests on; nothing here is implemented unless marked so.

The one number to keep in mind: at batch 1 a step is **6355 device ops at
~36 µs each**. Time is op count times a per-kernel floor, not arithmetic, and the
batch axis is free up to 64 rows (`docs/iterations/011`, Observation 9). Every
idea below is worth exactly what it does to one of those two facts -- fewer
launches, or more useful tokens per launch.

---

## Part A -- chunked prefill

### A0. Where correctness stands (updated 2026-09-02)

`TTModel.prefill` runs 11.3x faster than the step path and was wrong from the
first chunk. Two bugs are now found and fixed by per-layer bisection
(`model.probe` hook + `scratchpad/t_bisect.py`, both described in the handoff):

| bug | symptom | fix |
|---|---|---|
| `_linear_attention_chunk` returned the `ssm_out` partial sums without the `all_reduce` the decode step does | every device saw a quarter of the DeltaNet output; recurrent state (which does not pass through `ssm_out`) stayed correct, which is why the earlier narrowing found "layer-0 state within 0.06, final hidden off by 45" | `return self.all_reduce(out)` |
| prefill conv-tap cache keyed `("ssm_chunk", channels)` -- no layer | every DeltaNet layer after the first ran with layer 0's conv weights; layer 0 matched decode to 0.005, layer 1 was off by 30 % | key by layer, as decode does |

After both, per-layer agreement with decode on a 4-token prompt:

```
layer  0-2  (DN, DN+PLE, DN)   max|Δ| 0.001-0.008   scale 0.3-0.5    ~1 %
layer  3    (first QSA)        max|Δ| 0.014-0.022   scale 0.43       ~5 %   <- first jump
layer 11    (QSA)              max|Δ| 0.07          scale 1.36       ~5 %
layer 27    (QSA)              max|Δ| 0.22          scale 3.0        ~7 %
layer 47    (QSA)              max|Δ| 1.2-1.6       scale 4.4       ~30 %
```

Final hidden maxdiff 36.9 -> 13.3, and the greedy token still differs at every
length tried (4, 16, 64). The remaining error is small per layer and grows with
depth, which is what both "bf16 noise through 48 MoE layers" and "a small bug in
the attention chunk" look like. **Decide which before touching code:** run the
same per-layer diff of the *decode* path against the CPU reference
(`ttrunner_qwen38_flash_next.reference`) on the same 4 tokens. If decode-vs-reference is also ~5 %
at layer 3 and ~30 % at layer 47, prefill is at the noise floor and the token
mismatch is routing sensitivity (top-10 of 512 experts flips on small
perturbations) -- then measure agreement rate over a few hundred tokens instead
of demanding token-for-token equality. If decode-vs-reference is 10x tighter,
`_attention_chunk` has a bug; its differences from the step are `repeat_interleave`
GQA expansion + explicit mask + `scaled_dot_product_attention` (vs `sdpa_decode`
handling GQA internally), and `fill_cache` (vs `paged_update_cache`).

**Procedure that found both (reuse it, do not re-derive):** record the hidden
after every layer on both paths for the *same* 4-token prompt, print
`max|Δ|` per layer per position next to the tensor's scale. The first layer
whose error jumps is the culprit; everything after it is propagation. The
one-token-at-a-time path is the oracle because it is verified against the CPU
reference.

### A1. Make it fast for *short* sequences too (needed by MTP, Part B)

The chunk path is built for 128 tokens: `gated_delta_attn_seq` fixes the chunk at
128, so at 4 tokens nothing amortises. At 128 tokens it is now **7.2 ms/token**
(925 ms a chunk). Measure `prefill(seq=k)` for k in {2, 4, 8, 16} before
designing anything on top of it -- MTP verification needs a *cheap* multi-token
step, and "cheap" here means comparable to one 236 ms decode step.

> **Lever 1 below is done**, and the host round trips this section was written
> around are gone: `deltanet.prepare_device` builds the op's eight inputs on
> device and `_attention_chunk` reads a paged cache, so the chunk path contains
> no host->device copy at all. `short_chunk_bench.py` measures the remaining
> fixed cost at 850 ms for k=1, which is still the wrong shape for verifying two
> or three drafted tokens -- so lever 2 is the live one. See handoff 5.2 and
> `docs/iterations/017`.

Levers, in the order I would try them:

1. ~~**Do `prepare()` on device.**~~ **Done.** `deltanet.prepare_device` does
   the l2-norm, the `beta` scaling, the decay cumsums and the triangular inverse
   in ttnn, in the op's own `[H, NC, C, D]` layout, so each device prepares the
   heads it already holds and nothing is gathered. The inverse is *not* the
   Newton-Schulz iteration proposed here: that form is what
   `deltanet.block_inverse` already used and it turned out unstable on real data
   -- 620875 % on device, where a matmul carries ~1e-3 rather than ~1e-7. It was
   replaced by a blocked `D + C` recursion that never grows. `017` has the
   story; the fix applied to the host path too, which had been quietly 0.27 out.
2. **Short-sequence DeltaNet without the chunk op.** For seq ≤ ~8 the recurrence
   unrolled `seq` times (the decode `linear_attn.decode_step`, ~40 ops) is
   36 × seq × 40 × 36 µs ≈ 52 ms × seq -- for seq = 4 that is ~200 ms on top of a
   step whose MoE/attention/HC parts run batched over the 4 rows for free. That
   is a 4-token step at ~430 ms, i.e. ~110 ms/token, *without* fixing anything
   in the chunk path. It also gives the intermediate recurrent states for free,
   which A4/B1 need.
3. **MoE chunking.** The default is now `moe_chunk=32`, measured rather than
   reasoned from the waste figures, and **capped** there: past one row tile the
   answer changes (handoff 5.8, invariant 13). For a short chunk
   `moe_chunk=seq` is right and needs no thought, since seq <= 32.
4. **PLE and attention chunks already batch over seq**; nothing to do until the
   profile says otherwise.

### A2. Prefix reuse across turns (largest real-world win for one user)

A single-user chat re-sends the whole conversation every turn. Today every turn
re-prefills it. The state a slot holds -- 36 recurrent states (12 heads × 128 ×
128 bf16 per device = 4.7 MB per layer), 36 conv rings, 12 K/V caches, one PLE
ring, the n-gram history -- *is* the prefix cache. Implementation in
`tt/engine.py` only:

* when a request arrives, compare its `prompt_token_ids` with
  `slot.request.prompt_token_ids + emitted tokens` of the slot that finished
  last; if the new prompt extends the old one, do not `reset_slot`, set
  `prompt_pos` to the shared length, and feed only the new suffix.
* the K/V cache is not cleared on reset already (`reset_slot` docstring); the
  recurrent state and rings are what a reset zeroes, so the check must happen
  *before* `reset_slot`.
* a mismatch anywhere before the shared length invalidates everything (the
  recurrence is not positional), so this is exact-prefix only. That is the
  common chat case.

Cost: zero device work. Payoff: a turn's time-to-first-token drops from
`(conversation length) × step` to `(new message length) × step`. Also add a
**snapshot/restore** of the same state (a dozen `ttnn.copy`s) so a system prompt
can be prefilled once at startup and restored on every fresh conversation.

### A3. Overlap and scheduling

Single user means there is nothing to overlap prefill *with*. Skip
prefill/decode interleaving entirely; it only matters at concurrency > 1.

### A4. Prefill correctness harness (keep it, extend it)

`t_prelen.py` compares final tokens; `t_bisect.py` compares per-layer hidden.
Add a third: per-sub-block probe inside one layer (after PLE, after attention
branch, after MoE) so that when a *layer* is identified, the *block* is one
run away. Every future prefill change runs all three before it is believed.

### A5. Long-context fidelity: the QSA indexer is not on the device path

Found while reading `_attention_chunk` for A0. The device model attends
**densely** over the whole history in every sparse-attention layer, both in
`step` and `prefill`. The header of `tt/model.py` explains why that is exact
below 2048 tokens (the indexer keeps 512 blocks of 4; with fewer blocks it keeps
them all). Past 2048 tokens the real model keeps the top-512 blocks by indexer
score and the device model keeps everything -- different outputs, and the
divergence grows with the prompt. The indexer weights are loaded
(`plan.py`, `blk.N.indexer.*`), the scoring op `experimental.indexer_score_dsa`
was validated at 1.0 % (`README`, op table), `LayerState.indexer_blocks` and `indexer_ring` exist,
and nothing uses them. The CPU reference (`reference/layers.py`) implements the
selection and is the oracle.

So "262144 context" today means: the cache holds it and the step time is flat;
the *attention semantics* match the model only for the first 2048 tokens. This
is the most important open correctness item for long prompts and belongs ahead
of prefill speed. Implementation sketch for decode: per QSA layer, keep the
pre-norm/pre-rope indexer keys `[1,1,T,128]` in the existing slot, score the
current query against them (`indexer_score_dsa`), `ttnn.topk` 512 block scores,
expand to a token mask or a gathered K/V, then `sdpa_decode` over the selection.
Static shapes (always 512 blocks, padded/masked below 2048) keep it traceable.

---

## Part B -- latency and throughput at batch < 4

### B1. Multi-token prediction (MTP) speculative decoding

**What exists.** The GGUF repo ships the MTP head as a separate file:
`MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` (2.79 GB, includes its own
`token_embd`/`output` copies) and `-shared-Q4_K_M.gguf` (1.91 GB, reuses the
main model's). It is **one extra layer, `blk.48`**: a full sparse-attention
block (`attn_{q,k,v,output}`, q/k norms, `indexer.*`), a full MoE block (512
routed + shared expert), both hyper-connection gates, plus the NextN glue:

```
blk.48.nextn.enorm        (2560,)         rms weight for the next-token embedding
blk.48.nextn.hnorm        (10240,)        rms weight for the 4-stream hidden (per stream)
blk.48.nextn.eh_proj      (5120 -> 2560)  projects concat(enorm(emb), hnorm(h)) back to width
blk.48.nextn.hc_head_{norm,down,up}       the head's own stream collapse (same shapes as output_hc_*)
```

Quant types in the file are Q4_K / Q5_0 / Q6_K / Q8_0 / bf16 -- Q4_K and Q5_0 are
implemented in `gguf/quants.py` but were never exercised by the main model
(`docs/iterations/003`). Bit-exactness against the codebook must be re-checked
for them first.

**Wiring (hypothesis, from llama.cpp's DeepSeek-V4 MTP graph, which is the other
hyper-connection model with NextN; the upstream HF code ignores `mtp.*` and the
vendored llama.cpp has no `qwen4exp` MTP graph yet).** For draft position i:

```
h      = 4-stream hidden of the main model *before* its final collapse   [1,1,1,10240]
e      = embed(token_{i+1})                                              [1,1,1,2560]
x      = eh_proj( concat( enorm(e) repeated to 4 streams,  grouped_rms_norm(h, hnorm) ) )  per stream
x      = layer_48(x)          # HC gate -> QSA attention -> reinject -> HC gate -> MoE -> reinject
draft  = greedy( output.weight @ hc_collapse(x, nextn.hc_head_*) )
```

Verify this against llama.cpp the moment it grows a `qwen4exp` NextN graph, or
against the HF `mtp.*` module if one is published. Until then treat the order of
`concat` (embedding first) and "hnorm is per-stream" as the two assumptions.

**Why it fits this hardware.** A draft step is ~1/48 of the model's ops (~5 ms
traced) plus an LM head. Verifying k drafted tokens is a k+1-row step, and rows
are free up to 64. The whole gain is therefore (accepted tokens per verify) ÷
(cost of a verify step ÷ 236 ms). Published acceptance for one NextN head is
around 0.7-0.85 per position on text; with k = 3 that is ~2.3 tokens per
iteration.

**The obstacle is the recurrence, not the KV cache.** Rewinding sparse
attention is trivial (rewind `positions`; the cache slots are overwritten). The
36 DeltaNet layers have *no positional cache*: after a verify step the state
has absorbed all k+1 tokens, and if only j < k are accepted there is no state
for "after j tokens". Three ways out, cheapest first:

1. **Unrolled recurrence inside the verify step (A1.2).** Run the DeltaNet
   recurrence token-by-token *within* the batched step and keep every
   intermediate state (k+1 × 4.7 MB × 36 layers -- fine). On acceptance of j
   tokens, `ttnn.copy` state_j into the live state. One extra copy per layer,
   no re-execution. Cost per iteration ≈ 236 ms + 52 ms × k for the unrolled
   part. At k = 3, α = 0.75: ~2.3 tokens / ~385 ms ≈ **170 ms/token, 1.35x**.
   Honest, and the unrolled path is also the correct short-sequence prefill.
2. **Snapshot + re-run.** Snapshot recurrent/ring state before verify; on
   partial acceptance restore and re-run the accepted prefix as a short chunk.
   Expected extra cost `(1 - P(all accepted)) × one short step`. Simpler, but
   pays a full step on every partial acceptance, which at α = 0.75 and k = 3 is
   58 % of iterations.
3. **Chunked op returning intermediate states.** `gated_delta_attn_seq` only
   returns the final state. A custom op that also emits the state at each
   position removes the unroll cost entirely -- this is the version worth the
   kernel work once (1) shows the acceptance rate is real.

**Also needed, whichever way:** a `TracedDecoder` for the k+1-row verify graph
(a second trace; rows are a shape, and a different shape is a different trace),
and a sampling rule. Greedy: accept while `argmax(verify logits_i) == draft_i`.
Temperature > 0: standard rejection sampling against the full distribution --
`greedy_tokens` does not apply and the logits gather (~1 MB at batch 1) is
needed anyway.

**Measurement plan before writing the engine glue:** (a) load `blk.48`, run the
head eagerly on hidden states dumped from the main model over ~200 tokens of
real text, report top-1 agreement with the *actual* next token. If that is below
~0.6 the rest is not worth building. (b) Time `prefill(seq=4)` per A1 -- if a
4-row verify step costs 2+ decode steps, fix A1 first.

### B2. Free speculation without the MTP head

The same verify machinery accepts drafts from anywhere. **Prompt-lookup /
n-gram drafting** (find the last 2-3 tokens in the prompt, propose what
followed) costs nothing and is strong on tasks that copy from the input --
summarisation, code edits, retrieval answers. Build the verify step for B1 and
this comes for free; ship this first because it has no head to load and no
weights to validate.

### B3. Fewer launches per step (the op-count floor)

From `docs/iterations/012`, Observation 5, 6355 ops per step:

| op | count | | op | count |
|---|---|---|---|---|
| multiply | 1334 | | permute | 317 |
| reshape | 914 (−98 done) | | sum | 302 |
| linear | 760 | | silu | 230 |
| add | 461 | | rms_norm | 160 |
| slice | 444 | | sigmoid | 326 |

At 36 µs a launch, each 100 ops removed is ~3.6 ms (1.6 %). Ranked by ops
removed per unit of risk:

1. **Hyper-connection gate as one op.** `gated_residual_mix` + `reinject` run
   97 times a step at ~11 ops each (~1070 ops, 17 % of time measured with
   syncs). The gate is `rms_norm -> linear(10240->320) -> linear(320->10240)
   -> sigmoid -> split -> weighted stream sum`. Fusing to 3-4 ops (norm,
   two matmuls with fused activation, one `ttnn.experimental` weighted sum)
   removes ~700 launches: **~25 ms, 11 %**. Pure ttnn composition first; a custom
   kernel only if the composed version leaves > 5 ops.
2. **Layout pass.** 1133 remaining reshape/permute. Most come in pairs around a
   matmul (`[1,1,B,H] -> [1,B,C,1] -> ... -> back`). Pick one layout per block
   and hold it: the DeltaNet conv already showed "flat layout" *loses* when done
   piecemeal (011, Obs. 12) because the multiply broadcasts get worse -- so this
   is a block-at-a-time refactor with a per-block A/B, not a sweep. Ceiling
   ~40 ms if every layout op vanished; realistic 15-20 ms.
3. **Sigmoid/silu fused into the matmul.** `ttnn.linear(activation=...)` exists
   for some configs; 556 activations, the ones directly after a linear can fold.
4. **`sum`+`multiply` mean patterns** (302 sums): the mean over hc streams in
   `gated_residual_mix` was already tried as slice+add and lost (012). Leave.
5. **MoE grouped GEMM.** `sparse_matmul` has a 4.5 ms floor per call at M = 1
   (011, Obs. 5), 48 layers × 2 calls (fused gate-up, down) ≈ 430 ms
   *instrumented*, which is the largest single block. A gather-based grouped
   GEMM over the 10 selected experts would be bounded by weight traffic
   (10 × 3 × 2560 × 640 × 0.5 B ≈ 25 MB, sub-millisecond). This is the biggest
   prize and a real kernel project; it is last because it is the only item on
   this list that cannot be done by composing ttnn ops.

### B4. Throughput at batch 2-3 is already free -- expose it

Step time is flat to 64 rows, so 3 concurrent sequences cost the same 236 ms.
For a single user that means:

* **`n > 1` / best-of-n sampling** in the OpenAI API: n candidates in one step.
* **Parallel tool-call branches / self-consistency** from the client side.
* The server already batches slots; `max_concurrency=3` with `max_seq_len ≈
  87k` fits the DRAM budget (6.4 GB per slot at full context). Document the
  trade rather than change the default.

### B5. Device-side sampling

With temperature > 0 the engine gathers `[B, 248320]` logits to host. At B = 1
that is ~1 MB and a few ms, but it also forces `greedy_tokens` off. Do top-k on
device (`ttnn.topk` over the per-device vocab shard, then a 4-way merge of
k × 4 candidates), gather only those, sample on host. Removes the gather and
lets speculative verification (B1) stay on device for the common top-k/top-p
case.

### B6. Things measured and refuted -- do not retry without a new idea

From 011/012: slice+add stream mean, flat conv layout, matmul batch broadcast
for head tiling, DeltaNet linear fusion (0.45 %), MoE chunking at decode,
`ttnn.concat` of bfloat4_b weights (requantises; fuse at conversion instead).

---

## Priority for one person-week

1. Settle prefill correctness (A0: one reference-vs-decode run decides noise vs bug).
2. QSA indexer on the device path (A5) -- without it the long context is a cache, not a capability.
3. Prefix reuse across turns (A2) -- engine-only, largest felt win for chat.
4. Unrolled short-step (A1.2) -- doubles as MTP verify and short prefill.
5. n-gram speculation on top of (4) (B2) -- proves the verify path with no new weights.
6. MTP head bring-up: quant check -> agreement measurement -> wire in (B1).
7. HC gate fusion (B3.1), then the layout pass block by block (B3.2).
