# 020 — The record had drifted from the code

Five of section 5's eight items were marked done. This iteration set out to
finish the other three, could not — they wait on one tt-metal defect — and
instead asked a different question of the five: *what does each claim rest on,
and does it still hold?* That found four defects, two dead harnesses, four wrong
published numbers, and one fact that explains three caps nobody had connected.

The finished work is in `017` (the DeltaNet inverse and the prepare port),
`018` (the trace-alternation defect) and `019` (the MoE row-group cliff). This
note is about the rest.

## Observation 1 — a paged K/V cache, and the four checks that were missing

5.2's stated blocker was an on-device `prepare()`, done in `017`. What remained
were four host dependencies in `_attention_chunk`, and all four have a paged
answer: `paged_fill_cache` takes the page table as a device tensor,
`chunked_scaled_dot_product_attention` takes `chunk_start_idx_tensor` and is
causal internally (so the tile-padded mask, the growing `kv_len` slice and the
`repeat_interleave` all go), and decode has paged variants of both its ops, so
one layout serves both paths. Every op measured at the model's real shapes
first; the refactor followed.

It came out bit-identical at every gate — decode, prefill at two widths, both
latency configurations, batch throughput, and the engine end to end emitting the
same 48 tokens on two prompts. That last one mattered: everything had been
checked through harnesses and *nothing* through `TTEngine`, while the cache
layout changed underneath it.

Two things needed measuring rather than reasoning. The chunked op's two chunk
sizes buy different things — `q_chunk_size` is the outer iteration count and
drives speed, `k_chunk_size` fixes the accumulation order and therefore the
answer; all-32 is correct but 36 % slower, all-128 is fast but moves prefill's
NLL, and q=128 with k=32 is both. And a misaligned `chunk_start_idx` is
**silent**: at q_chunk_size=128 a start of 32 returns 257 % nonsense rather than
an error.

Step 5, the capture itself, is blocked by `018`'s defect. Steps 1–4 stand on
their own.

## Observation 2 — a fix aimed at prefill was silently repairing decode

`019` found `ttnn.sparse_matmul` dropping rows past the first 32-row tile when K
spans more than one block, and fixed it by giving the config a single K block
whenever `per_core_M > 1`. That threshold is a *batch-size* threshold in decode,
which nobody noticed, and the README claimed batch 64 was "bit-exact vs
single-sequence" with no harness behind it.

`batch_equivalence_check.py` gives every row the same prompt, so all rows must
agree with each other and with a batch-1 run. Against HEAD and a worktree at the
pre-fix commit:

| | rows agreeing with row 0 | matching batch 1 |
|---|---|---|
| batch 64, before | **32/64** | 0/64 |
| batch 64, after | **64/64** | 0/64 |
| batch 32, either | 32/32 | **32/32** |

Batch 64 had been corrupting exactly half its rows — the ones past the first
tile — so the bit-exactness claim was false and the 97.4 tok/s the README
advertised was measured on output where half the sequences were wrong.

## Observation 3 — one row tile, and the three caps it explains

Batch 64 still does not reproduce batch 1, and the obvious next move was to make
it. Threading a flag so decode's expert config stops depending on the batch
changes nothing at all: 0 of 64 rows still match.

The reason is smaller and more general than the MoE. `row_tile_boundary_check.py`
takes a plain `ttnn.linear` and shows it returning a given row *identically* at
m = 1, 8, 16, 32 and *differently* at 33, 48, 64 and 128 — one bf16 ulp, the
same amount each time. No experts, no sparse matmul, no routing.

That single fact is behind three caps recorded separately:

* `moe_chunk` cannot exceed 32 (`019`)
* batch 64 decodes differently from batch 1
* `step_n` cannot reproduce k sequential steps past k=32

Each compares a row computed among ≤ 32 rows against the same row computed among
more, and that comparison cannot come out equal. It is not fixable in this
repository: splitting `expert_ffn` into 32-row groups so `per_core_M` stays 1
leaves `step_n` at k=33 exactly as far out, because the ops *before* the experts
have already diverged.

I got this wrong once in between — attributing it to the sparse-matmul config,
publishing that, and being disproved by the very fix it implied. The twenty-line
probe should have come first.

## Observation 4 — half of `step_n`'s advertised range was wrong

`step_n`'s guard read `1..64`, the batch cliff where 65 rows overflow L1, and
nothing had exercised it past k=8. It is exact at k=1, 2, 4, 8, 16 and **32**,
and 35.68 % out at 33, 48 and 64 — the same figure at all three. The guard now
stops at 32.

The test that was supposed to protect this asserted `"0 < k <= 64" in src`. It
passed throughout, and fixing the bug required editing the test that was
guarding it. That is the failure mode of a source-text test, and 50 of this
suite's 113 tests are source-text tests.

## Observation 5 — two cited harnesses had stopped working

`docs/HANDOFF.md` cites harnesses as evidence. Two of them did not run:

* `indexer_select_check.py` called `_indexer_select` with its pre-`q_cos`
  signature and raised `TypeError` after minutes of setup. Repaired, it reports a
  *better* result than 5.3 records: 2048 of 2048 tokens, nothing missing,
  nothing extra.
* `traced_step_n_check.py` hangs, on a pre-paged worktree as well as on HEAD, so
  it is `018`'s defect and not the refactor. Its 255 ms at k=2 is now marked
  historical.

A citation that no longer runs still reads as evidence, which is worse than no
citation. The API half of that is now caught by `pytest`
(`harness_api_check.py`, over all 53 harnesses); the device half cannot be, so
invariant 16 says to run a harness before quoting its number.

## Observation 6 — four published numbers were measured on a model that was wrong

The README's headline throughput predated `013`–`014`, i.e. the model that
emitted the wrong token, and was still on the front page. Re-measured with new
harnesses: batch 64 at **107.6 tok/s** (was 97.4), the server at **69.3 tok/s**
sustained generation at 32 concurrent (the old 37.84 had no recorded
methodology, so it is retired rather than beaten), prompt intake at **7.2
ms/token** (was 8.6, and ~45 before this session), and prefix reuse cold at
11.40 s with chunked prefill or 41.31 s without — where the old figure of 7.49 s
named no configuration and matches neither.

`docs/roadmap_low_batch.md` was worse than stale: its top-priority lever was
already done, and the implementation it suggested — a short Newton–Schulz
iteration for the chunk inverse — is precisely the unstable form `017` had just
replaced. Anyone following it would have re-implemented the bug.

## Observation 7 — and one finding of my own that did not survive either

The closing sweep did not reproduce a prefill number I had published hours
earlier from what the git history says is identical code: 27.1 % where I had
recorded 25.2 %, and 21.5 % on the six runs after that. I wrote that up as
nondeterminism in the 128-row path, corrected the README and two judgements in
5.7 around it, and added an invariant.

Then I tested it directly, which is what I should have done first.
`prefill_determinism_probe.py` prefills 128 rows in three separate processes, on
a synthetic prompt and on the exact tokens `device_quality` scores, and returns
**bit-identical** logits and next-eight-decode-steps every time -- cold or after
prior device work. The path is deterministic. The three historical readings are
unexplained, and calling them a property of the code was wrong.

What survives is the procedural half, which cost nothing to keep: pair a prefill
A/B in one batch of invocations rather than against a number from an earlier
turn, and prefer decode as the regression gate, since it has returned
83.0 % / NLL 0.682 in every run across every board reset. The two 5.7 judgements
stay re-framed onto dispatch count and wall clock, which is where they should
have rested regardless.

Third retraction of an explanation in this session, after the trace-hang
mechanism and the row-tile cause. All three were plausible, published, and
disproved by an experiment that took one run.

## The lesson

This project is unusually disciplined about recording evidence, and that created
the failure mode: the record outlived the evidence. None of the above was visible
from the item list, which said five of eight done and was correct. All of it was
visible from asking what each claim rests on.

Three of the weak tests found here were written earlier in this same session,
which is the useful part of the finding. The audit was worth running on my own
work and not only on what I inherited.
