# 015 — QSA selection on device

The last correctness gap. Attention was dense causal in both device paths, which
is exactly right below the 2048-token budget -- QSA retains every complete block
there, so dense is the same thing -- and a different model above it. The
selection now runs on device, verified against the reference past the budget.

The work divided cleanly into "what to select", which is model semantics and
where the bugs were, and "how to apply it", which is kernel economics and where
three designs died.

## Observation 1 — probe the ops before designing around them

Four capability probes, ~3 minutes each, decided the whole shape of this
(`scripts/dev/probe_indexer_ops.py` and the micro-benchmarks in 5 below). The
one that mattered most corrected an earlier note in the handoff: `sdpa_decode`
*does* take an `attn_mask`, and the earlier probe that said otherwise was
missing `program_config`. Designing around that wrong result would have cost a
day.

Worth doing first, every time: the alternative is discovering a dead end half
way through an implementation, with sunk code arguing for itself.

## Observation 2 — unconditional writes, because a trace cannot branch

The pooled block for position p is `rms_norm(mean(raw_k[4j..4j+4]), k_norm)`
roped at `4j`, and it never changes once complete. So it is stored already
normed and roped, and written *every* step at index `p // 4` over a shift ring
of the last four raw keys.

For p not at a block boundary that value is wrong. It does not matter: the block
is not yet eligible, and it is overwritten before it becomes so. The pool is a
mean, so the ring's order is irrelevant.

That is the shape every per-step decision here takes -- compute it always, let
eligibility sort it out -- because a captured trace replays one graph and cannot
test p.

## Observation 3 — the trailing block is extra to the budget, not part of it

First version forced the block p sits inside into `topk` by giving it a large
positive score. Verified against `_indexer_mask` at position 2598 it allowed
2047 tokens where the reference allows 2051.

Forcing spends a selection slot. The reference allows the top 512 blocks *and*
the trailing partial one, so the device was keeping 511 of the former. Now the
straddling block is pushed *below* the other ineligible ones -- `topk` can never
return it -- and appended separately, in a whole `k_chunk_size` slot because
`sdpa_decode` asserts `mask_shape[3] % k_chunk_size == 0`.

Its padding entries point at `min(p + 1, max_seq - 1)` rather than 0. They are
scattered with a zero, and index 0 is visible from the very first step whenever
block 0 is selected -- pointing padding there would silently un-select it.

## Observation 4 — one comparison covers every case

`topk` returns 512 indices whatever the scores are. Below the budget fewer than
512 blocks exist, so the surplus is filled from the -inf mass; a filled block
lies entirely beyond p, the straddling block keeps its visible prefix, and an
eligible block keeps everything. So the mask is just `token <= p`, and the path
is exact at every position rather than only past the budget -- which is what
makes it safe to have on unconditionally in a traced graph.

## Observation 5 — three designs, and the one the kernels allow

Selecting is cheap. Applying the selection is where this was decided, and only
measurement decided it:

| approach | cost | verdict |
|---|---|---|
| gather the selected K/V into a 2048-token buffer | `ttnn.gather` 35 ms at S=4096, 291 ms at 32768, 1161 ms at 131072 -- **linear in the source** | 24 calls a step: 53 s at full context |
| paged attention, page size 4 | `paged_scaled_dot_product_attention_decode` 0.25 ms | page 4 pads to a tile: **8x** the K/V memory |
| scatter into a full-length additive mask | scatter 0.3-0.7 ms, additive + broadcast to 24 heads 0.6 ms | **this one** |

The first is the design the handoff had written down, and it looked obviously
right: constant-cost attention, a 24 x 2048 mask instead of 24 x 262144. It is
unusable in this ttnn build for a reason nothing about the design suggests.

What remains is `ttnn.topk` with k=512, which scales with k and with n:

| n blocks | k=32 | k=64 | k=512 |
|---|---|---|---|
| 2048 | 0.40 ms | 0.81 ms | 4.48 ms |
| 8192 | 0.19 ms | 0.35 ms | 19.5 ms |
| 32768 | 0.25 ms | 0.41 ms | 79.6 ms |
| 65536 | 11.0 ms | 20.9 ms | 157 ms |

Rows are free (32 rows cost the same as 1), so batch is not the problem; k=512
is. Twelve QSA layers at 2048 blocks is 54 ms a step, and at 65536 blocks 1.9 s.

## Where it leaves us

| | traced | eager |
|---|---|---|
| dense, 8192 context | 236.1 ms | 513.2 ms |
| sparse selection, 8192 context | **297.5 ms** (+26 %) | 576.9 ms (+12 %) |

Verified against the reference's `_indexer_mask` on the same random hidden
states, driven one position at a time on the device and in one call on the host:

* position 2599 (a block boundary): 2048 allowed, 2048 selected, **exact**.
* position 2598 (mid-block): 2051 allowed, 2051 selected, differing by one block
  -- rank 511 against rank 512, a score gap of 8.2e-4, below what bf16 resolves.
* below the budget: numerically identical to dense, NLL 0.682 either way and
  83.0 % next-token top-1.
* traced matches eager token for token, reproducing `004`'s llama.cpp output.

Two limits, both from kernels rather than from the model. `ttnn.scatter` takes
uint16 indices -- int32 and uint32 both assert -- so the selection can address
65536 cache positions; and `topk` at k=512 makes anything past ~16384 tokens
expensive. So it runs for `budget < max_seq_len <= 65536`, and `TTEngine` says
so when a longer context puts it out of reach, because dense beyond the budget
is a different model and not something a caller should have to infer from the
output.

Lifting either is a kernel question: a `topk` that scales with k, or a `scatter`
that takes a wider index. Both are worth raising upstream; neither is worth
working around here.
