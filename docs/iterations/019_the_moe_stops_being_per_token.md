# 019 — The MoE stops being per-token past one tile of rows

`moe_chunk` groups rows for the MoE, and `017` set its default from a clock
instead of a FLOP count — 1.85x on prompt intake — while capping it at 32
because 64 and 128 scored worse. This is the follow-up that says what the cap is
actually for, and it starts by demolishing the evidence that produced it.

## Observation 1 — the evidence for the cliff was one sample per setting

`017` reported next-token top-1 of 53.1 / 53.1 / 53.1 / 43.8 / 21.9 % for
`moe_chunk` 8 / 16 / 32 / 64 / 128 and concluded that 8, 16 and 32 "agree on
every token". Each of those was **one run**. Two runs of the identical
configuration then gave:

    moe_chunk=32, run A   50.0 % top-1   NLL 3.110   median 1.756
    moe_chunk=32, run B   53.1 % top-1   NLL 3.108   median 1.225

`device_quality.py --prefill 128` scores 32 positions and is not repeatable at
that width, so the reading "8, 16 and 32 agree" was luck, and a 10-point gap on
a 32-sample score is not on its own a cliff. The cap happened to be right; the
argument for it was not.

**Remedy.** Measure something deterministic. Prefill's own final logits repeat
to **0.0000 %** across three runs at each setting, and across settings they move
55.9 % at 64 and 144.9 % at 128, with a different argmax (1105 → 1154 → 50).
`moe_chunk_noise.py`. The effect is real, large, and now measured against a
noise floor instead of against nothing.

## Observation 2 — the invariant that needs no reference

The MoE is per-token: every row picks its own experts by its own top-k and is
weighted by its own probabilities. Grouping rows is therefore allowed to change
speed and nothing else, which makes the answer computable one row at a time and
compared with no reference implementation and no oracle. `moe_rows_check.py`:

| group | worst row | rows over 1 % |
|---|---|---|
| 1, 8, 16, 32 | 0.000 % | 0/64 |
| 64 | 102.9 % | **33/64** |

So `moe.moe_block` is exactly per-token up to one tile of rows and wrong on half
its rows past it. That is the cap, stated as a property of the code rather than
as a score on some text.

## Observation 3 — I checked row 0 and cleared the MoE of a bug it had

The first version of that harness compared **row 0** across group sizes, got
0.0000 % everywhere, and concluded the MoE was innocent — sending the search off
into `prefill`'s slicing and concat, where the offsets that *fail* (64) are
tile-aligned and ones that *work* (8, 16) are not, which made no sense for
several minutes. Row 0 agrees at every group size. The rows that move are the
later ones.

This is the same error as the compaction-era "verified token-for-token against
the reference" that meant one short prompt, and as `013`'s per-layer distances.
A spot check that happens to sit at the one index the bug spares is worse than
no check, because it actively redirects the search.

## Observation 4 — everything reduces, and nothing reproduces

With routing exonerated at 64 (`keep`, `weights` and the expert union all match
the host) the defect had to be in `expert_ffn`. Each of its ops, in isolation, at
every row count from 8 to 128:

| op | result |
|---|---|
| `ttnn.topk` + threshold + `ge` mask | exact |
| `ttnn.max` over the row axis | exact at 64; drops 1 expert of 473 at 128 |
| `ttnn.permute(0,3,2,1)` | exact |
| expert-axis `ttnn.sum` | bf16 floor |
| `ttnn.sparse_matmul`, 1–64 experts selected | bf16 floor |
| 4 `sparse_program_config` variants | byte-identical error; `mcast_in0=False` rejected |

`per_core_M` is `ceil(m/32)` — 1 up to 32 rows, 2 at 64 — so the boundary is
real, but the program config is not the mechanism: changing `out_subblock_h` to
2 or 4, or setting `fuse_batch=True`, changes not one bit of the error.

The defect appears only in `expert_ffn` at production scale (E=512, K=2560, the
fused gate|up), and the reduced probes are all clean. Left untested and named in
5.8 for whoever picks it up: `ttnn.repeat(x, (1, 512, 1, 1))`, which is 168 MB
at 64 rows and 335 MB at 128, and `sparse_matmul`'s device-side `nnz` counting
at that size — the op's own docstring already warns that an `nnz` disagreeing
with the mask *hangs* the device, so that accounting is known to be delicate.

## Where it leaves 5.8

Cap kept at 32, now justified by a measured invariant violation sitting in
`_moe_block` where the next reader looks, with the search space narrowed to one
function and the six places it is *not* written down so they are not searched
again. The remaining prize is 1.2x on prompt intake; `moe_rows_check.py` coming
back clean is the condition for taking it.

## The lesson

Two, and both are about the shape of a check rather than its subject.

A measurement that has never been repeated has no noise floor, and a
single-sample comparison across settings cannot tell an effect from variance —
`017` drew a cliff from five such samples and was lucky that one existed.

And a spot check must be chosen to *include* the case that could be wrong. Row 0
is where a per-row bug is least likely to show, because it is the row every
grouping treats identically; checking it first and stopping cost more time than
having no harness at all would have.
