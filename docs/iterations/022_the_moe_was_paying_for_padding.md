# 022 — The MoE was paying for padding, not for experts

The traced decode step was 266.3 ms. Two changes took it to 204.0, and neither
of them made the model do less arithmetic — both stopped it moving bytes that
were never data. The measurements that found them are more reusable than the
changes.

## Observation 1 — ablate the trace, don't count its ops

5.7 established that the traced step is not dispatch-bound: removing 97 ops moved
it 236.1 → 236.0 ms. So `op_count.py`, which is the right instrument for prefill,
says nothing about where decode's time goes — its 6095 calls have a flat
distribution and the largest QSA site does not appear in the top sixteen.

What does work is ablation on the traced path. `decode_ablation_check.py`
replaces one component with an identity of the same shape — *zero* extra ops, so
the delta is that component's device time and nothing else — and runs one
ablation per process, because alternating two traces hangs this build.

| ablated | median | its cost |
|---|---|---|
| none | 266.4 ms | — |
| **routed MoE** | 93.5 | **172.9 ms (65 %)** |
| attention (both kinds) | 211.6 | 54.8 |
| shared expert | 260.5 | 5.9 |
| all-reduce | 261.1 | 5.4 |
| reinject | 263.8 | 2.7 |
| PLE | 265.4 | 1.0 |

## Observation 2 — the experts are not what the expert block costs

The obvious reading of "65 % in the MoE" is that 512 experts at top-10 are
expensive. A top-k sweep says otherwise, and it is the single most useful number
here:

| top_k | 1 | 4 | 10 (real) | 20 |
|---|---|---|---|---|
| step | 259.6 ms | 262.9 | 266.4 | 272.0 |

Nineteen extra expert-FFNs across 48 layers cost 12.4 ms, so one is 13.6 us. At
top-10 the selected experts do about **7.5 ms of work inside a 173 ms block**.
Everything else scales with E = 512, not with how many experts run.

The cause is `TILE_LAYOUT`, which pads the row axis to 32. At decode M is 1, so
every [1, 512, M, K] tensor carries 32 rows of storage per real row:
[1, 512, 1, 2560] is **84 MB holding 2.6 MB of answer**.

## Observation 3 — two places were paying for it, and both had an exit

**The input copy.** `expert_ffn` fed `sparse_matmul` in its (sparse a, sparse b)
mode, which wants the rows already laid out per expert, so it called
`ttnn.repeat(x, (1, E, 1, 1))` — materialising 512 padded copies of one row, 84 MB
a layer. The op's own modes table has a (dense a, sparse b) mode taking
[1, 1, M, K] against [1, E, K, N]: the same computation with no copy. Bit-identical
by construction and checked element-wise, and 1.94x at M=1 in isolation.

It reverses at width, which is why it ships as a threshold rather than a
replacement: 1.82x at M=32 and 2.35x at M=64, but at M=128 it *loses* in the
model, 919.5 → 930.0 ms a chunk. `_BROADCAST_MAX_M = 64`, and a prefill chunk
keeps the old path byte for byte.

    traced decode 266.3 → 238.2 ms

**The combine.** The weighted sum over experts read the 84 MB, wrote a product
the same size, and read it back to reduce — about 254 MB a layer, of which the
permute was 0.5 ms and the rest was the multiply and the sum. `sparse_matmul`
cannot be told to stop padding (`output_tile` of [1,32] through [16,32] is
rejected outright by `matmul_utilities.cpp:231`), but the consumer can stop
paying: at M = 1 each expert contributes exactly one row, so reshaping the
expert axis into the row axis packs it exactly, and the weighted sum becomes one
matmul against the router weights in the shape routing already produced them.

    combine 55.1 → 20.8 ms      traced decode 238.2 → 204.0 ms

## Observation 4 — the faster form was also the more accurate one

The combine change is *not* bit-identical, so "is it still right" needed an
answer rather than an assurance. The elementwise form rounds all 512 products to
bfloat16 before adding them; the matmul accumulates them in fp32. Against the
exact weighted sum in float64, computed from the device's own operands:

| form | max abs err | mean abs err |
|---|---|---|
| sum (old) | 1.563e-03 | 2.527e-04 |
| matmul (new) | **1.138e-03** | **1.650e-04** |

27 % better on the worst element, 35 % on the mean. End to end the difference is
below what 127 scored positions resolve, which is the expected result for one
bf16 ulp in a model where any perturbation reroutes something: top-1 101/127 →
102/127, top-5 unchanged, mean NLL 0.873 → 0.884, median NLL 0.163 → 0.158.

The general point: when a change is not bit-identical, comparing the two
candidates to each other says only that they differ. Comparing both to an exact
reference built from the same operands says which one to keep.

## Observation 5 — what the step is now, and what is left

| component | cost | share |
|---|---|---|
| expert down projection | 51.7 ms | 25 % |
| QSA, 12 layers | 36.1 | 18 % |
| expert gate\|up projection | 26.1 | 13 % |
| combine | 20.8 | 10 % |
| DeltaNet, 36 layers | 18.0 | 9 % |

Two of these were surprises worth recording. **QSA's cost is not the attention.**
Stubbing `paged_scaled_dot_product_attention_decode` saves 0.7 ms and stubbing
`paged_update_cache` saves 0.5; the 36.1 ms is the projections around them, at a
context of ~25 tokens where the sparse selection path is not even engaged. And
the slices, silu and multiply between the two expert matmuls are free — the
`gateup` ablation's 26.1 ms is within rounding of the matmul measured alone
(0.551 ms x 48 = 26.4), which is why fusing them into `ttnn.swiglu` was dropped
without being built.

## The lever that is measured but not taken

The down projection writes [1, 512, M, 2560] on every device because the expert
weights are EXPERT_COLUMN-sharded: each card holds all 512 experts at a quarter
of the intermediate width. Sharding on the *expert* axis instead would give each
card 128 whole experts — identical weight bytes, identical FLOPs per card, a
quarter of the output tensor, and one full-width contraction per expert instead
of four partial sums to add.

Measured directly, as the two real configurations rather than an E sweep
(`expert_axis_scaling_check.py`):

| per layer | gate\|up | down | sum |
|---|---|---|---|
| now: E=512 x N=160 | 0.551 ms | 1.215 ms | 1.766 ms |
| expert-sharded: E=128 x N=640 | 0.425 ms | 0.431 ms | **0.856 ms** |

**2.06x**, about 44 ms across 48 layers, plus the combine shrinking with the
tensor it reads. It is not taken here because it needs the weight cache rebuilt
and the routing mask resharded per device — the router must still see all 512
logits to normalise and rank, so each card needs its own 128-column slice of a
mask computed replicated, which is a collective per layer. Both are real work
with real risk, and rebuilding a 60 GB cache is not a thing to start without
asking.

## The lesson

Three ideas in this iteration were killed by measurement before being built —
tiny output tiles (rejected by the op), the SwiGLU fusion (the ops it would
replace are already free), and 64-row prefill groups (0.90x, i.e. slower, though
it did confirm `expert_ffn` is exactly per-token at max diff 0.000e+00). Two
survived. The ratio is the point: on this hardware the intuitive cost model —
count the arithmetic — is wrong often enough that a cheap isolated harness
before a change is worth more than a careful implementation of the wrong one.
