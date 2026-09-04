# 023 — The attention config was the collapse

Decode's next-token accuracy fell from ~80 % over the first 128 positions to 0 %
past 224. The cause was four lines of program config that had been sitting in
`TTModel.__init__` since the paged refactor, and the reason nobody found it is
that nobody had ever scored past 128 positions.

## Observation 1 — the measurement that had never been taken

`device_quality.py` defaults to 48 tokens and its passage is ~200 long. Every
accuracy figure in the handoff -- the 83 % included -- came from at most 128
scored positions. Scoring per block of positions instead of in aggregate is what
made it visible, and an aggregate would have hidden it: the same run that shows
0 % past 224 averages to a respectable-looking 62 % overall.

    positions    0- 127   78-84 %
    positions  128- 159   53.1 %
    positions  160- 191   12.5 %
    positions  192- 223    6.2 %
    positions  224+        0.0 %

Zero top-1 on ordinary English prose is not hard text. A working language model
gets the common tokens right whatever else it is doing.

## Observation 2 — the control first, and it earned its keep

Before calling this a defect, the float32 CPU reference had to be run on the
*same passage*, because a passage can simply get harder. It does get harder here:
the reference falls from ~78 % to ~50 % over the same positions. Without that
control the fix below would have been declared a 0 %-to-50 % triumph and the
remaining 30 points would have been chased as a second bug that does not exist.

## Observation 3 — what moved the cliff

The onset tracked `sdpa_k_chunk`, the decode attention's `k_chunk_size`:

| k_chunk | accuracy holds until | then |
|---|---|---|
| 64 | ~64 | 0 % by 128 |
| 128 (the default) | ~128 | 0 % by 224 |
| 256 | ~128, gently | sharp fall at 256 |

A parameter that only sets how a reduction is *blocked* should not move where a
model stops working. That is enough to indict it, but not enough to know what is
wrong, because the model also has a DeltaNet recurrence, conv and PLE rings and a
K/V cache, all of which carry position.

## Observation 4 — test the op, not the model

`sdpa_decode_accuracy_check.py` builds a paged K/V cache of known length and runs
`paged_scaled_dot_product_attention_decode` on it, with no model and no state.
The first version compared against hand-written softmax attention and called
every length wrong, including L=32, which the model handles at 84 % -- the
reference's head-to-KV mapping was wrong, not the op. That is worth recording:
an "everything fails" result is nearly always the harness.

Comparing the op **against itself** needs no such assumption. For fixed inputs
the answer must not depend on how the online softmax is chunked:

| cache length | k=32 | k=64 | k=128 | k=256 |
|---|---|---|---|---|
| 32 | 0 | 0 | 0 | 0 |
| 128 | 6.2e+00 | 1.1e+00 | 0 | 0 |
| 320 | 6.8e+05 | 8.9e+05 | 5.9e+01 | 0 |
| 448 | 2.1e+07 | 4.4e+07 | 6.9e+04 | 0 |

Relative, against the widest config. Identical while everything fits one chunk,
and then divergence that grows with the chunk count until it is seven orders of
magnitude. The online softmax across k-chunks is wrong.

## The fix

Pass no program config and let ttnn choose. The pinned one -- q_chunk_size=32,
k_chunk_size=128, an 8x8 grid -- was hand-set and never validated past one chunk.

    positions      0-127  128-159  160-191  192-223  224-255  256-287
    reference      75-81%   62.5%    62.5%    50.0%    46.9%    50.0%
    device after   78-84%   59.4%    59.4%    50.0%    40.6%    46.9%
    device before  78-84%   53.1%    12.5%     6.2%     0.0%     0.0%

The device now tracks the float32 reference the whole way. End to end,
`device_quality.py 200` goes **62.3 % -> 73.4 %** top-1 and NLL 2.209 -> 1.267.
Short contexts improve slightly too (128 positions: 80.3 -> 81.1 %), because the
last few of 128 already crossed one k-chunk.

`sdpa_k_chunk` stays as a constructor argument because the QSA indexer sizes its
window from it; it no longer reaches the attention op. `pin_sdpa_config=True`
restores the old behaviour for an A/B.

## What it costs

Nothing at the canonical configuration and 3 % on one path:

| max_seq_len | indexer | before | after |
|---|---|---|---|
| 262144 | off | 173.7 ms | **173.4 ms** |
| 4096 | on | 204.0 ms | 210.3 ms |

Prefill is untouched -- `_attention_chunk` uses its own chunked configs, which
were separately validated when they were written (and, unlike this one, validated
against something).

## The lesson

The pinned config was added for speed, measured for speed, and never checked for
correctness at a size larger than the one it was measured at. Its own docstring,
two fields away, says the thing that should have raised the question:
"k_chunk_size sets the accumulation order and therefore the answer". That
sentence was written about the *prefill* config, where the answer was checked at
several starts. The decode config got the same kind of parameter and none of the
checking.

The general form: a knob that changes the answer needs a correctness test at
every size it will see, not at the size it was tuned on. And when a metric is
reported as one number over a range, look at it *across* the range once before
trusting it -- the aggregate here read 62 % while half the range read zero.
