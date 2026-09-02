# 017 — The inverse was stable only in float32

Moving `gated_delta_attn_seq`'s preparation onto the device was supposed to be a
transcription: `block_inverse` had been written as matmuls and adds precisely so
that it would port, and its docstring said so. The transcription was faithful and
the result was wrong by 620875 %, because the algorithm had never been stable —
float32 was hiding it.

## Observation 1 — a faithful port can expose a latent bug

`prepare_device` is `prepare` in ttnn ops, working in the op's own
`[H, NC, C, D]` layout so each device prepares the heads it already holds and
nothing is gathered. Against random inputs it agreed with the host to 0.13 %,
which is the hardware floor: a plain float32 `ttnn.matmul` at HiFi4 with
`fp32_dest_acc_en` is 0.16 % off torch for a 128x128, because the Tensix
multiplier decomposes fp32 into bf16 pieces. Every elementwise op was at ~1e-5.

Quality said otherwise. Prefilling 128 tokens scored 21.9 % next-token top-1
where stepping the same tokens scored 51.6 %, and the host preparation scored
50.0 %. Comparing the eight prepared tensors on *real* model data rather than
random data put all of it in one:

| tensor | rel. error, real data |
|---|---|
| `L_inv` | **620875 %** |
| `L_unit` | 0.2688 % |
| `intra_attn` | 0.1392 % |
| everything else | ≤ 0.07 % |

## Observation 2 — the growth was in the intermediates, not the answer

`block_inverse` summed the telescoping Neumann series, exact for a unit lower
triangular `I + N` because `N` is nilpotent:

    (I + N)^-1 = sum_{k=0}^{C-1} (-N)^k

in five squarings for `C = 32`. The trouble is what those squarings hold on the
way. `l_unit` is `I + (k_beta @ k^T) * pairwise` masked strictly lower, and both
factors are bounded by 1, so entries never exceed 1 — but when a head's keys
point nearly the same way and its decay is slow, *every* inner product is near
+1 and the strictly lower part is dense, positive and near 1. Head 7 of layer 0
reaches 0.983. Then `N^16` counts paths: ~1e8, cancelling back to an answer whose
largest entry is exactly 1.0.

Float32 carries ~1e-7 relative, so it survived that at 0.27 absolute error on an
answer bounded by 1 — wrong, and quietly wrong, for as long as the routine had
existed. The device carries ~1e-3 per matmul. Same cancellation, 6209.

**Remedy.** `block_diag_inverse` does it blocked instead. Doubling the block size
from 1 to 32, write the matrix as `D + C` — `D` the diagonal blocks already
inverted, `C` the newly included lower-left quadrants. `C` maps the high half of
a block to the low half, so `(D^-1 C)^2 == 0` and

    (D + C)^-1 == D^-1 - D^-1 C D^-1

*exactly*, in two matmuls per level. Ten matmuls, same as before, and nothing
grows: every intermediate is the inverse of a unit-triangular submatrix, bounded
by the answer. It runs once over the whole 128-wide matrix, so one pass yields
all four 32x32 diagonal inverses where the loop dispatched forty matmuls.

`L_inv` went to 0.2942 %, in line with every other tensor.

## Observation 3 — the test passed because of how the fixture was built

`test_block_inverse_matches_linalg_inv` scaled its off-diagonals by 0.4, well
inside the stable range. That is the obvious lesson and it is not the useful one.
The useful one is that *magnitude alone does not reproduce the failure*: a random
matrix scaled to max 0.98 also passes, because random signs cancel inside the
powers. What breaks it is the **correlation** the model produces, so the
replacement test builds `l_unit` from nearly parallel keys the way the model
does, where the old routine is 2.5 off on an inverse bounded by 1.
`deltanet_prepare_device_check.py --correlated` does the same for the device.

## Observation 4 — the op was carrying half of someone else's error

`_attention_chunk`'s comment recorded the chunked DeltaNet as 0.36 % from the
reference at position 0 of a 128-token chunk and 60 % by position 127, and
attributed it to the device op. Re-measured after the fix: 0.35 % at position 0
and ~32 % at position 127. Half of the op's reputation was the inverse. The
growth's *shape* is still there and is presumably the op's own accumulation.

None of which the token noticed, which is 013's lesson again: 128 tokens
prefilled now score 53.1 % against a same-positions decode control of 51.6 %, and
prefilling 32 and scoring 128 gives 71.9 % against 71.7 %. Prefill leaves state
as good as stepping the same tokens. It did not before.

## Observation 5 — the default was set from the FLOP count

With the host round trip gone the chunk still cost ~1 s, so: 19733 device calls
for 128 tokens (`op_count.py --prefill 128`) against 6355 for a *single-token*
step. Dispatch-bound. `moe_chunk` groups rows for the MoE, and its default of 16
had been chosen from waste figures — M=128 wastes ~51x the FLOPs, M=16 about 8x —
without a clock. Measured, the waste is worth paying:

| moe_chunk | wall clock | tok/s | top-1 | NLL |
|---|---|---|---|---|
| 8 | 3332.8 ms | 38.4 | 53.1 % | 3.108 |
| 16 | 2032.6 ms | 63.0 | 53.1 % | 3.038 |
| 32 | **1101.0 ms** | 116.3 | 53.1 % | 3.108 |
| 64 | 936.1 ms | 136.7 | 43.8 % | 4.021 |
| 128 | 998.7 ms | 128.2 | 21.9 % | 5.477 |

1.85x, bit-identical output. And a second finding fell out of checking rather
than assuming: `moe_chunk` is **not** a pure speed knob, though the reasoning
that it must be — the split is over rows, each row picks its own experts — is
what made it tempting to skip the measurement. 8, 16 and 32 agree on every
token; 64 and 128 degrade monotonically. A cliff between 32 and 64 is a limit
crossed, not precision accumulated, and `ttnn.scatter`'s uint16 indices (reach
65536) against the broadcast's |union| x M rows is the standing candidate.
`_MAX_MOE_CHUNK = 32` refuses the rest rather than documenting it, because the
fastest setting is on the wrong side and it fails silently. See §5.8.

## Where it leaves 5.2

The stated blocker is gone: the chunk path has no host round trip at all, and a
contract test forbids one returning. Its replacement is known precisely — four
host dependencies, all in `_attention_chunk` (host rope, `fill_cache`'s integer
`update_idx`, the host causal mask, and a `kv_len` that grows per chunk) — and so
is the route: `ttnn.experimental.paged_fill_cache` takes the page table as a
device tensor, and `chunked_scaled_dot_product_attention` takes
`chunk_start_idx_tensor`, whose own documentation names trace capture as the
reason it exists. Both want a paged K/V cache, which decode currently does not
use, so that repaging is its own change with its own measurement.
