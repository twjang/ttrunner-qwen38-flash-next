# 007 — Validating the ttnn ops the engine will stand on

Date: 2026-09-01

## Goal
Before writing a 48-layer device model, establish which ttnn ops exist, what
their real contracts are, and whether each reproduces the CPU reference. Every
number below is measured on the 4 × Blackhole p150a mesh.

---

## Observation 1 — the op inventory changed the plan twice

Iteration 005 assumed Gated DeltaNet and QSA both needed hand-written TT-Lang
kernels. Grepping the bundled tt-metal tree instead found:

- `ttnn.transformer.gated_delta_attn_seq` — the chunked gated delta rule
- `ttnn.experimental.indexer_score_dsa` / `_msa` — lightning-indexer scorers
- `ttnn.experimental.moe_compute` + `all_to_all_dispatch_metadata` — a full MoE pipeline
- `ttnn.sparse_matmul` — expert-skipping batched matmul

Two traps in the same inventory:

- **`ttnn.moe` is not a MoE layer.** Its docstring: *"Returns the weight of the
  zero-th MoE expert."* It is a routing helper, not the expert FFN. Reaching for
  it by name would have wasted a day.
- **`ttnn.experimental.hc_sum_reduce` is not hyper-connections.** It lives under
  `operations/experimental/ssm/` — it is a Mamba-era "hidden channel" reduce. The
  `hc_` prefix collides with this model's hyper-connection tensors (`hc_attn_down`
  etc.) by coincidence.

There is **no** qwen4exp model in tt-metal — only the primitives.

---

## Observation 2 — `gated_delta_attn_seq` works, and its contract is derivable

The op takes eight prepared tensors rather than (q, k, v, g, beta): it wants the
triangular system `L_unit`, its per-block inverses `L_inv`, and the decayed
q/k/v, then does the blocked forward substitution and the inter-chunk scan.

Hard constraints, from its own `device_operation.cpp`:

```c
constexpr uint32_t kRequiredDim = 4 * TILE_HEIGHT;   // 128
TT_FATAL(C  == kRequiredDim, ...);   // chunk_size
TT_FATAL(Dk == kRequiredDim, ...);   // key_dim
TT_FATAL(Dv == kRequiredDim, ...);   // val_dim
```

The model's `linear_key_head_dim == linear_value_head_dim == 128`, so only the
chunk size was a free choice, and it is forced to 128 (the reference used 64).

`L_inv` is documented only as *"4 precomputed diagonal block inverses per chunk"*
with shape `[BH, NC, C, 32]`. Reading the validator settles it: `C = 128 = 4 × 32`,
so rows `32b..32b+31` hold the inverse of the `b`-th 32×32 diagonal block.

`L_unit` had two plausible sign conventions. Rather than guess, both were tried:
`I + strictly_lower(k_beta @ kᵀ · pairwise_decay)` reproduced the reference, which
a wrong sign could not have done to within 1 %.

**Result** — against the reference (which iteration 004 matched to llama.cpp):

```
OUT   max_rel=0.01006  rms_rel=0.00813
STATE max_rel=0.00734  rms_rel=0.00822
0.16 ms for BH=8, NC=4 (512 tokens x 8 heads)
```

Accuracy is identical across `MathFidelity` LoFi→HiFi4 and with
`fp32_dest_acc_en` on or off — the kernel's DEST is bf16 internally (its sibling
`indexer_score` docs say the custom LLK is *"validated for bf16 DEST
half-sync"*). ~0.8 % rms is bf16-class and does not improve by asking.

---

## Observation 3 — MoE needs no token permutation

The textbook device MoE sorts tokens by expert, gathers them into a padded
capacity buffer, runs a grouped matmul, then scatter-adds back. That permutation
is data-dependent, so it either round-trips to the host every layer or needs its
own kernel.

`sparse_matmul`'s `(a_sparse, b_sparse)` mode avoids it entirely:

```
a [1, E, M, K] @ b [1, E, K, N] -> [1, E, M, N],  sparsity [1, 1, 1, E]
```

Broadcast the M tokens into all E slots and mark only the selected experts: the
op skips the rest, so the cost is `|selected| × M` rows. For **decode**, M = 1 per
sequence, which means exactly the 10 routed experts and nothing else — optimal,
with no gather, no capacity padding, and no host round-trip. Prefill pays waste
proportional to M and is therefore chunked.

Two contract details that are not optional:

- `program_config` is **required**, and only
  `MatmulMultiCoreReuseMultiCast1DProgramConfig` with `mcast_in0=True` is accepted.
- `nnz` must equal `count_nonzero(sparsity)` *exactly*. The docs are blunt about
  the failure mode: the compute and receiver kernels loop `nnz` times while the
  sender multicasts once per non-zero, so a mismatch **hangs the device**. It is
  data-dependent and cannot be checked on the host, so `sparsity_from_indices`
  returns the count alongside the mask rather than letting a caller infer it.

Measured on real layer-0 experts (1.84 GB resident: gate/up `bfp4_b`, down `bfp8_b`):

```
M=1  nnz= 10   1.79 ms/layer
M=8  nnz= 59   3.21 ms/layer
```

---

## Observation 4 — bfloat4_b error does not average down, and that breaks the plan

The iteration-005 policy put expert gate/up in `bfloat4_b`, reasoning from
Unsloth's choice of 3.4 bpw IQ3_S for the same tensors. On real weights the
device MoE came out at **13 % rms** error.

That is not a bug — it is `bfloat4_b`. The format is a sign plus 3 mantissa bits
against a shared exponent per 16 datums, so each weight carries ~2⁻³ ≈ 12 %
relative error (measured round-trip: 0.1206).

The mistake was assuming a matmul would average it away. It does not, and the
reason is worth stating precisely: block-float error is *multiplicative*. For
`sum_k w_k x_k` the perturbation is `sum_k w_k ε_k x_k`, whose magnitude scales
with `‖wx‖₂` — exactly as the signal does. So a 12 % per-element error yields
~12 % output error at any contraction length. Additive noise would have shrunk
by √K; this does not.

This also explains why 4.5 bpw `bfloat4_b` is much worse than 3.4 bpw IQ3_S:
IQ3_S spends its bits on a 512-entry, 4-dimensional codebook fitted to the weight
distribution, while `bfloat4_b` spends them on 8 magnitude levels.

**Remedy** — `tt/blockfloat.py` reproduces both formats in numpy so the device's
numerics can be simulated inside the CPU reference and the policy chosen from
end-to-end evidence instead of a plausible analogy:

```
bfloat8_b  mantissa_bits=7: elementwise agreement with ttnn = 1.000
bfloat4_b  mantissa_bits=3: elementwise agreement with ttnn = 1.000
39 Melem/s (ttnn's own host bfp8 path is 5 Melem/s)
```

Bit-exact and 8× faster, so a `quant_sim` hook in `WeightStore` can now run the
whole reference model at device precision. The resulting policy study is in
iteration 008.

The memory consequence is already clear: all-`bfloat8_b` experts would be
120.8 B × 1.0625 = **128.4 GB against 128 GB of device DRAM**, so "just use
bfp8 everywhere" does not fit and the policy has to be earned per tensor.

---

## Observation 5 — `indexer_score_dsa` is QSA's scoring rule, but not for prefill

The op computes

```
score[b,0,s,t] = sum_h relu(q[b,h,s,:] · k[b,t,:]) * weights[b,h,s]
```

QSA computes `relu(q @ blockKeysᵀ).sum(over heads) / sqrt(d)`. Setting
`weights = 1/sqrt(d)` makes these the same expression — including the unusual
detail that heads only ever vote *positively* for a block.

Validated with pooled block keys standing in for `k`:

```
Sq= 32  T= 512  max_rel=0.01004   0.299 ms
Sq=128  T= 512  max_rel=0.01104   0.285 ms
Sq= 32  T=2048  max_rel=0.01191   0.394 ms
```

**But** the op asserts `chunk_start_idx + Sq <= T`: it assumes at least as many
keys as queries, which is true for token-level attention with history and false
for QSA in *block* space, where `T = S/4` blocks against `Sq = S` queries. A
whole-sequence prefill cannot satisfy it.

So the split is: `indexer_score_dsa` for **decode** (Sq = 1, every complete block
eligible — precisely QSA's decode semantics), and a plain
matmul + relu + sum for prefill, where scoring is cheap anyway
(Sq × T × D × Hi = 2048 × 512 × 128 × 4 ≈ 0.5 GFLOP per layer).

---

## Observation 6 — the hyper-connection path, on device

`ttnn.rms_norm` normalises the last dimension only and its `weight` must match
that dimension. The hyper-connection norm needs independent 2560-wide groups
inside a 10240-wide tensor, with a distinct 10240-wide gamma — so it is a
weightless grouped normalise (reshaped so the group is last) followed by a
full-width multiply.

Against the reference, on device:

```
grouped_rms_norm       rel_err=0.00128
gated_residual mixed   rel_err=0.00153
gated_residual inject  rel_err=0.00086
reinject               rel_err=0.00000
```

---

## Status

| piece | state |
|---|---|
| hyper-connections | validated on device |
| Gated DeltaNet (36/48 layers) | validated via `gated_delta_attn_seq` |
| MoE expert FFN | validated via `sparse_matmul`; precision policy open |
| QSA indexer scoring | validated; decode path only |
| block-float quantiser | bit-exact vs ttnn |
| full model / async core | not yet |

## Next
Iteration 008: the precision policy study, then the full device model.
