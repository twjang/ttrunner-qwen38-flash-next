# 008 — Choosing precision and sharding from measurement

Date: 2026-09-01

## Goal
Iteration 005 set a residency plan by analogy to Unsloth's bit allocation.
Iteration 007 showed that analogy was wrong for block float. This iteration
replaces both the precision policy and the sharding layout with measured
decisions.

---

## Observation 1 — simulate the device, don't reason about it

`bfloat4_b` gave 13 % rms error on the device MoE. The question that actually
matters is not "how big is the weight error" but "does the model still produce
the same tokens", and that cannot be answered from a per-tensor error figure.

**Remedy** — `tt/blockfloat.py` reproduces ttnn's `bfloat8_b`/`bfloat4_b` in
numpy (bit-exact, 39 Melem/s vs ttnn's 5), and `WeightStore` gained a
`quant_sim` hook. The whole CPU reference can now run at any device precision
policy, on real weights, and be compared against f32 and against llama.cpp.

**Result** — `"The capital of France is"`, all 48 layers quantised:

| policy | top-1 | logit | Δlogit rms | per-device |
|---|---|---|---|---|
| f32 (reference) | `' Paris'` | 16.800 | — | — |
| all experts `bfloat8_b` | `' Paris'` | 17.009 | 0.365 | 28.3 GB |
| **bfp4 gate/up, bfp8 down** | `' Paris'` | 16.285 | 0.600 | **24.0 GB** |
| gate bfp8, up bfp4, down bfp8 | `' Paris'` | 16.367 | 0.594 | 28.3 GB |

Three things fall out:

1. **Every policy keeps the top-1 token.** bfp4 on gate/up roughly doubles the
   logit perturbation (0.600 vs 0.365) but does not change the prediction.
2. **All-bfp8 does not fit.** 120.8 B expert params × 1.0625 B = 128.4 GB against
   128 GB of device DRAM, before any KV cache. So "just use bfp8" was never an
   option; the question was only where to spend 4 bits.
3. **The asymmetric middle ground is worthless.** Upgrading *only* gate to bfp8
   moves the logit shift from 0.600 to 0.594 — noise — while costing 4.3 GB per
   device. The error is dominated by having any bfp4 in the gate/up pair, so the
   4-bit choice is all-or-nothing.

The plan therefore stands, now on evidence: **bfp4 gate/up, bfp8 down.** Not
because Unsloth spends fewer bits there, but because it is the only 4-bit
placement that buys memory, and it demonstrably preserves the prediction.

### Why bfp4 is worse than a 3.4 bpw codebook quant
`bfloat4_b` spends its 4 bits on 8 magnitude levels against a shared exponent
per 16 datums. IQ3_S spends 3.4 bits on a 512-entry, 4-dimensional codebook
fitted to the weight distribution. At *fewer* bits the codebook wins decisively
on rate-distortion — which is why the earlier reasoning ("4.5 bpw beats 3.4 bpw,
so bfp4 must be safe") inverted the truth.

---

## Observation 2 — the all-reduce cost inverted the sharding decision

With fabric enabled (`ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)` —
without it every CCL op dies on `fabric_context_ != nullptr`), measured on the
4-device mesh:

```
all_reduce Linear  0.230 ms   rel_err 0.0077
all_reduce Ring    0.233 ms   rel_err 0.0050
all_gather         works
```

0.23 ms is latency, not bandwidth (8 KB payload). The original plan sharded
attention and DeltaNet by head as well as the experts, which costs an all-reduce
on **every** layer's attention branch *and* its MoE branch: ~96 collectives per
step ≈ 22 ms of pure collective latency.

But the experts are 88 GB of the 93 GB. Replicating every dense weight and
sharding only the expert stacks costs ~1.7 GB per device and halves the
collectives:

| | original | revised |
|---|---|---|
| per-device weights | 24.02 GB | **26.34 GB** |
| headroom (32 GB) | 7.98 GB | 5.66 GB |
| collectives / step | ~96 | **48 all-reduce + 1 all-gather** |

**Result** — 26.34 GB/device with all 1224 tensors planned. Worth 2.3 GB to
remove ~11 ms per step, and the latency is amortised further by batching (a
0.23 ms all-reduce is nearly batch-independent at these payloads).

### The bonus: a whole class of bug disappears
Head-sharding `attn_qkv` is not a plain split. Its 10240 outputs are
`[q(2048) | k(2048) | v(6144)]`, so a flat 4-way split at 2560 cuts *through* the
q/k boundary and mis-pairs heads — with correct shapes and no error anywhere.
`ssm_conv1d` is indexed by the same 10240 channels and would have to be permuted
identically or the depthwise conv applies the wrong filter per channel. And the
per-v-head parameters (`ssm_a`, `ssm_dt.bias`, `ssm_alpha/beta`) have a 48-wide
head axis that does not tile-align when split 4 ways (12 < 32).

Replicating the dense weights makes all of that moot.

---

## Observation 3 — the weight cache

Dequantising 93 GB of IQ3_S/IQ4_NL and re-encoding to block float takes ~85
minutes, so it is done once into `.tensorbin` files.

Verified before committing to it:

```
bfloat8_b   7.0 MB (expected 7.0)  dump 0.00s  load 0.001s  lossless=True
bfloat4_b   3.7 MB (expected 3.7)  dump 0.00s  load 0.000s  lossless=True
```

File sizes confirm 1.0625 and 0.5625 bytes/element exactly. Load paths:

- **replicated** — `ttnn.load_tensor(path, device=mesh)` gives every device an
  identical copy (verified: composed shape `(4, ...)`, all devices equal).
- **sharded** — `ttnn.load_tensor(path)` then `ttnn.distribute_tensor(...,
  ShardTensorToMesh(dim))`. Sharding a *tiled block-float* tensor is lossless
  (verified max_err 0.0) because every shard dim here is a multiple of 32.

`ReplicateTensorToMesh` is not accepted by `distribute_tensor` (it is not a
`CppTensorToMesh`), which is why the replicated path uses `load_tensor` directly
rather than one uniform call.

---

## Observation 4 — two decisions that keep the MoE off the host

The MoE block runs 48 times per token, so any host round-trip inside it is fatal
to decode latency. Two details avoid one:

- **`nnz=None`.** `sparse_matmul` can count the sparsity mask's non-zeros on
  device. Passing an explicit count would mean reading the top-k indices back to
  the host — and a count that disagrees with the mask does not error, it
  **hangs the device** (the compute kernel loops `nnz` times while the sender
  multicasts once per non-zero; the docs note it cannot be validated host-side).
- **Threshold instead of scatter.** The routing mask is built as
  `probs >= kth_largest_value` rather than by scattering top-k indices, so no
  index tensor is ever materialised and no gather/scatter op is needed. Verified
  identical to a `topk` reference, including the renormalised weights.

The union mask for `sparsity [1,1,1,E]` is then just `max` over the token axis.

---

## Observation 5 — decode needs no chunk preparation at all

`gated_delta_attn_seq` wants eight prepared tensors including a triangular system
and its per-block inverses. Building those on the host every layer would mean 36
PCIe round-trips per token.

For S = 1 the delta rule is simply a recurrence:

```
state <- state * exp(g)
delta  = (v - k @ state) * beta
state <- state + kᵀ delta
out    = q @ state
```

Three batched matmuls over the (batch × heads) axis plus elementwise ops, all on
device. `gated_delta_attn_seq` is kept for chunked prefill, where the preparation
cost amortises over a whole chunk instead of one token.

---

## Status
- Precision policy: measured, not assumed.
- Sharding: experts only; 48 + 1 collectives per step.
- Weight cache: converting (~16 GB of 93 GB at time of writing).
- Device model written: `tt/model.py`, `tt/moe.py`, `tt/linear_attn.py`,
  `tt/weights.py`, `tt/ops.py`, `tt/deltanet.py`, `tt/blockfloat.py`.

Device validation is blocked until the converter releases the device cluster —
tt-metal allows one process to hold the devices, and a device test launched
alongside the converter simply hangs on open.

## Next
Iteration 009: end-to-end device validation against the CPU reference, then the
async batched core.
