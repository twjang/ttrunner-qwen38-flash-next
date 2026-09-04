# 005 — ttnn engine: hardware survey and residency plan

Date: 2026-09-01
Status: **in progress** — this iteration establishes the design and derisks it
against real hardware. The engine itself is not yet built.

---

## Observation 1 — most of the "custom kernels" already exist

The assumption going in was that Gated DeltaNet and QSA would both need kernels
written from scratch in TT-Lang. Surveying the installed `ttnn` first:

| need | what exists |
|---|---|
| gated delta rule | **`ttnn.transformer.gated_delta_attn_seq`** — blocked forward substitution + sequential inter-chunk scan |
| block-sparse attention | `ttnn.transformer.sparse_sdpa` (token indices), `sparse_sdpa_msa` (block indices) |
| dense attention | `scaled_dot_product_attention`, `..._decode`, paged variants |
| MoE | `ttnn.moe`, `ttnn.sparse_matmul` |
| rope | `ttnn.experimental.rotary_embedding_hf` |
| norms / topk / softmax | `rms_norm`, `topk`, `softmax` |

`gated_delta_attn_seq`'s contract maps directly onto the chunked formulation
already validated in iteration 004 — it wants `L_unit`, `v_beta_sc`, `k_bd_sc`,
`intra_attn`, `q_decay`, `k_decay_t`, `dl_exp`, `L_inv`, which are the same
intermediates `chunk_gated_delta_rule` computes. Since that reference is now
known-correct, it can generate the exact inputs and check the op's output.

**Where a custom kernel is still genuinely needed:** the QSA indexer. QSA pools
keys into blocks of `compress_ratio=4` and keeps a 512-block budget.
`sparse_sdpa_msa` requires `block_size` to be a multiple of 32, and
`sparse_sdpa` assumes the MLA layout where V is a prefix of K. Neither matches
2 KV heads at block granularity 4, so the block-score-and-topk step (and
possibly the sparse gather) is the one piece to write in `ttl`.

---

## Observation 2 — measured, not assumed, on 4 × Blackhole

```
mesh                    : MeshShape([1, 4]), 4 devices, 11x10 Tensix grid each
matmul 512x2560x2560, bf16 activations:
  weights bfloat16      rel_err 0.0112    103.0 us   65.1 TFLOP/s/dev
  weights bfloat8_b     rel_err 0.0133     94.5 us   71.1 TFLOP/s/dev
  weights bfloat4_b     rel_err 0.1132     84.0 us   79.9 TFLOP/s/dev

weight round-trip error (quantisation only):
  bfloat8_b             0.0074
  bfloat4_b             0.1184
```

The first bfp4 measurement was misleading: quantising *both* operands gave
0.2498. The number that matters is bf16 activations against 4-bit weights, which
is 0.1132. This is why the precision policy below is mixed rather than uniform.

---

## Observation 3 — the n-gram table decides the budget

93.7 GB of weights against 128 GB of device DRAM looks like it fits, and Q8_0's
188 GB looks like it does not. Both readings miss the point: **28.8 GB of the
checkpoint is `per_layer_token_embd`, a pure gather touching 16 rows per token.**
It belongs in host RAM regardless of quantisation, and only the gathered rows
(16 × 160 floats per token) cross PCIe.

`src/ttrunner_qwen38_flash_next/tt/plan.py` encodes the resulting policy. Precision per tensor
follows the sensitivity ordering measured from Unsloth's imatrix-calibrated
UD-IQ4_XS in iteration 002, rather than being invented:

| tensor | device dtype | sharding | why |
|---|---|---|---|
| `per_layer_token_embd` | host, stays IQ4_NL | — | gather; f32 would be 205 GB |
| `token_embd` | host, stays Q8_0 | — | gather |
| `ffn_gate/up_exps` | `bfloat4_b` | expert-column | IQ3_S upstream: least sensitive |
| `ffn_down_exps` | `bfloat8_b` | expert-row (all-reduce) | IQ4_NL/Q8_0 upstream |
| `ffn_gate_inp` (router) | `float32` | replicate | F32 upstream; wrong expert ≠ small error |
| `indexer.*` | `bfloat16` | replicate | BF16 upstream; changes *which* tokens are seen |
| `attn_q`, `attn_qkv`, `attn_gate` | `bfloat8_b` | column (by head) | 24 q / 48 v / 16 k heads all divide by 4 |
| `attn_k`, `attn_v` | `bfloat8_b` | replicate | only 2 KV heads — cheaper to replicate |
| `attn_output`, `ssm_out` | `bfloat8_b` | row (all-reduce) | |
| `ssm_a`, `dt`, norms, `hc_*_norm/inject` | `float32` | replicate | tiny; decay constants must stay exact |
| `output.weight` | `bfloat8_b` | column (vocab) | all-gather logits |

**Budget — all 1224 tensors planned, none unaccounted:**

```
host (mmap, stays quantised) :  29.48 GB
device total                 :  92.91 GB
per device                   :  24.02 GB of 32 GB
headroom per device          :   7.98 GB
```

by device dtype: `bfloat8_b` 47.26 GB, `bfloat4_b` 45.30 GB, `float32` 0.31 GB,
`bfloat16` 0.04 GB.

### What the 8 GB of headroom buys

- DeltaNet recurrent state: 36 layers × 48 heads × 128 × 128 × f32 = 113 MB
  *per sequence*, but head-sharded 4 ways → 28 MB/device/sequence.
- KV cache: 12 full-attention layers × 2 KV heads × 256 × 2 × bf16 =
  24.6 KB/token, replicated (KV heads are not sharded).
- Indexer keys: 12 × 128 × bf16 = 3 KB/token.

That is roughly **8 sequences at 8 k context**, or 2 at 32 k, per device before
paging is needed. The DeltaNet state is a fixed per-sequence cost independent of
length, which is the architecture's whole point — but it makes the *batch* limit
bind sooner than the context limit.

---

## Remaining work for deliverable (3)

1. **Weight conversion** — GGUF → per-device `bfp8_b`/`bfp4_b` tiles, written to
   a load-time cache. The dequantiser and the plan are both in place; this is
   mechanical.
2. **ttnn layer implementations**, each checked against the now-trusted CPU
   reference: hyper-connections, MoE (`ttnn.moe` / `sparse_matmul`), gated
   attention, DeltaNet via `gated_delta_attn_seq`.
3. **QSA indexer kernel in `ttl`** — the one piece with no ttnn equivalent.
4. **Async inference core** — batched scheduler over the mesh, exposed through
   the `Engine` interface that iteration 006 already defines.

The CPU reference exists precisely so each of these can be validated
independently rather than debugged as a whole.
