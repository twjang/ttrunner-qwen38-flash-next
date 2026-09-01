# 003 — The plain PyTorch CPU reference engine

Date: 2026-09-01

## Goal
Deliverable (2): a plain, obviously-correct PyTorch implementation that runs on
CPU and can serve as the numerical oracle for the ttnn engine.

---

## Observation 1 — three checkpoint conventions differ from the upstream model

The natural approach is to port `modeling_qwen4_exp.py` and point it at the
GGUF weights. That produces a model that loads cleanly, runs without error, and
emits garbage — because llama.cpp's converter rewrites three things on the way
into the GGUF, and none of them change any tensor's *shape*, so nothing would
catch it.

**Remedy** — read the converter (`conversion/qwen4exp.py` and `conversion/qwen.py`)
rather than only the model.

**Result**, in order of how quietly they would have broken things:

1. **Norm weights carry a folded `+1`.**
   ```python
   elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
       data_torch = data_torch + 1
   ```
   Upstream `Qwen4ExpTextRMSNorm` computes `normed * (1 + weight)` while
   `RMSNormGated` computes `normed * weight`. The converter folds the `+1` into
   the former and leaves the latter alone, plus patches five norms the suffix
   rule misses (`ple.norm_{key,query,conv}`, `indexer.{q,k}_layernorm`). The net
   effect is uniform: **every** norm in the GGUF is applied as `normed * weight`.
   Had this been missed, the weights are near zero-centred, so `1 + w` vs `w` is
   roughly "multiply by 1" vs "multiply by 0.02" — output would be noise.

2. **DeltaNet V heads are stored tiled, not grouped.** With 48 value heads over
   16 key heads, HF stores `[k0v0, k0v1, k0v2, k1v0, ...]`; the converter
   permutes to `[k0v0, k1v0, ..., k15v0, k0v1, ...]` so ggml's broadcast can be a
   plain repeat. Consequently the query/key heads must be expanded with
   `repeat` (tile), **not** `repeat_interleave`. Same shapes either way; the
   heads are simply paired up wrongly, and 2 of every 3 pairings would be wrong.

3. **The indexer's fused projection is split.** `indexer.index_qk_proj` (640
   rows) becomes `indexer.q_proj` (512) + `indexer.k_proj` (128).

---

## Observation 2 — the reference implementation's QSA indexer is a double Python loop

Upstream iterates `for batch_idx ... for query_idx ...`, calling `nonzero` and
`topk` per query. For a 4 k-token prefill that is 4 000 sequential iterations
per full-attention layer × 12 layers.

**Remedy** — a vectorised formulation for the (overwhelmingly common) dense
causal case: pool keys into blocks of `compress_ratio=4`, RoPE each block at its
*start* position, score all queries against all blocks in one einsum, mask
blocks that are not yet complete for a given query, then `topk` over the block
axis. The trailing partial block is always visible, matching upstream's `tail`.

**Result** — one einsum + one topk per layer instead of a per-token loop, with
the same selection semantics. The scoring rule is preserved exactly, including
its unusual shape: `relu(scores).sum(over heads) / sqrt(dim)` — heads *vote* for
blocks and can only ever vote positively.

---

## Observation 3 — the model cannot be materialised, so the weight store slices

Dequantising everything to f32 would need ~700 GB. Two tensors dominate:

| tensor | quantised | as f32 |
|---|---|---|
| `per_layer_token_embd` (320 M × 160) | 28.8 GB | 205 GB |
| `ffn_{gate,up,down}_exps` × 48 layers | 58 GB | 484 GB |

Both are accessed a few rows at a time — 10 of 512 experts per token, 16 n-gram
rows per token — so `WeightStore.get_rows()` dequantises *row ranges* straight
out of the mmap. This is only valid if no quant block straddles a row boundary;
rather than assume it, `_row_geometry` checks `row_elems % block_size == 0` and
refuses otherwise. It holds here: n-gram rows are 160 elements (5 IQ4_NL blocks)
and expert rows are 2560 (10 IQ3_S blocks).

Dense weights (~9 GB quantised) go through an LRU cache instead.

---

## Observation 4 — the chunked delta rule needed an independent check

`chunk_gated_delta_rule` is the subtle part of the model: it inverts a unit
lower-triangular matrix by forward substitution, tracks per-chunk cumulative
decay in log space, and carries a recurrent state across chunks. A transcription
error there yields plausible-looking numbers.

**Remedy** — the recurrent form is a completely different algorithm (a literal
token-at-a-time loop) that must produce the same answer. Both were run on the
same random inputs.

**Result**
```
out   max abs diff: 1.19e-07
state max abs diff: 3.58e-07
```
Float32 round-off. The two implementations agree.

---

## Observation 5 — GGUF and HF token ids agree, but the vocab is padded

The GGUF embedding rows are indexed by GGUF token id, so encoding with HF's
`tokenizer.json` is only safe if the id spaces match.

**Result** — ids 0..248076 agree exactly. The remaining 243 entries are
`[PAD248077]`..`[PAD248319]`, padding the vocabulary from 248077 to 248320; they
have no HF counterpart and never appear in text. Safe.

Stop tokens are `[248046, 248044]` per `generation_config.json` — note there are
two, and the GGUF only records the first.

---

## Two more formats

The MTP draft head turned out to use Q5_0 and Q4_K, neither exercised by
UD-IQ4_XS. Both implemented and validated the same way:

```
type         elems   max_abs_err  verdict   tensor
BF16          2048     0.000e+00  EXACT   blk.48.indexer.k_proj.weight
F32            256     0.000e+00  EXACT   blk.48.attn_k_norm.weight
Q4_K        524288     0.000e+00  EXACT   token_embd.weight
Q5_0         65536     0.000e+00  EXACT   blk.48.hc_attn_up.weight
Q6_K        524288     0.000e+00  EXACT   output.weight
Q8_0         65536     0.000e+00  EXACT   blk.48.ffn_down_exps.weight
FAILURES: 0
```

---

## Modules written

```
src/twtest/reference/
  config.py     Qwen4ExpConfig.from_gguf() + validate() cross-checks
  weights.py    WeightStore: LRU dense cache + block-aligned row slicing
  layers.py     norms, interleaved mRoPE, causal conv1d, both delta rules
  model.py      PLE/n-gram, DeltaNet, QSA indexer, gated attention, MoE,
                hyper-connections, full forward
  cache.py      HybridCache: conv + recurrent + KV + indexer keys + PLE history
  tokenizer.py  HF tokenizer + GGUF chat template
  generate.py   temperature / top-k / top-p sampling loop
  loader.py     load() entry point
```

## State
Config parses and validates against the real checkpoint. All modules import.
llama.cpp built for cross-validation. Download at 93/98 GB.

## Next
Iteration 004: run against llama.cpp on the same GGUF and compare token by
token — the first end-to-end correctness evidence.
