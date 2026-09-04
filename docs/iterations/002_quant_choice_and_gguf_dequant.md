# 002 — Quant choice, and a bit-exact GGUF dequantiser

Date: 2026-09-01

## Goal
Pick the weight format the whole stack is built on, and get a verified
dequantisation path for it.

---

## Observation 1 — Q8_0 does not fit the device, and the reason is one tensor

Q8_0 is 188 GB against 128 GB of total device DRAM (4 × Blackhole p150a). The
obvious read is "too big by 60 GB", but the breakdown matters:

| component | params | note |
|---|---|---|
| n-gram / PLE embedding | 51 B | pure gather, ~10 rows touched per token |
| MoE experts | ~121 B | 10 routed + 1 shared of 512 active per token |
| attention + hyper-connections + norms | ~4 B | dense, every token |

The n-gram table is a **lookup**, not a matmul. It never needs to be resident on
an accelerator — the model card explicitly calls it "amenable to offloading".
Excluding it, Q8_0 still needs ~134 GB on device, so it genuinely overflows; but
the honest budget to compare against is ~67 GB, not 94 GB.

---

## Observation 2 — reading UD-IQ4_XS's headers cost 90 MB, not 94 GB

To evaluate the user's UD-IQ4_XS suggestion without downloading it, only the
GGUF headers were needed. HTTP range requests for the first 40 MB of each shard
were enough to parse every tensor's name, shape and quant type.

**Result** — Unsloth's dynamic per-tensor bit allocation, in full:

| tensor role | size | quant | bpw |
|---|---|---|---|
| `per_layer_token_embd` | 28.8 GB | IQ4_NL | 4.5 |
| `ffn_down_exps` | 24.0 GB | IQ4_NL (layers 0-4 kept Q8_0) | 4.5 |
| `ffn_gate_exps` / `ffn_up_exps` | 33.9 GB | IQ3_S (1 layer IQ4_XS) | 3.4 |
| `attn_*`, `hc_*`, `ssm_out`, `token_embd` | ~9 GB | Q8_0 | 8.5 |
| `ffn_gate_inp` (router), norms, inject | 0.31 GB | F32 | 32 |
| `indexer.*` | 0.04 GB | BF16 | 16 |
| `output.weight` | 0.52 GB | Q6_K | 6.6 |

This is an imatrix-calibrated sensitivity map: `down_proj` gets a bit more than
`gate/up`, the routers stay in F32, and the QSA indexer — which selects *which*
tokens are attended to, so an error there is not a small perturbation but a
wrong token set — is kept at BF16.

**Decision (user):** cancel Q8_0, support 4-bit only. The Q8_0 download was
stopped at 47/196 GB and its partials deleted (60 GB reclaimed); UD-IQ4_XS plus
the Q4_K_M MTP heads (~98 GB) is now downloading. `mmproj-F16.gguf` was already
complete and was kept.

---

## Observation 3 — the byte accounting was off by 5.45 GB, and that was a real bug

Summing every tensor gave **99.13 GB**; HF reports the repo directory as
**93.68 GB**. A 5.5 % discrepancy is not rounding.

**Remedy** — rather than trust memory for the block layouts, fetched
`ggml-common.h` and checked the `static_assert`s that ggml itself compiles
against:

```c
static_assert(sizeof(block_iq4_nl) == sizeof(ggml_half) + QK4_NL/2, ...);
```

`2 + 32/2 = 18` bytes per block, not the 20 in the table. IQ4_NL is 55 % of the
file, and 54.54 × 18/20 = 49.09 GB — exactly the 5.45 GB gap.

**Result** — after the fix, computed 93.67 GB vs 93.68 GB reported (the
remainder is header bytes). The size arithmetic now self-validates, which is
what caught the error in the first place.

All layouts verified against upstream: Q8_0 34, Q4_K 144, Q6_K 210, IQ3_S 110,
IQ4_NL 18, IQ4_XS 136.

---

## Observation 4 — codebook constants must not be hand-transcribed

IQ3_S dequantisation needs `iq3s_grid`, a 512-entry `uint32` table, and IQ4_NL
needs the 16-entry `kvalues_iq4nl`. Retyping 512 hex constants is a silent
corruption waiting to happen — a single wrong digit degrades one weight in 512
in a way no shape check would catch.

**Remedy** — `scripts/gen_codebooks.py` parses the `GGML_TABLE_BEGIN(...)`
blocks straight out of `ggml-common.h`, asserts the parsed count matches the
declared count, and emits `src/ttrunner_qwen38_flash_next/gguf/_codebooks.py`.

**Result** — 512/512 and 16/16 entries parsed and asserted.

---

## Observation 5 — the dequantiser is bit-exact

Implemented vectorised numpy dequantisation for all seven formats in the
checkpoint (`src/ttrunner_qwen38_flash_next/gguf/quants.py`), following `ggml-quants.c` exactly.
IQ3_S is the delicate one: a 9th grid-index bit comes from `qh` via
`qs[2l] | ((qh << (8-2l)) & 256)`, and each group of eight values carries its
own sign byte.

Validation had to be against an *independent* implementation, not self-
consistency. The `gguf` package ships its own numpy dequanters, so both were run
on the same bytes. Only ~90 MB had been downloaded, so representative tensors
were pulled by byte range — dequantisation is block-independent, so a
few-hundred-block prefix validates the format completely.

```
type      blocks    elems   max_abs_err  verdict   tensor
BF16         512      512     0.000e+00  EXACT     blk.3.indexer.k_proj.weight
F32          512      512     0.000e+00  EXACT     output_hc_norm.weight
IQ3_S        512   131072     0.000e+00  EXACT     blk.0.ffn_gate_exps.weight
IQ4_NL       512    16384     0.000e+00  EXACT     per_layer_token_embd.weight
IQ4_XS       512   131072     0.000e+00  EXACT     blk.2.ffn_gate_exps.weight
Q6_K         512   131072     0.000e+00  EXACT     output.weight
Q8_0         512    16384     0.000e+00  EXACT     output_hc_down.weight

FAILURES: 0
```

Q4_K is implemented but not yet exercised — it appears only in the MTP head,
which is still downloading.

---

## Observation 6 — llama.cpp already defines the tensor namespace

Rather than reverse-engineering GGUF names from the checkpoint, `gguf-py`'s
`constants.py` has an explicit `MODEL_ARCH.QWEN4EXP` tensor list, including a
comment confirming two things worth knowing before writing the model:

- there is **no** `OUTPUT_NORM` / `ATTN_NORM` / `ATTN_POST_NORM` — the
  hyper-connections replace every layer norm;
- `ATTN_Q` holds `[q | gate]` interleaved per head, matching the reference
  implementation's `torch.chunk(q_proj(x), 2, dim=-1)`.

---

## State at end of iteration
- `src/ttrunner_qwen38_flash_next/gguf/reader.py` — GGUF v3 parser, mmap-backed, split-shard aware.
- `src/ttrunner_qwen38_flash_next/gguf/quants.py` — 7 formats, bit-exact vs `gguf`.
- `src/ttrunner_qwen38_flash_next/gguf/_codebooks.py` — generated from ggml source.
- UD-IQ4_XS downloading (~24/98 GB at time of writing).

## Next
Iteration 003: weight-map the checkpoint onto the `qwen4_exp` module tree and
build the plain PyTorch CPU reference engine.
