# Qwen3.8-Flash-Next on Tenstorrent Blackhole

A from-scratch inference stack for `unsloth/Qwen3.8-Flash-Next-GGUF` (`qwen4exp`:
48 layers of Gated DeltaNet + Qwen Sparse Attention, 512-expert MoE, a 51 B-param
n-gram embedding, and hyper-connections in place of every layer norm).

| # | deliverable | status |
|---|---|---|
| 1 | download the 4-bit checkpoint | **done** — UD-IQ4_XS, 93.68 GB, verified |
| 2 | plain PyTorch CPU reference engine | **done** — token-exact vs llama.cpp |
| 3 | ttnn engine + custom kernels + async core | **done** — 97.4 tok/s at batch 64, bit-exact vs single-sequence |
| 4 | OpenAI-compatible server | **done** — streaming, continuous batching, 37.84 tok/s at 32 concurrent |

## Measured performance

Four Blackhole p150a, UD-IQ4_XS weights (24.94 GB/device). Median of 25 samples
after 5 warmup steps — a thinner harness (2 warmup, 3 samples) reported 107 tok/s
for the same configuration, 26 % optimistic, because kernel JIT landed inside the
measured window.

| | before | after |
|---|---|---|
| weight load | 2373 s | **6.0 s** (395×) |
| decode, batch 64 | 0.75 tok/s | **97.4 tok/s** (130×) |
| decode, batch 1 (eager) | 1338 ms | **518 ms** (2.6×) |

```
   B  median ms      p10      p90     tok/s   ms/tok
   1      518.3    511.2    529.8      1.93   518.27
  16      549.0    544.7    562.1     29.14    34.32
  32      550.5    547.4    553.3     58.13    17.20
  64      742.8    739.6    757.2     86.16    11.61
```

Batch 64 is the last valid step: batch pads to multiples of 32, so 65..96 all
allocate as 96 and overflow L1 by ~22 %. The table above is the split-expert path;
fusing the experts' gate and up projections into one `sparse_matmul` (built by
`scripts/fuse_expert_gate_up.py`) gives **97.4 tok/s** at batch 64 with identical
tokens, and is neutral-to-better at every batch size, so the engine uses it
whenever the prebuilt weights are present:

| batch | split | fused |
|---|---|---|
| 1 | 522.4 / 520.3 ms | 504.8 / 523.0 ms |
| 32 | 57.57 / 57.97 tok/s | 59.51 / 58.49 tok/s |
| 64 | 86.66 / 86.60 tok/s | **97.47 / 97.36 tok/s** |

Trace capture replays a step with one dispatch. Through `TTModel` it is verified
token-for-token against eager:

| batch | eager | traced | gain |
|---|---|---|---|
| 1 | 515.7 ms | 255.3 ms | 2.02× |
| 16 | 552.7 ms | 364.3 ms | 1.52× |
| 32 | 538.7 ms | 417.8 ms | 1.29× |

It is **off by default in the server** (`use_trace=False`): the same decoder
driven through `TTEngine` emits corrupted text. Ten candidates were excluded by
standalone reproductions that passed, and differential instrumentation showed the
trace reads bit-identical inputs — only interleaving a full eager step restores
it, which costs exactly what the trace saves. See `docs/iterations/011`.
Correct and slower beats fast and wrong.

End-to-end through the engine, greedy, 12 output tokens after a 5-token prompt
(so 17 device steps produce 12 tokens — prompts are fed one token per step):

The server defaults to a single user: one slot at the model's full 262144-token
context, traced. Per-token cost is flat in position (496 ms at 4, 501 ms at
65536) because the step is dominated by the MoE and DeltaNet and the
sparse-attention budget is fixed at 2048 — long context costs memory, not time.
The K/V cache is 6.4 GB per sequence at full length, so slots and context trade
directly and the engine checks them against the DRAM budget at construction.

| slots | context | ms/token | ms/step |
|---|---|---|---|
| 1 | **262144** | 300.5 | **236** |
| 2 | 131072 | 319.0 | — |
| 3 | 65536 | 323.7 | — |

(229 ms before `docs/iterations/014`; the model was wrong then. Fixing the
DeltaNet head pairing cost 3 ms, and only because `attn_qkv` now carries twelve
q/k heads per device instead of four. With QSA's sparse selection on -- contexts
in `(2048, 65536]` -- a step is 297.5 ms at 8192 tokens against 236.1 dense; the
extra is almost all `ttnn.topk` at k=512, see `docs/iterations/015`.)

For a prompt-heavy single user, `chunked_prefill=True` consumes the prompt at
~45 ms/token instead of ~500, at the cost of the trace (each generated token
then costs ~513 ms). A 2000-token prompt with 200 output tokens is ~194 s that
way against ~1047 s traced-and-stepped; short prompts invert it. Prefix reuse
across chat turns is on unconditionally — a follow-up turn re-feeds only what it
added, measured 7.49 s to first token cold against 2.07 s warm.

Throughput-oriented configurations (more slots, shorter context):

| concurrency | tok/s | vs single |
|---|---|---|
| 8 | 10.18 | 7.2× |
| 32 | **37.84** | 30.5× |

Prompts are fed one token per step. `TTModel.prefill` is 11.3× faster
(45.0 vs 509.0 ms/token) but disagrees with the decode path from the first
chunk, so it is not enabled — see `docs/iterations/012`.

## Layout

```
src/twtest/
  gguf/       GGUF v3 reader + 10 dequantisation formats (bit-exact vs `gguf`)
  reference/  the plain PyTorch CPU model
  tt/         the Blackhole engine:
                plan.py        residency + per-tensor precision policy
                convert.py     GGUF -> .tensorbin device weight cache
                blockfloat.py  bit-exact bfp8_b/bfp4_b quantiser
                weights.py     cache loader (replicate vs shard)
                model.py       48-layer forward: batched decode + chunked prefill
                moe.py         on-device routing + sparse_matmul experts
                linear_attn.py DeltaNet decode recurrence
                deltanet.py    prep for ttnn's gated_delta_attn_seq
                ops.py         grouped RMS norm, hyper-connection gate
                engine.py      async engine (one device thread, continuous batching)
                bench.py       per-section timing
  server/     OpenAI-compatible FastAPI app (backend-agnostic)
scripts/      codebook generation, Telegram progress notifier
tests/        94 tests
docs/iterations/  observation -> remedy -> result log for every step
```

## Quick start

```bash
# CPU reference, greedy
PYTHONPATH=src .venv/bin/python -c "
from twtest.reference.loader import load
from twtest.reference.generate import generate, SamplingParams
m, tok, cfg = load('/home/twjang/models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS',
                   '/home/twjang/models/Qwen3.8-Flash-Next-tokenizer/tokenizer.json')
ids = tok.encode('The capital of France is')
print(''.join(tok.decode([t]) for t in generate(m, ids, SamplingParams(max_tokens=8, temperature=0))))
"

# convert weights for the device (once, ~30 min)
PYTHONPATH=src .venv/bin/python -c "
from twtest.tt.convert import convert
convert('/home/twjang/models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS',
        '/home/twjang/models/qwen38-tt-cache')"

# server on the CPU reference
PYTHONPATH=src .venv/bin/python -m twtest.server \
  --model /home/twjang/models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS \
  --tokenizer /home/twjang/models/Qwen3.8-Flash-Next-tokenizer/tokenizer.json

# server on the 4 x Blackhole mesh
PYTHONPATH=src .venv/bin/python -m twtest.server --backend tt \
  --model /home/twjang/models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS \
  --tokenizer /home/twjang/models/Qwen3.8-Flash-Next-tokenizer/tokenizer.json \
  --tt-cache /home/twjang/models/qwen38-tt-cache --max-concurrency 8

# tests
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

## Correctness

The metric is next-token accuracy on real prose, measured against the text
itself rather than against another implementation
(`scripts/dev/device_quality.py`, 47 positions):

| | top-1 | top-5 | mean NLL | perplexity |
|---|---|---|---|---|
| device, decode | **83.0 %** | 97.9 % | 0.682 | 1.98 |
| device, chunked prefill | 87.5 % | 100 % | 0.335 | 1.40 |
| device, QSA selection on | 83.0 % | 97.9 % | 0.682 | 1.98 |
| float32 CPU reference | 80.9 % | 97.9 % | 0.703 | 2.00 |

QSA attends to 2048 *selected* tokens, not to the whole context. The selection
runs on device and was verified against the reference past the budget --
exact at a block boundary, one block out mid-block at a score gap of 8.2e-4 --
and below the budget it is numerically identical to dense, which is what the
third row shows. It is on for contexts in `(2048, 65536]`; outside that the
device attends densely, and `TTEngine` prints a notice when that is a fidelity
caveat rather than an exact simplification. See `docs/iterations/015`.

Agreement *between* implementations is deliberately not the headline: with 512
experts and top-10 routing any perturbation flips a selection somewhere, so two
correct implementations of the same weights still disagree on a good fraction of
greedy tokens, and every precision configuration measured lands in the same
30-47 % band. `docs/iterations/014` has that table, and the story of how a wrong
head pairing survived being eyeballed for several iterations because a damaged
model still writes fluent English.

The reference engine reproduces llama.cpp's greedy output on the same GGUF,
token for token:

```
MINE     : 'The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is'
LLAMA.CPP: 'The capital of France is Paris. The capital of Germany is Berlin. The capital of'
```

Getting there required finding three things the checkpoint does differently from
the upstream HF model, none of which change any tensor's shape:

1. **Norm weights carry a folded `+1`** (all except `linear_attn.norm.weight`).
2. **`ssm_a` stores `A = -exp(A_log)`**, not `A_log` — so the decay is
   `ssm_a * softplus(...)`, with no second `exp`.
3. **DeltaNet V heads are stored tiled**, not grouped — Q/K expand with
   `repeat`, not `repeat_interleave`. Upstream's HF code interleaves
   (`modeling_qwen4_exp.py:594`) because the converter permutes the head order;
   following it instead of measuring cost a day (`docs/iterations/014`). The
   arbiter is next-token accuracy on real text: 80.9 % tiled, 12.8 % grouped.
   Both read as fluent English, which is why a greedy sample cannot settle it.

Plus two of our own: the DeltaNet output gate is **sigmoid** (`output_gate_type`
is set in `config.json` but never written to the GGUF), and the chunked delta
rule must form decays by **subtraction in log space** — as a ratio of
exponentials it yields `0/0 = NaN` in the carried state for the head with
`A = -158`, while leaving the chunk output finite. That one is invisible to any
prefill-only test.

## Device engine

4 × Blackhole p150a, 26.34 GB of weights per device of 32 GB. The 28.8 GB n-gram
table stays in host RAM (it is a gather touching 16 rows per token), experts are
`bfp4_b` gate/up and `bfp8_b` down, and only the expert stacks are sharded — so
there are 48 all-reduces plus one all-gather per step rather than ~100.

Most of what looked like it would need hand-written kernels already exists in
ttnn and was validated against the CPU reference on real hardware:

| piece | op | vs reference |
|---|---|---|
| Gated DeltaNet (36 layers) | `transformer.gated_delta_attn_seq` | 0.8 % rms |
| QSA indexer scoring | `experimental.indexer_score_dsa` | 1.0 % max |
| MoE experts | `sparse_matmul` (expert-skipping) | bfp4-bound |
| hyper-connections | `rms_norm` + matmuls | 0.15 % |
| all-reduce | `all_reduce` (needs `FABRIC_1D`) | 0.23 ms |

The precision policy was chosen by simulating the device's exact block-float
arithmetic inside the CPU reference rather than by analogy — see
`docs/iterations/008`.

See `docs/iterations/` for the full log.

## Notes

- Model license is `qwen-community-1.0` (not OSI-approved); review before any
  commercial deployment.
- Dependencies used here are MIT/Apache-2.0/BSD (`gguf`, `tokenizers`, `fastapi`,
  `uvicorn`, `torch`, `ttnn`). Quantisation constants are generated from
  llama.cpp's `ggml-common.h` (MIT) by `scripts/gen_codebooks.py` rather than
  hand-copied.
