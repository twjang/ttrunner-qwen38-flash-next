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
| 1 | **262144** | 300.5 | **229** |
| 2 | 131072 | 319.0 | — |
| 3 | 65536 | 323.7 | — |

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

**Read `docs/iterations/013` before trusting any number in this section.** The
device engine agrees with the float32 CPU reference on 40 % of greedy tokens of
real text (64 % where the reference is confident), measured teacher-forced over
47 positions with `scripts/dev/three_way_agreement.py`. Chunked prefill is worse
again at 27 %, and stays disabled. Earlier claims of "verified token-for-token"
in this file and in the iteration log rested on a single 5-token prompt
producing plausible text; they were not verification, and the model they were
checking had the wrong DeltaNet head pairing.

Every individual piece now sits at its dtype floor -- each branch of each layer
kind within 0.2-4 % of the reference, weights round-tripping at theirs, state
carried between steps not drifting -- so the gap is 48 layers of accumulated
precision rather than a defect anyone has found. Closing it is the current
priority; see `docs/HANDOFF.md`.

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
3. ~~**DeltaNet V heads are stored tiled**, not grouped~~ — **wrong**, and it
   cost 41 % in the first DeltaNet layer until `docs/iterations/013` caught it.
   V heads are *grouped*: v-head j reads k-head `j // reps`, expanded with
   `repeat_interleave`, exactly as upstream does it. Tiling also cannot be
   head-sharded, which is why the device was a third thing again.

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
