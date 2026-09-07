# Qwen3.8-Flash-Next on Tenstorrent Blackhole

A from-scratch inference stack for `unsloth/Qwen3.8-Flash-Next-GGUF` (`qwen4exp`:
48 layers of Gated DeltaNet + Qwen Sparse Attention, 512-expert MoE, a 51 B-param
n-gram embedding, and hyper-connections in place of every layer norm).

| # | deliverable | status |
|---|---|---|
| 1 | download the 4-bit checkpoint | **done** — UD-IQ4_XS, 93.68 GB, verified |
| 2 | plain PyTorch CPU reference engine | **done** — token-exact vs llama.cpp |
| 3 | ttnn engine + custom kernels + async core | **done** — 31.3 tok/s at batch 1, short context |
| 4 | OpenAI-compatible server | **done** — streaming, continuous batching, chunked prefill |

## Measured performance

**Batch 1, traced, at a genuinely filled context.** This model is served to a
coding agent, so batch 1 *is* the product and the batch axis is not measured:
one user, one stream, latency per token. Each row prefills a real prompt through
`TTModel.prefill` and then times `TracedDecoder.step` at that position, median
of 15 after 3 warm-up steps (`scripts/dev/bench_matrix.py <ctx>`).

| context | position | QSA selection off | QSA selection on |
|---|---|---|---|
| 256 | 192 | **31.99 ms** · 31.3 tok/s | not built below the budget |
| 2048 | 1984 | **34.93 ms** · 28.6 tok/s | not built below the budget |
| 4096 | 4032 | **39.74 ms** · 25.2 tok/s | 67.26 ms · 14.9 tok/s |
| 8192 | 8128 | **45.77 ms** · 21.9 tok/s | 105.80 ms · 9.5 tok/s |

The engine chooses the column: below `indexer_budget` (2048) it decodes with the
selection off, above it on. The right-hand column is therefore what a long
session actually costs, and the two columns together are the price of the
sparse-attention path -- 60 ms a token at 8192, more than the entire rest of the
model.

End to end through `TTEngine.generate` -- scheduling, sampling, detokenisation
and the async loop on top of the same step (`bench_matrix.py <ctx> --e2e`):

| context | e2e | vs the step alone |
|---|---|---|
| 256 | 35.50 ms · 28.2 tok/s | +3.5 ms |
| 2048 | 39.76 ms · 25.2 tok/s | +4.8 ms |

Those are the runs that returned. **`TTEngine.generate` hangs intermittently**
-- the identical ctx-256 command produced 35.50 ms twice and then hung for 30
minutes -- so the two rows above are single successful samples, not a
characterisation, and 4096/8192 have none yet. `TracedDecoder` at the same
contexts is entirely reliable, so the fault is the engine's loop around the
step, not the step; `docs/PRINCIPLES.md` 6b has what is known and the false
trail it laid first. This path was dead code until the conv-ring fix below, so
it has never run before.

Prefill, measured on the same runs: 117 tok/s at 1984 tokens, 188 at 4032,
157 at 8128 -- roughly 5x the rate the same prompt would cost stepped through
the decode path.

### How to read these

Two things make a decode number mean nothing if they are not stated, and both
were got wrong here before they were got right.

**The position, not the allocation.** A context opened at 8192 but decoded from
position 0 runs a *different amount of work* than the same graph at position
8128: the paged attention reads fewer pages, and the indexer's `topk` searches
only the blocks a position could have filled. Measured at position 0 the
8192 selection-on step reads 61 ms; at 8128 it is 105.8 ms. Only the second is
a number a user would experience.

**Which regime.** `selection_active` is a Python bool the trace bakes at
capture, and it defaults on while the engine runs it off below the budget. A
timing taken with the default that then gets compared against an engine
measurement is comparing two different models.

### Pointing an agent harness at it

The server speaks OpenAI chat completions, including tool calls. This checkpoint
does not emit OpenAI tool-call JSON -- its chat template asks for XML and the
server translates both directions -- and it is a reasoning model, so the answer
arrives after a `</think>` the server splits into `reasoning_content`.

Two things a client has to be told, because they are not the OpenAI defaults:
send `chat_template_kwargs: {"enable_thinking": false}` to skip the reasoning
pass (4.4x faster on short turns), and use the `system` role rather than
`developer`. For [pi](https://pi.dev), `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ttrunner": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsUsageInStreaming": false,
        "maxTokensField": "max_tokens",
        "thinkingFormat": "qwen-chat-template"
      },
      "models": [{ "id": "Qwen3.8-Flash-Next", "name": "Qwen3.8-Flash-Next",
                   "reasoning": true, "input": ["text"],
                   "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
                   "contextWindow": 32768, "maxTokens": 4096 }]
    }
  }
}
```

`contextWindow` is set well under the model's own because ingestion is ~13
ms/token: 32k of context is about seven minutes of prefill, and a bigger window
mostly buys a longer wait.

For a prompt-heavy single user, `chunked_prefill=True` consumes the prompt in
one pass per 128-token chunk rather than one step per token, and keeps the traced
decode step. The two used to be mutually exclusive -- a prefill allocating while
a trace is live corrupts the replay -- so the engine releases the trace around
each prefill and captures it again afterwards, restoring the state the capture
dirties. That costs ~2.6 s on a request that ingests and pays for itself after a
handful of generated tokens.

`speculate=k` drafts from the prompt and verifies k tokens in one pass. It is
**exact** — the output is identical to decoding one token at a time — and helps
only on text that quotes its context; on open prose the drafter rarely fires.
Speculation amortises the decode step, so its value shrinks every time that step
gets faster. Off by default; re-measure before enabling it.

Prefix reuse across chat turns is on unconditionally — a follow-up turn re-feeds
only what it added, measured **2.27 s** to first token warm against **11.40 s**
cold with `chunked_prefill=True` and **41.31 s** cold without, for the same
73-token prompt (`scripts/dev/prefix_reuse_check.py`). Both cases reuse exactly
the 69 tokens they should, miss correctly on an unrelated prompt, and return
token-identical output warm and cold.

Throughput-oriented configurations (more slots, shorter context):

| concurrency | sustained generation | end to end, 128 tokens out |
|---|---|---|
| 32 | **70.3 tok/s** | 46.9 tok/s |
| 64 | 72.3 tok/s | — |

Two numbers because they answer different questions. *Sustained generation* is
tokens over the span in which generation is actually running, which is what a
loaded server settles at; *end to end* divides by the whole wall clock, so it
carries prompt ingestion and is a function of how long the prompts are.
`scripts/dev/bench_server.py` is the harness.

The server feeds prompts one token per step unless chunked prefill is on.
`TTModel.prefill` is far faster per token and is used for one-slot engines; it is
refused above that because it consumes a whole prompt before returning, which a
shared lockstep batch cannot express.

## Layout

```
src/ttrunner_qwen38_flash_next/
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
  dev/        the measurement harnesses. The ones to reach for first:
                bench_matrix.py         the table above: one traced batch-1
                                        step per context, function or --e2e
                bench_step.py           batch-1 step, eager and traced
                                        (TTRUNNER_SELECTION=0 for the
                                        below-budget regime; pass a seq length,
                                        the default 262144 is not what you want)
                device_quality.py       next-token accuracy on real prose --
                                        the correctness gate, run it in BOTH
                                        regimes before believing a timing
                cumulative_ablation.py  what each component costs
                step_op_census.py       every ttnn call a step issues, by name
                                        and call site, with bytes
tests/        235 tests
docs/PRINCIPLES.md  the machine, the principles, the pitfalls -- read this first
docs/HANDOFF.md     the chronological log behind it
docs/iterations/    observation -> remedy -> result log for every step
```

## Quick start

```bash
# CPU reference, greedy
PYTHONPATH=src .venv/bin/python -c "
from ttrunner_qwen38_flash_next.reference.loader import load
from ttrunner_qwen38_flash_next.reference.generate import generate, SamplingParams
m, tok, cfg = load('/home/twjang/models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS',
                   '/home/twjang/models/Qwen3.8-Flash-Next-tokenizer/tokenizer.json')
ids = tok.encode('The capital of France is')
print(''.join(tok.decode([t]) for t in generate(m, ids, SamplingParams(max_tokens=8, temperature=0))))
"

# convert weights for the device (once, ~30 min)
PYTHONPATH=src .venv/bin/python -c "
from ttrunner_qwen38_flash_next.tt.convert import convert
convert('/home/twjang/models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS',
        '/home/twjang/models/qwen38-tt-cache')"

# server on the CPU reference
PYTHONPATH=src .venv/bin/python -m ttrunner_qwen38_flash_next.server \
  --model /home/twjang/models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS \
  --tokenizer /home/twjang/models/Qwen3.8-Flash-Next-tokenizer/tokenizer.json

# server on the 4 x Blackhole mesh
PYTHONPATH=src .venv/bin/python -m ttrunner_qwen38_flash_next.server --backend tt \
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
| device, QSA selection on | 83.0 % | 97.9 % | 0.682 | 1.98 |

Re-checked after `HANDOFF.md` 45.44 narrowed the selection's search: A/B'd in one
binary over 64 stepped tokens at `max_seq_len` 2049, the old full-range `topk`
and the new eligible-prefix one score **50/63 top-1, 61/63 top-5, NLL 0.794** --
identical to the digit -- and `traced_vs_eager` matches token for token.
| float32 CPU reference | 80.9 % | 97.9 % | 0.703 | 2.00 |

Chunked prefill is judged separately, against a decode control over the *same*
positions — the question is whether the state a prompt leaves still predicts the
text, not whether it matches decode token for token:

| prefilled | scored | prefill | decode control |
|---|---|---|---|
| 32 | 128 | **71.9 %** top-1, NLL 1.43 | 71.7 %, NLL 1.48 |
| 128 | 107 | 21.5 % top-1, NLL 6.35 | 22.6 %, NLL 6.00 |

Both pairs measured back to back in one batch of runs, which matters: readings
of the 128-row row taken at different points in one session spanned 21.5–27.1 %
from what appears to be identical code, and that spread is unexplained. Direct
testing finds the path deterministic — three processes return bit-identical
logits and decode steps — so the rule is to pair an A/B rather than to distrust
the path. `docs/HANDOFF.md` invariant 8.

On these numbers a 32-row prefill leaves state slightly *better* than stepping
the same tokens and a 128-row prefill slightly worse.

(Absolutely lower than the decode row above because the passage runs on into
dates and proper nouns; both paths fall together, which is the point of having a
control rather than an absolute target.)

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
