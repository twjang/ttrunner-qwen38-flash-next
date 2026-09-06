# Qwen3.8-Flash-Next on Tenstorrent Blackhole

A from-scratch inference stack for `unsloth/Qwen3.8-Flash-Next-GGUF` (`qwen4exp`:
48 layers of Gated DeltaNet + Qwen Sparse Attention, 512-expert MoE, a 51 B-param
n-gram embedding, and hyper-connections in place of every layer norm).

| # | deliverable | status |
|---|---|---|
| 1 | download the 4-bit checkpoint | **done** — UD-IQ4_XS, 93.68 GB, verified |
| 2 | plain PyTorch CPU reference engine | **done** — token-exact vs llama.cpp |
| 3 | ttnn engine + custom kernels + async core | **done** — 107.6 tok/s at batch 64; bit-exact vs single-sequence up to batch 32 |
| 4 | OpenAI-compatible server | **done** — streaming, continuous batching, 70.3 tok/s at 32 concurrent |

## Measured performance

Four Blackhole p150a, UD-IQ4_XS weights (24.94 GB/device). Median of 25 samples
after 5 warmup steps — a thinner harness (2 warmup, 3 samples) reported 107 tok/s
for the same configuration, 26 % optimistic, because kernel JIT landed inside the
measured window.

| | before | after |
|---|---|---|
| weight load | 2373 s | **6.0 s** (395×) |
| decode, batch 64 | 0.75 tok/s | **107.6 tok/s** (143×) |
| decode, batch 1 (eager) | 1338 ms | **546 ms** (2.5×) |
| decode, batch 1 (traced) | — | **32.0 ms** — 31.3 tok/s |

**Batch 1 is now 32.0 ms a token traced**, and that number is only meaningful
with its regime attached — see the two regimes below. Tracing is worth 17x here
(546 ms eager against 32.0 traced, both measured in one process at
`max_seq_len 512`), so any batch-1 figure taken eagerly is describing a
different program's cost structure.

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
`scripts/fuse_expert_gate_up.py`) gives **107.6 tok/s** at batch 64 with identical
tokens, and is neutral-to-better at every batch size, so the engine uses it
whenever the prebuilt weights are present:

| batch | split | fused |
|---|---|---|
| 1 | 522.4 / 520.3 ms | 504.8 / 523.0 ms |
| 32 | 57.57 / 57.97 tok/s | 59.51 / 58.49 tok/s |
| 64 | 86.66 / 86.60 tok/s | **97.47 / 97.36 tok/s** |

(That comparison is from before `docs/iterations/013`-`014`; what it establishes
is the split-vs-fused *relationship*, which is why it is kept. The absolute
figures on the fixed model are below.)

Re-measured on the model that works, 5 warmup + 25 samples
(`scripts/dev/bench_batch.py`), which is also the first run of the paged K/V
cache above one sequence:

| batch | ms/step | tok/s |
|---|---|---|
| 1 | 479.0 | 2.09 |
| 8 | 521.9 | 15.33 |
| 32 | 517.1 | 61.88 |
| 64 | 594.8 | **107.59** |

A sequence decodes bit-identically in a batch as it does alone **up to batch
32**, checked by `scripts/dev/batch_equivalence_check.py` (8/8 and 32/32 rows).
At batch 64 the rows all agree with each other but not with a batch-1 run: 64
rows put `per_core_M` at 2 in the experts' `sparse_matmul`, which needs the
single-K-block program config to be correct at all, and that config accumulates
in a different order. It is a bf16 difference, not a defect.

It *was* a defect until recently, and a bad one: with the narrow config at
`per_core_M = 2`, `sparse_matmul` silently drops rows past the first 32-row
tile, so batch 64 had **32 of its 64 rows corrupted** and the "bit-exact"
claim above was simply false. See `docs/HANDOFF.md` 5.8. Batch 32 was and
remains exact.

Making batch 64 match batch 1 was tried and does not work: forcing the experts'
program config to be batch-independent leaves the divergence exactly where it
was, because the *router's* `ttnn.linear` is batch-dependent too — its
probabilities move up to 0.53 % between groupings, which is enough to reorder
experts at the top-k boundary. So a sequence's output does depend on how many
others share its batch, above 32. That is a property to know about, not a bug to
fix at this level.

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
context, traced.

**There are two decode regimes and they differ by 3x.** `indexer_budget` is 2048.
Below that position the sparse selection is provably a no-op, so `TTEngine` runs
with `selection_active = False`; at 2048 it flips. Measured in one harness at one
sequence length, changing only that flag:

| `selection_active` | ms/step | tok/s |
|---|--:|--:|
| False — positions 0..2047 | **32.3** | 31.0 |
| True — positions >= 2048 | **61.5** | 16.3 |

So the *attention* is budget-limited at 2048 selected tokens, but the selection
itself still costs ~29 ms a token. Long context costs memory *and* time, and a
batch-1 number quoted without its regime says little. The `ms/step` column below
is the below-budget regime.

(105.8 ms until `docs/HANDOFF.md` 45.44. `ttnn.topk` there is O(nb x k) — 4.4 ms
for k=512 of nb=2048, twelve QSA layers a token — and it was searching every
block in the allocated context when only the first `p // 4` can ever be
selected; the rest carry a -inf bias. Restricting the search to the eligible
prefix is **-44 ms** with next-token accuracy identical to the digit.)

### Where the 32 ms goes, and how far the roofline is

Single-part ablation of the 32.65 ms step (`scripts/dev/cumulative_ablation.py`),
below the budget:

| component | cost |
|---|--:|
| DeltaNet (36 layers) | **8.36 ms** |
| `gated_residual_mix` | **6.68 ms** |
| MoE (48 layers) | **5.67 ms** |
| collectives | 3.86 ms |
| QSA (12 layers) | 3.66 ms |
| shared expert | 1.18 ms |
| `reinject`, PLE | 0.04, 0.03 ms |

The weight read is ~3.17 GB a device a token, which is **8.2 ms at the measured
388 GB/s — 122 tok/s**. That is a floor, not a target: a perfect implementation
still spends it. At 32.0 ms the step is **26 % of its own roofline**, and the
gap is structural rather than incidental — at M = 1 `ttnn.linear` gives each
output tile to one core, so a matmul with 16 output tiles (the router) runs on
16 of 110 cores. 460 such calls carry 80 % of the bytes moved.

Two things follow, both measured rather than argued:

* **Cutting bytes does not buy time at M = 1.** A thirteenfold byte reduction
  buys 20 % of the time, and one shard measured *slower* than the tensor it
  replaced.
* **1.69 of the 3.17 GB is replicated** — read four times over, once per device.
  A single memory holding the same weights would read 7.61 GB of unique bytes
  per token; the mesh reads 12.7. That penalty is architectural, not a kernel
  problem.

`docs/PRINCIPLES.md` is the distilled version of all of this — the machine, the
two regimes, the four principles that decide what is worth trying, the
measurement discipline, and the framework pitfalls. Read it before touching the
decode path. `docs/HANDOFF.md` is the chronological log behind it.
The K/V cache is 6.4 GB per sequence at full length, so slots and context trade
directly and the engine checks them against the DRAM budget at construction.

| slots | context | ms/token | ms/step |
|---|---|---|---|
| 1 | **262144** | 300.5 | 173.7 |
| 2 | 131072 | 319.0 | — |
| 3 | 65536 | 323.7 | — |

(That table predates the work in `docs/HANDOFF.md` 44-46 and has not been
re-measured at 262144 since; the current traced step is **32.0-32.7 ms** at
`max_seq_len` 512, 4096 and 8192, consistent across all three. Opening the model
at 262144 allocates a 6.4 GB K/V slot and a 65536-entry indexer block cache, so
that configuration is slow to *set up* whatever the step costs.)

(236.2 ms until `docs/iterations/022`, which found that the routed MoE was 65 %
of the step and that almost none of that was the experts -- at M = 1 the row axis
pads to 32, so every [1, 512, 1, 2560] tensor was 84 MB carrying 2.6 MB. Removing
the per-expert input copy and combining the experts with one matmul took it to
173.7 without giving up any accuracy: the first change is bit-identical, the
second measurably closer to an exact float64 reference than what it replaced.
Before that, 229 ms pre-`014`, when the model was wrong. With QSA's sparse
selection on -- contexts in `(2048, 65536]` -- a step was 297.5 ms at 8192 tokens
against 236.1 dense; the extra is almost all `ttnn.topk` at k=512, see
`docs/iterations/015`.)

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

For a prompt-heavy single user, `chunked_prefill=True` consumes the prompt at
**~13 ms/token** instead of ~177, and keeps the traced decode step. The two used
to be mutually exclusive -- a prefill allocating while a trace is live corrupts
the replay -- so the engine now releases the trace around each prefill and
captures it again afterwards, restoring the state the capture dirties. That costs
~2.6 s on a request that ingests and pays for itself after about seven generated
tokens. A 1051-token prompt with 200 tokens out takes **50.7 s**, against 125.6 s
with the trace off and 221.4 s with no chunked prefill.

`speculate=k` drafts from the prompt and verifies k tokens in one pass. It is
**exact** — the output is identical to decoding one token at a time — and worth
1.10x on text that quotes its context (160.3 ms/token against 176.4 at
`speculate=8`), and neutral on open prose, where the drafter rarely fires. It
was worth 1.37x until the traced step itself got 1.36x faster; speculation
amortises that step, so most of its advantage went with it. Off by default. Prefix reuse
across chat turns is on unconditionally — a follow-up turn re-feeds only what it
added, measured **2.27 s** to first token warm. Cold depends on how the prompt is
fed and the earlier figure here did not say: **11.40 s** with
`chunked_prefill=True`, **41.31 s** without it, for the same 73-token prompt
(`scripts/dev/prefix_reuse_check.py`, with and without `--chunked`). Both cases
reuse exactly the 69 tokens they should, miss correctly on an unrelated prompt,
and return token-identical output warm and cold.

Throughput-oriented configurations (more slots, shorter context):

| concurrency | sustained generation | end to end, 128 tokens out |
|---|---|---|
| 32 | **70.3 tok/s** | 46.9 tok/s |
| 64 | 72.3 tok/s | — |

Two numbers because they answer different questions. *Sustained generation* is
tokens over the span in which generation is actually running, which is what a
loaded server settles at; *end to end* divides by the whole wall clock, so it
carries prompt ingestion and is a function of how long the prompts are. The
earlier figure here was 37.84 tok/s with no recorded methodology and was taken
before the correctness fixes in `docs/iterations/013`-`014`, so it is not
directly comparable to either. `scripts/dev/bench_server.py` is the harness.

The server feeds prompts one token per step. `TTModel.prefill` is ~71× faster
(7.2 vs 509.0 ms/token) and is on for one-slot engines; it is refused above that
because it consumes a whole prompt before returning, which a shared lockstep
batch cannot express.

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
