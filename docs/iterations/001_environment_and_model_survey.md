# 001 — Environment & model survey

Date: 2026-09-01

## Goal
Establish the ground truth needed for all four deliverables: does the requested
model exist, what is its architecture, and what hardware/toolchain is actually
available on this box.

---

## Observation 1 — the model name looked like a typo, but is real

`unsloth/Qwen3.8-Flash-Next-GGUF` reads like a garbled `Qwen3-Next`, so it was
worth verifying before committing to a ~190 GB download.

**Remedy** — queried the HF API directly rather than guessing:

```
curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/api/models/unsloth/Qwen3.8-Flash-Next-GGUF
```

**Result** — `200`. Real repo, public (not gated), 431 k downloads, last
modified the same day. Base model is `Qwen/Qwen3.8-Flash-Next`, license
`qwen-community-1.0` (not OSI; a per-use review is needed before any commercial
deployment — flagged, not blocking for local work).

The 8-bit weights are `Q8_0/`, six shards, **188.2 GB**:

| shard | size |
|---|---|
| 00001-of-00006 | 0.01 GB |
| 00002-of-00006 | 0.68 GB |
| 00003-of-00006 | 54.40 GB |
| 00004-of-00006 | 49.45 GB |
| 00005-of-00006 | 49.67 GB |
| 00006-of-00006 | 34.02 GB |

Also pulled: `mmproj-F16.gguf` (0.90 GB, vision tower — the model is
image-text-to-text) and the two Q8_0 MTP heads (4.14 + 2.79 GB) for later
speculative decoding.

---

## Observation 2 — `--include` silently swallowed all but its first pattern

First download invocation:

```
hf download ... --include "Q8_0/*" "mmproj-F16.gguf" "MTP/*Q8_0.gguf" "README.md"
```

emitted `UserWarning: Ignoring --include since filenames have being explicitly
set.` The trailing patterns bound to the positional `FILENAMES` argument, which
*disables* `--include` entirely — so the 188 GB `Q8_0/` tree was never queued.
Only the 0.9 GB mmproj came down. A silent no-op that would have looked like a
successful download.

**Remedy** — one `--include` flag per pattern:

```
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --include "Q8_0/*" --include "mmproj-F16.gguf" --include "MTP/*Q8_0.gguf" \
  --local-dir /home/twjang/models/Qwen3.8-Flash-Next-GGUF --max-workers 8
```

**Result** — all six `Q8_0` shards now streaming into
`.cache/huggingface/download/Q8_0/*.incomplete`. Disk: 455 GB free vs 196 GB
needed, so headroom is fine.

Secondary lesson: `pkill -f "hf download unsloth"` killed the *tool call's own
shell* (its command line contains the pattern). The relaunch uses `setsid` so
the download is detached and survives shell teardown.

---

## Observation 3 — `telegram-send` times out under download load

Every `telegram-send` invocation failed with `Connection timed out`, escalating
its own suggested `--timeout` to 160 s. But `curl https://api.telegram.org/`
connected in 0.3 s — the API is reachable; the eight parallel HF download
workers are simply starving the client.

**Remedy** — `scripts/notify.py`: reads the bot token from the existing
`~/.config/telegram-send.conf` (nothing hardcoded, token never printed or
placed in `argv`), posts via `sendMessage` with a 30 s timeout and four
retries with linear backoff.

**Result** — messages deliver reliably while the download saturates the link.

---

## Observation 4 — ttnn is already installed; it was just invisible to system Python

`python3 -c "import ttnn"` → `ModuleNotFoundError`, which suggested a multi-hour
tt-metal source build stood between us and deliverable (3). But system Python is
3.10 and the project venv is 3.12.

**Remedy** — checked the venv instead of the system interpreter.

**Result** — `.venv/lib/python3.12/site-packages` already ships `ttnn`, `ttl`
(the TT-Lang kernel DSL), `tt_lang`, `tt_torch`, `tt_jax`, `pjrt_plugin_tt`
1.4.0 and `torch_xla`. No build required. Live device probe:

```
num_devices : 4
arch        : Arch.BLACKHOLE
compute grid: 11 x 10  (110 Tensix cores per device)
```

Boards are 4 × **p150a** (Blackhole), FW bundle 19.13.1.0, TT-KMD 2.10.0,
32 GB GDDR6 each → **128 GB total device DRAM**. Host: 48 cores, 503 GB RAM.

**Consequence for deliverable (3):** 188 GB of Q8_0 weights do not fit in 128 GB
of device DRAM. A residency strategy is required — see iteration 002.

---

## Observation 5 — an authoritative architecture spec exists

Qwen3.8-Flash-Next is not a Qwen3 variant that can be inferred from existing
code. Per the model card it introduces Gated DeltaNet + **Qwen Sparse Attention**
(block-level sparse w/ MQA indexer), **Gated Residual**, and a 20 M-entry
**n-gram embedding** (51 B params), on top of a 512-expert MoE.

**Remedy** — pulled the config and the upstream reference implementation
(Apache-2.0, safe to consult):

- `refs/config.json` — from `Qwen/Qwen3.8-Flash-Next`
- `refs/qwen4_exp/modular_qwen4_exp.py` (1186 lines)
- `refs/qwen4_exp/modeling_qwen4_exp.py` (2755 lines)
- `refs/qwen4_exp/configuration_qwen4_exp.py` (334 lines)

**Result** — full hyperparameters in hand. Architecture is `qwen4_exp`
(`Qwen4ExpForConditionalGeneration`):

| field | value |
|---|---|
| hidden_size | 2560 |
| num_hidden_layers | 48 |
| layer_types | 12 × (3 × linear_attention → 1 × full_attention) |
| linear attn heads | 48 V / 16 QK, head_dim 128, conv kernel 4 |
| full attn heads | 24 Q / 2 KV, head_dim 256, partial_rotary_factor 0.25 |
| indexer | 4 Q heads / 1 KV head, dim 128, budget 2048, compress ratio 4 |
| MoE | 512 experts, 10 routed + 1 shared, intermediate 640 |
| gated residual | hc_count 4, hc_lowrank 320 |
| n-gram embedding | ngram_size 3, vocab base 20 M, 8 heads/ngram, at layer 2 |
| vocab | 248320 |
| rope | theta 1e7, mrope interleaved, sections [11,11,10] |
| context | 262144 native |
| vision | depth 27, hidden 1152, patch 16, merge 2 |

---

## State at end of iteration
- Download running (detached), ~196 GB queued.
- Reference spec + config vendored under `refs/`.
- ttnn confirmed working against 4 Blackhole devices.
- `scripts/notify.py` in place for progress reporting.

## Next
Iteration 002: GGUF tensor-name/shape audit against the config, then the plain
PyTorch CPU reference engine.
