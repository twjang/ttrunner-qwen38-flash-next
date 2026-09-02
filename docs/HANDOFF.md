# Handoff — Qwen3.8-Flash-Next on 4× Tenstorrent Blackhole

Written 2026-09-02 for whoever picks this up next, human or agent. It is
optimised for an agent with less context and less patience than the one who
wrote it: facts first, recipes second, judgement calls last. Read this, then
`docs/roadmap_low_batch.md` for *what* to do next, then the iteration log only
for the specific observation you are about to build on.

---

## 0. The rules that do not bend

* Feature branches and PRs only. **Never** push to `main`/`master`, never
  force-push, never bypass tests or CI. Current branch: `feat/ttnn-inference-stack`
  (no remote configured; the user decides when/where it goes).
* No secrets in code, logs or docs. Telegram credentials live in
  `~/.config/telegram-send.conf`; `scripts/notify.py` reads them and never
  prints them. If you ever see a token in a file or in git history, stop, say
  so, and recommend rotation.
* No real customer data or PII anywhere. Test prompts are synthetic
  (`scripts/dev/_device_model.py::synthetic_prompt`) or public facts.
* New dependency → state its license; MIT/Apache-2.0/BSD only. The one
  external code we hold is a vendored *copy* of upstream HF modeling
  (`refs/qwen4_exp`, for provenance) and a llama.cpp checkout in the session
  scratchpad — read-only references, not dependencies.
* Do not run anything destructive on hardware you did not bring up yourself
  (no `tt-smi -r` on a device another process holds; check `pgrep -af python`
  first).

## 1. What this is, in one screen

Four engines for `unsloth/Qwen3.8-Flash-Next-GGUF` (arch `qwen4exp`, UD-IQ4_XS):

| piece | where | state |
|---|---|---|
| GGUF reader + bit-exact dequant | `src/twtest/gguf/` | done, tested (`tests/test_quants.py`) |
| CPU PyTorch reference | `src/twtest/reference/` | done; **the oracle** for everything device-side |
| Device model (ttnn, 4 × p150a) | `src/twtest/tt/model.py` (+ `moe.py`, `linear_attn.py`, `deltanet.py`, `ops.py`) | decode verified token-for-token vs reference; prefill *not* |
| Engine + trace | `src/twtest/tt/engine.py`, `traced.py` | continuous batching, device argmax, single-user trace |
| OpenAI-style server | `src/twtest/server/` | works on both backends |

Model facts you will need constantly: 48 layers = 36 Gated DeltaNet + 12
sparse attention (every 4th: layers 3, 7, …, 47); 512 experts, top-10,
`expert_intermediate` 640; hidden 2560; hyper-connections `hc_count` 4
(so the residual stream is 10240 wide until the final collapse); vocab 248320;
context 262144; attention `head_dim` 256, 2 KV heads, 24 q heads; DeltaNet
48 v-heads / 16 k-heads globally = 12 / 4 per device (head-sharded), head dim
128; PLE (n-gram embedding) injects at layer 1 only; conv kernel 4.

Hardware facts: 4 devices, ~32 GB DRAM each, 24.94 GB weights per device,
~7-8 GB free. The K/V cache costs **6.4 GB per slot** at full context, so slots
and context trade one for one (`TTEngine` refuses combinations over budget).

## 2. Current numbers (the ones you must not regress)

| configuration | result |
|---|---|
| single user, 1 slot, 262144 ctx, traced, fused experts | **229 ms/step**, `' Paris.\n\nThe French city in Europe'` for `The capital of France is` |
| same, eager | 496 ms/step |
| batch 64, eager, fused experts | 97.4 tok/s aggregate |
| server, 32 concurrent | 37.84 tok/s |
| prefill, 128 tokens | 5.77 s (11.3× the step path) — **not correct yet**, see §5 |
| unit tests | `uv run pytest -q tests` → 108 passed, ~3 s, no hardware needed |

Step time is flat in position (496 ms at pos 4, 501 ms at pos 65536) and flat
in batch up to 64 rows. It is **op-count bound**: 6355 device ops × ~36 µs
traced. Any change is judged by ops removed or useful rows added per step.

## 3. Environment and how long things take

```bash
cd ~/twtest
uv run pytest -q tests                           # 3 s, CPU only, run before every commit
uv run python scripts/dev/prefill_check.py 4 16  # ~4 min: 1.5 min weight load + steps at ~0.5 s each
uv run python scripts/dev/prefill_bisect.py 4    # ~3 min
```

* Python is `uv run python` (there is no bare `python`); `.venv/bin/python`
  with `PYTHONPATH=src` also works and is what the README shows.
* Weights: GGUF shards in `~/models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS`, device
  cache in `~/models/qwen38-tt-cache` (130 GB, per-device `.tensorbin` files,
  includes the prebuilt fused `blk.N.ffn_gateup_exps.weight`). MTP head:
  `~/models/Qwen3.8-Flash-Next-GGUF/MTP/*.gguf`. Override with
  `TWTEST_GGUF_DIR` / `TWTEST_TT_CACHE`.
* Opening the mesh + loading all weights ≈ 90 s. One eager step ≈ 0.5 s. A
  traced step ≈ 0.23 s after ~40 s of warmup/compile. The CPU reference is
  minutes per token — use it for a handful of tokens only.
* One process owns the devices at a time. Before starting a device job:
  `pgrep -af "uv run|python" | grep -v grep`. Kill by **PID**, never
  `pkill -f <pattern>` (it matched its own shell twice in this project).
* Long runs: redirect to a log and poll with `until grep -q "^exit" log; do
  sleep 3; done`. Print `RESULT ...` lines and grep for them; never rely on
  the last N lines of a log (tt-metal appends noise after your output).
* Progress to the user goes through `scripts/notify.py "message"` (Korean is
  the user's preference for those messages).

## 4. Invariants that bit us — check these before you believe a result

1. **Trace = fixed addresses.** Everything the steady-state loop touches must
   be allocated before `begin_trace_capture`, including the LM head
   (`logits`/`greedy_tokens` are warmed in `TracedDecoder.__init__`). Host-side
   Python (list rebinding, ring rotation by index) is *invisible* to capture —
   hence `trace_safe_rings`, which shifts rings by device copies at fixed
   indices. Any new per-step buffer must go through `model._input(name, ...)`.
2. **Two ring conventions.** `trace_safe_rings=True` reads `state[age-1]`
   (newest first); the rotating path at step 0 reads `state[0]` as oldest.
   `_causal_conv_chunk`/`_ple_chunk` hand back whichever the following decode
   mode expects. Getting it wrong is silent (taps permuted).
3. **Head-sharding.** DeltaNet weights are head-sharded across the 4 devices.
   `from_dev(t)` returns *device 0's* copy — for a sharded tensor that is a
   quarter of the heads. Gather with `ConcatMeshToTensor` and shard back with
   `ShardTensorToMesh`. `ssm_out` and the MoE `down` are row-sharded on their
   contraction dim → each device holds a **partial sum** until `all_reduce`.
4. **`update_cache` takes one int for the whole batch**; continuous batching
   needs `traceable_kv=True` (`paged_update_cache`, per-sequence positions).
   The engine constructs the model that way; the other path raises if
   positions differ.
5. **`ttnn.concat` of bfloat4_b requantises** (0.055 max error). Fuse weights
   at conversion time (`scripts/fuse_expert_gate_up.py`), never on device.
6. **Batch ceiling 64.** 65–96 rows pad to 96 tiles and overflow L1.
7. **Timing hygiene.** 5 warmup + 25 samples minimum (`tt/bench.py`
   defaults). A cold single run once misreported a 1.3× win as 0.94×. Discard
   the first request when timing the server (it includes JIT). To time a
   position, set `state.positions` directly — generating to 65536 is 9 hours.
8. **Instrumented profiles are inflated ~40 %** by per-section syncs; read the
   shares, not the totals.
9. **The QSA indexer is not on the device path.** Attention is dense causal in
   both `step` and `prefill`. Exact below 2048 tokens, *different from the
   model* beyond. See roadmap A5. Do not claim long-context correctness until
   this exists.
10. **Verification standard.** A change to the model is done when (a) unit
    tests pass, (b) the single-user text above is reproduced byte-for-byte,
    (c) the number it claims to move is measured with the hygiene in (7), and
    (d) if it touches prefill, `prefill_check.py` and `prefill_bisect.py` are
    both run and their output pasted into the iteration note.

## 5. Open work, with definition of done

Ordered as in the roadmap. Each has an entry point, a first command, and what
"done" means. Estimates are for an agent that already has this file loaded.

### 5.1 Prefill correctness — decide noise vs. bug (½ day)

State: two structural bugs fixed on 2026-09-02 (missing `all_reduce` in
`_linear_attention_chunk`; conv-tap cache keyed without the layer). Layers 0–2
match decode to ~1 %; the first jump is layer 3 (first QSA, ~5 %), growing to
~30 % at layer 47; greedy token still differs at lengths 4/16/64.

Do: write `scripts/dev/decode_vs_reference.py` — run the CPU reference
(`twtest.reference.model`) on `synthetic_prompt(4)` capturing the hidden after
each layer (the reference has per-layer modules; add a hook list), and the
device `step` path with `model.probe`, and print the same per-layer table as
`prefill_bisect.py`. Then:

* if decode-vs-reference ≈ prefill-vs-decode at every layer → prefill is at
  the noise floor. Switch the acceptance criterion to *agreement rate* over
  ≥ 200 generated tokens on a real prompt (≥ 95 % is the bar the decode path
  itself would meet), enable `chunked_prefill` in `TTEngine`, remove
  `test_engine_refuses_chunked_prefill_while_it_is_wrong`.
* if decode-vs-reference is ≫ tighter → bug in `_attention_chunk`. Add a
  sub-block probe (after attention branch, before `reinject`) at layer 3 and
  diff against `_attention_step`. Suspects in order: GQA expansion
  (`repeat_interleave` vs `sdpa_decode`'s internal mapping), the explicit
  `-inf` mask in bf16, `fill_cache` vs `paged_update_cache` layout.

Done when: `prefill_check.py` prints MATCH at 4/16/64/128 **or** the agreement
measurement is in an iteration note and the engine flag is on.

### 5.2 QSA indexer on device (2–3 days; correctness for > 2048 tokens)

Entry: `_attention_step` in `tt/model.py`; the reference implementation is the
QSA block in `src/twtest/reference/layers.py`; the scoring op
`ttnn.experimental.indexer_score_dsa` was validated at 1.0 % in iteration 007;
the weights `blk.N.indexer.{q_proj,k_proj,q_norm,k_norm}` are already on
device (`plan.py`). `LayerState.indexer_keys` is the reserved slot.

Shape it so the step stays traceable: always select exactly 512 blocks (mask
the surplus below 2048), so shapes are static. Validate against the CPU
reference on a prompt > 2048 tokens (reference is slow: generate the prompt
hidden once and cache it to disk).

Done when: device and reference agree on the next token after a 3000-token
synthetic prompt, and the < 2048 result is unchanged, and step time is within
5 % of 229 ms.

### 5.3 Prefix reuse across turns (1 day, engine only)

Entry: `TTEngine._device_loop`, the admission block that calls
`reset_slot`. Keep the last finished sequence's `prompt_token_ids + emitted`
per slot; if the new prompt has it as an exact prefix, skip `reset_slot`, set
`prompt_pos` to the prefix length. Add snapshot/restore of a slot's state
(recurrent, conv rings, PLE ring, positions, histories) for a system prompt.

Done when: a two-turn chat's second turn has time-to-first-token proportional
to the *new* message only, and tokens equal the cold-prefill result.

### 5.4 Short multi-row step (the MTP verify step) (2 days)

Roadmap A1.2/B1.1. A `step_n(tokens[k])` that runs the DeltaNet recurrence
unrolled k times *inside* one batched step (MoE/attention/HC over k rows),
keeping the intermediate recurrent states so a prefix of j ≤ k tokens can be
committed by `ttnn.copy`. First measurement: time it for k = 2, 4, 8 against
k × 229 ms.

Done when: `step_n` on k tokens reproduces `step` called k times (hidden
maxdiff at the bf16 floor you measured in 5.1), and its time for k = 4 is
< 2 × one step.

### 5.5 Speculation: n-gram first, MTP second (2 + 4 days)

* n-gram drafting (roadmap B2) needs only 5.4 plus an accept loop in the
  engine. Greedy accept rule: accept while `argmax(verify_i) == draft_i`.
* MTP (roadmap B1): the head is `blk.48` in `MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`.
  Order of work: (i) exercise Q4_K/Q5_0 dequant against the codebook (never
  exercised — `docs/iterations/003`), (ii) load `blk.48` through `convert.py`
  into the device cache (residency: replicate everything but the experts,
  shard the experts like the main layers), (iii) implement the head eagerly
  with the wiring in roadmap B1 and measure top-1 agreement with the *actual*
  next token over ≥ 200 tokens of a real prompt, (iv) only if ≥ 0.6, wire it
  behind 5.4 with a second trace for the k+1-row verify graph.

Done when: tokens/second at batch 1 improves with greedy output identical to
non-speculative decode (speculation with greedy acceptance is exact).

### 5.6 Fewer launches (ongoing; each item is a self-contained PR)

Roadmap B3. Start with the hyper-connection gate (`ops.gated_residual_mix`
and `reinject`; 97 calls × ~11 ops). Every PR: op count before/after (count
with the profiler recipe in `docs/iterations/012` Observation 5), traced step
time with hygiene, single-user text unchanged.

## 6. Recipes

**Bisect any prefill/decode disagreement** — `scripts/dev/prefill_bisect.py`.
Read top-down; the first layer whose relative error jumps is the culprit.
To bisect inside a layer, set `model.probe` to a closure and also call it from
inside `_layer`/the prefill loop after the sub-block you suspect (temporary
edit, revert after).

**Count device ops in a step** — wrap `ttnn` ops with a counting decorator (see
012 Obs. 5) or use tt-metal's op profiler with `TT_METAL_DEVICE_PROFILER=1`;
run one eager step; group by op name.

**Verify a device weight against GGUF** — `TTWeights.get(name)` →
`ttnn.to_torch(..., mesh_composer=ConcatMeshToTensor(dim=<shard dim>))` vs
`WeightStore(gguf).get(name)`; bfloat4_b tolerates ~0.06 max error, bf16 ~1e-2.

**Trace something new** — allocate every per-step input via `model._input`,
run one eager step *and* the LM head, `synchronize_device`, then capture.
If the first traced output is garbage but the second is right → an allocation
happened after capture. If *every* output is garbage → a host-side Python
mutation is inside the loop.

**Add a test without hardware** — the `tests/` style is source-contract tests
(`inspect.getsource` asserts) for device code plus real numeric tests for
anything that runs on CPU (`test_quants`, `test_delta_rule`). Device-touching
tests `pytest.importorskip("ttnn")`.

## 7. Where things are written down

* `docs/iterations/NNN_*.md` — the observation → remedy → result log. 012 is
  the single-user work (trace root cause, op-count floor); 011 is continuous
  batching, expert fusion, the tile cliff; 009/010 the device bring-up;
  008 the precision policy; 007 the op validation table.
* `docs/roadmap_low_batch.md` — what to do next and why, with the numbers.
* `README.md` — measured performance tables and quick start.
* `refs/qwen4_exp/` — vendored upstream HF modeling (ignores `mtp.*`).

## 8. Judgement calls the previous agent made, so you do not re-litigate them

* One code path for the prompt (the step path) until prefill is *verified*,
  because four bugs in this project were invisible at batch 1 or in a
  timing-only harness. Speed is easy to see; wrongness is not.
* `max_concurrency=1` and full context by default: the user said single
  user, and every slot costs 6.4 GB of context.
* Expert fusion happens in the conversion script, not on device, because
  `concat` requantises bfloat4_b.
* The layout/kernel work (B3) was deliberately not started at the end of a
  long session; it spans the whole model and needs the A/B discipline in §4.7
  for every block.
* When something fails, ask "is it per-step or one-time?" before "what is
  different about my setup?" — the former halved the trace search space after
  thirteen null results from the latter.
