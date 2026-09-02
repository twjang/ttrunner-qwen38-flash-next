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
| single user, 1 slot, 262144 ctx, traced | **236 ms/step**; eager 469 ms |
| same, QSA selection on (context in (2048, 65536]) | 297 ms/step at 8192 |
| same, eager, chunked prefill on | prompt at ~45 ms/token |
| `step_n` verifying k tokens, traced | 255 ms at k=2, 276.6 at k=4, 323.9 at k=8 |
| same, eager | 496 ms/step |
| batch 64, eager, fused experts | 97.4 tok/s aggregate |
| server, 32 concurrent | 37.84 tok/s |
| prefill, 128 tokens | 5.77 s (11.3× the step path) — **not correct yet**, see §5 |
| unit tests | `uv run pytest -q tests` → 130 passed, ~3 s, no hardware needed |
| **next-token accuracy on real prose** | **decode 83.0 % top-1 / 97.9 % top-5, perplexity 1.98; chunked prefill 87.5 %; float32 reference 80.9 % / 2.00** |

Step time is flat in position (496 ms at pos 4, 501 ms at pos 65536) and flat
in batch up to 64 rows. The *eager* path is dispatch-bound -- measured at 0.30 ms
per device call, so 6143 calls is most of its 469 ms -- and the traced path is
not: removing 97 calls moved eager by 29.6 ms and traced by nothing. Judge a
change on the path it is meant to help (§5.7).

**Read `docs/iterations/014` before anything else.** It is the story of how a
wrong head pairing survived five iterations of review, and its lesson is a rule:

> Judge the model against the **text**, never against another implementation.

`scripts/dev/device_quality.py` is that measurement — next-token top-1 and NLL
on real prose, one forward, no oracle. Run it after any change to the model and
put the number in the iteration note.

Everything relative failed to catch the bug. Device-versus-reference agreement
cannot: with 512 experts and top-10 routing, any perturbation flips a selection,
so two *correct* implementations still disagree on half the greedy tokens, and
every precision configuration measured landed in the same 30-47 % band (014 has
the table). Greedy samples cannot either — a model with the wrong head pairing
still writes fluent English, which is exactly why it survived. And synthetic
token ids are worse than nothing: out of distribution, every path disagrees with
every other on them.

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
9. **V heads are tiled over K heads** -- v-head j reads k-head `j % n_k`, so
   Q/K expand with `repeat`. Upstream interleaves; the GGUF converter permutes
   the head order, so upstream is not authoritative here and the measurement is
   (`reference_quality.py`: 80.9 % tiled, 12.8 % grouped). The device cannot
   serve that pairing from its shard -- it all-gathers the sixteen heads and
   selects twelve (`TTModel.head_select`).
10. **A per-layer distance is a weak metric.** The tensor between layers is the
   hyper-connection stream -- four redundant copies the output mixer averages --
   so a stream can wander far while the mixed result the LM head sees does not.
   Compare the mixed hidden or the token. And an additive attention mask in
   TILE_LAYOUT pads with *zeros*, which means "attend to me": slice to a whole
   tile and let the causal condition mask the pad.
11. **QSA attends to selected tokens, not to everything.** The selection is on
    for `budget < max_seq_len <= 65536` and off outside that -- below the budget
    dense is exactly right, above 65536 the selection cannot address the cache.
    Off *and* over the budget means the device is running a different model, and
    `TTEngine` prints a notice; do not quote a long-context result without
    checking which regime it came from.
12. **Verification standard.** A change to the model is done when (a) unit
    tests pass, (b) `device_quality.py` is run and next-token accuracy does not
    regress from 83 %, (c) the number it claims to move is measured with the
    hygiene in (7), and (d) if it touches prefill, `device_quality.py --prefill`
    and `prefill_state_check.py` are run too -- an output can be right while the
    state left behind is not. Paste the output into the iteration note.
    Reproducing one prompt's text is not (b), and neither is agreement with the
    reference; both of those stood while the head pairing was wrong.

## 5. Open work, with definition of done

The model is correct as of `docs/iterations/014`, so this is no longer gated on
accuracy: 5.1 and 5.2 pay back work that iteration left on the table, 5.3 is the
remaining correctness gap (long context), and the rest is speed.

Each has an entry point, a first command, and what "done" means. Estimates are
for an agent that already has this file loaded.

### 5.1 Fold the head selection into the shard layout — **done**

`split_qkv_channels` now gives device d exactly the twelve q/k heads its twelve
v heads pair with, `(12d + i) % 16`, instead of chunking q/k four ways. The
pairing is local, so the decode step needs no expansion, no all-gather and no
selection matrix. Quality unchanged at 83.0 % next-token top-1; the traced step
is back to 236.1 ms from 239.1; attn_qkv costs ~200 MB more per device.

`convert(..., force=True)` re-writes tensors already in the manifest, which is
what any change to the plan or the shard layout needs.


### 5.2 Chunked prefill inside the trace — blocked on an on-device `prepare()`

Chunked prefill works and is on for one-slot engines, but it turns the trace off:
it runs eagerly and allocates gigabytes of temporaries per call, and after a few
of them the trace replay came back as token 0 repeated. Eager prefill is correct
-- warm and cold turns agree token for token -- so this is about getting both at
once.

Capturing a second trace for the prefill graph does not work as things stand:
`_linear_attention_chunk` builds `gated_delta_attn_seq`'s eight inputs on the
**host** (`deltanet.prepare()`, ~30 MB round trip per layer per chunk), and host
work is invisible to a capture. So the real prerequisite is the roadmap's A1.1,
moving `prepare()` onto the device; only then is there a graph to capture.

Until then the flag is the right choice for prompt-heavy work and the wrong one
for generation-heavy work: a 2000-token prompt with 200 output tokens is ~194 s
eager-and-prefilled against ~1047 s traced-and-stepped, and short prompts invert
it. Prefix reuse makes later turns of a chat cheap either way.

Done when: `use_trace=True` and `chunked_prefill=True` together reproduce the
eager result token for token over
`scripts/dev/prefix_reuse_check.py --chunked --trace`.


### 5.3 QSA indexer on device — **done**

Attention was dense causal, so the device was exact only below the 2048-token
budget. The selection now runs on device -- pooled block cache written
unconditionally, scores relu(q . block) summed over the four indexer heads,
`topk` over eligible blocks, the trailing partial block appended, and the result
scattered into an additive mask for `sdpa_decode`. `docs/iterations/015` has the
design and the three approaches that measurement killed.

Verified against the reference's `_indexer_mask` past the budget: exact at a
block boundary (2048/2048), and one block out mid-block, at rank 511 against
rank 512 with a score gap of 8.2e-4 -- below what bf16 resolves. Below the budget
it is numerically identical to dense. Traced matches eager token for token.

Cost at 8192 context: 297.5 ms traced against 236.1 dense.

Two kernel limits, both recorded in 015: `ttnn.scatter` takes uint16 indices, so
the selection reaches 65536 cache positions; and `ttnn.topk` at k=512 costs
4.5 ms at 2048 blocks but 157 ms at 65536, so past ~16384 tokens it is
expensive. It runs for `budget < max_seq_len <= 65536`, and `TTEngine` prints a
notice when a longer context puts it out of reach -- dense beyond the budget is a
different model. Lifting either limit is an upstream kernel request: a `topk`
that scales with k, or a `scatter` that takes a wider index.


### 5.4 Prefix reuse across turns — **done**

Implemented in `TTEngine._device_loop`: a slot keeps the token sequence its
state consumed, and a prompt that continues it skips `reset_slot` and starts at
`prompt_pos = len(prefix)`. A slot fed a filler token stops claiming a prefix,
so it is only ever reused when it really holds what it says. `EngineStats`
counts `cached_prompt_tokens`. Tests in `tests/test_prefix_reuse.py`.

Not done: snapshot/restore of a slot, which is what would let several slots
share one system prompt. With `max_concurrency=1` -- the configuration this is
tuned for -- slot reuse already covers that case, because an idle engine does
not step and the state simply stays.


### 5.5 Short multi-row step — **done** (`step_n`)

`TTModel.step_n(tokens, state)` advances one sequence by k tokens in a single
step and returns the mixed hidden at all k positions. The k tokens ride the
batch axis for everything per-token; only the DeltaNet convolution and the
recurrence are unrolled. Verified to reproduce k sequential steps **exactly**
(0.00 % at k = 1, 2, 4, 8) with `scripts/dev/step_n_check.py`.

Warmed, against 517.8 ms for one eager step (`step_n_bench.py`):

| k | step_n | k steps | speedup | per token |
|---|---|---|---|---|
| 1 | 531.5 ms | 517.8 | 0.97x | 531.5 |
| 2 | 672.3 | 1035.5 | 1.54x | 336.1 |
| 4 | 864.4 | 2071.0 | 2.40x | 216.1 |
| 8 | 1233.1 | 4142.1 | 3.36x | 154.1 |
| 16 | 2000.4 | 8284.1 | 4.14x | 125.0 |

Measure warmed: each k is a new set of kernel shapes, and the first call read
3103 ms at k=4 where the warmed figure is 864.

The chunked path is *not* the verifier, and `short_chunk_bench.py` is why: it
carries a ~1.3 s fixed cost per call whatever k is (`deltanet.prepare()`
round-trips ~30 MB a layer to the host), so it only beats eager decode past
k = 3 and never beats a 236 ms traced step -- and being host-bound it cannot be
captured either.

Two pieces are deliberately left to 5.6, because they are speculation wiring
rather than a multi-row step: `step_n` is not yet captured as a trace (each k
needs its own capture, and its `_input` names need entries in
`TracedDecoder._fill_inputs` the way the indexer's did), and committing only a
prefix of the k tokens needs a state snapshot -- or a replay, which the numbers
above make affordable.


### 5.6 Speculation — runs, but neither exact nor faster yet

Built end to end and opt-in via `TTEngine(speculate=k)`, where k is the tokens a
verify *feeds* and it drafts `k - 1`. `docs/iterations/016` has the whole
iteration. Two findings there matter more than the code:

**It is not identical to token-by-token decoding**, despite greedy acceptance.
The verifier batches k rows where the stepper runs one, bf16 rounding differs in
the last bits, and argmax amplifies it -- measured divergence at token 29 and 39.
Every emitted token is still the argmax of the verifier's own logits, so it is
*a* greedy decode, not the same one. Do not repeat the "exact by construction"
claim; it is standard and it is wrong here.

**It is not yet faster.** Generation-only, against a 240 ms baseline:

| `speculate` | drafted | copy-heavy | open prose |
|---|---|---|---|
| 2 | 1 | 414.6 ms/tok | 492.3 ms/tok |
| 9 | 8 | 253.9 ms/tok | 507.9 ms/tok |

Open prose is 2.1x worse than baseline while drafting *less* often than
copy-heavy, which is backwards from any drafting-cost model -- so the overhead is
not in the drafter, and where it is remains unknown.

**The instrumentation now exists** -- `TTEngine.speculation_report()` breaks a
round into snapshot / verify / restore / replay and reports the drafter's own
cost. It took one run to show that a *plain* round was costing 487.5 ms against
a traced 236, because a harness was still disabling the trace; the drafter
itself costs five microseconds. Start every measurement with it.

**Do not propose an eleventh cause for the hang; bisect instead.** Ten are
excluded in `docs/iterations/016`, including everything the standalone harnesses
do differently. Take `scripts/dev/traced_step_n_check.py`, which works, and move
it toward the engine one step at a time -- its own state object, then the
admission loop, then the asyncio queue -- until it breaks.

Verified independently and worth keeping: `step_n` reproduces k sequential steps
(0.00 % on the hidden), `TracedStepN` replays at 255 ms for k=2 against 472 for
two traced steps, `snapshot`/`restore` roll a discarded draft back identically,
and the drafter and accept rule are unit-tested. Offline pricing says
1.70-2.14x on prompts that quote their context, so the ceiling is real.

**One engine per process.** A second `TTEngine` built after closing the first
hangs on its own capture -- an engine-lifecycle bug, not a speculation one, and
it blocks anything that opens a mesh twice. `close` now releases its captured
traces, which was part of it but not all.


### 5.7 Fewer launches — and what it is actually worth

**Read this before spending a day on it.** `docs/iterations/012` framed the
single-user step as op-count bound, "6355 ops x ~36 us traced". That arithmetic
describes the *eager* path. Measured directly, by removing 97 ops and timing
both paths at the same configuration:

    eager    498.7 -> 469.1 ms   (-5.9 %)
    traced   236.1 -> 236.0 ms   (unchanged)

29.6 ms for 97 ops is **0.30 ms apiece, and that is a host dispatch**. A trace
replays with one dispatch, which was its whole point, so removing launches buys
nothing there. Fewer launches pays for chunked prefill and for eager decode --
which is what speculation currently needs -- and not for traced decoding.

`scripts/dev/op_count.py` wraps the ttnn namespace and counts one step. The
current distribution, 6143 calls, 128 a layer:

| op | share | op | share |
|---|---|---|---|
| multiply | 20.1 % | permute | 5.2 % |
| linear | 12.4 % | sum | 3.3 % |
| reshape | 12.1 % | silu | 3.7 % |
| add | 7.5 % | rms_norm | 2.6 % |
| slice | 7.2 % | sparse_matmul | 1.6 % |
| sigmoid | 5.3 % | all_reduce | 1.4 % |

Done: the hyper-connection mix averages in one op instead of summing and
scaling.

Two warnings for whatever is next. Op count is a poor proxy for cost -- the
`reinject` docstring records a three-op form running **9.5x slower** than the
nine-op one it replaced, because `repeat_interleave` is pathological on those
shapes. And every PR here needs the same A/B: `device_quality.py` unchanged,
`op_count.py` before and after, and `bench_step.py` at one configuration for
both paths.


## 6. Recipes

**Judge a change to the model** — `scripts/dev/device_quality.py 48`
(add `--prefill 32` for the chunked path, `--score-from N` for a matched
control). Next-token top-1 and NLL against the text itself: absolute, one
forward, no oracle. `reference_quality.py` is the same thing on the CPU
reference, for changes to the model rather than to the device.

`three_way_agreement.py 48 16` still exists and reports agreement with the
float32 oracle, split by the oracle's confidence, plus both paths' perplexity.
Treat its agreement figure as a weak signal -- it cannot exceed ~50 % even for
correct code -- and its NLL comparison as the useful part.

**Isolate a block instead of bisecting the model** — `attn_block_check.py`
(chunk vs step), `attn_pos0_oracle.py` (both vs a closed form),
`deltanet_block_check.py`, `layer_decode_bisect.py` (a layer's sub-blocks vs the
reference's own methods, one token from an empty state),
`branch_sequence_check.py` (the same across steps, which is the only way to see
a ring, cache or recurrent-state fault), `deltanet_step_trace.py` and
`attn_chunk_trace.py` (every intermediate of one block),
`weight_fidelity.py` (device weights vs GGUF -- note ttnn stores linear weights
transposed, and an rms of ~141 % means the comparison is misaligned, not the
weight), `routing_overlap.py` (does the router pick the same experts),
`simulated_precision.py` (price a precision-policy change on CPU, with
`--group`/`--experts`/`--dense`/`--all-dtype`), `bench_step.py`,
`prefix_reuse_check.py` (the engine's chat-turn behaviour end to end),
`prefill_state_check.py` (what a prompt leaves behind),
`reference_prefill_vs_decode.py` and `reference_greedy.py` (the oracle against
itself, CPU only).

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
* Every claim about the model is measured against the **text**. Two paths
  agreeing says nothing when both are built from the same wrong assumption, and
  neither does agreeing with the reference when the reference was given the same
  assumption in the same change — which is exactly what happened with the head
  pairing (`docs/iterations/014`).
* When a control comes back implausible — a "verified" path 52 % from float32 at
  layer 0, or an oracle with perplexity 4445 — chase it. Every real defect this
  project has found announced itself as a number someone was tempted to explain
  away.
* Upstream's HF source is evidence, not authority: the GGUF converter permutes
  and folds things (the head order, a `+1` in the norm weights, `A = -exp(A_log)`).
  Where they disagree, measure.
