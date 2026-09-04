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
  first). Never `pkill -f <pattern>` -- the pattern matches the killing shell
  and has taken this session down twice; use
  `ps -eo pid,args --no-headers | awk '/name/ && !/awk/ {print $1}'`. Never
  `kill -9` a job mid-capture; SIGTERM, then reset.
* **"It stopped failing" is not "it works."** Verify the thing still produces
  the right answer, not just that the symptom went away. A second command queue
  made the speculation hang disappear and was committed as the fix; the replay
  was in fact not executing at all, and returned `[201058, 0]` where the eager
  path returns `[75, 220]` (`docs/iterations/018`). A no-op passes every test
  except the one that matters.
* **When a bisection keeps coming back green, make the failure describe
  itself.** Sixteen candidate causes of that same hang were excluded by
  rebuilding the working setup rung by rung, and all sixteen were correct and
  useless, because every rung tested the setup and the setup was never broken.
  Three `print` statements and `faulthandler.dump_traceback_later` located it in
  one run each. Instrument the failing side before enumerating differences.

## 1. What this is, in one screen

Four engines for `unsloth/Qwen3.8-Flash-Next-GGUF` (arch `qwen4exp`, UD-IQ4_XS):

| piece | where | state |
|---|---|---|
| GGUF reader + bit-exact dequant | `src/ttrunner_qwen38_flash_next/gguf/` | done, tested (`tests/test_quants.py`) |
| CPU PyTorch reference | `src/ttrunner_qwen38_flash_next/reference/` | done; **the oracle** for everything device-side |
| Device model (ttnn, 4 × p150a) | `src/ttrunner_qwen38_flash_next/tt/model.py` (+ `moe.py`, `linear_attn.py`, `deltanet.py`, `ops.py`) | decode verified token-for-token vs reference; prefill *not* |
| Engine + trace | `src/ttrunner_qwen38_flash_next/tt/engine.py`, `traced.py` | continuous batching, device argmax, single-user trace |
| OpenAI-style server | `src/ttrunner_qwen38_flash_next/server/` | works on both backends |

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
| single user, 1 slot, 262144 ctx, traced | **173.7 ms/step** (236.2 before `022` stopped the MoE paying for tile padding); eager 489 ms, unchanged, because the eager path is dispatch-bound and the op count barely moved |
| same, QSA selection on (context in (2048, 65536]) | **297 ms/step** at 8192 (297.4 re-measured) |
| same, eager, chunked prefill on | prompt at **7.2 ms/token** (was ~45) |
| `step_n` verifying k tokens, **eager** | 490 ms at k=1, 797 at k=2, 991 at k=4, 1404 at k=8, 6651 at k=32 (`step_n_check.py`) |
| same, traced | *historical*: 255 ms at k=2, 276.6 at k=4, 323.9 at k=8. Its harness no longer completes — see 5.5 |
| batch 64, eager, fused experts | **107.6 tok/s** aggregate (`bench_batch.py`); 61.9 at batch 32, 15.3 at 8 |
| server, 32 concurrent | **70.3 tok/s** sustained generation (median of 3 rounds after a discarded warm one: 70.19 / 70.34 / 70.40); end to end depends on output length — 36.1 tok/s at 32 tokens out, 46.9 at 128 (`bench_server.py`) |
| prefill, 128 tokens, `moe_chunk=32` | **925 ms** (138.4 tok/s); 2033 ms at the old `moe_chunk=16` default, 1070 ms before routing and expert compute were split |
| unit tests | `uv run pytest -q` → 202 passed, ~3 s, no hardware needed |
| **next-token accuracy on real prose** | **decode 83.0 % top-1 / 97.9 % top-5, perplexity 1.98; float32 reference 80.9 %**. Holds across context now: `023` fixed a decode attention config that took accuracy to 0 % past ~224 tokens, and 200 scored positions went 62.3 % → **73.4 %** |
| chunked prefill, judged against a same-positions decode control | measured back to back in one batch: 32 prefilled, 128 scored **71.9 %** / NLL 1.43 against 71.7 % / 1.48; 128 prefilled, 107 scored **21.5 %** / 6.345 against 22.6 % / 5.996. So a 32-row prefill is slightly ahead of stepping and a 128-row one slightly behind. Earlier readings of the 128 row spanned 21.5–27.1 % and are unexplained — invariant 8 |
| a sequence's output vs the same sequence alone | identical to **batch 32**; differs above it, and that is arithmetic, not a bug (invariant 13) |

Step time is flat in position (496 ms at pos 4, 501 ms at pos 65536) and flat
in batch up to 64 rows. The *eager* path is dispatch-bound -- measured at 0.30 ms
per device call, so 6143 calls is most of its 486 ms -- and the traced path is
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
cd ~/ttrunner_qwen38_flash_next
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
  `TTRUNNER_GGUF_DIR` / `TTRUNNER_TT_CACHE`.
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
8. **Pair a prefill A/B in one batch of runs; do not compare across turns.**
   `device_quality.py 250 --prefill 128` was recorded at 21.5 %, 25.2 % and
   27.1 % top-1 (NLL 6.345, 5.823, 5.614) at different points in one session,
   from what the git history says is the same model code and the same harness.
   That looked like nondeterminism and was written up as such. Direct testing
   says otherwise: `scripts/dev/prefill_determinism_probe.py` prefills 128 rows
   in three separate processes, on both a synthetic prompt and the exact tokens
   `device_quality` scores, and the final logits *and* the next eight decode
   steps come back **bit-identical** every time -- warm or cold, whatever the
   process did first. Six consecutive `device_quality` runs now agree to the
   digit.

   So the three historical readings are **unexplained**, not evidence of a
   property. Something in the environment differed and has not been identified.
   The practical rules stand either way, because they cost nothing:

   * A/B a prefill change **back to back in one batch of invocations**. Every
     cross-turn prefill comparison in this file's history is suspect, and two
     judgements in 5.7 were re-framed onto axes that hold still -- dispatch
     count, wall clock, exact-token equality.
   * Prefer something deterministic where one exists: prefill's logits within
     one process (`prefill_determinism_probe.py`, `moe_chunk_noise.py`), a
     per-row invariant needing no reference (`moe_rows_check.py`), or token
     equality (`batch_equivalence_check.py`).
   * Decode is rock solid: `device_quality.py 48` has returned 83.0 % / NLL
     0.682 in every run of this session, across board resets. Prefer it as the
     regression gate.

9. **Instrumented profiles are inflated ~40 %** by per-section syncs; read the
   shares, not the totals.
10. **V heads are tiled over K heads** -- v-head j reads k-head `j % n_k`, so
   Q/K expand with `repeat`. Upstream interleaves; the GGUF converter permutes
   the head order, so upstream is not authoritative here and the measurement is
   (`reference_quality.py`: 80.9 % tiled, 12.8 % grouped). The device cannot
   serve that pairing from its shard -- it all-gathers the sixteen heads and
   selects twelve (`TTModel.head_select`).
11. **A per-layer distance is a weak metric.** The tensor between layers is the
   hyper-connection stream -- four redundant copies the output mixer averages --
   so a stream can wander far while the mixed result the LM head sees does not.
   Compare the mixed hidden or the token. And an additive attention mask in
   TILE_LAYOUT pads with *zeros*, which means "attend to me": slice to a whole
   tile and let the causal condition mask the pad.
12. **QSA attends to selected tokens, not to everything.** The selection is on
    for `budget < max_seq_len <= 65536` and off outside that -- below the budget
    dense is exactly right, above 65536 the selection cannot address the cache.
    Off *and* over the budget means the device is running a different model, and
    `TTEngine` prints a notice; do not quote a long-context result without
    checking which regime it came from.
13. **One row tile is the reproducibility boundary.** A `ttnn.linear` returns a
    given row *identically* for any row count that fits in one 32-row tile, and
    differently past it — one bf16 ulp, the same at 33, 48, 64 and 128
    (`scripts/dev/row_tile_boundary_check.py`). So a row computed among ≤ 32
    rows and the same row computed among more cannot be compared for equality,
    whatever the surrounding code does. This sits behind `moe_chunk`'s cap
    (5.8), batch 64 decoding differently from batch 1, and `step_n` stopping at
    k=32 (5.5). Do not try to remove it in the model: splitting `expert_ffn`
    into 32-row groups so `per_core_M` stays 1 changes nothing, because the ops
    before the experts have already diverged. Distinct from `sparse_matmul`
    dropping rows past the first tile, which was a real defect and is fixed.
14. **A device matmul is not a float32 matmul.** `ttnn.matmul` on float32
    inputs, at HiFi4 with `fp32_dest_acc_en`, is **0.16 %** off torch for a
    128x128 -- the Tensix multiplier decomposes fp32 into bf16 pieces. Every
    elementwise op measured is at ~1e-5. So an algorithm that is merely
    *correct* in float32 can be unusable on device: porting one, check whether
    it cancels large intermediates against each other. `block_inverse` summed
    `sum_k (-N)^k`, whose terms reach ~1e8 for an answer bounded by 1; float32
    hid that at 0.27 absolute error and the device turned it into 620875 %.
    `block_diag_inverse` replaced it with a blocked form that never grows.
    Corollary for test design: the old test scaled its off-diagonals to 0.4 and
    passed, and *magnitude alone does not reproduce the failure* -- random signs
    cancel inside the powers. Build the fixture the way the model builds the
    tensor (`test_block_inverse_survives_correlated_keys`).
15. **A per-layer distance still cannot tell noise from a bug** (013's lesson,
    re-earned). Chunked prefill's hidden state is ~32 % from the reference by
    position 127 of a chunk and the *token* is fine: 128 tokens prefilled score
    53.1 % against 51.6 % for stepping the same ones. Judge prefill by a
    same-positions decode control, which is what `--score-from` is for.
16. **A cited harness can stop working, and the citation will not notice.** Two
    did: `indexer_select_check.py` called `_indexer_select` with its pre-`q_cos`
    signature and raised `TypeError` after minutes of setup, and
    `traced_step_n_check.py` hangs on the device. Both were quoted in items
    marked **done**, and a citation that no longer runs still reads as evidence.
    `pytest` now runs `scripts/dev/harness_api_check.py` over every harness
    (`tests/test_harness_api.py`) and fails on a name or arity that has drifted;
    the device half it cannot see, so before quoting a harness's number, run it.
17. **Half this suite tests source text, not behaviour.** 50 of 113 tests read
    `inspect.getsource` and assert on strings. That is a deliberate trade -- it
    costs no hardware and it catches a refactor that quietly undoes a fix -- but
    it locks in *text*, and text can be wrong. One of them was: the `step_n`
    guard test asserted `"0 < k <= 64" in src` and so stood guard over a range
    half of which was broken (5.5), passing all the while and requiring the test
    to be edited before the bug could be fixed. Two rules follow. A source test
    must anchor on something *positive*, or a gutted function satisfies it for
    the wrong reason -- three such tests existed and were fixed. And when a
    source test pins a number or a limit, check the number, because the test
    cannot.
18. **Verification standard.** A change to the model is done when (a) unit
    tests pass, (b) `device_quality.py` is run and next-token accuracy does not
    regress from 83 %, (c) the number it claims to move is measured with the
    hygiene in (7) and (8) -- repeat it, one run has no noise floor -- and
    (d) if it touches prefill, `device_quality.py --prefill`
    and `prefill_state_check.py` are run too -- an output can be right while the
    state left behind is not. Paste the output into the iteration note.
    Reproducing one prompt's text is not (b), and neither is agreement with the
    reference; both of those stood while the head pairing was wrong.

19. **The traced step is not measured by counting its ops; ablate it.** 5.7
    established the traced path is not dispatch-bound, so `op_count.py` -- the
    right instrument for prefill -- says nothing about decode. Replace a
    component with an identity of the same shape, which adds no ops, and the
    delta is its device time (`decode_ablation_check.py`, one ablation per
    process because alternating two traces hangs). Doing that found 65 % of the
    step in the routed MoE, and a top_k sweep then found that almost none of
    *that* was the experts: 1 expert costs 259.6 ms and 20 cost 272.0, so the
    selected experts do ~7.5 ms of work inside a 173 ms block. The rest was
    TILE_LAYOUT padding the row axis to 32, making every [1, 512, 1, 2560] an
    84 MB tensor holding 2.6 MB. Corollary for anything at M = 1: suspect the
    padding before the arithmetic. See `022`.

20. **When a change is not bit-identical, compare both candidates to an exact
    reference, not to each other.** Two device forms differing says only that
    they differ. Building the same computation in float64 from the *device's
    own* operands -- bf16 converts exactly, so the reference has no error of its
    own -- says which to keep. That is how `_combine`'s matmul form was shown to
    be 27 % better on the worst element and 35 % on the mean rather than merely
    different, and it is the check to run whenever a rewrite moves where the
    rounding falls. End-to-end next-token accuracy cannot do this job: 127
    scored positions do not resolve one bf16 ulp.

21. **A knob that changes the answer needs a correctness test at every size it
    will see, not at the size it was tuned on.** The decode attention's program
    config was added for speed, measured for speed, and never checked past one
    k-chunk; it took accuracy to zero past 224 tokens of context and survived
    because no measurement went that far (`023`). Two habits follow. Look at an
    accuracy metric *across* its range once before trusting the aggregate -- the
    run that read 62 % overall read 0 % on half its positions. And when a
    harness says everything fails, including cases known to work, suspect the
    harness: the first version of `sdpa_decode_accuracy_check.py` called L=32
    wrong, where the model scores 84 %, because its reference had the wrong
    head-to-KV mapping. Comparing an op against *itself* under a parameter that
    must not matter needs no reference at all, and is what found this.

22. **There is no matrix-vector op, and the tile is not a way to fake one.**
    ttnn has no `matvec`/`gemv`. It has a configurable tile -- `ttnn.Tile([1,32])`
    constructs and `linear`/`matmul` take an `output_tile` -- which looks like the
    way to stop padding a 1-row decode activation up to 32. It is not, twice over
    (`tiny_tile_matvec_check.py`). Below 16 rows the dense op **silently returns
    the wrong answer**: 16x32 matches the exact product as well as 32x32 (2.97e-03)
    while 8x32 and 1x32 are off by ~0.9, with no error raised; matching the
    weight's tile to the activation's is rejected outright; and `sparse_matmul`
    refuses a tiny `output_tile` entirely. And it would buy nothing anyway -- a
    dense matvec at M=1 is bound by reading the weight ([2560, 3072] against one
    row), so every tile from 1x32 to 32x32 times the same to within noise. The
    padding only ever mattered where it *multiplied*: the MoE's [1, 512, 1, 2560]
    was 84 MB carrying 2.6 (`022`), and that is precisely where the op refuses
    tiny tiles -- which is why the fix there was to restructure the consumer
    instead.

23. **A device call on the eager prefill path is worth ~57 us, and its size does
    not matter.** Injecting a known number of ops and reading the slope gives
    90 +/- 15 us for the marginal op, and the one clean removal on record
    (`moe_chunk` 32 -> 128, 1008 calls, 918.6 -> 861.7 ms) gives 57 us for a real
    one; price reductions with the smaller figure. An injected op on a 128x2560
    activation costs 1.11x one on a 32x32 tile, so what is being paid for is the
    launch and not the arithmetic. Two corollaries. `op_count.py`'s totals are
    python calls into `ttnn`, not dispatches -- an injected `reshape` measures
    2.1 us, and `reshape` is 1094 of the 11681 a chunk reports -- so a count is
    an upper bound on the dispatches it stands for. And the 0.30 ms per dispatch
    that 5.7 used for four iterations is retired: it was one A/B on the decode
    path and it over-predicts both paths by 4-6x (`021`).

## 4b. Fixed: decode quality collapsed past ~128 positions

**Found and fixed 2026-09-03/04 (`023`).** Decode's next-token accuracy fell from
~80 % over the first 128 positions to **0 % past 224**. The cause was the decode
attention's pinned SDPA program config, and the reason it survived so long is
that no measurement had ever scored past 128 positions -- `device_quality.py`
defaults to 48 tokens and its passage is ~200 long.

The op's online softmax over the K/V cache must give the same answer however it
is chunked. It did not: for one fixed set of inputs, changing only
`k_chunk_size` moved the result by up to **1e7 relative** once the cache spanned
more than one chunk, growing with the chunk count
(`sdpa_decode_accuracy_check.py`). The collapse tracked the setting -- k=64 broke
from ~64 tokens, k=128 from ~128, k=256 decayed from 128 and fell sharply at 256.

**The fix is to pass no program config and let ttnn choose.** Against the float32
CPU reference on the same passage, which itself declines on this text, the device
now tracks it the whole way:

| positions | 0–127 | 128–159 | 160–191 | 192–223 | 224–255 | 256–287 |
|---|---|---|---|---|---|---|
| reference | 75–81 % | 62.5 | 62.5 | 50.0 | 46.9 | 50.0 |
| device now | 78–84 % | 59.4 | 59.4 | 50.0 | 40.6 | 46.9 |
| device before | 78–84 % | 53.1 | **12.5** | **6.2** | **0.0** | **0.0** |

`device_quality.py 200` goes 62.3 % → **73.4 %**, NLL 2.209 → 1.267. It costs
nothing at the canonical configuration (262144, indexer off: 173.7 → **173.4 ms**)
and 3 % on the indexer path (4096: 204.0 → 210.3). Prefill is untouched;
`_attention_chunk` has its own chunked configs.

`sdpa_k_chunk` remains a constructor argument because the QSA indexer sizes its
window from it. It no longer reaches the attention op. `pin_sdpa_config=True`
restores the old behaviour for an A/B, and it is wrong.

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


### 5.2 Chunked prefill alongside the trace — **done**, by releasing the trace around each prefill
> **Two corrections, in order.** This was first marked done in 4a43a7a on a
> `prefix_reuse_check.py --chunked --trace` run that never had a trace: the
> engine forced `_use_trace` off whenever `chunked_prefill` was on, printed that
> it was doing so, and the improvement credited to the trace came from prefill
> getting faster on its own. Removing the exclusion then showed the two really do
> corrupt each other -- with the trace genuinely live, turn 2 replayed cold
> returned `[2250, 10478, ...]` against `[10782, 303, ...]` everywhere else, and
> `turn2 warm == cold` went YES -> NO. Warming one chunk before the capture, the
> fix attempted at the time, does not cover it.
>
> **What does work is not having a trace live while a prefill allocates.**
> `_device_loop._reprefill` releases the trace, prefills, and captures again.
> Capturing consumes two warm-up tokens and so dirties the state the prefill just
> built, and `TracedDecoder.reset()` would fix that by zeroing everything --
> which throws the prompt away -- so `snapshot`/`restore` puts it back in place,
> keeping the addresses the new trace just recorded valid.
>
> Costs, measured (`recapture_after_prefill_check.py`, 128-token prompt): release
> 9.1 ms, snapshot 39.3, capture 2537.9, restore 3.4 -- about 2.6 s per request
> that ingests, against 411 ms saved on every token generated afterwards. It pays
> from roughly the seventh token. The tokens it produces are identical to the
> eager path's, and `prefix_reuse_check.py --chunked --trace` now returns
> `turn2 warm == cold: YES` with the right tokens.
>
> End to end on a fresh 1051-token prompt, through the server:
>
> | | ingestion | generation |
> |---|---|---|
> | now | **13.4 ms/token** | **183-194 ms/token** |
> | chunked prefill, trace off | 7.6 | 588 |
> | traced, no chunked prefill | 177 | 177 |
>
> A 1051-token prompt with 200 generated is **50.7 s**, against 125.6 s and
> 221.4 s for the two configurations that were previously the only choices. A
> turn whose prompt is already cached skips ingestion entirely and pays no
> capture.

Chunked prefill works and is on for one-slot engines, but it turns the trace off:
it runs eagerly, and a captured prefill graph replayed as token 0 repeated.
Eager prefill is correct -- warm and cold turns agree token for token -- so this
is about getting both at once.

**The blocker this item was filed with is gone.** `deltanet.prepare_device`
builds `gated_delta_attn_seq`'s eight inputs on device, in the op's own
`[H, NC, C, D]` layout, so each device prepares the heads it already holds and
nothing is gathered. The chunk path has no host round trip at all, and
`test_chunked_deltanet_never_leaves_the_device` keeps it that way. Doing it
turned up an unstable inverse in both engines; `docs/iterations/017`.

**What remained: four host dependencies, all in `_attention_chunk`.**

1. `self.rope(range(start, start + seq))` builds cos/sin on the host.
2. `ttnn.fill_cache(..., update_idx=start)` takes a Python int, so a capture
   bakes in one slot.
3. The causal mask is built with `torch.arange`.
4. `kv_len` grows with the chunk index, so shapes are not static.

**All four have a verified answer.** A paged K/V cache. Every op it needs exists
in this build and was measured at this model's real shapes -- b=1, 24 q heads,
2 kv heads, head_dim 256, block 32 -- by
`scripts/dev/paged_attention_probe.py`:

| op | position comes from | result |
|---|---|---|
| `ttnn.experimental.paged_fill_cache` | page table (device tensor) | cache round-trips **exactly** (max abs 0) |
| `ttnn.transformer.chunked_scaled_dot_product_attention` | `chunk_start_idx_tensor` (device) | **1.0 / 1.5 / 1.9 %** from the dense device path at chunk 0 / 128 / 256 |
| `ttnn.transformer.paged_scaled_dot_product_attention_decode` | `page_table_tensor` + `cur_pos_tensor` | 3.7 % from a float32 reference |
| `ttnn.experimental.paged_update_cache` | `update_idxs_tensor` + `page_table=` | writes the position **exactly** (max abs 0) |

Two things that table settles. The chunked op is **causal internally**, so it
needs no mask and no growing slice -- (3) and (4) go together, and the
`repeat_interleave` that expands 2 kv heads to 24 goes with them, since it takes
`nkv` directly. And *decode has a paged variant too*, so one layout serves both
paths -- which matters because K/V is 6.4 GB a slot at full context and there is
no room to keep two.

On the accuracy column: do not read the float32 numbers as error growth in the
paged route. The dense path in use today is itself 1.9 / 4.3 / 6.5 % from
float32 at those chunks -- that is bf16 accumulating over more keys. Paged and
dense agree with *each other* far more closely than either agrees with float32,
and at chunk 256 paged is the closer of the two (6.119 vs 6.456).

**Steps 1-4 are done.** The K/V cache is paged, every position in both paths is
device data, and the chunk path contains no host->device copy at all. Verified
bit-identical to the flat path at every gate, including through the real serving
stack rather than only the harnesses:

| | before | after |
|---|---|---|
| decode | 83.0 % top-1, NLL 0.682 | **identical** |
| prefill=32, 128 scored | 71.9 %, NLL 1.433 | **identical** |
| prefill=128, 107 scored | 24.3 %, NLL 5.648 | **identical** |
| chunk wall clock | 1060.5 ms | 1070.5 ms |
| `TTEngine`, 48 tokens x 2 prompts | 240.1 / 240.5 ms/token | **176.4 / 177.3 ms/token** after `022`; was 239.6 / 240.4 |
| `TTEngine` + `chunked_prefill`, prefix reuse | — | **11.40 s cold, 2.27 s warm, warm == cold** |
| the same **with the decode trace on** | — | **6.2–6.4 s cold**, same tokens, twice over |
| `TTEngine(speculate=8)`, copy-heavy prompt | 176.4 ms/token plain | **160.3 ms/token** (1.10x), tokens identical |
| the same, open prose | 177.3 ms/token plain | 176.5 ms/token (1.00x) |
| the same, run twice in separate processes | — | **token-identical on all three turns** |
| traced step, 262144 ctx | 236 ms | **236.2 ms**, since taken to **173.7** by `022` |
| traced step, QSA on at 8192 | 297 ms | **297.4 ms** |
| decode with the **QSA indexer on** (8192) | 83.0 %, NLL 0.682 | **identical** |
| `step_n`, k = 1, 2, 4, 8 | 0.00 % on the hidden | **identical** |
| prefix reuse over three turns | 69 of 73 reused, warm == cold | **identical** |
| `snapshot`/`restore` after a discarded draft | rolled == clean | **identical** |

The indexer row matters more than its size suggests: every other quality check
here runs at `max_seq_len=512`, where the selection is **off**, so the branch
that hands `paged_scaled_dot_product_attention_decode` an `attn_mask` was
untested until it was run deliberately at 8192.

The engine row is `speculation_check.py` with `TTRUNNER_SPEC_ONLY=0`, and it
covers the paged *decode* path end to end. Chunked prefill is off in that
configuration, so the MoE routing/compute split (5.7) is covered by
`device_quality --prefill`, not by this.

Two things that took measuring. `chunked_scaled_dot_product_attention`'s two
chunk sizes buy different things -- `q_chunk_size` is the outer iteration count
and drives speed, `k_chunk_size` fixes the accumulation order and therefore the
answer. All-32 is correct at every start but 36 % slower (1442 ms); all-128 is
fast but moved prefill's NLL to 5.980; **q=128, k=32** is both, and is selected
only when the chunk is a full 128 at a 128-aligned start, with the all-32 config
as the fallback. And a misaligned start is *silent*, not an error: at
q_chunk_size=128 a start of 32 returns 257 % nonsense.

**What is left is step 5 alone: the capture -- and it is blocked by 5.6's
defect, which is the same defect.** Alternating two distinct model-scale traces
hangs this ttnn build, and a traced prefill would alternate with the traced
decode step exactly as a traced verifier does. See 5.6; the mechanism is *not*
settled, so do not design around a particular one. A traced prefill would alternate
with the traced decode step exactly as a traced verifier does, so it cannot work
until that is fixed upstream.

That also explains this item's original symptom, recorded long before the cause
was known: "after a few of them the trace replay came back as token 0 repeated".
A trace that is not executing but also not blocking returns stale buffers -- the
same signature the second-command-queue experiment produced in 5.6.

So steps 1-4 stand on their own merits: the cache is paged, both paths take
their positions from device memory, and the chunk path is free of host work,
all bit-identical and speed-neutral. Step 5 is one `execute_trace` fix away, and
that fix lands 5.6 with it.

The original plan, for the record:

1. Allocate `st.keys`/`st.values` paged, `[T/32, n_kv, 32, head_dim]`, plus a
   per-slot page table as a device tensor.
2. Move decode to `paged_update_cache(page_table=...)` and
   `paged_scaled_dot_product_attention_decode`. **Gate on decode quality
   (`device_quality.py`, 83.0 % top-1 / NLL 0.682) before touching prefill** --
   this is the step that can regress a working path, and it is worth its own
   commit.
3. Move `_attention_chunk` to `paged_fill_cache` + `chunked_scaled_dot_product_attention`,
   deleting the host mask, the `kv_len` slice and the `repeat_interleave`.
4. Make rope's cos/sin a bound buffer written before replay, as
   `TracedDecoder._fill_inputs` already does for the decode inputs. That is
   dependency (1) and the only one left after step 3.
5. Capture. `chunk_start_idx` must be a multiple of both `q_chunk_size` and
   `k_chunk_size` (an upstream workaround the op documents), so fix the prefill
   chunk at 128 for the traced path.

**API notes, all found the hard way:**

* `paged_update_cache` takes `page_table=`; `page_table_tensor=` is rejected as
  an unknown argument. `chunked_scaled_dot_product_attention` and
  `paged_scaled_dot_product_attention_decode` take `page_table_tensor=`.
* `paged_update_cache` keeps its flat-mode precondition: the input must be L1
  height-sharded, one shard per core, **tile-high**, shard width == head_dim,
  counted from the *padded* height. A `(1, 256)` shard fails with "Physical
  shard shape must be tile {32, 32} sized". `TTModel._l1_height_sharded` already
  builds exactly this, so the existing call site needs only the extra kwarg.
* `paged_fill_cache` writes its input at the *start* of the page table it is
  given, so a chunk at offset `start` passes a page table sliced from block
  `start / block_size` onward, not the whole one.

**The QSA indexer is not a complication after all**, which is worth saying
because 5.2 previously claimed it was. It builds its mask from
`st.indexer_blocks` over *logical* positions and never touches `st.keys`, and
`paged_scaled_dot_product_attention_decode` accepts `attn_mask` with the same
`[b, 1, s, s]` shape the flat op does. Its uint16 `ttnn.scatter` ceiling is a
limit on sequence length (65536), not on cache layout.

**Worth it?** A 128-token chunk issues 12293 device calls
(`op_count.py --prefill 128`) at ~1060 ms, so it is essentially all dispatch,
which is exactly what a trace removes. See also 5.7.

Done when: `use_trace=True` and `chunked_prefill=True` together reproduce the
eager result token for token over
`scripts/dev/prefix_reuse_check.py --chunked --trace`.

**Met — and read what was actually done, because it is not step 5.** The two
flags conflicted for the reason the eager verifier first hung in 5.6: a
prefill's first call allocates gigabytes of its own temporaries, and doing that
while a trace is live corrupts the replay. That is what "token 0 repeated" was.
`_device_loop` now warms one chunk before capturing the decoder, so the buffers
exist and nothing allocates under a live trace, and `_use_trace` no longer
switches itself off when chunked prefill is on.

Run twice, identical both times and identical to the untraced chunked run:

| | |
|---|---|
| four requests, tokens | identical to `--chunked` without the trace |
| prefix reuse | 69 of 73 as expected; the unrelated prompt correctly misses |
| warm == cold | yes |
| cold time to first token | **6.2–6.4 s**, against 11.40 s untraced |

**The prefill chunk itself is still not captured.** It runs eagerly, still
issues 12293 device calls, and steps 1–4 above are what make that cheap rather
than a trace. What changed is that it no longer costs you the *decode* trace,
which is where the 11.40 → 6.3 s comes from. Capturing the chunk would need a
second trace and so waits on the defect in 5.6's report; whether it is worth it
once that lands is a fresh question, since prompt intake is already
7.2 ms/token.


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

**Re-verified, after repairing the harness.** `indexer_select_check.py` had gone
stale against the code -- `_indexer_select` grew `q_cos`/`q_sin` parameters and
the harness still called the old signature, so it raised a `TypeError` rather
than checking anything. Fixed, it now reports 2048 of 2048 tokens selected with
**nothing missing and nothing extra**, no block off in either direction, and the
trailing partial block present. So the mid-block caveat above is conservative
at this position rather than a standing error.

Separately, the branch that hands the selection to
`paged_scaled_dot_product_attention_decode` as an `attn_mask` is exercised only
when `max_seq_len` exceeds the budget, which no other quality check here does:
at 8192 with the selection on, decode scores 83.0 % top-1 / NLL 0.682, identical
to dense (5.2's gate table).

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


### 5.5 Short multi-row step — **done** (`step_n`), but only to k = 32

`TTModel.step_n(tokens, state)` advances one sequence by k tokens in a single
step and returns the mixed hidden at all k positions. The k tokens ride the
batch axis for everything per-token; only the DeltaNet convolution and the
recurrence are unrolled. Verified to reproduce k sequential steps **exactly**
(0.00 % on the hidden, tokens matching, positions right) at k = 1, 2, 4, 8 and
16, with `scripts/dev/step_n_check.py`.

**Half its advertised range was wrong.** The guard read `1..64` -- the batch
cliff, where 65 rows pad to 96 tiles and overflow L1 -- and nothing had ever
exercised it past 8. At k = 33, 48 and 64 it is **35.68 % out on the hidden**
with a different token stream, and the same figure at all three, so it is a
structural break at one tile of rows rather than anything that accumulates. The
guard now stops at 32 and says why.

**And it is the same boundary as everything else.** `step_n_layer_bisect.py`
shows the divergence starting at 0.385 % in layer 0 and rising smoothly, not
jumping -- accumulation, not corruption -- and the boundary is exactly one row
tile: k=32 is exact, k=33 is not. The cause is not the MoE: a plain
`ttnn.linear` already returns a row identically at m <= 32 and differently past
it (`row_tile_boundary_check.py`), so every op in the layer has diverged before
the experts are reached. Splitting `expert_ffn` into 32-row groups to keep
`per_core_M` at 1 was tried and changes nothing (35.68 % becomes 36.72 %).

`step_n` exists to reproduce k sequential steps *exactly*, so a path that cannot
is no use to it whatever the cause. Nothing needs k > 32 -- speculation caps its
widths at 17 -- so it refuses, with `TTRUNNER_ALLOW_WIDE_STEP_N=1` to lift the cap
for investigation.

One row tile turns out to be the reproducibility boundary throughout: it caps
`moe_chunk` (5.8), it is why batch 64 decodes differently from batch 1, and it
is why `step_n` stops at 32. Three findings, one cause.

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


### 5.6 Speculation — **done**: exact, and 1.10x on text that quotes itself

> **Re-measured after `022`, and the headline moved.** Speculation amortises the
> traced step, so making the step 1.36x faster takes most of its advantage with
> it: copy-heavy went 174.7 -> **160.3 ms/token**, but plain decode went
> 240.1 -> 176.4, so the ratio fell from **1.37x to 1.10x**. Open prose is now
> neutral (176.5 against 177.3) where it used to cost 10 %. The scheme is
> unchanged and still exact -- what changed is what it is being compared against,
> which is the thing to re-check whenever the baseline moves.

Built end to end and opt-in via `TTEngine(speculate=k)`, where k is the tokens a
verify *feeds* and it drafts `k - 1`. `docs/iterations/016` has the iteration and
`018` the diagnosis. Set `TTRUNNER_ALLOW_SPECULATION=1` to lift the refusal.

**The capture was never the problem.** For four board resets the record said
capturing `step_n` inside the engine hangs the device. It does not: the setup
completes, both traces capture, and the engine enters its serve loop. What hangs
is the **first replay of the verifier**, and a stack dump says so directly --
`ttnn.execute_trace` in `TracedStepN.step_n` (traced.py:130), from
`speculate_round` (engine.py:565).

**It is the interleaving, and it is general.** A speculating engine replays the
decoder's trace on plain rounds and `step_n` on drafted ones.
`traced_step_n_check.py` was thought not to do that -- it releases both captures
before replaying `step_n` alone -- and that was the standing explanation for why
no harness hung. The explanation was wrong: it *does* hang, in its k-loop, and
releasing the earlier traces does not help (see the exclusions below). The defect
is not
about *which* two graphs, either: `spec_capture_ladder.py two_stepn` alternates
two `step_n` captures (k=2 and k=4) with no decoder involved and hangs the same
way. **Any two traces replayed alternately hang**, which is also what blocks
5.2's step 5, and what produced its "token 0 repeated" symptom.
`scripts/dev/spec_capture_ladder.py` is the sixty-line reproduction, no engine,
no asyncio, no admission loop:

| configuration | outcome |
|---|---|
| cq 0, replay `step_n` alone | correct, 265 ms |
| cq 0, replay decoder, then `step_n` | **hangs** in `ttnn.execute_trace` |
| cq 1, replay decoder, then `step_n` | returns in **12 ms**, **wrong tokens** |

**Two non-fixes, so nobody spends a day on them again.** A device sync between
the replays does nothing -- both calls already pass `blocking=True` and
`synchronize_device` with no `cq_id` waits on every queue. A second command
queue stops the hang and is *worse*: the replay does not execute, it merely
stops blocking, returning `[201058, 0]` where the eager `step_n` returns
`[75, 220]`, which showed up first as the engine accepting 0 of every 5 drafted
tokens with a 9.6 ms verify. It was briefly committed as the fix on the evidence
that the hang stopped; do not repeat that.

**A mechanism that looked established and is not.** `FDMeshCommandQueue::enqueue_trace`
ends with `trace_dispatch::update_worker_state_post_trace_execution`, which
**sets** -- not increments -- the host launch-message write pointer to *that
trace's* program count, while capture (`record_begin` ->
`reset_host_dispatch_state_for_trace`) zeroes it. That reads like an exact
explanation: each trace is recorded against a zeroed pointer and leaves it at
its own count, so alternating two traces of different sizes desynchronises host
and workers and the dispatcher waits forever.

It predicted correctly once -- `spec_capture_ladder.py two_same` captures one
graph twice, identical counts, and alternates A/B/A cleanly where `two_stepn`
(k=2 against k=4) hangs -- **and that control is confounded**: two captures of
the same graph share their program count *and* their kernel binaries, so it
cannot separate the two. `scripts/dev/repro_trace_program_count.py` then
alternates tiny traces of 2 and 5 programs with no trouble at all.

Both candidates are retired by controls that separate them. `plus_one` captures
the same graph twice with the second one `ttnn.add` longer -- counts differ by
one, binaries identical -- and alternates cleanly; the same with
`TTRUNNER_NEW_KERNEL=1`, where the extra op is `ttnn.atan` so trace B references a
kernel trace A does not, also alternates cleanly. And magnitude is not it either:
k=2 against **k=3** hangs.

The line the controls actually draw: trace B may be trace A *plus appended
operations* -- one or fifty of them (`TTRUNNER_EXTRA_OPS=50`), new kernels
included -- and the pair alternates fine; if the two traces hold
differently-shaped versions of the same operations, they hang. **B superset of A
is safe; B and A disagreeing about a shape is not.** Changing k re-shapes every
layer rather than adding anything.
`docs/upstream/trace_alternation_hang.md` has the full control set.

**What is actually established**, and all of it at model scale:

* replaying one trace repeatedly is fine, however many times;
* replaying a second trace *alone* after capturing both is fine;
* alternating two *distinct* model-scale traces hangs in `ttnn.execute_trace`;
* alternating two captures of the *same* model-scale graph does not;
* tiny traces do not reproduce any of it, so the defect needs scale.

**Four workarounds excluded by experiment, so they are not re-tried:**

1. `ttnn.synchronize_device` between the replays — no effect. Both
   `execute_trace` calls already pass `blocking=True`, and with no `cq_id` the
   sync waits on every queue.
2. A second command queue — stops the hang and is *worse*: the replay does not
   execute, it merely stops blocking, returning `[201058, 0]` where the eager
   `step_n` returns `[75, 220]`, in 12 ms against 265.
3. An ordinary (non-trace) program between the replays, on the theory that
   normal dispatch maintains what the trace path leaves stale
   (`TTRUNNER_EAGER_BETWEEN=1`) — still hangs.
4. `MeshDevice.reset_sub_device_stall_group()` between the replays, since
   `enqueue_trace` updates worker state per sub-device (`TTRUNNER_RESET_STALL=1`)
   — still hangs.
5. A 1 GB `trace_region_size`, against the 256–384 MB the two traces need, in
   case they were colliding in an undersized region — still hangs, so it is not
   region pressure.
6. `ttnn.release_trace` on the first trace before capturing and replaying the
   second — still hangs. This is the strongest hint available: what
   `enqueue_trace` leaves behind is not undone by releasing the trace that left
   it.

**`traced_step_n_check.py` does not currently complete**, and this item cites it
as the standalone measurement that works. It replays the decode trace many
times, releases both captures, then captures a fresh `step_n` and replays it —
which is the alternation, release or no release. It reaches its k-loop and hangs
there, verified on a pre-paged worktree as well as on HEAD, so this is not the
cache refactor. The numbers it produced (255 ms at k=2, 276.6 at k=4, 323.9 at
k=8) were taken in an earlier session and cannot be reproduced by it today —
treat them as historical. `step_n`'s *correctness* is unaffected and
independently checked: `step_n_check.py` passes at k=1, 2, 4 and 8, 0.00 % on
the hidden, tokens matching, positions right.

There is no host-side lever left that I can see. The one candidate that still
fits — per-program config-buffer state — cannot be varied from Python; it needs
an instrumented tt-metal build, which is what
`docs/upstream/trace_alternation_hang.md` is for.

**Next step is upstream.** The report must carry the model-scale harness
(`spec_capture_ladder.py`, stages `both_replay` and `two_stepn`, with `two_same`
and `stepn_only` as controls) rather than a tidy standalone script, because the
standalone one does not reproduce it. Point at
`update_worker_state_post_trace_execution` as a *suspect*, with the caveat above
-- do not present it as the cause. It is worth more than one item: the same fix
lands 5.2's traced prefill, since that alternates a prefill replay with the
decode step's. Until then the flag stays refused, because it takes the boards
with it.

**It *is* identical to token-by-token decoding**, which reverses what this item
said for most of its life. `speculation_exactness_check.py` drives the round
loop directly -- draft, snapshot, verify, accept, restore, replay -- against a
plain sequential decode from the same start, in one process and all eager, so it
needs neither the trace nor the fix above. The token streams are **identical**
at k=2, 8 and 17 over 64 tokens, with 30, 53 and 64 drafted tokens accepted.
17 is the widest width speculation uses.

The old claim rested on "the verifier batches k rows where the stepper runs one,
so bf16 rounding differs". That premise is false: `step_n` reproduces k
sequential steps exactly at every k up to 32, and invariant 13 says why -- k <=
32 rows and 1 row are both inside one 32-row tile. The 0.00 % that
`step_n_check.py` reports was read as a rounded maximum hiding something; it was
not.

The divergence at tokens 29 and 39 that prompted the old claim is unexplained.
It was measured through the engine across two processes, which was unavoidable
then and is not a comparison this file trusts elsewhere. If the engine adds
divergence of its own that is an engine bug, not a property of the scheme, and it
cannot be tested while the engine hangs.

**Instrumentation to start from.** `TTEngine.speculation_report()` breaks a round
into snapshot / verify / restore / replay and reports the drafter's own cost; the
`mark()` lines print each setup step while speculating; `TTRUNNER_STACK_DUMP=<s>`
in `speculation_check.py` dumps every thread's stack on a timer. The first of
those once showed a *plain* round costing 487.5 ms against a traced 236 because a
harness was disabling the trace, and the third ended this hunt in one run.

Verified independently and worth keeping: `step_n` reproduces k sequential steps
for **k up to 32** (0.00 % on the hidden; it is 35.68 % out from k=33 and the
guard now refuses that -- see 5.5), `TracedStepN` replays at 255 ms for k=2 against 472 for
two traced steps, `snapshot`/`restore` roll a discarded draft back identically,
and the drafter and accept rule are unit-tested. Offline pricing says
1.70-2.14x on prompts that quote their context, so the ceiling is real.

**One engine per process.** A second `TTEngine` built after closing the first
hangs on its own capture -- an engine-lifecycle bug, not a speculation one, and
it blocks anything that opens a mesh twice. `close` now releases its captured
traces, which was part of it but not all. Note this is *not* the speculation
hang: that reproduces with the speculative engine as the only one in the process
(`TTRUNNER_SPEC_ONLY=2`).

Done when: the interleaved replay works -- upstream fix or a formulation that
avoids two traces -- and `speculation_check.py` shows a round cheaper than
`speculate=0` on the copy-heavy prompt.

**Met, by the second route.** Nothing required the verifier to be *captured*.
Run eagerly it leaves the decoder's trace as the only one in the process, never
alternated against anything, and the defect has nothing to bite. A verify
amortises over the tokens it checks -- warmed eager `step_n` costs 150 ms per
token verified at k=8 against a traced step's 236 -- so from k=4 up it is ahead
whenever the draft is accepted.

At `speculate=8`, 48 tokens, both prompts:

| | speculative | baseline | |
|---|---|---|---|
| copy-heavy | **174.7 ms/token** | 240.1 | **1.37x** |
| open prose | 268.1 | 240.5 | 0.90x |

163 rounds, 42 of 49 drafted tokens accepted, and the output is **identical to
the non-speculative engine token for token on both prompts**. Open prose loses
10 % -- the drafter fires on 4 % of rounds there and each drafted round pays a
full verify -- so it stays opt-in per `speculate=k`, off by default.

Every width the rounds can use is warmed before the decoder is captured. Without
that the first attempt hung anyway, because an eager `step_n` allocates its own
48-layer graph and doing that under a live trace is the hazard above.

The upstream report stands regardless: the defect is real, it is just no longer
in the way.


### 5.7 Fewer launches — **done**: a launch is ~57 us, and that prices the item out

**Read this before spending a day on it.** `docs/iterations/012` framed the
single-user step as op-count bound, "6355 ops x ~36 us traced". That arithmetic
describes the *eager* path. Measured directly, by removing 97 ops and timing
both paths at the same configuration:

    eager    498.7 -> 469.1 ms   (-5.9 %)
    traced   236.1 -> 236.0 ms   (unchanged)

29.6 ms for 97 ops is 0.30 ms apiece. A trace replays with one dispatch, which
was its whole point, so removing launches buys nothing there.

> **That 0.30 ms is not a dispatch cost, and `021` retired it.** It is the cost
> of those 97 particular ops on that path, and it was then applied to every
> other path for four iterations. It does not survive contact with either:
> extrapolated over the step it was measured on it predicts 1873 ms for a 498.7
> ms step, and over a prefill chunk 3.50 s for 925 ms. Measured directly on
> prefill by injecting a known number of ops and reading the slope
> (`dispatch_cost_check.py`), a launch is **90 +/- 15 us** marginal and **57 us**
> realisable. Everything below that prices a candidate in dispatches is
> therefore quoting a number five to six times too large. Fewer launches pays for chunked prefill and for eager decode --
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

**Prefill is where this item is now worth something.** The framing above was
written when the eager path was a fallback. It is not any more: chunked prefill
is on for one-slot engines, it is *not* traced, and it issues **11681** device
calls for a 128-token chunk against 6143 for a single-token step
(`op_count.py --prefill 128`). Op-count reduction pays here at full rate --
but at 57 us a call, not at 0.30 ms.

Note that "11681 calls" overstates the dispatches: `op_count.py` counts python
calls into `ttnn`, and an injected `reshape` measures **2.1 us**, i.e. it is a
host-side view that never reaches the device. `reshape` alone is 1094 of that
total.

**Where a chunk's time actually goes**, measured by ablation rather than inferred
from the call counts below (`decode_ablation_check.py <part> 128`, baseline
922.5 ms):

| component | cost | share |
|---|---|---|
| DeltaNet chunk, 36 layers | 229.5 ms | 25 % |
| — of which `prepare_device` | **158.9 ms** | 17 % |
| — of which `gated_delta_attn_seq` | 18.0 ms | 2 % |
| the whole MoE compute half | 65.8 ms | 7 % |
| QSA chunk, 12 layers | 18.5 ms | 2 % |

So prefill and decode are almost inverted: the MoE is 65 % of a traced step and
7 % of a chunk, and the largest single item here is this project's own DeltaNet
preparation, not a vendor op.

**And `prepare_device` is compute-bound, not dispatch-bound**, which was worth
finding out the hard way. It is vectorised over the chunk axis, so N chunks cost
the same ~59 dispatches as one; if its 158.9 ms were mostly launches, batching
four chunks would have removed three quarters of it. `prefill(deltanet_batch=4)`
does exactly that, is bit-identical (max diff 0.000e+00 on the logits, identical
8-token continuations), and buys **0.7 %** -- 3735.2 -> 3710.3 ms on 512 tokens.
The op count stays and the data each op moves quadruples, so the saving never
appears. Its masks are also already cached, and `block_diag_inverse_device` is 20
ops for five algebraic levels (`017` picked that recursion over a cheaper
unstable one). There is no dispatch fix in it.

Which relocates the 10.9 % that a *wider* chunk buys: it is not dispatch
amortisation, it is the dense matmuls getting wider, and that changes their
blocking and so their rounding. Measured across the ten dense shapes this path
uses (`row_count_stability_check.py`), five are bit-identical at 512 rows and
128, and the other five all land slightly *closer* to the exact float64 product
at 512 -- five of five, none worse. So it is not an accuracy loss. It is still a
change, and it cannot be validated end to end because the regime it applies to
(prompts over 128 tokens) is the one section 4b says is broken.

**So the chunk is wide and the linears are not.** `PREFILL_CHUNK = 512`, and every
dense linear on the prefill path goes through `ops.linear_rows`, which never
hands the op more than 128 rows. Each row keeps exactly the company it kept when
the chunk was 128, so the arithmetic is untouched, while everything width-neutral
-- the DeltaNet scan (NC=4 in one call), the expert FFN (per-token exact), PLE,
the embedding, the elementwise work -- amortises over four times the tokens.

    512 tokens: 3559.9 -> 3276.5 ms, 8.0 %, and bit-identical:
    logit max diff 0.000e+00 against chunk=128, first 8 continuation tokens 8/8.

`deltanet_batch` stays at 1 because a 512-wide chunk already gives the scan its
four chunks; grouping further measured no gain.

`--by-caller` attributes each call to the `ttrunner_qwen38_flash_next` line that issued it, which is
what says where to cut; "multiply, 1993" does not. The distribution is flat --
the largest single site is 4.2 % -- so expect many small wins rather than one
big one.

**A worked example, measured and kept: split routing from expert compute.**
The MoE's two halves want different row-group sizes, and only one of them is
fussy. Routing at a wider group *changes the answer* -- bf16 moves the router's
probabilities ~0.5 %, which reorders experts across the k-th boundary (5.8) --
while the expert FFN is exactly per-token at any width, now that
`sparse_program_config` gives it a single K block above one row tile. So
`prefill` routes in `moe_chunk`-sized groups and calls `expert_ffn` **once** for
the whole chunk instead of `seq / moe_chunk` times.

12293 -> 11681 dispatches and **1070.5 -> 925.1 ms** a chunk, 13.6 %, taking
prompt intake to 7.2 ms/token.

It is *not* bit-identical, which I expected it to be: batching the expert
compute to 128 rows forces the single-K-block program config (the narrow one is
the buggy one above a row tile), and that changes the accumulation order.

**The quality side of this comparison does not hold, and the decision does not
rest on it.** The 24.3 % -> 25.2 % and NLL 5.648 -> 5.823 quoted here were
measured in different batches of invocations, and a 128-row prefill is not
repeatable across those -- the same binary spans 21.5-27.1 % top-1 (invariant
8). Both numbers are inside that spread, so the change is *quality-unmeasured*,
not quality-positive. What does hold is 612 fewer dispatches and 1070.5 -> 925.1
ms, neither of which moves, and prefill=32 reducing exactly to the old path at
71.9 % / 1.433, which is deterministic and is the check that the restructuring
itself is faithful.

**A worked example, measured and rejected.** The shared expert is dense and has
no routing, so `moe_chunk` -- which exists to bound the routed MoE's
|union| x M waste -- buys it nothing, and it was running once per sub-chunk:
eight dispatches x 4 x 48. Hoisting it to the whole chunk saves 1296 calls
(9.5 %) and takes a chunk from 1089 to 1017 ms. It is also genuinely per-token,
unlike `moe_block` (5.8): `shared_expert_rows_check.py` finds no row over 1 % at
any group size to 128.

It still lost. "Per-token within 1 %" is not "identical" -- against the per-row
answer a 32-row group is 0.000 % and a 128-row group 0.500 % -- and over 107
scored positions whole-chunk gave 20.6 % top-1 / NLL 6.389 against 24.3 % /
5.648 sub-chunked. Reverted, with the reasoning left in `prefill` so it is not
re-attempted.

**Re-measured after the routing/compute split**, because that moved the baseline
and a stale rejection is worth as little as a stale acceptance: the hoist buys
925.1 -> 887.6 ms (4 %) for NLL 5.823 -> 6.650 with top-1 unchanged at 25.2 %.

That NLL gap is at the edge of what a 128-row prefill can resolve across batches
(spread 0.73, invariant 8), so read the rejection as *4 % of wall clock for a
quality question this measurement cannot settle* rather than for a proven loss.
The conservative call on a path this session spent its effort making correct is
to leave the dispatches. Revisit it with a back-to-back A/B if 4 % ever matters. The lesson generalises: on this path a dispatch saving that
changes any group size is buying speed with bf16 accuracy, and 7 % is not a good
price. Look for savings that leave every group width alone.

**A second worked example, also rejected, and it failed both ways.** The
DeltaNet chunk builds q, k and v with three separate reshape/permute/pad/cast
passes; they sit back to back on the channel axis with equal widths, so one pass
over `3 * n_v` heads plus three slices should replace them. It removed 360
dispatches (11681 -> 11321) exactly as predicted, and ran **30 % slower**
(925.1 -> 1205.2 ms): the single permute is over a tensor three times the size,
and permute cost is not linear in it. It also did not come out bit-identical the
way pure data movement should (top-1 25.2 % -> 21.5 %, NLL 5.823 -> 6.345),
which was never chased down because the wall clock had already disqualified it.
Reverted.

Three warnings for whatever is next. Op count is a poor proxy for cost -- the
`reinject` docstring records a three-op form running **9.5x slower** than the
nine-op one it replaced, because `repeat_interleave` is pathological on those
shapes, and the q/k/v permute above is the same lesson a second time. Measure
wall clock, never the call count alone. Accuracy on this path is not repeatable at 32 scored positions, so use
107+ and quote NLL (invariant 8). And every PR here needs the same A/B:
`device_quality.py` unchanged, `op_count.py` before and after, and
`bench_step.py` at one configuration for both paths.

**Done: the conversion factor is measured and it closes the item.** 5.7 never
had a definition of done for most of its life, and the two it acquired were both
wrong — "5.2's step 5 lands" (5.2 closed by another route, without capturing the
prefill) and then "the chunk is captured" (that is blocked upstream, so it makes
the item permanently open on someone else's schedule). Neither asks the question
5.7 is actually for, which is *what is a launch worth here*. That now has an
answer, and the answer prices the category out:

| what | calls | worth at 57 us |
|---|---|---|
| the largest single call site (`shared_expert`) | 576 | 33 ms, 3.6 % |
| every site in the by-caller top twelve | 3712 | 212 ms, 23 % |

The second row is the ceiling on the whole item, not an opportunity — those
calls are doing the model's work. With the distribution flat (no site above
4.9 %) every *individual* remaining reduction is under 4 %, against a required
A/B of `device_quality.py`, `op_count.py` and `bench_step.py`, on a path where
three of three attempts so far either changed the answer or ran slower.

So: **stop grinding op count here.** The measurement, the size control that
shows it is the launch and not the arithmetic, and the calibration against a
real removal are in `021`; the harness is `dispatch_cost_check.py` and it
re-runs in a few minutes if the baseline moves.

What would still buy the category is capturing the chunk, which takes the launch
count out of the equation wholesale rather than by percentages. That is blocked
on the trace-alternation defect in `ttnn_bug_report/` and is tracked there, not
here.

### 5.8 The MoE row-group cliff — **done**: defect fixed, cap measured

`prefill(moe_chunk=)` groups rows for the MoE. Speed wants the biggest group;
the answer changes past 32, which is one tile:

| moe_chunk | wall clock (1 draw) | tok/s | logits vs 16 | argmax |
|---|---|---|---|---|
| 16 | 2032.6 ms | 63.0 | — | 1105 |
| 32 | **1101.0 ms** | 116.3 | 0.0000 % | 1105 |
| 64 | 936.1 ms | 136.7 | 55.9 % | 1154 |
| 128 | 998.7 ms | 128.2 | 144.9 % | 50 |

The wall-clock column is one unwarmed draw per row, taken before
`moe_chunk_sweep.py` was fixed to warm twice and take a median of nine. **Ignore
it.** Re-measured properly (`TTRUNNER_LIFT_MOE_CAP=1` to reach past the cap):

| moe_chunk | median | min | max | tok/s |
|---|---|---|---|---|
| 16 | 1214.1 ms | 1206.2 | 1227.4 | 105.4 |
| 32 | **918.6 ms** | 914.8 | 923.4 | 139.3 |
| 64 | 881.6 ms | 879.5 | 970.0 | 145.2 |
| 128 | 861.7 ms | 858.3 | 960.7 | 148.6 |

Two things the single draws got wrong. 16 -> 32 is **1.32x**, not the 1.85x they
implied. And they had 128 *slower* than 64 (998.7 against 936.1) where it is in
fact faster, so the ordering was wrong too, not just the ratio -- larger chunks
are monotonically quicker.

**Which makes the cap nearly free.** Lifting it from 32 to 128 buys **6.6 %**,
not the 1.18x claimed from the raw draws. For that it changes the answer past one
row tile. The logits column below is the evidence that matters and is unaffected;
it was measured within one process.

**Measure this on prefill's logits, not on next-token accuracy.** The cap was
first set from one `device_quality.py --prefill 128` run per setting, and two
runs at the *same* setting then gave 50.0 % and 53.1 % top-1 — so that evidence
could not tell a real effect from run noise, and the reasoning built on it ("8,
16 and 32 agree on every token") was not supported. Prefill's logits *are*
bit-deterministic: three repeats per setting agree to 0.0000 %
(`moe_chunk_noise.py`). Use that.

**What is actually wrong.** `moe_block` must be exact per row — each row selects
its own experts and is weighted by its own probabilities, so grouping may only
change speed. `scripts/dev/moe_rows_check.py` builds the answer one row at a
time, which needs no reference implementation, and compares:

| group | worst row | rows over 1 % |
|---|---|---|
| 1, 8, 16, 32 | 0.000 % | 0/64 |
| 64 | 102.9 % | **33/64** |

**Where it is not**, so this is not re-searched:

* Routing is exact at 64 — `keep`, `weights` and the expert union all match the
  host. (Check *all* rows, not row 0: row 0 agrees at every group size, which is
  what made the MoE look innocent for a while.)
* In isolation, at every row count from 8 to 128: `ttnn.topk` and the threshold
  mask (exact), `ttnn.max` over the row axis (exact at 64; drops 1 expert of 473
  at 128), `ttnn.permute(0,3,2,1)` (exact), the expert-axis `ttnn.sum` (bf16
  floor), and `ttnn.sparse_matmul` across every combination of 1–64 selected
  experts (bf16 floor).
* Four `sparse_program_config` variants — `out_subblock_h` 2 and 4,
  `fuse_batch=True` — give byte-identical error; `mcast_in0=False` is rejected
  by the op. So `per_core_M > 1` is the boundary but the config is not the bug.

**Found, and fixed.** `ttnn.repeat` is exact at production scale, so it was the
sparse matmul after all — the earlier probe missed it by running at E=64, K=256,
N=128, about a hundredth of the real shape.
`scripts/dev/sparse_matmul_rows_check.py` isolates it with real weights:
`ttnn.sparse_matmul` returns **wrong rows past the first 32-row tile** whenever
`per_core_M > 1` and K spans more than one block, silently. Neither the expert
count nor N nor the mask density matters — it is wrong at a union of 1. K is the
trigger, through the block count:

| in0_block_w | K blocks (K=2560) | rows wrong |
|---|---|---|
| 8 (the old default) | 10 | 32/64 |
| 16 | 5 | 32/64 |
| 40 | 2 | 16/64 |
| **80 = k_tiles** | **1** | **0/64** |

and by K at the default width: K=256 (one block) clean, K=512 16/64, K≥1024
32/64.

`sparse_program_config` now uses `in0_block_w = k_tiles` — one block, no K loop —
**when `per_core_M > 1`**, keeping the original candidate list below that. The
condition is load-bearing: at one row tile there is nothing to fix, and widening
the block changes the accumulation order, which moved prefill's NLL from 5.648 to
6.301 when applied unconditionally. `expert_ffn` also derives both configs from
the *tensors* now: the down-projection had been given the global intermediate
size where the weights are sharded, so its nominal Kt (20) was four times the
real one (5) — and the old candidate list landed on 5 by luck, which is why only
the gate/up matmul was ever wrong.

Result: `moe_block` at 64 rows goes from **33 of 64 rows wrong (worst 102.9 %) to
1 of 64 (worst 6.8 %)**, with decode (83.0 % / NLL 0.682) and prefill (24.3 % /
5.648) bit-identical to before.

**It fixed the decode path too, which was not the point of it.** `per_core_M`
exceeds 1 for decode as soon as the batch does 32, so batch 64 was running its
experts through the broken configuration. `batch_equivalence_check.py`, same
prompt in every row:

| | rows agreeing with row 0 | matching a batch-1 run |
|---|---|---|
| batch 64, before | **32/64** | 0/64 |
| batch 64, after | **64/64** | 0/64 |
| batch 32, either | 32/32 | **32/32** |

So batch 64 was silently corrupting half its rows, and the README's "bit-exact
vs single-sequence" claim was false there -- including for the 97.4 tok/s figure
it advertised. Batch 64 still does not reproduce a batch-1 run, because the
single-K-block config accumulates in a different order, but the rows are now
consistent and the difference is bf16 rather than corruption. Batch 32 was and
remains exact.

**One row tile is the reproducibility boundary throughout this engine**, and the
cause is simpler than the MoE. A plain `ttnn.linear` returns a given row
*identically* for any row count fitting in one 32-row tile and differently past
it -- by one bf16 ulp, the same amount at 33, 48, 64 and 128
(`scripts/dev/row_tile_boundary_check.py`). Nothing about the experts is
involved.

That one fact is behind three findings recorded separately: this cap, batch 64
decoding differently from batch 1, and `step_n` refusing past k=32 (5.5). Each
compares a row computed among <= 32 rows against the same row computed among
more, and that comparison cannot come out equal.

It is not fixable at the model level. Splitting `expert_ffn` into 32-row groups
so `per_core_M` stays 1 was tried and changes nothing (`step_n` at k=33 goes
from 35.68 % to 36.72 %), because the ops *before* the experts have already
diverged. Expect any new row-grouping knob to have the same ceiling.

**Restoring batch invariance was attempted and abandoned.** Threading a
`single_k_block` flag so decode's expert config no longer depends on the batch
leaves batch 64 diverging from batch 1 by exactly as much, because the expert
matmul is not the only batch-dependent op: the *router's* `ttnn.linear` moves
its probabilities up to 0.53 % between groupings, which is enough to reorder
experts at the k-th boundary (the same effect that caps `moe_chunk`). The change
was neutral on quality (85.1 % / NLL 0.683 against 83.0 % / 0.682, one token on
47) and on step time (234.4 ms against 236.2), so it bought a parameter and a
comment that claimed something untrue, and was reverted. Above batch 32 a
sequence's output depends on its batch; document that rather than chase it.

**The residual is not a bug at all**, which took one more measurement to
establish and is the reason this item can close. The last differing row is
routing, and the cause is upstream of `topk`: `probs` themselves differ between
group 64 and per-row on 38 of 64 rows — by at most **0.529 %, with no row over
1 %**. That is bf16 non-associativity, the router's matmul accumulating in a
different order under a different tiling, and it is irreducible. `topk` is
almost stable under it (indices differ on 3 rows, the k-th value on 1), and what
turns a 0.5 % perturbation into a changed expert set is our own
`keep = ge(probs, threshold)`, which admits ties: on the one row that moves, the
threshold is *bit-identical* and a twelfth expert simply crossed it.

**So the cap stays at 32, permanently, and it costs 6.6 % of a chunk.** With the
matmul fixed, over 107 scored positions:

| moe_chunk | top-1 | NLL |
|---|---|---|
| 32 | **24.3 %** | **5.648** |
| 64 | 23.4 % | 6.443 |
| 128 | 22.4 % | 6.136 |

6.6 % of prompt intake is not worth that, and unlike before the reason is
understood rather than mysterious.

**And replacing the threshold would not help either**, which was the obvious
next idea and is now measured rather than assumed. Exact top-k selection would
be immune to *tie admission*, but the instability is not tie admission: at group
64 the top-k **set** itself differs on 1 row in 64 -- the same one -- because a
0.53 % perturbation genuinely reorders two experts across the k-th boundary.
(Index *order* differs on 3 rows; only one of those changes the set.) Any
selection rule built on these probabilities inherits that, so scatter-based
exact top-k would buy nothing and cost the gather/scatter the design avoids.

**Done, and closed.** The defect that made this an item -- `sparse_matmul`
dropping rows past the first tile -- is found, fixed and locked by a test. What
remains is inherent float behaviour: measured (0.53 % on the router's
probabilities), traced to its consequence (one row in 64 routes differently),
priced (NLL 5.648 at 32 against 6.443 at 64), and shown not to be fixable by
changing the selection rule. The cap is set accordingly and should stay.


## 6. Recipes

**Every environment switch**, because they are otherwise scattered through this
file and several are undiscoverable:

| variable | what it does |
|---|---|
| `TTRUNNER_GGUF_DIR`, `TTRUNNER_TT_CACHE`, `TTRUNNER_TOKENIZER` | where the weights, the converted cache and the tokenizer live |
| `TTRUNNER_MAX_SEQ` | `max_seq_len` for the dev harnesses (default 512). **Above 2048 turns the QSA selection on**, which no other check here does |
| `TTRUNNER_TRACE_REGION_MB` | `trace_region_size` for `spec_capture_ladder.py`; unset takes ttnn's default, which is what the working harnesses use |
| `TTRUNNER_ALLOW_WIDE_STEP_N=1` | lifts `step_n`'s k≤32 guard for investigation. Does not make it correct (5.5) |
| `TTRUNNER_ALLOW_SPECULATION=1` | lifts `TTEngine`'s speculation refusal. **Hangs the boards**; expect `tt-smi -r all` (5.6) |
| `TTRUNNER_SPEC_ONLY=<k>` | `speculation_check.py` builds only that one engine, so a process holds one |
| `TTRUNNER_STACK_DUMP=<s>` | dumps every thread's stack on a timer — what located the trace hang |
| `TTRUNNER_SECOND_K`, `TTRUNNER_NEW_KERNEL`, `TTRUNNER_NO_DEC_REPLAY`, `TTRUNNER_SYNC_BETWEEN`, `TTRUNNER_EAGER_BETWEEN`, `TTRUNNER_RESET_STALL`, `TTRUNNER_STEPN_CQ` | `spec_capture_ladder.py` knobs, one per hypothesis it tests or excludes — see 5.6's exclusion list |
| `TTRUNNER_MOE_CHUNK` *(none — use `--moe-chunk`)* | `device_quality.py --moe-chunk N` lifts `_MAX_MOE_CHUNK` for the measurement that decides whether the cap should move |

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

* `docs/iterations/NNN_*.md` — the observation → remedy → result log. Read
  **014** first (why the metric was the bug), then **020** (what the record had
  drifted from) if you are about to trust a number in here. 019 is the MoE
  row-group cliff and the `sparse_matmul` row defect; 018 the trace-alternation
  hang; 017 the DeltaNet inverse and the on-device prepare; 015 the QSA
  selection; 013 the four correctness bugs; 012 the single-user work (trace root
  cause, op-count floor); 011 continuous batching, expert fusion, the tile
  cliff; 009/010 the device bring-up; 008 the precision policy; 007 the op
  validation table.
* `docs/upstream/trace_alternation_hang.md` — filing-ready report for the one
  defect that blocks 5.2's step 5, 5.6 and (by consequence) 5.7.
* `ttnn_bug_report/` — the runnable half of that report: `repro.py` hangs the
  device in a few minutes and needs only the checkpoint, and
  `minimal_attempt_does_not_reproduce.py` records the self-contained version
  that does *not*, so the search is not repeated.
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
