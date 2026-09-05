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

23. **DRAM-sharding the weights is real, correct, and worth almost nothing
    here.** The standard advice for decode GEMV is to hold the activation in L1
    and width-shard the weight across the DRAM banks with
    `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`. It works: the answer
    matches the interleaved one to 3.906e-03, one ulp from the different
    accumulation order. On this model's projections with the shipped `bfloat4_b`
    weights it measures **1.15x / 1.00x / 1.01x** -- a wash
    (`dram_sharded_gemv_check.py`). At `bfloat16`, four times the bytes, it does
    better (1.06 / 0.95 / 1.23), which is the tell: these projections are 1-4 MB
    at 4 bits and are not DRAM-bandwidth-starved, so fetching those bytes better
    has little to recover.

    Two measurement notes that matter more than the result. Timing one call at a
    time measures **dispatch**, not the op: the same GEMVs read 0.137 ms per call
    and 0.035-0.060 ms when a run of them is enqueued and synchronised once, and
    inside a trace it is the latter that applies. And the projections are not
    where decode's time is -- QSA is 3.0 ms a layer of which its four projections
    are ~0.13 ms and SDPA plus the cache update are ~0.10; the remaining ~2.8 ms
    is the small ops around them. Matmul tuning cannot reach that.

24. **Decode uses a thirty-second of the row capacity, and the way to get it
    back is to fill the rows, not to remove the padding.** At M=1 the activation
    pads to 32 rows, and a fixed weight takes *identical* time for M=1 through
    M=32 -- 0.060 ms, 16.6 rows/ms at M=1 against 531 at M=32
    (`gemv_padding_cost_check.py`). That is why prefill is ~23x faster per token,
    and the instinct that decode therefore runs at 1/32 speed is right about the
    waste.

    It is wrong about the remedy. Removing the padding would not make M=1 faster:
    at M=1 the op is fetching weights at 73.6 GB/s, far short of the arithmetic
    limit, so the 31 empty rows are not what sets the 0.060 ms. They are spare
    capacity, and spare capacity is reclaimed by *filling* it.

    `step_n` fills it, and traced it is worth a lot (`traced_step_n_scaling_check.py`):

        k        1      2      4      8     16     32
        ms   175.1  228.5  254.1  302.2  400.1  572.5
        /tok 175.1  114.3   63.5   37.8   25.0   17.9

    k=32 costs 3.27x the time of k=1 for 32x the tokens -- **9.8x per token**. So
    the ceiling on speculation here is not 1.1x (what prompt-lookup drafting
    currently delivers) but close to an order of magnitude, and a drafted round at
    k=32 pays for itself from about 3.3 accepted tokens. The bottleneck is the
    drafter, not the machine.

    What blocks collecting it today: the engine's verifier is *eager* because
    alternating two traces hangs this build (`ttnn_bug_report/`), and eager
    `step_n` is 122-318 ms/token against these traced figures. Capturing in
    sequence is fine -- release, capture, replay -- but a per-round switch costs a
    2.6 s capture, so this needs either the upstream fix or a design where
    `step_n` is the only trace.

25. **A GEMV does reach DRAM bandwidth; a small one is hidden behind a launch
    floor, and only outside a trace.** Sweeping the weight of a decode projection
    (`gemv_bandwidth_check.py`) shows the kernel is fine -- it streams at
    **273 GB/s** for bfloat4_b and **393 GB/s** for bfloat16 once the weight is
    big enough, against 332 GB/s for a plain elementwise read+write. What it also
    shows is a fixed **~33 us per op** eagerly: a 0.37 MB weight and a 4.42 MB
    weight take 0.033 and 0.061 ms. Our projections are 1-4 MB per device --
    4-bit and sharded four ways -- so eagerly they sit in the floor-dominated
    regime and *look* like 73 GB/s.

    Inside a trace that floor is **1.4 us**, measured by injecting ops into the
    capture and reading the slope (`traced_step_op_floor_check.py`): 2328 extra
    ops move a 210 ms step by 3.3 ms. So 6095 ops account for ~9 ms of it, and
    invariant 19 stands -- the traced step is real device work, not per-op
    overhead. An arithmetic coincidence (6095 x 28 us = 171 ms against a 173 ms
    step) suggested otherwise and did not survive being measured.

    And "a GEMV runs at a thirty-second of a GEMM" is about FLOPs, not bandwidth.
    A GEMV has an arithmetic intensity of ~1 op/byte where a 32-row GEMM has ~32,
    so at the *same* bandwidth it does 1/32 the arithmetic. Saturating DRAM is
    the ceiling for a GEMV, not a shortfall from one. So the launch floor on small
    weights is the one real loss.

    > **The bfloat4_b "loss" was a bad framing, corrected.** Reading 273 GB/s
    > against bfloat16's 393 compares *bytes*, and bfloat4_b moves 3.6x fewer of
    > them for the same matrix. In elements it is the fastest of the three (493 G
    > against bfloat8_b's 342 and bfloat16's 196), and in wall clock the same
    > 2560x32768 matmul is **0.170 ms at bfloat4_b against 0.427 at bfloat16** --
    > 2.5x faster, not 30 % worse. What caps bfloat4_b is elements per second, not
    > bytes: the unpack and math pipeline, which is hardware. There is nothing to
    > recover.

26. **HiFi4 is free, so stop treating fidelity as a speed knob.** The unpack of a
    quantised weight happens in the core, fed from L1, and the math passes behind
    it are entirely hidden by the memory transfer. Across bfloat4_b, bfloat8_b
    and bfloat16, at a floor-bound size and a bandwidth-bound one, LoFi through
    HiFi4 all take the same time to within 2 % (`math_fidelity_check.py`) -- e.g.
    bfloat4_b at N=32768: 0.170 / 0.170 / 0.171 / 0.173 ms. And the fidelity is
    not cosmetic even on a 4-bit weight: against the dequantised weight, LoFi is
    7.4e-03 and HiFi3/HiFi4 4.0e-03. Blanket HiFi4 costs nothing and buys
    accuracy, which is the right default and now a measured one.

27. **The bandwidth is being used; it is being used on an output nobody wants.**
    The arithmetic that exposes this: four p150a have aggregate DRAM bandwidth in
    a top consumer GPU's class, and a decode step genuinely needs ~1 GB per
    device -- the dense projections plus the ~11.5 experts of 512 that routing
    selects. At the 273-393 GB/s these boards demonstrably stream that is a couple
    of milliseconds. The step takes 173.

    `sparse_matmul_reads_check.py` locates the gap. Holding the weight fixed and
    varying only the mask's non-zero count on the down projection:

        nnz      1      2      8     32    128    512
        ms   1.052  1.052  1.060  1.092  1.222  1.743

    The sparsity is doing something -- 512 non-zeros cost 1.66x what 1 does -- but
    it sits on a **1.05 ms floor that does not depend on the mask at all**, where
    a single expert needs 0.23 MB of the 118 MB present.

    That floor is the *output*. The op's contract is `[1, E, M, K]`, so it writes
    512 slots x 32 padded rows x 2560 = **84 MB a layer**, of which the ~11.5
    selected experts at one real row are 59 KB -- a factor of ~1400. 84 MB /
    1.05 ms is 80 GB/s, which is write bandwidth being spent in full, and 48
    layers of it is the 51.7 ms the ablation attributes to the down projection.

    So the machine is not slow and the bandwidth is not idle. Decode writes 4 GB
    per token per device to carry 3 MB of answer. `_combine` (`022`) fixed the
    *consumer* of that tensor; the producer still writes all of it, and the only
    lever that shrinks it is fewer experts per device -- expert-axis sharding,
    priced at 2.06x on the two matmuls in `022` and blocked on a cache rebuild.

28. **A device call on the eager prefill path is worth ~57 us, and its size does
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

## 4c. Why the `moe_compute` port hangs: CCL-on-Blackhole is not functional

Found by reading upstream rather than by bisecting the device, after four runs
that each wedged the card.

**Symptom.** The first `all_to_all_dispatch_metadata` + `moe_compute` pair
appears to succeed -- it returns five tensors with plausible shapes -- and then
the *next device operation of any kind* hangs. Not the second `moe_compute`:
a bare `ttnn.add` hangs too, at ~120 % CPU with the log frozen. The reason the
first pair "succeeds" is that ttnn is asynchronous, and reading a shape does not
synchronise. Nothing had actually run yet. The first genuine synchronisation
point is where the already-broken dispatch surfaces.

**Cause.** `all_to_all_dispatch` is known-broken on Blackhole upstream:

- tenstorrent/tt-metal#27859, *All to all dispatch not functional on blackhole* --
  "both 2D and 1D fabric fail, either with a hang or with an error reporting that
  the node does not contain any neighbours", specifically on 4-device configs.
- tenstorrent/tt-metal#30030, the same for all-to-all combine.
- The official single-card test says so in its own header: it exists to be the
  regression net "without requiring a 6U Galaxy host **or working CCL-on-BH**".

So the hang is not in our wiring. Every multi-device CCL path through this op is
unsupported on our hardware, and #41827's note that only DeepSeek V3 and Kimi K2.5
are tested end-to-end means our shapes are untested territory besides.

**What was ruled out along the way**, so nobody re-runs these:

- *Ring size.* `effective_matmul_ring_size()` returns a hardcoded 8 without
  querying the device, while the op auto-detects from the live DRAM-bank count
  ("7/8 on BH"). A mismatch would corrupt both the weight layout and the drain
  core. Measured on our p150a: `get_optimal_dram_bank_to_logical_worker_assignment`
  gives **8**, so ring 8 and `output_width_shard_dim` 4 were right all along.
- *Output lifetime.* Freeing the outputs first changes nothing. (Note for later:
  slots 3 and 4 share a backing buffer -- deallocate 0, 1, 2, 4 only, never 3.)
- *Input consumption.* Re-dispatching each round changes nothing.
- *`compute_only` being a partial mode.* Full mode hangs earlier, not later.

**The supported shape of the port.** `tests/ttnn/nightly/unit_tests/operations/
experimental/test_moe_compute_single_card.py` runs the op with no CCL at all:
it builds the four inputs directly on device -- `gen_sparse_buffer_and_indices`
exists precisely to simulate "output from all_to_all_dispatch" -- and passes
`cluster_axis=None, topology=None, num_links=None, mux_core_range_set=None,
optional_cross_device_semaphore=None`.

That fits our layout rather than fighting it. We already shard the 512 experts
across the four cards (`Shard.EXPERT`, 128 local each) and all-reduce afterwards,
so there is nothing for a device-to-device dispatch to do: each card needs only
its own tokens against its own experts. Build the sparse buffer locally, call
`moe_compute` per device with `cluster_axis=None`, combine locally, and keep the
existing all-reduce. No fabric, no semaphore, no drain core.

INVARIANT 28 (corrected -- see §4c.3): what hangs is `all_to_all_dispatch`,
not CCL. `ttnn.all_reduce(cluster_axis=1, topology=Linear)` runs in every layer
of the live model, and `moe_compute` with a `cluster_axis` runs too once given
mux cores. The general claim first recorded here was wrong. What does hold is
the failure *mode*: a broken CCL op does not error, it hangs at the next
synchronisation point, which makes it look like a bug in whatever ran after.

### 4c.1 What `moe_compute` costs, and the two constraints that shape the port

Measured on the local path (`scripts/dev/moe_compute_sweep.py`, one row count per
process -- an unsupported one aborts the interpreter rather than raising, so a
single in-process loop loses every measurement after the first bad one).

| rows | per layer |
|------|-----------|
| 1    | aborts in `CircularBufferConfig::set_page_size` |
| 32   | **1.47 ms** |
| 64   | 2.22 ms |
| 128  | **2.50 ms** |
| 256  | rejected, L1 circular buffers over budget |
| 512  | rejected: "grow to 2532240 B which is beyond max L1 size of 1572864 B" |

Against the 2.78 ms/layer the `sparse_matmul` path costs at M=1, so the op is
worth having even though decode has to pad 1 row up to 32.

**Constraint 1: the row count must be 32..128.** Below a tile row the op aborts;
above ~128 its circular buffers exceed L1. Decode pads to 32. Prefill's 512-row
chunks split into four 128-row calls (4 x 2.50 = 10.0 ms/layer), which is worth
re-measuring against the current path before committing to it.

**Constraint 2: the fused combine is out of reach.** `compute_only=False` looked
like it would replace our `_combine` too, and upstream's single-card test does
run it with `cluster_axis=None`. This build disagrees:

    TT_FATAL: moe_compute(compute_only=false) requires cluster_axis to be provided
    (moe_compute_device_operation.cpp:473)

and any `cluster_axis` pulls in the CCL path that hangs this box (invariant 28).
So we stay on `compute_only=True` and keep our own combine -- which is no loss:
it is one matmul against the router weights at M=1 and measured *more* accurate
than the alternative it replaced (invariant 25).

INVARIANT 29: `moe_compute` is usable here only as `compute_only=True` with
32..128 rows and `cluster_axis=None`. Both bounds are hard: one aborts the
process, the other fails compilation.

Still open before this can replace `moe_block`: output slot 4 (`matmul_output`,
shape `[110, 2, 32, 2560]` at M=32) is sharded per core, and nothing yet maps
its (core, slot) layout onto the `[1, E, M, K]` our `_combine` consumes. That
mapping, and a numerics check against float64, are what remain.

### 4c.2 Why the port stops here: `compute_only` has no consumable output

Attempting the integration killed it. `matmul_output` (slot 4, `[110, 2, 32, 2560]`)
is not the per-expert result for this device's experts. The `2` is a **double
buffer**, and after the op has walked all `experts_per_device` experts only the
last two survive in it. Upstream's own validator says so and checks nothing else:

    reshape_func = functools.partial(
        prepare_output_tensor_from_combine_writer,
        experts_per_device=2,  # always 2 for double buffer
        ...)
    # Calculate which experts are still in the double buffer
    experts_to_check = [(experts_per_device - 2, 0), (experts_per_device - 1, 1)]

With 128 local experts we would get experts 126 and 127 and nothing else. The
per-expert results are consumed in place by the fused combine stage as they are
produced; `compute_only=True` simply skips that stage and lets them be
overwritten. It is a mode for benchmarking the compute kernels, not a building
block -- which is consistent with what it is used for upstream (§4c: a hermetic
regression net for the kernels).

So the only output that carries a whole layer is the combine, slot 5, and that
needs a `cluster_axis`, which needs CCL, which hangs this box (invariant 28).
The two constraints close on each other and there is no path between them.

There are three modes, not two -- worth stating exactly, because the third is
newer than the build we run:

    enum class MoEComputePath : uint8_t { FullCcl = 0, FullLocal = 2, ComputeOnly = 1 };

- `FullCcl` (compute_only=false + a cluster_axis): matmul plus fused
  selective_reduce_combine over fabric. The production path, and the one that
  hangs here.
- `FullLocal` (compute_only=false, cluster_axis=None): matmul plus a *local*
  combine, no fabric, no semaphores. Returns 6 tensors like FullCcl. Guarded by
  `TT_FATAL(num_devices == 1, ...)` -- 1x1 meshes only.
- `ComputeOnly`: no combine cores, no fabric; 5 tensors, and see above for why
  the 5th is not usable.

Our build predates `FullLocal`. The failure we saw quoted
`moe_compute_device_operation.cpp:473`, which on current main is the 1x1-mesh
assert; on ours that line was still the unconditional "requires cluster_axis".

INVARIANT 30 (corrected -- see §4c.3): `compute_only=True` has no consumable
output, and that part stands. But the conclusion drawn from it here -- that the
port was closed -- did not: `FullCcl` runs on this box and its combine output is
exactly what we need.

### 4c.3 The port is open: FullCcl works

Two earlier conclusions in this section were wrong, and the correction matters
more than either of them.

CCL is not broken here. The evidence was in our own model the whole time:
`model.py` calls `ttnn.all_reduce(t, cluster_axis=1, topology=ttnn.Topology.Linear)`
in every one of the 48 layers, and decode runs. Reading two upstream issues about
all-to-all and generalising them to "no CCL on Blackhole" was unfounded -- both
issues are in fact closed, and #30030 records Quietbox and Deskbox passing.

What hangs is `all_to_all_dispatch_metadata` specifically, and the reason is
almost certainly topology. It takes no topology argument, so it uses whatever
`get_usable_topology()` resolves, and that helper marks any tensor spanning a
full mesh row as Ring. Four p150a cards are physically a *line*, so the traffic
is forwarded across a wrap edge that does not exist and waits forever. The op's
own source says so and names the workaround:

    // BH LB callers must pass topology=Linear explicitly; the kernel-side
    // `Topology` template ... multicast through the line-aware code path.

We do not need that op anyway -- the inputs are built locally (§4c).

And the combine kernel supports both topologies, so the guess that MoE was
Ring-only is not it either:

    TT_FATAL(resolved_topology == Topology::Linear || resolved_topology == Topology::Ring,
             "moe_compute: combine kernel only supports Topology::Linear or Topology::Ring")

**FullCcl, fed by locally-built inputs, measured 1.86 ms/layer at M=32** (6
outputs, 1578 ms on the first call for JIT). Three things it needs that the
local path does not:

- `mux_core_range_set` is **not** optional. Passing None gets you
  "Not enough mux cores! Needed: 1 (num_links=1 * neighbors.size()=1),
  Available: 0". `((1,1),(3,3))` is the upstream default.
- The same mux set must go to `get_moe_tilize_drain_core`, which places the
  drain core around the mux workers.
- A global semaphore, and `topology=Linear` passed explicitly.

INVARIANT 31: on this box `moe_compute` runs as FullCcl -- `compute_only=False`,
`cluster_axis=1`, `topology=Linear`, a mux core range set, and a global
semaphore -- with inputs built locally rather than dispatched. Never call
`all_to_all_dispatch*`; it cannot be told the topology and hangs on a line mesh.

Open before this replaces `moe_block`: the combine output is
`[k, tokens_per_device, hidden]`, **token-sharded across the mesh** -- M=32 came
back as `(10, 8, 2560)` per device. Our hidden state is replicated and reduced
with all-reduce, so the data flow changes: sum over k, then all-gather on the
token axis instead. That plus a float64 numerics check is what remains.

### 4c.4 The integration contract, read off the reference flow

Taken from `tests/nightly/tg/ccl/moe/test_moe_compute_6U.py`, which the module
docstring names as the full flow. It should have been read end to end before any
of §4c was written; grepping it piecemeal is what produced the wrong turns above.

**It traces.** The reference captures the op with `begin_trace_capture` /
`execute_trace`, which is the thing that matters most here -- our decode is a
single traced replay, and an op that could not be captured would be useless
regardless of its kernel time.

**One packed weight tensor for every layer.** `layer_id` selects the layer
inside it, so the packers take `num_layers` and all 48 layers' experts live in
one DRAM-sharded tensor per projection. Use
`ttnn.experimental.get_weight_mem_configs(mesh_device, num_layers=,
experts_per_device=, hidden_size=, intermediate_size=, has_bias=)`, which
returns `.w0_w1` / `.w2` -- the intended API, not the Python
`moe_compute_utils.get_weight_mem_configs` that takes shard maps.

**The semaphore belongs on the combine cores, not the whole grid:**

    output_shard_cores = ttnn.experimental.get_moe_combine_cores(
        mesh_device, output_height_shard_dim, output_width_shard_dim,
        hidden_size, mux_core_range_set=mux_core_range_set)
    combine_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in output_shard_cores])
    combine_barrier_semaphore = ttnn.create_global_semaphore(mesh_device, combine_core_range_set, 0)

`mux_core_range_set` feeds three placement helpers -- `get_moe_tilize_drain_core`,
`get_moe_combine_cores`, `get_moe_worker_mcast_bounding_box` -- so it has to be
decided first and passed to all of them. Upstream default `((1,1),(3,3))`,
`num_links=4` (we measured with 1).

**L1 is the binding constraint for a 48-layer model.** The reference says it
outright:

    # NOTE: we're extremely tight on L1 for a single invocation of the op.
    # When running multiple layers, all inputs go to DRAM and get moved to L1
    # per-layer via to_memory_config.

So per layer: inputs `to_memory_config` DRAM->L1, run, deallocate the L1 inputs,
move outputs back to DRAM, free the L1 outputs. Outputs unpack as
`(per_expert, activation, e_t, _, matmul, combine)` -- slot 3 skipped, which is
the shared-buffer rule from §4c.2 seen from the other side.

**The combine output is preallocated per layer**, `[k, total_tokens, hidden]`
with `ShardTensorToMesh(dim=1)`, and passed as `optional_output_tensor`.

INVARIANT 32: read the reference flow a docstring points at before wiring
against the op, not after. Every wrong turn in §4c -- "CCL is broken here",
"the port is closed", the missing mux cores -- was already answered in
`test_moe_compute_6U.py`.

### 4c.5 FullCcl validated, and the weight path that makes it work

Numbers, all at M=32 on the 1x4 mesh: **1.65 ms/layer** eager, **57.7 ms for all
48 layers inside a trace** (1.20 ms/layer; tracing buys only 1.03x over eager,
so this is device-bound, not dispatch-bound). Against a whole traced decode step
of 109.2 ms today.

Correctness took three fixes, each of which was already written down upstream:

**1. The Python packers are not the production path.** `moe_compute_utils`
describes itself as "executable specifications", and using them produced a
combine output that was 96 % non-finite. The real path is on-device:

    _prep = ttnn.experimental.prepare_w0_w1_tensor_for_moe_compute(w0, w1, L=, E=, K=, N=)
    host  = ttnn.experimental.quantize_weights_via_host(_prep, dtype=ttnn.bfloat4_b,
                                                        memory_config=None)
    dev   = ttnn.to_device(host, mesh, memory_config=wmc.w0_w1)

with `wmc = ttnn.experimental.get_weight_mem_configs(mesh, num_layers=,
experts_per_device=, hidden_size=, intermediate_size=, has_bias=)`. Raw weights
go up replicated to DRAM first; `memory_config=None` on the quantise is what
makes it hand back a host tensor.

**2. `bfloat4_b` is mandatory.** Quantising to `bfloat16` instead is not a
precision trade -- the output comes back all zeros. Fine for us, since the
experts are 4-bit anyway.

**3. The combine applies no router weights.** `compute_matmul_golden` upstream
takes no scores argument at all, and the combine golden assigns `contrib`
unweighted. Slot k holds the raw output of the token's k-th selected expert; the
score weighting is the model's job. Applying it host-side moved cosine from
0.728 to 0.980.

INVARIANT 33: the combine output is `[k, tokens, hidden]`, token-sharded across
the mesh, holding *unweighted* per-expert results. The MoE output is
`sum_k score[t,k] * out[k,t]` after concatenating on the token axis -- and it is
already summed across devices, so it needs an all-gather on tokens, **not** the
all-reduce our current path uses.

Residual error against float64 with 4-bit weights: max abs 2.35e-02 on values up
to 1.19e-01, cosine 0.980. Whether that is acceptable is an end-to-end question
(quality today is 102/127 top-1, NLL 0.887), not one this harness can settle,
because our own `sparse_matmul` path is 4-bit too and has never been measured
this way on the same inputs.

## 4d. The real bottleneck is read amplification, and it needs a custom kernel

Measured from the actual weight files, not estimated:

| | |
|---|---|
| expert weights per layer | 1.8 GB (459 MB per device) |
| `moe_compute` at 1.20 ms/layer | ~382 GB/s per device -- i.e. DRAM bandwidth |
| what decode actually needs | top-10 = 35.8 MB/layer, **2.0 % of the bytes** |
| floor if only those are read | 0.033 ms/layer -> **1.6 ms for all 48 layers** |

`moe_compute` is dense over the expert axis: it reads all 512 experts every
layer and saturates DRAM doing it. That is the right design for throughput,
where a large batch activates most experts, and the wrong one for batch-1
latency, where 10 of 512 are wanted. No amount of tuning removes a 50x read
amplification.

Which settles the ceiling of that whole avenue: **the MoE alone costs 57.7 ms
while an RTX 5090 emits an entire token in 32.6 ms.** Even with everything else
free, this path cannot reach the target. llama.cpp is fast there because it
gathers the selected experts and does a small dense matmul.

INVARIANT 34: for batch-1 decode the MoE is bound by *how many expert bytes get
read*, not by kernel efficiency. 2 % of the weights are needed. Any design that
touches the whole expert axis is already 50x off the floor, however well it runs.

### 4d.1 A custom kernel is possible from Python, and trace-compatible

No tt-metal rebuild is needed. `ttnn.generic_op(io_tensors, program_descriptor)`
runs user kernels, described entirely from Python:

- `ttnn.KernelDescriptor(kernel_source, source_type=FILE_PATH | SOURCE_CODE,
  core_ranges, compile_time_args, named_compile_time_args, defines,
  runtime_args, common_runtime_args, config, compiler_include_paths)`
- `ttnn.CBFormatDescriptor(buffer_index, data_format, page_size, tile)`
- `ttnn.ComputeConfigDescriptor(math_fidelity=...)` -- HiFi4 costs nothing here
  (invariant 12), so use it
- `ttnn.ProgramDescriptor(kernels, semaphores, cbs)`, with
  `custom_program_hash` for the program cache, and `MeshProgramDescriptor` when
  devices need different programs
- output tensors are pre-allocated and passed last in `io_tensors`

The trace objection does not apply. A trace fixes the program and the buffer
addresses, not the data in them. Pass the top-k indices as an **io_tensor**
rather than as host-set runtime args, and the kernel recomputes expert addresses
from that tensor on every replay -- so one capture serves any routing. This is
the property that makes sparse expert reads and tracing compatible, and it is
why the dense expert axis was never actually forced on us.

`moe_compute`'s own kernels ship in the wheel and are the model to work from:
`.../operations/experimental/ccl/moe_compute/device/kernels/{dm0,dm1,compute,
tilize_*}.cpp` plus `moe_ring_common.h`. Its `dm0.cpp` bank-run loop is exactly
the dense read to make index-driven.

Staging, so the premise is tested before the arithmetic is built on:

1. A read-only `generic_op`: take the index tensor, read just those experts'
   tiles, reduce to a checksum. If it lands near 0.033 ms/layer the premise
   holds; if it does not, nothing further is worth building.
2. Then gate/up + SwiGLU + down, scores applied per expert, accumulating into
   the hidden vector.

## 4e. moe_compute is rejected: it is slower than what we already have

The ablation harness settles it. Each part replaces one component with an
identity of the same shape, so the delta against `none` is that component's
device time inside the traced step and nothing else.

| ablate | median | delta |
|--------|--------|-------|
| none | 145.66 ms | -- |
| `expertffn` | 114.83 ms | **30.83 ms** |
| `moe` (whole block) | 102.73 ms | **42.93 ms** |

`expert_ffn` -- exactly the part `moe_compute` would replace -- costs **30.8 ms**
for all 48 layers. `moe_compute` measured **57.7 ms** for the same work. Adopting
it would cost us 27 ms a token.

So invariant 34's arithmetic was right about the floor but wrong about who was
far from it: at 0.64 ms/layer our `sparse_matmul` path is already ~2x better
than ttnn's fused op, which means it is *not* reading the whole expert axis.
The 50x figure was measured against `moe_compute`, not against us.

INVARIANT 35: do not port the MoE to `ttnn.experimental.moe_compute`. It is
2x slower than the hand-rolled `sparse_matmul` path for batch-1 decode
(57.7 ms vs 30.8 ms per token over 48 layers). Everything in section 4c is
still true and was worth learning -- it is simply the wrong tool at batch 1,
being dense over the expert axis where we are not. Revisit only for prefill,
where many rows make the dense read pay for itself.

### 4e.1 MoE is 29 % of the step; the other 103 ms is where the race is

The same table says the thing that matters more. The whole MoE block is 42.9 ms
of a 145.7 ms step, so **non-MoE work is 103 ms**. Driving the MoE to *zero*
would still leave 103 ms, against the 32.6 ms an RTX 5090 spends on an entire
token. No MoE change alone can close that.

The custom kernel is still worth it -- 30.8 ms against a 1.6 ms floor is 19x on
the table, and it is the single largest component. But it is not sufficient, and
anything that claims a 3x from MoE alone is arithmetically wrong.

CAVEAT, unresolved: this harness's baseline is 145.66 ms, while decode was
measured at 109.2 ms earlier in the session (§5.7 work). The two setups differ
somehow -- harness vs engine configuration -- and the numbers must not be mixed
until that is explained. All deltas above are internally consistent, being from
one harness in one session.

## 4f. The QSA selection is skippable below the budget: 146.2 -> 110.8 ms

The ablation harness, run component by component on the traced step, put the
cost somewhere nobody had looked:

| ablate | median | delta |
|--------|--------|-------|
| none | 146.17 ms | -- |
| `indexer` (all of it) | 109.91 ms | **36.8 ms** |
| `idxtopk` (its topk only) | 123.84 ms | **22.3 ms** |
| `idxscatter` | 144.36 ms | 1.8 ms |
| `noselect` (skip selection, keep the cache) | **110.82 ms** | **35.4 ms** |

`ttnn.topk` is 22.3 ms of a 146 ms step -- 15 % of decode -- because it takes
k=512 out of `max_blocks`=1024, which is half a sort, in each of the twelve QSA
layers. The scatter that the call-site comment worried about is 1.8 ms.

**And below `indexer_budget` none of it does anything.** Eligible blocks at
position p number `p // ratio`, so while `p < budget` there are fewer than
`indexer_topk` of them: `topk` returns every visible block, the -inf padding is
dropped by the `index <= p` filter, and the mask is exactly plain causal
attention. Skipping it is not an approximation.

What made this a change rather than an observation is that the block cache still
has to be filled on the skipped steps, or the first selecting step reads
garbage. So `_indexer_select` is split: `_indexer_update` (k_proj, ring, pool,
rope, `paged_update_cache`) always runs; `_indexer_mask` (q_proj, the per-head
scores, topk, scatter, repeat) runs only when `selection_active`.

`selection_active` is a plain Python bool, so a trace bakes in the regime it was
captured under -- which is why crossing the budget takes a new capture, not a
branch. `TTEngine._enable_selection` does the release/snapshot/capture/restore
that `_reprefill` already established, ~2.6 s, once per sequence that gets that
far. Sequences shorter than 2048 tokens never pay it.

INVARIANT 36: the QSA selection is exact *and* skippable below
`indexer_budget`, and the engine starts with it off. Two things keep that
honest: `_indexer_update` must keep running while it is off, and the trace must
be recaptured on the way past the budget. `tests/test_indexer_selection.py`
pins the arithmetic -- eligible blocks never outnumber the selection slots below
the budget -- so a change to the budget or the compression ratio cannot break
the fast path quietly.

Verified empirically, not just argued: `scripts/dev/selection_skip_equivalence.py`
opens the model above the budget so the indexer engages, greedily decodes the
same prompt with the selection on and off, and compares. Token ids identical,
hidden states **bitwise** identical, max abs diff 0.000e+00.

Where the step goes now, on this harness: MoE 42.9, QSA 41.8 (36.8 of it the
indexer), deltanet 16.4, shared 5.5, sdpa 5.1, allreduce 3.8, PLE 2.8.

## 4g. Stage 1: index-driven sparse expert reads work, at 1.87 ms for 48 layers

`scripts/stage1_expert_gather_bw.py` plus `scripts/kernels/expert_gather_checksum.cpp`
-- one data-movement kernel under `ttnn.generic_op`, no compute kernel, no
semaphores -- reads only the selected experts' tiles, driven by an index tensor
rather than host runtime args. Per device, gate|up and down summed:

| K | ms/layer | 48 layers | vs dense | ideal |
|---|----------|-----------|----------|-------|
| 0 (launch floor) | 0.0117 | 0.56 ms | -- | -- |
| **3** (the real per-device budget, ceil(10/4)) | **0.0390** | **1.87 ms** | 0.0274 | 0.0234 |
| 10 | 0.1171 | 5.62 ms | 0.0823 | 0.0781 |
| 128 (dense control) | 1.4236 | 68.33 ms | -- | -- |

246-389 GB/s per device, so it is bandwidth-bound rather than sitting on the
launch floor, and the ratio against the dense control lands within 17 % of the
ideal fraction. The read amplification is gone.

Against **30.8 ms** for the whole of `expert_ffn` today (§4e), reading just the
bytes the routing asks for costs **1.87 ms**.

The measurement is hardened, because four critics went at it before any device
time was spent and every one of them found something:

- The `ASSERT` guarding a bad expert index expands to `while (1) { ; }` under
  WATCHER_ENABLED. The safety check would have hung the card. Clamped instead.
- `tt::CBIndex` is declared in a header `dataflow_api.h` does not include --
  a compile error for an enum whose values `CBFormatDescriptor` already fixes.
- `generic_op` hashes `compile_time_args` by value but `runtime_args` only by
  *count*, so all three K values shared one program: the sweep would have
  measured K=3 three times and reported it as a scaling curve. `k_sel` now
  rides along as a trailing compile-time arg.
- `sum_data != 0` could not distinguish landed bytes from whatever L1 held.
  The landing slots are poisoned with `0xDEADBExx` before each read, `verify()`
  rejects that sentinel, and it now actually performs the per-device
  disagreement check its docstring had been claiming.

`sum_tid` -- every core folding `tile_id` for each page it reads, against a
closed form the host computes independently -- matched exactly at every K, which
is what makes "only the intended tiles were read" a measurement rather than an
assumption.

INVARIANT 37: `ttnn.generic_op` runs user kernels with no tt-metal rebuild, and
an index tensor passed as an io_tensor keeps expert selection data-dependent
under trace replay. Stage 1 proves the read side. What is unbuilt is stage 2:
gate/up, SwiGLU, down, and the router weighting, on top of these reads.

There is no existing ttnn op to use instead. `ttnn.gather` is element-wise --
its device op requires output shape == index shape and its writer reads the
whole tile row from DRAM, indexing only inside L1, which is precisely the
amplification being removed. `ttnn.tosa_gather` is a thin wrapper over it.
`ttnn.sparse_matmul` skips reads by a *mask*, walking E in order without
permuting, and that is the 0.64 ms/layer path we already run.

## 5. Where the decode step actually goes: small-N matmuls at 5-7 % of bandwidth

Component ablation was the wrong frame and cost most of a session to disprove.
It accounts for 113.5 ms of a 146.2 ms step -- attention 59.6, MoE 42.9, PLE
2.8, all_reduce 3.8, reinject 1.9 -- and every hunt for the remaining ~33 ms
inside a component came back small. Then two measurements reframed it.

**The host is not the problem.** `step_host_split.py`: whole step 144.81 ms,
`execute_trace` alone 141.70, `_fill_inputs` alone 2.56. The host is 1.8 %.
`model.embed` and the n-gram lookup, which read host memory, are 0.01 ms.

**The device is running at 5 % of its bandwidth.** Per device per token the
model must read 2.92 GB: dense 2.18 (the `hc_*` and router weights carry no
`.devN` suffix -- they are *replicated*, so they do not divide by four; an
earlier roofline in this file divided them and was wrong), top-10 experts 0.52,
head 0.22. In 141.7 ms that is **20.6 GB/s**, against 273 GB/s measured on
bfloat4_b and 388 GB/s reached by the stage-1 kernel.

`gemv_saturation.py` prices the real weight shapes inside a trace, and the
pattern is entirely about the output width N:

| shape (K x N) | GB/s | % of 388 |
|---------------|------|----------|
| lm_head 2560 x 62080 | 275.8 | 71 % |
| hc_up 640 x 10240 | 167.9 | 43 % |
| attn_qkv 2560 x 4608 | 128.3 | 33 % |
| attn_output 6144 x 2560 | 71.2 | 18 % |
| ssm_out 4096 x 2560 | 70.2 | 18 % |
| **hc_down 10240 x 640** | **26.6** | **7 %** |
| **router 2560 x 512** | **19.7** | **5 %** |

N sets how many output tiles there are to spread across cores, so a narrow
output starves the grid regardless of how much weight has to be read. And M=1
and M=32 time *identically* at every shape, which is the 32-row tile padding
showing up exactly as expected: decode wastes 32x of the compute, but since a
GEMV has arithmetic intensity ~1 op/byte that is not what binds -- the schedule
is.

The model is full of narrow-output matmuls. `hc_down` (10240 x 640, replicated)
runs **96 times a token**, two per layer inside `gated_residual_mix`, and
ablating it alone costs **18.77 ms** of the step -- consistent with 96 x 0.139 ms
standalone.

INVARIANT 38: decode is bound by matmul scheduling, not by bandwidth or compute.
Achieved bandwidth tracks the output width N: ~270 GB/s at N=62080, ~20 GB/s at
N=512. Any fix that does not widen the effective output or split the reduction
space across cores will not move the step, however few bytes it reads.

CAUTION on method: ablating a whole `gated_residual_mix` measured 2.54 ms while
ablating just its first linear measured 18.77 ms, which cannot both be right.
The stand-in returned a `ttnn.slice` view where the real function returns a
fresh tensor, and the downstream matmuls appear to have paid for that -- its
samples spread 135-145 ms where every other ablation held within 2 ms. Prefer
the narrower ablation, and distrust any delta whose spread is wide.

### 5.1 DRAM sharding does not fix the narrow-output matmuls either

Invariant 23 called `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` a
wash, but it sampled only N >= 2048 -- the shapes already running at 70-128
GB/s. Re-run with the narrow ones added:

| shape | interleaved | dram-sharded | ratio |
|-------|-------------|--------------|-------|
| qsa q\|gate 2560x3072 | 0.060 ms | 0.053 | 1.12x |
| qsa out 1536x2560 | 0.033 | 0.036 | 0.94x |
| deltanet qkv 2560x2048 | 0.048 | 0.049 | 0.99x |
| **router 2560x512** | 0.043 | 0.053 | **0.81x** |
| **hc_down 10240x640** | 0.139 | rejected (TT_FATAL, tensor spec) | -- |
| **hc_up 640x10240** | 0.051 | rejected | -- |

So the config that exists "for very narrow tensors stored in DRAM" is *worse*
on the narrowest shape we have and refuses the two that cost the most. Invariant
23 stands and now extends: DRAM sharding is not the lever at any N.

`hc_down` reads 0.139 ms here and 0.1387 in `gemv_saturation.py`, so that number
is solid: 96 calls a token is >= 13.3 ms, and ablating it measured 18.77.

INVARIANT 39: `gated_residual_mix` is the largest single addressable cost in the
step. It runs 96 times a token (twice a layer) and reads ~9.4 MB of replicated
weight per call -- hc_norm 1.3, hc_down 3.4, hc_up 3.4, hc_inject 1.3 -- which is
902 MB/device/token, 3.3 ms at 273 GB/s against the ~19 ms it costs now. The gap
is scheduling, not bytes: 10240x640 is too narrow to fill the grid and DRAM
sharding is refused for it. Fusing the whole function into one `generic_op`
kernel that streams all four weights is the shape of the fix, and the tooling is
proven (invariant 37).

Do NOT reach for sharding the `hc_*` weights across devices instead: it would cut
each device's read 4x but add an all_reduce to all 96 calls, and at 3.77 ms per
48 all_reduces today that is ~7.5 ms of new cost against ~10 ms saved. Measure
it before believing either number, but it is marginal by construction and it
does nothing about the narrow output.

### 5.2 The roofline, corrected again; and `sparse_matmul` zero-fills its whole output

Two corrections to numbers earlier in this file, both from a 15-agent audit that
worked from `plan.py`'s residency rules rather than from a division.

**The floor is 9.4 ms, not 5.2 and not 7.5.** Dense weight read per device per
token is **2.960 GB**, not 1.285 (which divided replicated tensors by four) and
not the 2.18 I estimated by hand from one layer's files. Everything with no
`.devN` suffix is REPLICATE: attn_q 401 MB, attn_qkv 451, router 252, hc
down/up 668, hc norm/inject 252, shared expert 251, attn_output 201. With the
lm head 0.169 GB and top-10 experts 0.516 GB that is **3.645 GB/device/token ->
9.39 ms at 388 GB/s, 13.35 at 273**.

So 110.8 ms is **11.8x the floor**, and 32.6 ms is **3.5x the floor**. The target
is demanding rather than comfortable, and every earlier framing in this file
that called it roomy was working from the wrong denominator.

**`ttnn.sparse_matmul` unconditionally zero-fills its entire output**, verified in
`sparse_matmul_device_operation.cpp`: `create_output_tensors` calls

    output_tensor = ttnn::zeros_like(output_tensor, ..., std::optional<Tensor>(output_tensor));

in **both** branches -- lines 268-278 and 286-295 -- so supplying your own output
tensor does not escape it. Our two calls per layer allocate
`[1, 128, 32, 1280]` and `[1, 128, 32, 2560]` in bfloat16, so the fill writes
**31.5 MB a layer, 1.51 GB/device/token**, independently of the sparsity mask.
At the fill rate measured on this box that is ~18.9 ms of `expert_ffn`'s 30.83.

This corrects INVARIANT 27, which blamed the matmul's writer for the [1, E, M, K]
waste. The writer is innocent: it `continue`s on a zero sparsity page and never
touches unselected slots. The cost is the *allocation*, and the fix is fewer
output bytes rather than fewer experts per device.

`dtype` is a bound kwarg on `sparse_matmul`, so halving the output would take
~10 ms off the step for one keyword in two places -- and it is **not the move**,
because the first call's output feeds SwiGLU and the standing constraint on this
work is not to spend accuracy. The principled fix is the same stage-2 kernel that
handoff 4g's read side already argues for: writing `[1, k_sel, M, N]` instead of
`[1, 128, M, N]` removes the fill entirely, 42x smaller, with no precision cost.

INVARIANT 40: `ttnn.sparse_matmul` costs 1.51 GB/device/token in zero-fill alone,
before it reads a single weight. Any accounting of the MoE that starts from the
weights is wrong by ~19 ms. Do not try to dodge it by passing
`optional_output_tensor` -- that path re-zeros too.

Other findings from the audit worth having, none of them measured by me yet:

- The seam semantics were verified against the harness source, and my earlier
  arithmetic double-counted. `moe` patches `moe_block` **only**, so
  `shared_expert` (5.47) and the 48 `_moe_block` all_reduces sit *outside* the
  42.93. `attn` patches both attention kinds, so its 59.57 **contains** the
  DeltaNet 16.35. Measured deltas sum to **74.64 ms of 110.8**, leaving 36.16
  unattributed -- not the ~16 I had been quoting.
- The hyper-connection block (97 `gated_residual_mix` + 96 `reinject`,
  ~2.30 GB/device/token) is modelled at 22-31 ms and is probably the largest
  single item in the model. My own measurement puts `hc_down` alone at 18.77,
  which is consistent, and section 5.1 already names it.
- `shared_expert`'s 5.47 ms includes 3.19 for a single N=1 sigmoid-gate column:
  328 KB read on **one** core to use 10 KB of weight.
- The LM head, argmax and host gathers are **not** in the 110.8 -- the harness
  times `dec.step` only. I measured them separately at 2.00 ms in situ.
- The traced `step_n` curve fits **t ~= 162 + 12.8k**, so k=8 accepted tokens per
  step would be 33.1 ms/token with no kernel work at all. That is the cheapest
  path to the target on paper, and it is blocked on the trace-alternation defect
  in `ttnn_bug_report/`. Worth more than any single op here if it can be unblocked.

### 5.3 Deployed: the hyper-connection down projection is a reduction shard

The first fix to actually ship from section 5's reframing. `hc_*_down` is
`[10240, 320]` in device layout -- 10240 of reduction against **10 output
tiles** -- and it was `Shard.REPLICATE`, so all four devices read the whole
3.48 MB and ran it 96 times a token (twice a layer inside
`gated_residual_mix`). Measured at 0.1209 ms a call, 11.61 ms a token.

Splitting the reduction four ways is the one structural fix that needed no
kernel. `hc_down_shard_check.py`, at the real shapes and dtypes read off the
manifest rather than guessed:

| | ms/call |
|---|---------|
| all_reduce 32x640 | 0.0397 |
| all_reduce 32x2560 | 0.0398 |
| all_reduce 32x10240 | 0.0446 |
| hc_down replicated [10240, 320] bf8 | 0.1209 |
| hc_down K-sharded [2560, 320] bf8 | 0.0324 |
| **K-sharded + all_reduce** | **0.0445** |
| hc_up [320, 10240] bf8 | 0.0157 |
| router [2560, 512] fp32 / bf16 | 0.0369 / 0.0368 |

`all_reduce` is **latency-bound, not bandwidth-bound** -- 640 and 2560 wide cost
the same -- which is what makes this pay: a quarter of the read plus one
collective is 2.72x better than the full read.

**Measured, deployed: 111.41 -> 104.10 ms** on the shipping config, and
146.18 -> 138.94 with the selection running. Predicted 7.33, got 7.31.

Quality is unchanged: 87.2 % top-1 against 83.0 before (41/47 vs 39/47 -- two
tokens on a small sample), top-5 97.9 % both, NLL 0.682 vs 0.675, against a
float32 reference of 80.9 % / 0.703. Same products, summed in a different order.

Three things this needed that are easy to get wrong:

- **The all_reduce goes inside the silu.** The matmul returns a partial sum and
  silu is nonlinear, so sum-then-silu and silu-then-sum are different functions.
  Getting that backwards would be silently wrong.
- **The activation has to be split on the same axis.** `normed` is replicated
  10240 wide; `ttnn.mesh_partition(normed, dim=-1)` gives device d its columns.
  Without it the op refuses outright ("width=10240 height=2560"), which is the
  good case.
- `up` stays replicated. Its 10240-wide output already fills the grid at
  0.0157 ms, and splitting its 320 of reduction would starve it and buy a
  collective that costs three times the matmul.

Two things this ruled out at no cost: the router's dtype (fp32 and bf16 time
identically, so it is not bandwidth-bound and there is no reason to spend
accuracy on it), and the shared expert's N=1 sigmoid gate, which an audit
modelled at 3.19 ms and **measured at 1.25** -- too small to restructure.

INVARIANT 41: `ttnn.all_reduce` on this box costs ~0.040 ms whatever the width,
up to 10240. It is latency-bound, so a reduction shard pays for any weight whose
replicated read costs more than ~0.055 ms -- and does not pay for one that is
already wide enough to fill the grid.

### 5.4 Where the 104 ms goes, measured: op overhead and full-E elementwise

`step_op_census.py` counts one decode step: **7064 ttnn calls**, and the top of
the list by *count* moves almost no bytes.

| op | calls | GB moved |
|----|------:|---------:|
| multiply | 1297 | 0.107 |
| reshape | 937 | 0.113 |
| linear | 784 | 2.419 |
| slice | 588 | 0.051 |
| add | 581 | 0.082 |
| sigmoid + silu | 556 | 0.021 |
| sparse_matmul | 96 | (22.1 if every expert were read) |

Weight reading is not the problem. Summing the manifest against `plan.py`'s
residency rules, the dense weights are 9.5 ms of reading and the top-10 experts
1.3 ms -- **about 11 ms of a 104 ms step**. Every replicated weight put together
is 6.21 ms, so sharding all of them perfectly would recover at most ~4.7 ms.
`attn_q` (1.03 ms), `hc_*_up` (1.72), the router (0.65), `attn_output` (0.52)
and the hc norms/injects (1.30) are the whole of it.

`tiny_op_cost.py` prices the ops **at M=1, which is what decode runs**:

| op | us/call |
|----|--------:|
| multiply / add / silu / sigmoid, [1,1,1,2560] | 5.5-6.0 |
| multiply [1,1,1,10240] | 6.46 |
| reshape [1,1,1,10240] -> [1,1,4,2560] | 4.43 |
| **multiply [1,128,1,1280]** | **73.03** |
| slice [1,128,1,1280] -> 640 | 29.26 |
| silu [1,128,1,1280] | 51.49 |

Two facts fall out, and they set the plan.

**Elementwise cost is fixed per call, not per byte.** 2560 wide and 10240 wide
cost the same 5.5-6.5 us, and M=1 costs what M=32 costs. So the 3959 glue calls
are **~21.8 ms of the step** regardless of how little data they touch, and the
only thing that removes them is fusing them.

**The expert-wide elementwise ops are driven by E, not by M.** The MoE's SwiGLU
chain -- two slices, a silu and a multiply over [1, 128, 1, 1280] -- is
183.0 us a layer, **8.79 ms a token**, on tensors that are 97 % zeros because
only ~3 of the 128 local experts are selected.

So the MoE block is `sparse_matmul` 30.8 (of which 1.51 GB is zero-fill,
section 5.2) plus 8.79 of full-E elementwise: **~39.6 ms of 104**, against
1.87 ms for reading the bytes the routing actually asks for (section 4g). It is
the largest single block and one fused kernel addresses all of it: gather the
selected experts, gate/up, SwiGLU, down, score-weighted accumulate -- replacing
~7 ops a layer and writing `[1, k_sel, 1, N]` instead of `[1, 128, 1, N]`.

INVARIANT 42: on this box an elementwise or data-movement op costs ~5.5 us
whatever its shape, up to at least 10240 wide, and expert-axis ops cost with E
whatever M is. Op *count* is therefore the currency at batch 1, not bytes --
7064 calls at 5.5 us is 39 ms of pure per-op cost. Optimising bytes without
reducing calls will keep missing.

CAUTION, and it cost a wrong prediction: **benchmark at M=1.** The same reshape
that costs 4.43 us at M=1 costs 92.30 at M=32, and pricing it at 32 predicted a
5.3 ms saving from replacing it where the measured result was 0.23 ms. Storage
pads a row to a 32-row tile; the op still walks only the tile-rows the logical
shape has.

### 5.5 Stock fusion is exhausted: what was tried and what it measured

Every fusion ttnn offers for these shapes was tried and none of it helps. Each
of these is a measurement, so none of them needs trying again.

| candidate | today | fused | verdict |
|-----------|-------|-------|---------|
| `ttnn.swiglu` vs slice+slice+silu+multiply, [1,128,1,1280] | 142.24 us | 142.20 us | **no gain** |
| `ttnn.linear(activation="sigmoid")` vs `sigmoid(linear(...))`, [320,10240] | 19.27 us | 19.27 us | **no gain** |
| `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` | see 5.1 | 0.81-1.12x | no gain, refused for hc shapes |

`ttnn.swiglu` and friends are *composites*: they issue the same ops behind one
Python call, so they cut the host's call count and nothing the device sees. At
batch 1 in a trace the host call count is already free -- what costs is the
per-op device cost (invariant 42) -- so composites buy exactly nothing.

INVARIANT 43: ttnn's fused-looking ops (`swiglu`, `glu`, `geglu`, `reglu`, and
`linear`'s `activation=`) are composites, not fused kernels. They do not reduce
device-side op count and measure identical to writing the chain out. Real fusion
here means `ttnn.generic_op` (invariant 37), not a stock kwarg.

Also measured while looking: the four-op SwiGLU chain costs 142 us as a unit,
not the 183 that summing its ops individually suggested -- consecutive ops
pipeline, so per-op timings over-add. Treat a sum of individual op costs as an
upper bound.

### 5.6 Where the 7413 calls live

Counted by stubbing one component at a time and diffing the call count:

| component | calls | at 5.5 us |
|-----------|------:|----------:|
| **everything else** (hyper-connections, PLE, norms, embed) | **3033** | 16.7 ms |
| DeltaNet, 36 layers (60 a layer) | 2160 | 11.9 ms |
| QSA, 12 layers (105 a layer) | 1260 | 6.9 ms |
| MoE block, 48 layers (20 a layer) | 960 | 5.3 ms |

DeltaNet's measured ablation is 16.35 ms against 11.9 of pure op cost, so
roughly three quarters of it is overhead rather than work -- but `decode_step`
itself is already tight (12 ops, with the k/q matmuls deliberately fused into
one). The 60 a layer is mostly the surrounding projections, the conv ring, and
four layout ops (`permute`+`transpose` there and back) that exist only to feed
the convolution.

"Everything else" being the largest bucket is the finding: no single component
owns it, so no single ablation was ever going to show it.

## 6. Stage 2 built: the gather works, and replacing the matmul is a regression

`scripts/kernels/expert_gather.cpp` + `scripts/stage2_gather_matmul.py`. The
stage-1 reader with its checksum replaced by a write, so the selected experts'
weights land in a compact `[1, K_SEL, K, N]` tensor and the arithmetic can be an
ordinary dense batched matmul over K_SEL instead of E.

It works, and it is **bit-exact**: gathered against `torch`'s own index, max abs
error **0.000e+00**. On the real layer-0 `ffn_gateup_exps` `[1, 128, 2560, 1280]`
bfloat4_b, at K_SEL=10 (the worst case -- all ten routed experts could land on
one device, and a trace needs a static shape):

| | ms/call | over 48 layers |
|---|--------:|---------------:|
| gather only | 0.1495 | 7.18 |
| dense matmul over K_SEL=10 | 0.3824 | 18.35 |
| gather + dense matmul | 0.5307 | 25.47 |
| **one `sparse_matmul` over E=128, today** | **0.321** | -- |

The gather is at bandwidth -- 18.4 MB read plus 18.4 written in 0.1495 ms is
246 GB/s, matching stage 1. The problem is the other half: **a dense matmul over
ten experts costs more than a sparse one over 128.** So gather-then-dense is a
regression, not a win, and `expert_ffn`'s 30.83 ms is not the matmul.

INVARIANT 44: `ttnn.sparse_matmul` is *good* at the matmul. At M=1 it beats a
dense batched matmul over the ten selected experts (0.321 ms against 0.382),
even though the dense one reads a twelfth of the weight. Do not replace it. Its
cost lives in the zero-fill of its `[1, E, M, N]` output (invariant 40) and in
everything downstream having to work over E rather than over the selection.

Which relocates the target. What is still worth removing, in order:

1. The SwiGLU chain over `[1, 128, 1, 1280]` -- 142 us a layer, 6.83 ms a token
   (measured as a unit; summing its four ops individually gives 183 and
   over-adds, because consecutive ops pipeline). A gather of the *output* is
   25 KB rather than 18 MB and would let the chain run over `[1, 10, 1, 1280]`.
   That needs the down projection's weights gathered too, or its expert axis
   will not match.
2. The zero-fill, 1.51 GB a token, which needs a smaller output and therefore
   the same restructuring.

Both are the same change: make everything after the first `sparse_matmul` work
over the selection instead of over E. The gather kernel is the piece that was
missing and it now exists and is exact; what it should gather is the activations
and the down weights, not the gate/up weights.

### 6.1 Deployed: the gate/up output is bfloat8_b

`sparse_matmul` zero-fills its whole `[1, E, M, N]` output on every call whatever
the mask (invariant 40), and the SwiGLU chain then reads that output over all 128
experts. Both costs are set by the element width, so `dtype=ttnn.bfloat8_b` on
the gate/up call halves both: **103.87 -> 101.69 ms**.

Quality held, and that is the only reason it stayed: top-1 83.0 % unchanged,
top-5 97.9 % unchanged, NLL **0.672** against 0.687 before, on a float32
reference of 80.9 % / 0.703. The tensor is the product of bfloat4_b weights, so
its low bits were already noise, and it feeds silu and a multiply rather than an
accumulation.

The same change on the **down** projection was tried and reverted:
101.69 -> 101.45 ms, inside the spread, because `hidden` is already bfloat8_b so
that matmul's input had halved already and only the output fill remained.
Quality was unchanged there too -- it is not a correctness call but a
proportionality one. That output is what `_combine` weights and sums into the
layer's answer, and 0.24 ms does not buy precision on the output path.

INVARIANT 45: element width on an intermediate is worth more than it looks,
because `sparse_matmul`'s mask-independent zero-fill and every downstream
expert-axis op are both sized by it. But the win is in the *intermediate*, not
the result: halving the gate/up output is 2.18 ms, halving the down output is
0.24.

## 7. Deployed: a real fused kernel for the MoE SwiGLU

`scripts/kernels/swiglu_{reader,compute,writer}.cpp` under `ttnn.generic_op`,
wired into `moe.expert_ffn` as `fused_swiglu`. **101.69 -> 99.38 ms a token.**

What it replaces: `silu(gate) * up` over the `[1, E, M, 2N]` gate|up matmul
output, which was two slices, a silu and a multiply. That chain is not op
overhead -- the four ops together move ~30 MB a layer because each reads and
writes the whole E=128 tensor, and 30 MB at 388 GB/s is the 87 us measured. One
fused pass reads both halves once and writes the result once:

| | us/layer | 48 layers |
|---|--------:|----------:|
| slice + slice + silu + multiply | 87.06 | 4.18 ms |
| **fused kernel** | **40.05** | **1.92 ms** |

2.17x standalone, 2.31 ms in the model against 2.26 predicted. Quality held:
top-1 83.0 % and top-5 97.9 % unchanged, NLL **0.663** against 0.672, on a
float32 reference of 80.9 % / 0.703.

Three kernels, because the compute has to be its own: a reader that turns each
output tile index into the two input tiles it needs (`gate` at column `nt`, `up`
at `nt + Nt`), a compute kernel that is `copy_tile` x2, `silu_tile`,
`mul_binary_tile` -- `silu_tile` is a real SFPU op, so no hand-rolled
`x * sigmoid(x)` -- and a writer. The intermediate lives in circular buffers and
never reaches DRAM, which is the entire saving.

Four things that were not obvious and cost a compile or a rerun:

- **Compute kernels use `kernel_main` in this build**, not
  `namespace NAMESPACE { void MAIN {`. The latter fails with "'kernel_main' was
  not declared in this scope".
- **The tile count must be a runtime arg**, not a compile-time one: the work
  split is balanced but not equal, so cores differ by one.
- **The output buffer has to be pre-allocated and reused.** `generic_op` takes
  its output as an io_tensor, and allocating inside a trace capture corrupts the
  replay. One buffer serves all 48 layers, because a layer's hidden is consumed
  by its own down projection before the next layer runs and a trace replays in
  order.
- **The program descriptor is rebuilt per call, deliberately.** `both` is a
  fresh allocation out of the matmul so its address is part of the runtime args.
  The cost is host-side and paid at capture, not at replay -- the recorded
  commands carry the addresses and a replay puts the tensors back in the same
  places.

INVARIANT 46: fusing is worth it exactly where a chain of ttnn ops round-trips a
large intermediate through DRAM, and the measure of the win is the bytes saved,
not the calls saved. Four ops over a 30 MB tensor became one pass over 16.7 and
went 2.17x; four ops over a *small* tensor would have gone nowhere, because a
ttnn op costs ~5.5 us whatever it touches (invariant 42) and a `generic_op`
launch is not cheaper than that.

Session so far: **146.18 -> 99.38 ms**, 1.47x, no precision spent that the
quality harness could see.

### 7.1 Deployed: a fused gate-and-average for the hyper-connection block

`scripts/kernels/gated_mean_{reader,compute,writer}.cpp`, wired into
`ops.gated_residual_mix` as `fused_gated_mean`. **99.38 -> 96.67 ms a token.**

It replaces the tail of that function -- `multiply(mix, normed)` over a
10240-wide pair, four tile-aligned slices to pull out the hc streams, three adds
and a scale. Nine ops, ~11 MB round-tripped between them, and each paying ~5.5 us
of fixed per-op cost whatever it touches.

| | us/call | over 97 calls |
|---|-------:|--------------:|
| multiply + 4 slices + 3 adds + scale | 45.62 | 4.42 ms |
| **fused kernel** | **8.12** | **0.79 ms** |

**5.62x** standalone, 2.71 ms in the model. One output tile reads eight input
tiles and writes one; the four products and three adds stay in the destination
registers. Quality held: top-1 85.1 % against 83.0, top-5 97.9 % both, NLL 0.675
against 0.663 -- all inside the spread of a 47-token sample, and well clear of
the float32 reference's 80.9 % / 0.703. The fused form accumulates in fp32
registers where the chain wrote bfloat16 between every step, so it is if anything
the better-conditioned of the two.

The averaging assertion in `tests/test_tt_plan.py` has now moved twice and the
sequence is worth keeping: `reshape`+`mean` (chosen for host dispatch) ->
four slices (because a TILE-layout reshape re-tiles) -> one fused kernel. Each
step was a measurement, and the middle one taught the lesson about benchmarking
at M=1.

INVARIANT 47: a `generic_op` launch costs ~8.1 us here, against ~5.5 for a ttnn
op. So fusing pays from **two** ops upward on the launch count alone, and more
where the ops round-trip a real intermediate. That is a much lower bar than it
looked: the two kernels deployed so far replaced four ops and nine.

Session: **146.18 -> 96.67 ms, 1.51x**, with no precision the quality harness
can see.

### 7.2 Deployed: `_l2norm` as an RMS norm identity, no kernel needed

`l2norm(x) = x * rsqrt(sum(x^2) + eps)` was five ops. An RMS norm is the same
reduction with a mean instead of a sum, and `ttnn.rms_norm` is a real fused op:

    rms_norm(x, eps/D) = x * rsqrt(sum(x^2)/D + eps/D) = sqrt(D) * l2norm(x)

so `l2norm(x) = rms_norm(x, eps/D) / sqrt(D)`, and any following scalar folds
into that one multiply -- which is why `_l2norm` now takes `scale` rather than
letting the caller do its own. Verified in float64 against the closed form: max
difference **5.6e-17**, so this is an identity, not an approximation.

Six ops become two at the q site and five become two at the k site, twice a
layer over 36 DeltaNet layers. **96.67 -> 96.21 ms**, quality unchanged (top-1
85.1 %, top-5 97.9 %, NLL 0.671 against 0.675).

Predicted 1.39 ms and got 0.46, which is the useful part: `ttnn.rms_norm` is not
a 5.5 us op -- it carries its own reduction -- so replacing five cheap ops with
one expensive one plus a multiply recovers about a third of what the op count
suggested. Invariant 42's 5.5 us floor is a floor for *elementwise* ops, not a
constant for every op.

### 7.3 Where this is going, arithmetically

Six changes deployed: the QSA selection skip (35 ms), the hc_down reduction
shard (7.3), the gate/up dtype (2.2), the fused SwiGLU (2.3), the fused
gate-and-average (2.7), and this (0.5). **146.18 -> 96.21 ms, 1.52x.**

The target is 32.6, so 64 ms still has to come out, and the individual fusions
are worth 0.5-3 ms each. Twenty more of them would not do it. What the numbers
say instead:

- DeltaNet issues **61 ops a layer** and measures 16.35 ms against a 12.08 ms
  per-op floor, so ~74 % of it is the floor rather than work.
- The whole step is ~6500 ttnn calls now. At 5.5 us that is 36 ms of floor.
- Actual weight reading is ~11 ms.

So a decode path fused down to a handful of kernels a layer would sit somewhere
near 16-20 ms -- comfortably past the target. Fusion *can* get there, but the
unit of work has to be a whole layer, not a chain inside one. Fusing chains one
at a time is a 0.5-3 ms-per-kernel grind and the arithmetic says it does not
converge in reasonable time.

The other route the audit found remains open and is much cheaper: the traced
`step_n` curve fits **t ~= 162 + 12.8k**, so **k=8 accepted tokens a step is
33.1 ms/token with no kernel work at all**, and every fusion above shrinks the
162 ms fixed term further. It is blocked on the trace-alternation defect in
`ttnn_bug_report/`, whose own conclusion is "B superset of A is safe; B and A
disagreeing about a shape is not" -- which means the ladder of capture widths is
what breaks it, not two traces as such.

## 8. The measurement that reframes the target: traced `step_n(8)` is 22.71 ms/token

`scripts/dev/step_n_traced_curve.py`, one trace per k so the alternation defect
is never touched, below `indexer_budget` where the selection is off (and where
`step_n` is willing to run at all):

| k | ms a step | **ms a token** |
|---|----------:|---------------:|
| 1 | 95.09 | 95.09 |
| 2 | 123.08 | 61.54 |
| 4 | 142.71 | **35.68** |
| 8 | 181.66 | **22.71** |

Fit: **t ~= 82.7 + 12.37k**. The audit predicted `162 + 12.8k`; the slope was
right and the fixed term is now half what it was, because the six deployed
optimisations shrank exactly that part.

**k=8 is 22.71 ms a token, 1.4x past the 32.6 ms target. k=4 is 35.68, already
within 10 %.** No kernel work is involved -- the marginal cost of a token is
12.37 ms and everything else is a per-step fixed cost that 32 empty tile rows
are paying for.

This says plainly what the fusion grind could not: the gap is not work, it is
occupancy. Every fusion so far helps twice over, because it comes off the fixed
term and is then amortised k ways.

### 8.1 What stands between this and shipping it

Speculation needs the state to advance by the number of tokens actually
*accepted*, and one captured trace advances by exactly k. Two widths would be
the obvious answer and are the one thing that provokes the hang -- the bug
report's own conclusion is "B superset of A is safe; B and A disagreeing about a
shape is not", and two widths differ in shape.

So the width has to stay fixed and the *advance* has to become data. Sketch,
to be measured rather than believed:

- The DeltaNet recurrence is `state = state * g + k^T delta`. Setting `g = 1`
  and `delta = 0` for a rejected position makes that step the identity, and both
  come from tensors (`g_exp`, `beta`) that are already bound buffers refreshed
  before each replay. An accept mask multiplied into them is **data, not shape**,
  so one capture serves every acceptance count.
- The QSA K/V cache is positional, so rejected writes land at indices the next
  round overwrites. Nothing to undo.
- The **conv ring is the open problem**: `_causal_conv_step` shifts a column in
  per token, and a rejected token still pollutes the window. Either the shift
  has to become maskable the same way, or the ring has to be restored -- and at
  batch 1 the recurrent state plus rings is small enough (~112 MB) that a
  snapshot/restore is ~0.3 ms, which is affordable against a 181 ms verify.

INVARIANT 48: the decode step costs `82.7 + 12.37k` ms for k tokens, so at batch
1 roughly 87 % of it is fixed cost paid for one token. Speculation is worth more
than any fusion on this list, and the two compound: fusion shrinks the fixed
term, speculation divides it.

### 8.2 The accept mask is exact; the conv ring is the last piece

`_linear_attention_step_n` now takes an optional `accept` tensor, and `step_n`
builds it from a `list[float]` bound like any other per-step input -- so it is
**data, not shape**, and one capture serves every acceptance count. That is the
whole point: two capture widths are what provokes the alternation hang.

    g_exp <- g_exp * acc + (1 - acc)    -> 1 where rejected
    beta  <- beta * acc                 -> 0 where rejected

so `state = state * 1 + k^T * 0 = state`. Verified in
`scripts/dev/accept_mask_identity.py`, and the control is the one that matters --
same k, same graph, same accepted prefix, only the *rejected* tokens changed:

| | diff |
|---|---|
| recurrent state, rejected tokens changed | **0.0000e+00** |
| the accepted rows' outputs | 0.0000e+00 |
| the same change **without** the mask | 2.3584e-01 |
| **conv ring, rejected tokens changed** | **1.4350e+01** |

So the recurrence is bit-exactly inert on rejected steps, and the convolution
ring was not -- until the rewind below. **Both are now 0.0000e+00.** It holds the last `kernel-1` qkv columns and a rejected step still
shifts one in.

Not worth masking the shift itself: the ring is three slots deep on 36 layers, so
a per-step blend is 3 x k x 36 blends -- 864 ops at k=8, ~14 ms, more than the
verify. The columns are all still in hand, though: `qkv_col` is `[1, k, C, 1]`
and the correct ring after accepting j is columns `j-2, j-1, j`. That selection
is one small matmul a layer against a `[3, k]` selection matrix -- 36 matmuls of
`[3,8] x [8,2560]`, and the matrix is a bound tensor, so it stays data.

One thing to be honest about: a masked k=4 state is **not** bit-identical to a
genuine k=2 run (measured 1.68e-02 relative). That is the k=2-vs-k=4 matmul
blocking, not the mask -- the isolating control above is exactly 0. It is the
same class of difference as any tile-blocking change, and the quality harness is
what should judge it.

### 8.3 The conv ring rewind closes it: a rejected token now costs nothing

The ring holds the last `kernel-1` qkv columns and a rejected step still shifts
one in. Masking the shift would be three blends a step a layer -- 864 ops at
k=8, ~14 ms, more than the verify. So instead the columns are kept and the ring
is rewound once at the end.

`_linear_attention_step_n` builds a `history` before the conv loop: the entry
ring, oldest first, followed by this call's k columns. History index `h` then
holds the column from time `h - depth`, so after accepting j tokens slot s --
0 is newest -- must hold time `j - 1 - s`, i.e. index `depth + j - 1 - s`. That
is a gather, written as one small matmul against a `[depth, depth + k]`
selection matrix, and the matrix is a bound tensor: **data, not shape**, so one
capture serves any j.

Cost is ~15 ops a layer, ~3 ms at k=8, against a 181 ms verify.

Measured, with the control that isolates it -- same k, same graph, same accepted
prefix, only the *rejected* tokens changed (`scripts/dev/accept_mask_identity.py`):

| | diff |
|---|---|
| recurrent state | **0.0000e+00** |
| conv ring | **0.0000e+00** |
| accepted rows' outputs | **0.0000e+00** |
| the same change with no mask (control) | 2.3584e-01 |

A rejected token has bit-exactly no effect on anything carried forward. One
capture at width k, and the state advances by exactly the number accepted.

Two things that cost a run each: the rewind must build the ring itself when
`st.conv` is None, because `_causal_conv_step` would otherwise create it *inside*
the loop and the history has to be captured before the loop touches it -- with a
fresh state the rewind silently did nothing. And `accept` must be a prefix of
ones; `step_n` rejects anything else rather than computing a `j` that does not
mean what the caller thinks.

INVARIANT 49: speculative acceptance is expressible entirely as bound tensors on
this model -- an accept mask for the recurrence and a selection matrix for the
convolution ring. Neither changes a shape, so the ladder of capture widths that
provokes the alternation hang is not needed and must not be reintroduced.

## 9. Speculation, end to end: 97.89 -> 67.53 ms a token

`scripts/dev/spec_decode_bench.py`, k=8, one capture width so the alternation
hang stays out of reach. `TracedStepN.step_n` now takes an `accept` prefix and
rebinds the mask and the conv-ring selection between replays via
`TTModel._bind_accept`, so the shape never changes.

| | ms a token |
|---|---:|
| plain traced step | 97.89 |
| **speculative, k=8** | **67.53** (1.45x) |

17 rounds for 68 tokens, 65 % of them with a draft, acceptance histogram
`{0: 7, 2: 3, 4: 1, 6: 1, 7: 5}`.

Verification is sound, checked separately in `scripts/dev/step_n_row_check.py`:
`step_n`'s row i predicts exactly what a plain step at position i does
(`[561, 1118, 13934, 17943]` both ways), so the drafter is being compared
against the right tokens.

**Why 67.53 and not the 22.71 the k=8 curve promises.** Two things, and neither
is the masking:

- A partial acceptance costs **two** verifies. j is only known after the step
  runs, so the first pass advances by k and a rewind has to replay the same
  width with the correct mask. Six of the eleven drafted rounds were partial.
- Seven of seventeen rounds had **no draft at all** -- `prompt_lookup_draft`
  only fires on a repeated n-gram -- and each of those yields one token for a
  full verify.

So the ceiling is the drafter, not the machinery. With one verify a round the
same trace would be ~47.8 ms a token; with a drafter that is usually right it is
the 22.71 the curve says.

The second verify is removable and the design is known: snapshot the recurrent
and conv state **per step** inside the graph (288 copies at k=8, ~1.6 ms) and
select slot j afterwards, instead of rewinding by replaying. That turns every
round into one verify whatever j is.

CAUTION on the harness, unresolved: the benchmark's *plain* baseline decodes a
different continuation from the speculative run ("Download Presentation - The
PPT/PDF document..." against a coherent echo of the prompt), and they agree on
0 of 64 tokens. Since `step_n` row i is exactly a plain step i, and the masking
is bit-exact, the fault is in how the benchmark builds its `TracedDecoder`
baseline rather than in speculation -- but it is not yet diagnosed, so the
1.45x is a timing result and the agreement check is **not** evidence of
correctness either way.

Session: **146.18 -> 96.21 ms** deployed, and **67.53 ms** with speculation on
top -- 2.16x from where the session started, against a 32.6 ms target.

### 9.1 Two policy results, both measured, both counter-intuitive

**Always drafting is worse.** Padding a short draft so every round "verifies
properly" cost **67.53 -> 86.08 ms a token**. The acceptance histogram was
identical -- the padded drafts were all rejected -- but a rejection has to
rewind, so seven rounds that had been one masked verify became two. The
no-draft path's single masked verify, which advances by exactly 1, is optimal
precisely when j is known in advance to be 0.

**Where the time actually goes**, from the 17-round run: seven no-draft rounds
at ~191 ms for one token each, five full acceptances at one verify for eight
tokens, five partial at two verifies. Mean 4.00 tokens a round.

That gives the ceiling of this design honestly:

| | ms a token |
|---|---:|
| today, two verifies on a partial | 67.53 |
| one verify a round (per-step state, not built) | ~47.8 |
| the k=8 curve, if every round accepted all 8 | 22.71 |
| **target** | **32.6** |

So removing the second verify is worth ~20 ms and is the last piece of
machinery. Past that it is the *drafter*: 32.6 ms needs a mean accepted prefix
near 5.9 of 7, against the 3.00 `prompt_lookup_draft` manages. That is a model
question -- the checkpoint carries a 51 B-parameter n-gram table for PLE that
has never been tried as a drafter -- not an engineering one.

INVARIANT 50: with a fixed capture width, emitting j+1 tokens requires the state
to have advanced j+1, and j is only known after the step runs. So a partial
acceptance costs either two verifies or per-step state saved in-graph. Do not
"fix" it by drafting harder: a rejected draft is strictly worse than no draft,
because no draft can use the single masked advance.

### 9.2 k=6 is the optimum, and what the 54.79 ms actually means

Swept on the same 64-token generation:

| k | ms a token | tokens a round | acceptance |
|---|-----------:|---------------:|------------|
| 4 | 63.58 | 2.44 | `{0:14, 3:13}` |
| 5 | 54.95 | 3.00 | `{0:11, 4:11}` |
| **6** | **54.79** | **4.00** | `{0:5, 1:2, 4:1, 5:9}` |
| 7 | 67.64 | 3.76 | `{0:7, 2:1, 3:2, 4:1, 5:1, 6:5}` |
| 8 | 67.53 | 4.00 | `{0:7, 2:3, 4:1, 6:1, 7:5}` |

k=6 wins because nine of seventeen rounds accept *everything* -- a full
acceptance is one verify for six tokens, where a partial is two verifies for
fewer. The optimum is where the drafter's reach and the verify width meet, and
past it the extra width buys rejections rather than tokens.

Making the drafter prefer the **longest** context match rather than the most
recent 3-gram measured neutral here (54.79 against 54.94, identical histogram),
because a prompt that quotes itself makes the 3-gram unambiguous. Kept anyway:
it is the better bet in general and costs only host-side scanning against a
~187 ms verify.

**The honest caveat, and it matters more than the number.** This benchmark's
prompt quotes itself, which is exactly the case `prompt_lookup_draft` is built
for -- the drafter's own docstring says it "earns 1.7-2.1x on prompts that quote
their context and nothing at all on open prose". So **54.79 ms a token is the
favourable case, and the general-text number is the plain step's 96.21.**

That splits the remaining work cleanly:

- For text that repeats, the ceiling is the second verify on a partial
  acceptance: one verify a round would be ~46.7 ms at k=6. The design is in 9.1
  and is the last piece of machinery.
- For text that does not, speculation contributes nothing and the only lever is
  the plain step -- which is fusion, and the untouched chains are DeltaNet
  (2196 calls a token, 61 a layer, ~74 % of its 16.35 ms is per-op floor) and
  QSA (1260 calls).

INVARIANT 51: the speculation width has an optimum and it is not "as large as
possible". At k=6 nine of seventeen rounds accept everything; at k=8 the same
drafter earns the same 4.00 tokens a round while every rejection costs a wider
verify. Sweep it against the actual drafter rather than assuming bigger is
better.

## 10. The MoE wall breaks: gather the experts side by side, one wide matmul

Stage 2 concluded that gather-then-dense was a regression, and that was true of
the layout it tried. Gathering into `[1, K_SEL, K, N]` gives a *batched* matmul
over K_SEL, and at M=1 that still cannot fill the core grid: 0.382 ms against
0.321 for the sparse matmul it would replace (invariant 44).

Concatenating the selected experts on the **output axis** instead gives one wide
`[K, K_SEL*N]` matmul -- the same shape as `attn_qkv`, which runs at 128 GB/s --
and that changes the answer completely. Measured on the real layer-0
`ffn_gateup_exps`, K_SEL=3 (the per-device budget, ceil(10/4)):

| | ms a layer | over 48 layers |
|---|----------:|---------------:|
| gather (bit-exact against torch, **0.000e+00**) | 0.0465 | 2.23 |
| one wide matmul `[2560, 3x1280]` | 0.0478 | 2.29 |
| **gather + matmul** | **0.0933** | **4.48** |
| `expert_ffn` today, both projections | -- | **30.83** |

And the down projection wants the *other* concatenation, on its input axis:
`[K_SEL*N, K]`, so that the matmul **sums over the experts** -- which is the
combine, for free. Stacking `[N, K]` slabs on rows is exactly the straightforward
`dst = w`, so one `WIDE` compile-time flag covers both layouts in the same
kernel. Priced separately at 0.0411 ms a layer.

So the whole MoE becomes: gather gate|up, one wide matmul, SwiGLU over a
`[1, 1, 1, K_SEL*2N]` tensor instead of `[1, 128, 1, 1280]`, scale by the router
weights, gather down, one wide matmul that also combines -- and it removes the
1.51 GB zero-fill (invariant 40) on the way.

**Correction to the headline above: 4.48 ms is K_SEL=3, which is the expected
per-device count and not a safe one.** Ten experts scattered over four devices
put more than three on one device often, and truncating would drop a routed
expert. The exact worst case is K_SEL=10, and the cost scales with it:

| K_SEL | gather | matmul | over 48 layers |
|-------|-------:|-------:|---------------:|
| 3 (expected) | 0.0465 | 0.0478 | 4.48 ms |
| 6 | 0.0907 | 0.0558 | 6.97 ms |
| **10 (exact)** | 0.1503 | 0.0719 | **10.60 ms** |

All three bit-exact against torch. At the safe width the gate/up half is 10.60
against 30.83 for *both* projections today, so the honest projection for the
whole MoE is **~19 ms against ~33**, about **14 ms a token** -- still the largest
single item left, but not the 24 the K_SEL=3 number implied.

The gather dominates at large K_SEL (0.150 against 0.072 for the matmul) because
it copies the weights before the matmul reads them. Removing that copy needs a
matmul kernel that reads scattered experts directly, which is the fused kernel
this approach was chosen to avoid.

A third layout, `WIDE=2`, puts every expert's gate half ahead of every expert's
up half, so the fused SwiGLU kernel -- which splits its input down the middle --
works on the result unchanged. Bit-exact, and free: 0.0466 ms against 0.0465.

INVARIANT 52: at M=1 the layout of a gather decides whether it is worth doing.
The same bytes, gathered into an expert *batch*, lose to `sparse_matmul`;
gathered side by side into one wide matrix, they beat it by 7x. Width is what
fills the grid, and an expert axis is not width.

Not yet wired into `expert_ffn`: the down-projection gather, the router scaling
between the two matmuls, and removing `_combine`. The kernel supports both
layouts and is bit-exact; what remains is the plumbing.

### 10.1 Deployed: the wide-gather MoE, and why it is 5 ms rather than 14

`moe.wide_expert_ffn`, gated by `moe.WIDE_EXPERTS` (the selection width, or 0
for the old path). **96.21 -> 91.22 ms a token.**

The path, for a device holding 128 of the 512 experts:

    topk on the localised router weights   -> k_sel local ids *and* their scores
    gather gate|up (WIDE=2)                   [1, 1, K, k_sel*2N], gates then ups
    one linear                                [1, 1, M, k_sel*2N]
    fused SwiGLU                              [1, 1, M, k_sel*N]
    scale by the scores                       (one small matmul broadcasts them)
    gather down (WIDE=0)                      [1, 1, k_sel*N, K], stacked on rows
    one linear                                [1, 1, M, K]   <- sums the experts

The last matmul *is* `_combine`: stacking the down slabs on their input axis
makes the contraction run over experts as well as over N. The caller's
all-reduce completes the sum across devices exactly as before, so nothing
outside the MoE changed.

Using `ttnn.topk` on the localised weights is what makes a fixed width safe: it
returns the ids to gather and the scores to scale by together, and entries that
were not selected have score zero and contribute nothing. k_sel=10 is the exact
worst case for top-10 over four devices.

Agreement with the path it replaces: **cosine 0.999903**, max abs error 7.6e-06
on values to 1.0e-03. Quality after: top-1 85.1 % against 83.0, top-5 97.9 %
both, NLL 0.674 against 0.663.

**Why 5 ms and not the 14 the component numbers implied.** Measured on one real
layer: `moe_block` went 0.9794 -> 0.8877 ms, 1.10x. The gathers and the two wide
matmuls do save what section 10 measured, but the path around them gives much of
it back -- a second `topk` (k=10 over 128, once a layer), the index typecast,
layout change and pad the kernel's uint32 page needs, the score broadcast, and
two `generic_op` launches. Roughly 8 extra ops a layer at ~5.5 us each plus the
topk.

Two of those are removable and neither is hard: the kernel could read uint16
indices directly, which drops the typecast and the pad, and the router's own
global `topk` already computes indices that are currently discarded -- using
them would need the local compaction this design sidesteps, but it would remove
the second topk.

INVARIANT 53: measuring a replacement's *components* against the components it
replaces overstates the win. The wide matmuls beat `sparse_matmul` 7x in
isolation and the whole block moved 1.10x, because a new path brings its own
glue. Price the block, not the piece.

Session: **146.18 -> 91.22 ms**, 1.60x.

### 10.2 What MoE routing costs, op by op

`scripts/dev/route_cost.py`, at the real shapes, inside a trace:

| op | us a layer | over 48 |
|----|-----------:|--------:|
| **topk k=10 of 512 (global)** | **94.52** | **4.54 ms** |
| router linear `[2560, 512]` fp32 | 36.54 | 1.75 ms |
| topk k=10 of 128 (local -- the wide path added this) | 33.23 | 1.59 ms |
| divide by sum over 512 | 16.19 | 0.78 ms |
| softmax over 512 | 9.65 | 0.46 ms |
| ge over 512 | 5.95 | 0.29 ms |
| mesh_partition 512 -> 128 | 2.08 | 0.10 ms |
| **total** | **198.2** | **9.51 ms** |

The global `topk` alone is half the routing, and it exists only to find the
k-th largest probability as an inclusion threshold. Replacing it with four local
topks plus a collective prices out at roughly the same (33 + 40 us), so it is
not obviously improvable without changing what the router selects.

The router's fp32 dtype is not the problem: `[2560, 512]` times identically at
bfloat16 (36.8 us against 36.9), which is the same finding as section 5.3 -- it
is not bandwidth-bound.

The 33.23 us local topk is the wide path's own glue, and it is the concrete
content of invariant 53: 1.59 ms a token added to buy the gathers.

CAUTION on a measurement that did not work: an ablation stubbing out the router
chain measured **117.98 ms against a 91.09 baseline** -- slower with work
removed, which cannot be right, so the stub differs from the real path in some
way that was not chased down. The op-by-op numbers above stand on their own; the
`wroute` seam does not, and should not be trusted until that is explained.

Also fixed while here: the wide path's fallback to `sparse_matmul` was a silent
`except: pass`, which would have left the old path running while every
measurement claimed the new one. It now warns once. Checked: it does not fire.

## 11. The unattributed time was never a component

Re-censused after everything above:

| | at the start | now |
|---|---:|---:|
| ttnn calls a step | 7064 | **6426** |
| bytes moved a step (per device, from shapes) | 25.29 GB | **4.72 GB** |
| step | 146.18 ms | **91.09 ms** |

`sparse_matmul` is gone from the list entirely -- the wide path turned it into
`ttnn.linear`, which is now 880 calls moving 4.142 GB. **Bytes fell 81 %.**

And the accounting closes, which answers the question that has been open since
section 5.4. Modelling each op as `max(5.5 us, its bytes / 388 GB/s)`, and
linears at the 25-33 % of bandwidth invariant 38 measured:

    ~5546 small ops x 5.5 us                     ~30 ms
    880 linears, 4.142 GB at ~4x their byte time ~43 ms
                                                 ------
                                                 ~73 ms   against 91.09 measured

So **the "unattributed 30 ms" is the per-op floor, spread across five and a half
thousand small operations.** It never belonged to a component, which is exactly
why no component ablation could show it: stubbing one out leaves every other
op's floor standing. Every hunt for it in sections 5.4 through 10.2 was looking
for the wrong kind of thing.

INVARIANT 54: the decode step's time is roughly `5.5 us x (number of ops)` plus
the linears at three to four times their byte time. At 6426 calls that floor
alone is 35 ms, so **no amount of byte-level work gets past it** -- the bytes are
already down to 4.72 GB, which is 12 ms at bandwidth. Only fusing ops away moves
the floor, and only widening the linears' output moves the other half.

What that implies for the 32.6 ms target, stated plainly: it needs the op count
roughly halved *and* the linears saturating. Both are known-possible -- the two
fused kernels deployed here each removed the ops they replaced, and the wide
gather showed what widening an output does -- but it is a long grind of the same
two moves, not one more insight.

## 12. The narrowest matmul in the model: `hc_inject` was 7.7 ms

`ttnn.linear` is 880 calls and roughly 43 ms of the step, and 288 of those are
the hyper-connection block. Priced at their real shapes
(`scripts/dev/hc_linear_cost.py`):

| | us a call | x96 a token |
|---|---------:|------------:|
| **`hc_inject` [10240, 4] fp32** | **80.35** | **7.71 ms** |
| `hc_down` [2560, 320] bf8 | 31.50 | 3.02 ms |
| `hc_up` [320, 10240] bf8 | 14.45 | 1.39 ms |

`inject_w` is 164 KB. At 388 GB/s that is 0.4 us, and it costs 80.35 -- **200x
its byte time**. Four output columns is one tile, so all but one core sits idle.
It is the sharpest instance of invariant 38 in the model and it had never been
measured because it is a tiny weight that looks harmless.

It also takes the *same* `normed` that `down_w` does, so the two are one matmul
with a wider output. Concatenating them takes 320 columns to 352 -- four real,
the rest padding to a tile -- and the pair then costs about what `down` alone
did. `inject_w` is mesh-partitioned to match `down_w`'s row shard, so its half is
a partial sum too, which is fine because the all_reduce `down` already needs
comes before anything nonlinear touches either.

**91.09 -> 84.01 ms**, against 7.1 predicted. The concatenated weight is built
once per layer at first use and cached; paying a device concat every token would
be the same mistake the fusion is fixing.

Quality held, and it had to be checked because `inject_w` goes from float32 to
`down_w`'s bfloat8_b: top-1 85.1 % and top-5 97.9 % unchanged, NLL **0.662**
against 0.674 -- slightly better, which is noise on 47 tokens but certainly not
a regression.

INVARIANT 55: a weight being small is not a reason to leave it alone. The
cheapest weight in the hyper-connection block was its most expensive matmul,
because cost here follows output *width* and not size. Look for narrow outputs,
not big tensors -- and prefer widening an existing matmul over adding one.

Session: **146.18 -> 84.01 ms**, 1.74x.

### 12.1 The rest of the narrow outputs, ranked and fused

Every linear weight in the model, sorted by output width (`_exps` excluded):

| tensor | K | N | tiles | calls a token |
|--------|--:|--:|------:|--------------:|
| `hc_*_inject` | 10240 | 4 | 0.1 | 96 (fixed, section 12) |
| `ssm_alpha`, `ssm_beta` | 2560 | 48 | 1.5 | 36 each |
| `hc_*_down` | 10240 | 320 | 10 | 96 |
| `ffn_gate_inp` | 2560 | 512 | 16 | 48 |
| `ffn_gate_shexp`, `ffn_up_shexp` | 2560 | 640 | 20 | 48 each |

The measurement that makes the rule concrete: **`[2560, 48]` and `[2560, 96]`
cost the same 31.6 us.** Width is free until it fills the grid, so two narrow
matmuls against one input are strictly worse than one wider one.

Two pairs qualified -- same input, adjacent outputs -- and both are now one
matmul:

| | before | fused | predicted saving |
|---|------:|------:|-----------------:|
| `ssm_alpha` + `ssm_beta` | 2.28 ms | 1.14 | 1.13 |
| `ffn_gate_shexp` + `ffn_up_shexp` | 3.48 ms | 1.93 | 1.55 |

**84.01 -> 82.73 ms.** Predicted 2.68, measured 1.28, and the difference is
invariant 53 again: each fusion adds two slices to split the result, 72 and 96
calls a token, about 0.93 ms of new glue. Quality unchanged -- top-1 85.1 %,
top-5 97.9 %, NLL 0.662, all identical.

So the rule has a threshold worth stating: fusing two same-input matmuls trades
one matmul call for two slices, and only pays when the matmul costs more than
about 11 us. At 31.5 it pays comfortably; at 8 it would not.

Session: **146.18 -> 82.73 ms**, 1.77x.

### 12.2 Fewer calls is not automatically less work

The DeltaNet convolution is `sum_t w_t * p_t` written as four multiplies and
three adds. Folding it into one multiply and one reduction -- concatenate the
four pieces, concatenate the four taps once, `sum` over the last axis -- halves
the call count, and **measured 4.5 ms a token slower**: 82.73 -> 87.26.

Each piece is `[1, 1, C, 1]`, which TILE layout pads to `[C, 32]`. Concatenating
four of them repacks four tensors' worth of tiles into one, and `sum` over that
padded width is a full-tile reduction. Neither is elementwise.

INVARIANT 56: invariant 42's ~5.5 us floor is for **elementwise** ops, whose cost
really is independent of shape. `concat`, `sum`, `reshape` and friends are data
movement and pay for the tile padding, so replacing seven elementwise ops with
two movement ops can cost more than it saves. Count calls only within the same
class of op.

Reverted, with the measurement kept at the call site so the "obvious"
simplification is not tried again.

## 13. Where this stands, and what 32.6 ms would actually take

**146.18 -> ~83 ms a token, 1.76x**, with no precision the quality harness can
see (top-1 85.1 % against 83.0 at the start, top-5 97.9 % throughout, NLL 0.662
against 0.703 for the float32 reference). 229 tests pass. Bytes moved a step fell
from 25.29 GB to 4.72.

Deployed, in order of size:

| change | saved |
|--------|------:|
| skip the QSA selection below `indexer_budget` (§4f) | 35 ms |
| fold `hc_inject` into `hc_down`'s matmul (§12) | 7.1 |
| shard `hc_down` on its reduction axis (§5.3) | 7.3 |
| the wide-gather expert path (§10.1) | 5.0 |
| fused gate-and-average kernel (§7.1) | 2.7 |
| fused SwiGLU kernel (§7) | 2.3 |
| `bfloat8_b` on the gate/up output (§6.1) | 2.2 |
| fuse the remaining same-input matmuls (§12.1) | 1.3 |
| `_l2norm` as an rms_norm identity (§7.2) | 0.5 |

Speculation is built and verified but not wired into the engine: k=6 measures
54.79 ms a token on a self-quoting prompt and nothing on open prose (§9.2).

### What the remaining 50 ms is made of

From the census and the op timings, the step is roughly `5.5 us x ops` for
elementwise work plus the linears at three to four times their byte time
(invariant 54). At 6426 calls that is ~35 ms of floor and ~43 of linears.

And most of the floor is **not** reducible, which took a failed experiment to
learn (invariant 56): `reshape`, `slice`, `permute` and `concat` are data
movement and pay for the tile padding, so removing them by restructuring can
cost more than it saves -- folding the DeltaNet convolution into a concat plus a
reduce halved its call count and lost 4.5 ms. Only *elementwise* clusters shrink
reliably, and only through `generic_op`, at roughly 1-2 ms a kernel.

So reaching 32.6 needs, concretely:

1. **The linears saturating.** They run at 25-33 % of bandwidth because their
   outputs are narrow (invariant 38), and every same-input pair that could be
   widened has been. What is left needs either DRAM-sharded matmuls -- measured
   a wash or worse at every width tried (§5.1) -- or a matmul kernel of our own.
2. **The elementwise floor roughly halved**, one `generic_op` kernel at a time.
   The two deployed here were worth 2.3 and 2.7 ms; the remaining clusters are
   smaller.
3. **Speculation wired**, which divides whatever the fixed cost has become --
   but only on text the drafter can predict.

None of those is blocked. All three are grinds, and the arithmetic says they are
long ones: the target is 50 ms away and the unit of progress is now 1-2 ms.

### The two things most worth doing next

Not the smallest remaining items, but the ones with the best ratio:

- **Wire speculation into the engine.** It is built, verified bit-exact
  (§8.2, 8.3) and measured; what is missing is engine plumbing, not physics. On
  repetitive text it is worth ~30 ms today and it compounds with everything
  above, because every millisecond taken off the step is divided by k.
- **Price a custom matmul kernel for the narrow shapes.** Invariant 38 is the
  single largest unaddressed effect -- 43 ms of linears at a quarter of
  bandwidth -- and it is the one thing on this list that has never been
  attempted. `generic_op` is proven; a GEMV that splits the reduction across
  cores is the shape to try, and `dram_sharded_gemv_check.py` already has the
  harness to compare against.

## 14. A GEMV that splits its reduction across cores: proven, not shippable yet

`scripts/kernels/ksplit_{reader,compute,writer}.cpp` and
`scripts/stage5_ksplit_gemv.py`. Core `(g, nt)` accumulates only K-tiles
`[kt_lo, kt_hi)` for output column `nt` and writes a partial into `[1, G, M, N]`;
the host sums over `G` with one `ttnn.sum`. No cross-core semaphore and no second
kernel pass -- the reduction rides on a stock op, which is what made this
buildable at all.

This is the first attempt at invariant 38, the largest effect in the model:
linears at 25-33 % of bandwidth because a narrow output leaves most cores idle.

**bfloat16 partials** (110 cores of an 11x10 grid):

| shape | `ttnn.linear` | k-split | ratio | rel err |
|-------|-------------:|--------:|------:|--------:|
| hc_down `[2560, 320]`, 11 groups | 31.80 us | **14.83** | **2.14x** | 1.72e-02 |
| router `[2560, 512]`, 6 groups | 36.52 | **24.27** | **1.51x** | 2.83e-02 |
| qsa out `[1536, 2560]`, 1 group | 33.23 | 41.98 | 0.79x | 4.14e-02 |

The narrow shapes win and the already-wide one loses, which is exactly what the
theory says: splitting the reduction only helps when the output cannot fill the
grid on its own.

**But the accuracy is not shippable**, and fixing it costs the win. With float32
partials and `fp32_dest_acc_en`:

| shape | ratio | rel err |
|-------|------:|--------:|
| hc_down | 1.02x | 1.29e-02 |
| router | 0.68x | 1.89e-02 |
| qsa out | 0.20x | 1.78e-02 |

So the error is **not** mainly the partials' precision -- it barely moved -- and
it is present at `groups=1`, where there is no cross-group summation at all.
That points at the tile matmul path itself: `matmul_tiles` with bfloat8_b weights
and bfloat16 activations is evidently not the same numerics as `ttnn.linear`'s
tuned kernel. Until that is understood, 1.7e-02 relative on a projection that
runs 96 times a token is not something to ship on the strength of a 1.63 ms
saving.

**Correction, and it nearly cost the whole thing.** The 1.7e-02 above is
divergence from `ttnn.linear`, which is not the same as error. Judged against
float64 on the same quantised operands:

| shape | kernel | `ttnn.linear` | |
|-------|-------:|-------------:|---|
| hc_down `[2560, 320]`, 11 groups | **9.71e-03** | 1.29e-02 | kernel is *more* accurate |
| router `[2560, 512]`, 6 groups | **1.64e-02** | 1.86e-02 | kernel is *more* accurate |
| qsa out `[1536, 2560]`, 1 group | 3.91e-02 | 1.68e-02 | 2.3x worse |

On both shapes where it wins on speed it is also the more accurate of the two,
and the reason is structural: splitting the reduction into groups *is* a
pairwise summation, which is better conditioned than one long serial
accumulation. That is why the only shape where it is less accurate is the one
with a single group, where no pairing happens.

INVARIANT 57: splitting a GEMV's reduction across cores is worth **2.12x** on
the model's narrowest projection, nothing on a wide one, and is *more accurate*
than `ttnn.linear` wherever it applies. Judge a replacement against float64, not
against the op it replaces -- comparing the two to each other made a better
kernel look like a broken one, and it was nearly discarded on that basis.

Deployed for the hyper-connection `down|inject` matmul: **~83 -> 82.25 ms**,
about 1 ms where 1.6 was predicted, with quality inside the run-to-run spread
(top-1 83.0 %, NLL 0.666). The guard is `groups >= 2`, so a shape whose output
already fills the grid falls through to `ttnn.linear`.

### 14.1 The k-split does not generalise on a `groups >= 2` guard

Routing every narrow linear in the model through `ksplit_linear` -- one change in
`linear_rows`, so it would apply everywhere at once -- measured **82.25 ->
83.68 ms** and moved NLL from 0.666 to 0.691. Reverted.

The guard is the problem. Two or three reduction groups do not earn back the
kernel launch and the `ttnn.sum`, and the shapes with few groups are also the
ones where the split's pairwise summation has nothing to pair. The three shapes
measured say where the line is:

| shape | output tiles | groups | ratio |
|-------|-------------:|-------:|------:|
| hc_down `[2560, 320]` | 11 | 10 | 2.12x |
| router `[2560, 512]` | 16 | 6 | 1.49x |
| qsa out `[1536, 2560]` | 80 | 1 | 0.79x |

So it wants roughly six groups or more, which on a 110-core grid means an output
of about sixteen tiles or fewer. That is a per-shape decision made where it has
been measured, not a rule to apply from `linear_rows`, and the call site now
says so.

INVARIANT 58: a kernel that beats a stock op on one shape is not an improvement
to the op. `ksplit_linear` is 2.12x where it was measured and a net loss applied
generally, because its win depends on how many reduction groups the grid affords
-- which is a property of the shape, not of the kernel.

### 14.2 Applied to the two other shapes that qualify

The rule from 14.1 -- roughly six reduction groups or more, so an output of
about sixteen tiles or fewer on a 110-core grid -- admits exactly two more
call sites, and both now use it:

- the MoE router, `[2560, 512]`, sixteen output tiles, six groups, measured
  1.49x standalone
- the fused `ssm_alpha|beta`, `[2560, 96]`, three output tiles and therefore the
  most reduction groups of anything in the model

**82.69 -> 82.41 ms**, which is inside the run-to-run spread against 1.15 ms
predicted, so the honest reading is that the timing gain is not resolvable here.
Quality moved the other way -- top-1 85.1 % against 83.0, NLL 0.663 against
0.666 -- which is consistent with the split being the better-conditioned
summation (invariant 57), and is the reason to keep it rather than the time.

Everything else in the model is either already wide enough that the split
declines, or has too few calls a token to matter. The narrow-output work is
finished at this grid size.

## 15. The component breakdown was wrong, and fixing it found the real targets

Every conclusion in sections 5.4, 10.2 and 11 about "where the step goes" rested
on `decode_ablation_check.py`, and the harness had a defect that made its numbers
uninterpretable. It is worth stating plainly because two of those sections drew
firm conclusions from it.

The decode ablations opened the model at `max_seq_len=4096`, above
`indexer_budget`, so the QSA sparse selection ran -- **except** in the one part
called `noselect`, which turned it off. `noselect` was then read as the baseline.
Every other part was therefore measured in a different regime from the thing it
was compared against, and the table came out like this:

    noselect   82.84 ms   (used as the baseline)
    moe        84.03      "delta" -1.19
    deltanet  101.60             -18.76
    shared    114.57             -31.73
    allreduce 116.01             -33.17
    ple       117.85             -35.01

Five components that cost less than nothing. The same shape had already appeared
once, recorded in 10.2 as a caution about the `wroute` seam (117.98 against a
91.09 baseline) and never chased. It was the same defect both times.

The fix is one line: pin the regime to the one the engine ships.
`engine.py` starts with `model.selection_active = False` and only turns the
selection on past `indexer_budget`, so a decode ablation with the selection on is
not measuring the step the engine runs. With that pinned, `none` measures 82.35
-- the shipping number -- and the deltas are these:

| component | ms a token | share |
|---|---:|---:|
| `moe_block` | **33.28** | 40 % |
| DeltaNet, 36 layers | 16.23 | 20 % |
| `gated_residual_mix`, 96 calls | 11.99 | 15 % |
| QSA, 12 layers | 7.28 | 9 % |
| shared expert | 4.62 | 6 % |
| PLE | 3.21 | 4 % |
| all-reduce | 2.31 | 3 % |
| **sum** | **78.92** | **96 %** |
| *(the selection, as a regime: +35.91)* | | |

**96 % of the step is attributed to components.** So section 11's headline --
"the unattributed time was never a component" -- was an artifact. There was no
unattributed 30 ms; there was a broken baseline. Invariant 54's cost model (a
per-op floor plus linears at 3-4x their byte time) still describes *why* a
component costs what it does, but the per-op floor lives inside the components
and the ablations do show it.

INVARIANT 59: an ablation harness must hold every regime fixed except the one
part it removes. A component that appears to cost less than nothing is not a
surprising result, it is a broken control -- and the tell is that *several*
components show it at once.

### 15.1 Inside the MoE, and what a fusion is actually worth

`moe_block` is 40 % of the step against a roofline of 1.33 ms (top-10 experts,
0.516 GB a device a token, at 388 GB/s), so it runs at about 4 % of bandwidth.
Splitting it the same way (`decode_ablation_check.py w:<piece>` and `r:<piece>`,
each piece replaced by a persistent buffer of the shape it returns, so the timed
region gains no op and no allocation):

| piece | ms a token |
|---|---:|
| the two expert gathers | 13.19 |
| the router chain | 9.35 |
| down matmul | 5.53 |
| gate\|up matmul | 3.45 |
| local `topk` | 2.00 |
| router scale | 0.37 |
| fused SwiGLU | ~0 |
| **sum** | **33.89** (against 33.28 for the block) |

and inside the router chain:

| piece | ms a token |
|---|---:|
| `ttnn.topk(probs, k=10)` over 512 | 5.00 |
| router GEMV \[2560, 512\], k-split | 2.29 |
| `softmax` | 1.12 |
| threshold / ge / multiply / sum / divide | 0.93 |
| `mesh_partition` | 0.73 |

The measurement that decided what to build: **removing four ops from the router
moved the step 0.24 ms.** `sparsity` (a `max`, a `typecast`, a `to_layout`) and
the partition of `keep` were computed on every wide-path layer and never read --
192 ops a token, genuinely discarded -- and deleting them was worth almost
nothing. So the chain's 9.35 ms is not op count. It is two sorts.

INVARIANT 60: before fusing a chain, ablate its pieces. A chain of a dozen small
ops can have eleven of them free and one that is the whole cost, and the fusion
that removes the eleven buys nothing. Op counts price dispatches; ablation prices
work.

### 15.2 Deployed: the routing tail in one kernel, 82.11 -> 74.94 ms

`scripts/kernels/router_select.cpp` replaces the global `topk`, the
threshold/mask/normalise, the `mesh_partition` and the local `topk` -- 8.66 ms of
the step -- with one launch on one core. Nothing in it needs the FPU: after a
softmax every probability is positive, and positive IEEE floats order exactly as
their bit patterns do, so the selection is unsigned integer compares. Only the
normalisation is arithmetic, about a dozen adds and ten divides a row.

`scripts/stage7_router_select.py` measures **151.59 -> 27.18 us, 5.58x**, and
checks it three ways, because "differs from the ttnn ops" is not "less accurate"
(invariant 57):

* against float64: kernel 6.914e-04, chain 6.914e-04 -- **identical**, both
  dominated by the bfloat16 rounding of the weights they emit;
* against the chain: 3.05e-05 on the weights, but 2 of 10 carrying slots hold a
  *different index*;
* against the chain on the **effective per-expert weight vector** -- sum over
  slots of weight x one-hot expert, which is the only thing the gather and the
  matmul downstream can see: 3.05e-05.

That third check is the one that matters. The index differences are ties: bf16
probabilities collide often, and two experts of equal weight in swapped slots
produce the same weighted sum. Padding slots differ too, and carry weight zero.

Quality, measured with the kernel switched off and on in the same session
(`TT_NO_FUSED_ROUTER=1`), on 191 tokens: **top-1 72.3 % against 71.2 %, NLL 1.194
against 1.221**, top-5 equal. The kernel is ahead. On the 47-token sample it read
0.674 against 0.663, which is why the longer run was done -- a 47-token NLL moves
by more than this effect.

INVARIANT 61: when a change alters a *selection*, compare the quantity the
consumer actually reads, not the intermediate. Two selections that disagree slot
by slot can be the same input to everything downstream, and a slot-by-slot
comparison will report a difference that does not exist.

### 15.3 The wide MoE path was exact at one row only

Found while wiring the above, and it is a correctness bug rather than a
performance note. `wide_expert_ffn` gathers **one** set of experts per call, and
`expert_gather.cpp` reads that selection from the index tile's first face -- row
0. At M > 1 every row was therefore routed to row 0's experts.

`scripts/dev/moe_rows_check.py` measures it directly: with the wide path on, row
groups of 8 got **55 of 64 rows wrong, worst row 96.3 %**, where the file's own
comment (written against the `sparse_matmul` path) records 1, 8, 16 and 32 as
exact. The check had been recording the regression since the wide path landed.

Decode at batch 1 runs M=1 and is unaffected, which is why nothing else showed
it. `step_n` -- the speculative verifier, sections 8 and 9 -- runs M = k and was
silently taking the wide path. **Any speculation result measured after the wide
path landed was measured on a wrong verifier.**

Now guarded on `x.shape[-2] == 1`; M > 1 takes `sparse_matmul`, which is per-row
exact. The check is back to its documented shape, with one residual: group sizes
8/16/32 now show a worst row of 1.961 % rather than 0.000 %, because the
single-row *reference* is itself built through the wide path while the groups run
sparse, so what is left is the wide-vs-sparse numerical difference and not a
row-grouping error.

INVARIANT 62: a kernel that reads its parameters from row 0 of a tensor is a
kernel that only works at one row. `generic_op` will not tell you; the shapes are
all valid. Guard it at the call site, in the code, not in a comment.

### 15.4 Rejected: keeping the gathered weights in L1

The gathers cost 13.19 ms to move weights the matmul then reads again -- 73.8 MB
a layer where 24.6 MB is the requirement. The cheap fix, if it worked, would be
to land the gather in L1 instead of DRAM and let `ttnn.linear` read it there, no
new kernel needed.

`scripts/stage6_l1_weights.py` measures it and the answer is no, twice over. The
gate|up shape is refused outright -- `MatmulMultiCoreProgramConfig: Input B memory
layout must be INTERLEAVED, got WIDTH_SHARDED`. The down shape is accepted, gives
a bit-identical answer, and runs at **exactly the same speed**: 128.94 us with
the weight in DRAM, 129.34 us with it L1-sharded across 80 cores.

INVARIANT 63: at M=1 these matmuls are not limited by where their weight lives.
`[6400, 2560]` bfloat8_b is 8.19 MB in 129 us -- 63 GB/s, a sixth of bandwidth --
and removing the DRAM read entirely changes nothing. Moving weights closer is not
the lever; issuing less work is.

### 15.5 Deployed: the experts shard on the intermediate axis

`WIDE_EXPERTS` was 10 because the top-10 experts land unevenly on four devices
and **all ten could arrive on one**. The expected count is 2.5, so three quarters
of both gathers and both matmuls was padding carrying zero weight -- 22.2 ms of
the 33.3.

Sharding the other way removes the worst case instead of paying for it. Every
device holds all 512 experts at a quarter of the intermediate width:

    was     gate|up  [128 experts, 2560, 1280]   gather 10 x 2560 x 1280 = 16.4 MB
    now     gate|up  [512 experts, 2560,  320]   gather 16 x 2560 x  320 =  6.6 MB

Same bytes resident either way, same arithmetic, and no approximation: the down
projection is already sharded on its contraction dim and the MoE already
all-reduces, so this only changes what the partial sum is over -- experts
becoming columns. `Shard.EXPERT_COLUMN` and `Shard.EXPERT_ROW` already existed
in `plan.py` and already do exactly this.

The global `k_sel` has to cover the tie admission rather than the mean, because
all four devices now select the same list. `scripts/dev/kept_expert_census.py`
measures it over 2256 real routing decisions: 10 experts kept 77 % of the time,
11 18 %, 12 3 %, and a maximum of **14**. Sixteen is above that tail and is also
the gather kernel's own limit, since the selection is read from one tile face.

`scripts/stage8_expert_shard_axis.py`, at k_sel 10 on the old axis against 16 on
the new one:

| | gathers | gate\|up matmul | down matmul | total |
|---|---:|---:|---:|---:|
| expert axis (was) | 11.22 | 3.30 | 6.19 | 20.70 ms |
| intermediate axis | 4.68 | 2.63 | 2.57 | **9.88 ms** |

**2.10x, 10.83 ms a token.**

This reverses `scripts/reshard_experts.py`, whose docstring records why the
expert axis was chosen: `sparse_matmul` zero-fills its whole `[1, E, M, N]`
output, and 512 experts a device made that 84 MB a layer against 21 (invariant
27). That reason is now **stale for decode** -- the wide gather path retired
`sparse_matmul` from decode entirely -- but it still holds for prefill, which
routes through `apply_experts`. Only the down projection's output grows, and it
grows four times.

INVARIANT 64: a sharding decision is a decision about *which op reads the
weight*, so it expires when that op is replaced. The expert axis was right while
`sparse_matmul` ran the decode MoE and wrong the moment the gather did, and
nothing in the code would have said so -- the plan entry outlives the reason
written next to it.

### 15.6 Landed: 82.35 -> 62.40 ms, and what each piece was worth

| | ms a token |
|---|---:|
| before this round | 82.35 |
| `router_select.cpp` (15.2) | **74.94** |
| dead `sparsity` ops, dead `gated` multiply | 74.70 |
| experts on the intermediate axis | **62.40** |

The dead-code half of that is worth saying out loud. Two computations were being
issued on every layer of every token and read by nothing:

* `sparsity` -- a `max`, a `typecast` and a `to_layout`, plus the partition of
  `keep` -- built for `sparse_matmul`, which the wide path stopped calling. 192
  ops a token.
* `gated = ttnn.multiply(mix, normed)` in `gated_residual_mix`, left behind when
  `fused_gated_mean` took the work over. A 10240-wide multiply on a tile padded
  from one row to thirty-two, 96 times a token.

An AST scan for assigned-and-never-read locals across the five hot modules found
no third one, which is the reason to run it rather than to keep looking by eye.

Quality is unchanged where it counts: **top-1 72.3 % and top-5 89.5 % on 191
tokens, identical before and after the reshard**; NLL moved 1.194 -> 1.211. The
reshard changes what each device's partial sum is over -- intermediate columns
rather than experts -- so a 1.4 % NLL move is summation order, not accuracy.
`moe_rows_check.py` also improved: worst row 1.961 % -> 1.562 %, and 6 of 64 rows
over 1 % where it had been 22.

### 15.7 Rejected: fusing attn_qkv with attn_gate

They read the same `mixed` and are both column shards, so `_fused_pair`
concatenates them into one `[2560, 4096]` matmul. Measured with a switch so both
ran in the same session: **62.40 ms fused, 62.37 ms unfused**, ranges
overlapping. It also keeps a third copy of both weights resident, ~376 MB a
device. Reverted.

That is invariant 62 a second time, and the sharper form of it: 2560 and 1536
output columns already spread over enough of a 110-core grid that widening to
4096 buys no bandwidth, and the two slices it costs are not free either. The
rule that *does* predict these is invariant 38 -- widening helps while the grid
is starved, and these were not.

### 15.8 The routing rule: the reference is exact top-k, and the threshold beats it

`router_select.cpp` can do either rule, and the switch is worth 4 ms:

| | step | top-1 (191) | top-5 | NLL |
|---|---:|---:|---:|---:|
| threshold, `k_sel` 16 | 62.40 ms | **72.3 %** | **89.5 %** | **1.211** |
| exact top-k, `k_sel` 10 | **58.42 ms** | 70.2 % | 89.0 % | 1.223 |

The reference is `torch.topk(probs, num_experts_per_tok)` then normalise
(`reference/model.py:401`) -- **exact top-k, no tie admission**. So the faster
rule is also the faithful one, and this project's threshold form is the
approximation. It nevertheless scores better here, by 4 tokens in 191, which is
inside one standard deviation (~6.2) and therefore does not settle anything.
Default stays on the threshold until a larger sample says otherwise;
`TT_ROUTER_EXACT_TOPK=1` picks the other.

Worth keeping in view: the tie admission is what forces `k_sel` to 16 rather than
10, and that 1.6x lands on the gather and both matmuls -- the largest single
item left in the MoE.

### 15.9 The op census, and what the 62 ms is made of

    one step: 5897 ttnn calls, 3.31 GB moved   (was 6426 calls, 4.72 GB)

| op | calls | GB | us/call |
|---|---:|---:|---:|
| multiply | 1044 | 0.077 | 0.2 |
| slice | 828 | 0.005 | 0.0 |
| reshape | 708 | 0.015 | 0.1 |
| **linear** | **484** | **2.745** | **14.6** |
| add | 473 | 0.081 | 0.4 |
| sigmoid | 326 | 0.004 | 0.0 |
| permute | 269 | 0.005 | 0.1 |
| rms_norm | 256 | 0.005 | 0.1 |

The budget closes, and this is the first time it has:

    5897 ops x 5.5 us (invariant 42's floor)          32.4 ms
    484 linears, 2.745 GB at 25-33 % of bandwidth     ~21 ms
    everything else's bytes                            ~9 ms
                                                      -------
                                                      ~62 ms   against 62.40 measured

So **62 % of the step is six op names that move almost no data**: multiply,
slice, reshape, add, sigmoid and permute are 3648 calls between them and 0.187
GB. They are not work; they are the floor, and only fusion removes them.

For the 32.6 ms target that is now a concrete arithmetic: the op count has to
reach roughly 2700 and the linears have to run at about 60 % of bandwidth
instead of 30. Each fused kernel so far has removed 300-800 calls, so the first
half is six or seven more of them.

### 16. The model's full context runs: a compact page table for QSA

At `max_seq_len` 262144 the decode attention was doing two wrong things at once.

`indexer_max_seq` is 65536 because `ttnn.scatter` takes uint16 indices and a
dense mask row is one column per cache position, so past 64 k tokens
`use_indexer` turned itself **off** and the twelve QSA layers ran plain causal
attention -- not the model. And that read the whole cache every layer: 262144
positions x 2 kv heads x 256 head_dim x 2 bytes x 2 (K and V) is 537 MB a layer,
**6.4 GB a token**, 16.6 ms at 388 GB/s before a weight is touched.

Both are the same problem, and the paged cache already had the fix in it. The
selection names `indexer_topk` = 512 blocks of 4 tokens, and those live in at
most 512 pages of the 32-token paged cache. Hand SDPA a page table listing
**only those pages** and:

* it reads `compact_slots * 32` = 17408 positions however long the context is;
* the mask is 17408 wide, which uint16 addresses with room to spare, so the
  selection runs to the model's maximum.

Two properties had to hold and neither is documented, so
`scripts/stage9_compact_page_table.py` checks them against a torch reference: a
page table **shorter than the cache** is accepted with `cur_pos` bounding the
compact window, and **repeated entries** are allowed -- two selected 4-token
blocks often share one 32-token page, and each slot then enables only its own
four columns, so nothing is counted twice and no de-duplication is needed. Both
hold; the answer matches torch over the selected positions to 2.7e-02 relative,
which is bfloat16 attention noise.

INVARIANT 65: a paged SDPA page table is a *selection mechanism*, not just an
address map. Listing a subset of pages, in any order, with repeats, re-addresses
the read -- which is how a sparse-attention model gets to read sparsely without
a gather.

### 16.1 It selects exactly what the dense mask does

`scripts/dev/compact_window_equivalence.py` decodes 3000 tokens at
`max_seq_len` 32768 -- the one regime where both paths can run, above
`compact_len` and below the uint16 reach -- and asks the same state for both
masks, since `_indexer_mask` has no side effects:

    compact: 2049 mask columns of 17408 open, 2049 distinct positions
    dense:   2049 distinct positions of 32768 columns
    compact-only 0, dense-only 0          -> IDENTICAL selection

2049 open columns mapping to 2049 *distinct* positions is the check that matters:
it is what proves the repeated pages are not double counting.

The same run prints the cost honestly. At 3000 tokens of context SDPA reads 11.7
MiB a layer dense and **68.0 MiB compact** -- compact is 5.8x worse, because the
window is fixed while the dense read grows with the sequence. Break-even is at
`compact_len` = 17408 tokens, which is exactly where the gate sits. At 262144 it
is 1024 MiB against 68, and the twelve QSA layers go from ~31.7 ms a token to
~2.1.

The gate is on `max_seq_len` and not on the current position, because the mask
width is a captured trace's shape. A session opened at 262144 therefore pays
~1.7 ms a token while it is short. It cannot do better with the dense mask
either -- that row is `max_seq_len` wide whatever the position, so at 262144 it
cannot be scattered at all.

The 8x inside the window is the last slack: a slot is a 32-token page and only 4
of its rows carry a selected block. Matching `KV_BLOCK` to the compression ratio
would make the window 2048 wide instead of 17408, and is the obvious next thing
to try if long-context decode matters more than it does today.

### 16.2 Deployed: the indexer scores all four heads in one matmul

The block scores were four matmuls of `[n_blocks, 128] x [128, 1]`, a relu each
and three adds -- nineteen ops a layer, and **four reads of the block cache**.
relu is elementwise and the sum is over heads, so relu-then-sum is the same
function with the heads on the output axis: one `[n_blocks, 128] x [128, 4]`,
one relu, and a matmul against a constant `[4, 1]` of ones (rather than
`ttnn.sum`, whose reduction axis would be 4 wide in a 32-wide tile).

Five ops instead of nineteen, 168 fewer a token, and one read of a cache that is
16.8 MB a layer at the model's maximum instead of four.
`scripts/dev/indexer_select_check.py` puts it against the reference mask at
position 2599: **2048 of 2048 tokens, 100 % overlap, no block missing or extra.**

## 17. What a fused kernel is actually worth, and the two halves of the 58 ms

The fused `reinject` (four ops a call, 96 calls a token) removed 288 ttnn calls
and bought **0.40 ms**, where the op floor said 288 x 5.5 us = 1.6 ms. That gap
is the useful result, and chasing it produced the first measurement in this
project that prices an op by its *shape*.

`scripts/dev/elementwise_shape_cost.py`, same element count at different padding,
inside a trace:

| shape | tiles | physical | multiply | sigmoid | GB/s |
|---|---:|---:|---:|---:|---:|
| [1,1,1,10240] | 320 | 640 KB | 6.75 us | 5.78 us | 291 |
| [1,1,32,10240] | 320 | 640 KB | 6.68 us | 5.80 us | 295 |
| [1,1,1,2560] | 80 | 160 KB | 5.82 us | 5.71 us | 85 |
| [1,1,32,320] | 10 | 20 KB | 5.80 us | 5.74 us | 11 |
| [1,1,1,32] | 1 | 2 KB | 5.83 us | 5.75 us | 1 |

**Invariant 42 is exactly right, and now it is explained.** A 320-tile multiply
costs 16 % more than a one-tile multiply while moving 320 times the data -- so it
runs at 291 GB/s, three quarters of peak, and its bytes are very nearly free. The
5.8 us is dispatch. The M=1 row padding that looked like a 32x tax on every
elementwise op costs almost nothing, because the op was never bandwidth-bound.

INVARIANT 66: at these shapes an elementwise op's bytes are free and its launch
is not. Do not fuse to save bytes; fuse to save launches. And a `generic_op` is
itself worth about two of those launches once its own work is counted, so
**fusing N ops saves roughly (N - 2) x 5.8 us a call**. A four-op chain saves
two ops' worth; a thirteen-op chain saves eleven.

That is why `reinject` returned 0.4 ms rather than 1.6: it replaced four ops with
a kernel that reads three tiles per output tile and synthesises a broadcast, and
that kernel costs about what two of the ops did.

### 17.1 The two halves, and what each needs

    5897 calls, of which 484 are linear
    5413 small ops x 5.8 us                    31.4 ms
    484 linears, 2.745 GB, 14.6 us each at peak, measured ~56    27.0 ms
                                               -------
                                                58.4 ms   against 58.4 measured

So the step is two roughly equal halves and they need opposite things.

**The small-op half is op count, and only long chains pay.** Ranked by what
`(N - 2) x 5.8 us x calls` would return:

| chain | ops | calls | worth |
|---|---:|---:|---:|
| `linear_attn.decode_step` | 13 | 36 | 2.3 ms |
| the `gated_residual_mix` tail (silu, sigmoid, slices, multiply, permute, mean) | ~7 | 96 | 2.8 ms |
| DeltaNet q/k/v prepare (slices, reshapes, two l2norms) | ~10 | 36 | 1.7 ms |
| DeltaNet gate chain (add, softplus, multiply, exp, sigmoid, reshapes) | 9 | 36 | 1.5 ms |
| `grouped_rms_norm` | 4 | 96 | 1.1 ms |
| DeltaNet output chain | 6 | 36 | 0.8 ms |

About 10 ms in six kernels, none of them large. The short chains are not worth
touching: a three-op fusion returns 5.8 us a call and costs a day.

**The linear half is bandwidth, and it is at 26 %.** 484 calls whose roofline is
14.6 us each measure about 56. Invariant 38 says why -- output width decides how
many cores get work -- and the k-split kernel (section 14) is the tool, but it
declines every shape whose output already fills the grid, which is most of them
now. Getting these to 50 % is worth ~13 ms and is the single largest item left in
the model.

For the 32.6 ms target: ~10 ms of fusion and ~13 ms of matmul efficiency lands at
about 35. Neither half is speculative -- both have a measured mechanism and a
worked example -- but neither is one insight either.

## 18. The k-split was returning zeros, and had been since it shipped

`ksplit_linear` computed its output-tile count as `w.shape[-1] // 32`. The fused
`ssm_alpha|ssm_beta` weight is **24 columns wide**, so that is zero, and
`_ksplit_build`'s plan --

    for g in range(groups):
        for n in range(nt):          # nt == 0
            plan.append(...)

-- produced no work items at all. Every core was padded to the idle entry, the
`generic_op` ran and wrote nothing, and the untouched zero output buffer went
through `ttnn.sum` and came back as zeros.

Nothing raised. The guard is `groups = max(1, min(kt, n_cores // max(nt, 1)))`,
which is 80 here -- comfortably past `groups >= 2` -- so `ksplit_linear` returned
a tensor rather than None and the caller's `linear_rows` fallback never ran.

So from "perf: apply the k-split to the router and alpha|beta" until this was
found, **all 36 DeltaNet layers ran with `a = b = 0`**: the decay was
`a_decay * softplus(dt)`, a constant per head, and `beta` was `sigmoid(0) = 0.5`.
The gate did not depend on the token.

Repairing it (192 tokens, everything else equal):

    top-1   70.2 % -> 72.3 %       top-5   89.0 % -> 92.1 %
    NLL     1.223  -> 1.202        median NLL 0.415 -> 0.236

The other two call sites were unaffected -- the router is `[2560, 512]` and the
fused down|inject is padded to 352, both whole tiles -- which is why only the
DeltaNet showed it, and it showed as *quality*, which nothing was watching
per-layer.

INVARIANT 67: a kernel that can be handed an empty work plan must refuse it. The
guard here tested whether the split was *worth* doing and never whether it would
*do* anything, and the two are not the same question. `ops.output_tiles` is now a
function precisely so a test can hold the property instead of a source line
(`tests/test_ksplit_plan.py`).

INVARIANT 68: a speedup measured on a path that was computing the wrong thing is
not a speedup. Re-timed once it worked, the alpha|beta split is **59.81/59.05 ms
against 59.67/58.92** for the `linear_rows` it replaced -- no gain either way --
and `ssm_alpha`/`ssm_beta` are float32 on purpose while the split accumulates in
the activation's bfloat16. It is off by default now; `TT_AB_KSPLIT=1` restores it.

### 18.1 Where the linears' 27 ms sits, by shape

`scripts/dev/linear_shape_census.py` records every `(K, N, dtype)` a decode step
issues and times each once, ranked by `calls x measured`:

| ms/token | calls | us | roofline | eff | shape |
|---:|---:|---:|---:|---:|---|
| 2.97 | 48 | 61.93 | 11.88 | 19 % | [2560, 3200] bf4 — MoE gate\|up gather |
| 1.96 | 48 | 40.79 | 8.97 | 22 % | [2560, 1280] bf8 — shared expert gate\|up |
| 1.87 | 36 | 51.88 | 32.30 | 62 % | [2560, 4608] bf8 |
| 1.67 | 48 | 34.80 | 11.22 | 32 % | [1600, 2560] bf8 — MoE down gather |
| 1.56 | 36 | 43.21 | 10.77 | 25 % | [2560, 1536] bf8 — attn_gate |
| 1.50 | 97 | 15.46 | 8.97 | 58 % | [320, 10240] bf8 — hc_\*_up |
| 1.49 | 12 | 124.18 | 43.07 | 35 % | [6144, 2560] bf8 — attn_output |
| 1.11 | 12 | 92.33 | 86.14 | **93 %** | [2560, 12288] bf8 — attn_q\|gate |
| 1.06 | 48 | 22.13 | 0.03 | **0.1 %** | [2560, 1] f32 — shared expert gate |

The shape at 93 % is the proof that nothing structural is in the way: a wide
enough output *does* saturate. The one at 0.1 % is the opposite extreme and was
the reason the tile-count bug was found at all.

That gate column is worth noting as a **negative** result: routed through the
k-split -- which the tile-count fix finally lets it reach -- it is worth nothing
measurable, 59.79 ms against 59.60 over two pairs. One output tile becomes 80
partials plus a `ttnn.sum`, and the sum is another op at the same 5.8 us floor.
Below about 20 us a call there is nothing left for a split to win.

The two that matter are the first and second: 4.9 ms between them at ~20 % of
bandwidth, both of them the MoE's own matmuls, and both fed by a gather.

## 19. The decode matmul was reading two k-tiles at a time: 59.2 -> 51.7 ms

The linears were 27 ms of a 59 ms step at about 26 % of bandwidth, and invariant
38 said why: output width sets how many cores get work. That turns out to be the
wrong diagnosis for most of them, and finding out took three eliminations.

**Not bandwidth.** `scripts/dev/matmul_dtype_width.py` runs the same widths at
three weight dtypes. At `[2560, 3200]`, bfloat4_b, bfloat8_b and bfloat16 all
take **61.4 us** -- identical, while the weight is 4.6, 8.7 and 16.4 MB. A
matmul whose time does not move when its operand quadruples in size is not
reading-bound.

**Not the FPU.** `scripts/dev/matmul_fidelity.py`: HiFi4, HiFi3, HiFi2 and LoFi
are within 2 % on every shape in the census. The four-pass mode costs nothing
here, which also means it buys nothing -- these operands are 7 and 3 mantissa
bits.

**It is the K loop, at ~0.65 us a k-tile** across every shape: `[2560, 1280]`
40 us for 80 tiles, `[1600, 2560]` 35 for 50, `[6144, 2560]` 124 for 192. That
is read *latency*, and `in0_block_w` -- how many k-tiles a core pulls before it
computes -- is the knob. `ttnn.linear` with no program config picks **two**.

`scripts/dev/matmul_program_config.py`, at `in0_block_w = 8`:

| shape | default | blk 8 | | error vs float64 |
|---|---:|---:|---:|---|
| MoE gate\|up [2560, 3200] | 61.35 us | **20.19** | 3.04x | 1.37e-02 (default 1.45e-02) |
| attn_output [6144, 2560] | 123.55 | **47.32** | 2.61x | 2.52e-02 (2.33e-02) |
| shexp gate\|up [2560, 1280] | 40.54 | **15.82** | 2.56x | 2.60e-02 (1.28e-02) |
| MoE down [1600, 2560] | 34.55 | 17.48 (blk 50) | 1.98x | 3.97e-02 (2.01e-02) |

Blocking the *whole* K at once is worse on both counts -- slower than 8 and
4.69e-02 -- so eight is a real optimum rather than the largest value that fits.
Four is the accuracy optimum (1.16e-02 on the gate\|up, better than either) at
about 1.8-2.4x, which is the setting to reach for if the error ever matters.

Deployed as `ops.fast_linear`, which builds and caches a
`MatmulMultiCoreReuseMultiCast1DProgramConfig` per shape and falls back to the
plain op for anything the config rejects. **Two A/B pairs: 51.68 and 51.72 ms
against 59.39 and 58.98.**

INVARIANT 69: at M=1 these matmuls are latency-bound on their weight reads, not
bandwidth-bound and not compute-bound. The knob is `in0_block_w`, the default is
2, and 8 is worth 2.5-3x. Invariant 38's "output width decides it" describes the
*narrow* shapes; for everything wider than a few tiles this is the real limit,
and the two are separate problems.

### 19.1 The quality harness is not deterministic, and never was

Chasing an inconsistency in the numbers above turned up something that
retroactively qualifies every quality claim in this document. Four runs of
`device_quality.py 192` on **identical code**:

    top-1  137, 140, 140, 143  of 191
    top-5  171, 172, 173, 174
    NLL    1.225, 1.211, 1.205, 1.213

A spread of six tokens in top-1 and 0.02 in NLL, with nothing changed between
runs. The likely cause is the collective: `ttnn.all_reduce` over four devices
sums its partials in whatever order they arrive, float addition is not
associative, and the MoE and the hyper-connection each all-reduce every layer.

INVARIANT 70: on this rig a difference of six top-1 tokens in 191, or 0.02 of
NLL, is **noise**. Any comparison closer than that needs repeated runs or a
larger sample, and several in this session were closer than they looked. The two
that survive it are the k-split zero bug (median NLL 0.415 -> 0.236, against a
0.226-0.232 spread) and the exact-vs-threshold router rule, which was
indistinguishable and is now known to have been *comfortably* indistinguishable.
