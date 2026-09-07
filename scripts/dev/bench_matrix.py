"""One traced decode measurement per (mode, batch, context).

    uv run python scripts/dev/bench_matrix.py <ctx> [--e2e]

Two modes, because they answer different questions:

* **function** -- `TracedDecoder.step` on `TTModel`. The device step and nothing
  else.
* **e2e** -- through `TTEngine.generate`, the path a served request takes:
  scheduling, sampling, detokenisation and the async loop on top of the step.

`ctx` is the context the model is **opened** at; the run decodes from near
position 0. Two attempts to measure at a *filled* context both failed and are
worth not repeating: setting the positions without filling the cache leaves the
paged attention with no page table and the run never returns, and
`TTModel.prefill` cannot hand its state to the decode path at all -- it leaves
the conv ring as `[1, 1, C, 1]` where a decode step wants `[1, 1, 1, C]`, which
is the mismatch its own docstring warns about ("not verified, do not wire this
into the engine"). Only `TTEngine` gets from a long prompt into a traced decode,
which is what the `--e2e` mode measures.

**batch 1 only.** This model is served to a coding agent, so the batch axis is
not the question; the same script took batch 2 and 8 and those columns were
dropped rather than kept as decoration.

**`selection_active` is swept, not fixed**, because it is the thing that makes a
long context expensive: below `indexer_budget` (2048) `TTEngine` runs with it
off, above it on. Each context is measured both ways and the difference is the
QSA selection's cost at that allocation.
"""
import asyncio
import os as _os
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device_model import open_model                                # noqa: E402

CTX = int(sys.argv[1])
E2E = "--e2e" in sys.argv
PREFILL = "--prefill" in sys.argv
TARGET = max(0, CTX - 64)          # room for the steps this run takes
BATCHES = [1]
ITERS = 15
GGUF = str(Path.home() / "models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS")
CACHE = str(Path.home() / "models/qwen38-tt-cache")
TOKENIZER = str(Path.home() / "models/Qwen3.8-Flash-Next-tokenizer/tokenizer.json")


def function_mode() -> None:
    mesh, cfg, m = open_model(max_seq_len=CTX)
    from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder

    print(f"RESULT ctx {CTX}  indexer {m.use_indexer}  ratio {m.indexer_ratio} "
          f"topk {m.indexer_topk}  blocks {m.max_blocks}", flush=True)
    state = m.new_state(batch=1)
    # A real prompt through the real chunked prefill, so the cache, the ring and
    # the position are all where `ctx` says they are. This matters for more than
    # tidiness: the indexer's `topk` searches only the blocks a position could
    # have filled, so a step taken near position 0 measures a much smaller search
    # than the same graph does at the end of the context.
    if TARGET > 0:
        t0 = time.perf_counter()
        m.prefill([1000] * TARGET, state)
        ttnn.synchronize_device(mesh)
        pre = time.perf_counter() - t0
        print(f"RESULT prefill ctx={CTX} tokens={TARGET} {1000 * pre:9.1f} ms "
              f"{TARGET / pre:8.1f} tok/s", flush=True)
    pos0 = int(state.positions[0])

    # `selection_active` is a Python bool the trace bakes at capture, so each
    # setting needs its own decoder and its own capture.
    # Below `indexer_budget` the indexer is not built, so `selection_active`
    # cannot do anything and the second arm would measure the same graph twice.
    sels = (False, True) if m.use_indexer else (False,)
    for sel in sels:
        try:
            m.selection_active = sel
            dec = TracedDecoder(m, state)
            for _ in range(3):
                dec.step([1000])
            ttnn.synchronize_device(mesh)
            ts = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                dec.step([1000])
                ttnn.synchronize_device(mesh)
                ts.append(1000 * (time.perf_counter() - t0))
            ts.sort()
            ms = ts[len(ts) // 2]
            print(f"RESULT function ctx={CTX} sel={int(sel)} pos={pos0} "
                  f"{ms:8.2f} ms/step  {1000 / ms:7.2f} tok/s", flush=True)
            dec.release()
        except Exception as exc:                                    # noqa: BLE001
            print(f"RESULT function ctx={CTX} sel={int(sel)} FAILED "
                  f"{type(exc).__name__}: {exc}", flush=True)
    ttnn.close_mesh_device(mesh)


async def e2e_mode() -> None:
    from ttrunner_qwen38_flash_next.tt.engine import TTEngine
    from ttrunner_qwen38_flash_next.engine import GenerationRequest

    # Inside the try: building the engine is itself a thing that fails, and
    # with the constructor outside it a failure printed nothing at all -- the
    # sweep just moved on to the next context leaving a blank cell.
    eng = None
    try:
        print(f"STAGE ctx={CTX} building engine", flush=True)
        eng = TTEngine(cache_dir=CACHE, gguf_dir=GGUF, tokenizer_path=TOKENIZER,
                       max_concurrency=1, max_seq_len=CTX, use_trace=True,
                       chunked_prefill=True)
        # A real prompt, long enough to put the decode at the context being
        # swept. This is the only route that reaches a genuinely filled cache,
        # so it is also the only place `selection_active` takes its real value
        # rather than one this script sets.
        print(f"STAGE ctx={CTX} engine up", flush=True)
        # PROMPT overrides the prompt length, to separate "long prompt"
        # from "large context" when the engine path misbehaves.
        want = int(_os.environ.get("PROMPT", 0)) or max(8, TARGET)
        ids = eng.encode(" the" * want)[:want]
        print(f"STAGE ctx={CTX} encoded {len(ids)} tokens", flush=True)

        marks = []
        async for ev in eng.generate(GenerationRequest(
                prompt_token_ids=list(ids), max_tokens=ITERS, temperature=0.0)):
            if ev.token_id >= 0:
                marks.append(time.perf_counter())
                if len(marks) == 1:
                    print(f"STAGE ctx={CTX} first token", flush=True)

        # Decode rate only: the first mark carries prefill, so measure between
        # marks, not from the request.
        if len(marks) < 3:
            print(f"RESULT e2e ctx={CTX} FAILED only {len(marks)} tokens", flush=True)
        else:
            span = marks[-1] - marks[0]
            ms = 1000 * span / (len(marks) - 1)
            print(f"RESULT e2e ctx={CTX} prompt={len(ids)} "
                  f"{ms:8.2f} ms/step  {1000 / ms:7.2f} tok/s", flush=True)
    except Exception as exc:                                        # noqa: BLE001
        print(f"RESULT e2e ctx={CTX} FAILED {type(exc).__name__}: {exc}", flush=True)
    finally:
        if eng is not None:
            await eng.close()


if __name__ == "__main__":
    if E2E:
        asyncio.run(e2e_mode())
    else:
        function_mode()
