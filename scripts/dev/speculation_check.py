"""Does speculation change the answer, and does it go faster?

Both engines run with the trace on. `speculation_report()` breaks a round down
by phase, which is the only reliable way to read this: inferring from end-to-end
rates hid a parameter mix-up for one cycle and an eager decoder for another.

Exactness needs two *processes*, not two engines -- a second engine in one
process still hangs -- so `TWTEST_SPEC_ONLY` runs one configuration and prints
its tokens for comparison across runs.

    uv run python scripts/dev/speculation_check.py [k]            (default 8)

Greedy acceptance *does* make speculation exact here, which this file denied
for a while. `speculation_exactness_check.py` drives the round loop directly
against a plain sequential decode in one process and gets identical tokens at
k=2, 8 and 17. So the token comparison below is a pass/fail after all -- and if
it ever fails, that is an engine bug rather than a property of the scheme.

Two prompts, because prompt-lookup drafting is a bet on repetition: one that
quotes its context back (where it should pay) and one that does not (where it
should cost nothing).
"""
import asyncio
import faulthandler
import os
import sys
import time
from pathlib import Path

from ttrunner_qwen38_flash_next.engine import GenerationRequest
from ttrunner_qwen38_flash_next.tt.engine import TTEngine

# TWTEST_STACK_DUMP=<seconds> dumps every thread's stack on a timer. The
# speculative engine spins somewhere in the serve loop -- setup completes and no
# token is ever emitted -- and six candidate causes were excluded by rebuilding
# the setup elsewhere before anyone simply asked the process where it was.
if os.environ.get("TWTEST_STACK_DUMP"):
    faulthandler.dump_traceback_later(
        float(os.environ["TWTEST_STACK_DUMP"]), repeat=True, exit=False
    )

K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
N = 48
GGUF = os.environ.get("TWTEST_GGUF_DIR", str(Path.home() / "models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS"))
CACHE = os.environ.get("TWTEST_TT_CACHE", str(Path.home() / "models/qwen38-tt-cache"))
TOKENIZER = os.environ.get(
    "TWTEST_TOKENIZER", str(Path.home() / "models/Qwen3.8-Flash-Next-tokenizer/tokenizer.json")
)

PROMPTS = {
    "copy-heavy": (
        "Document: The maintenance window runs from 02:00 to 04:00 UTC on the "
        "first Sunday of each month. During the window the write path is "
        "unavailable and reads are served from the replica set.\n\n"
        "Question: when does the maintenance window run and what happens to the "
        "write path?\n\nAnswer: the maintenance window runs from"
    ),
    "open prose": (
        "The Rosetta Stone is a granodiorite stele inscribed with three versions "
        "of a decree issued in Memphis in 196 BC. The top and middle texts are in"
    ),
}


async def run(engine, text):
    ids = engine.encode(text)
    t0 = time.perf_counter()
    first = None
    out = []
    async for ev in engine.generate(
        GenerationRequest(prompt_token_ids=ids, max_tokens=N, temperature=0.0)
    ):
        if ev.token_id >= 0:
            if first is None:
                first = time.perf_counter()
            out.append(ev.token_id)
    total = time.perf_counter() - t0
    # Generation rate, excluding prompt ingestion -- which dominates the total
    # and is not what speculation changes.
    gen = (time.perf_counter() - first) / max(len(out) - 1, 1) if first else 0.0
    return out, total, gen


async def main():
    results = {}
    # ONLY=2 runs just the speculating engine, so a single engine exists in the
    # process. Two engines in one process was a confound worth removing: the
    # first one's captures used to leak (TTEngine.close now releases them).
    only = os.environ.get("TWTEST_SPEC_ONLY")
    specs = (int(only),) if only else (0, K)
    for spec in specs:
        engine = TTEngine(
            cache_dir=CACHE, gguf_dir=GGUF, tokenizer_path=TOKENIZER,
            max_concurrency=1, max_seq_len=2048,
            # The trace stays on for both. It was briefly turned off here on the
            # theory that a step_n capture could not coexist with the decoder's
            # -- wrong, and it cost a measurement cycle: plain rounds ran eager
            # at 487.5 ms against a traced 236, which is the whole of the
            # overhead the earlier numbers showed.
            use_trace=True, speculate=spec,
        )
        try:
            for label, text in PROMPTS.items():
                toks, dt, gen = await run(engine, text)
                results[(spec, label)] = (toks, dt, gen)
                print(
                    f"RESULT speculate={spec} [{label}] {len(toks)} tokens, "
                    f"total {dt:6.2f}s, generation {1000 * gen:6.1f} ms/token",
                    flush=True,
                )
                # printed so two *separate* processes can be compared: two
                # engines in one process still hang, so exactness is checked
                # across runs
                print(f"RESULT TOKENS {spec} {label} {toks}", flush=True)
            if spec:
                print("RESULT --- where a round's time goes ---", flush=True)
                for line in engine.speculation_report().splitlines():
                    print(f"RESULT   {line}", flush=True)
        finally:
            await engine.close()

    if len(specs) < 2:
        return
    for label in PROMPTS:
        base, base_t, base_g = results[(0, label)]
        spec, spec_t, spec_g = results[(K, label)]
        print(
            f"RESULT [{label}] identical: {'YES' if base == spec else 'NO'}   "
            f"generation {base_g / spec_g:5.2f}x   total {base_t / spec_t:5.2f}x",
            flush=True,
        )
        if base != spec:
            print(f"RESULT [{label}]   base {base[:16]}", flush=True)
            print(f"RESULT [{label}]   spec {spec[:16]}", flush=True)


asyncio.run(main())
