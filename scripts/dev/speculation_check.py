"""Does speculation change the answer?

Not "does it go faster": it cannot yet. Capturing the `step_n` graph while the
decoder's trace is live wedges the device (docs/iterations/016), so speculation
runs with an eager decoder, where a non-drafted round costs 518 ms against a
traced 236. What can be established now is exactness -- and that is the property
the whole scheme rests on.

    uv run python scripts/dev/speculation_check.py [k]            (default 8)

Greedy acceptance makes speculation *exact*: the tokens it emits are the tokens
one-at-a-time decoding would have emitted, so the only honest test is to run both
and compare them token for token. Anything else is a benchmark of a different
model.

Two prompts, because prompt-lookup drafting is a bet on repetition: one that
quotes its context back (where it should pay) and one that does not (where it
should cost nothing).
"""
import asyncio
import os
import sys
import time
from pathlib import Path

from twtest.engine import GenerationRequest
from twtest.tt.engine import TTEngine

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
            # one live trace only: capturing step_n while the decoder's trace is
            # live wedges the device, so speculation runs with an eager decoder
            use_trace=not spec, speculate=spec,
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
