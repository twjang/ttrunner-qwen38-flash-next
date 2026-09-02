"""Does speculation change the answer, and does it go faster?

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
    out = []
    async for ev in engine.generate(
        GenerationRequest(prompt_token_ids=ids, max_tokens=N, temperature=0.0)
    ):
        if ev.token_id >= 0:
            out.append(ev.token_id)
    return out, time.perf_counter() - t0


async def main():
    results = {}
    for spec in (0, K):
        engine = TTEngine(
            cache_dir=CACHE, gguf_dir=GGUF, tokenizer_path=TOKENIZER,
            max_concurrency=1, max_seq_len=2048, use_trace=True, speculate=spec,
        )
        try:
            for label, text in PROMPTS.items():
                toks, dt = await run(engine, text)
                results[(spec, label)] = (toks, dt)
                print(
                    f"RESULT speculate={spec} [{label}] {len(toks)} tokens in {dt:6.2f}s "
                    f"({1000 * dt / max(len(toks), 1):6.1f} ms/token)",
                    flush=True,
                )
        finally:
            await engine.close()

    for label in PROMPTS:
        base, base_t = results[(0, label)]
        spec, spec_t = results[(K, label)]
        print(
            f"RESULT [{label}] identical: {'YES' if base == spec else 'NO'}   "
            f"speedup {base_t / spec_t:5.2f}x",
            flush=True,
        )
        if base != spec:
            print(f"RESULT [{label}]   base {base[:16]}", flush=True)
            print(f"RESULT [{label}]   spec {spec[:16]}", flush=True)


asyncio.run(main())
