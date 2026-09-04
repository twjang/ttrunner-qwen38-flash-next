"""Aggregate throughput of `TTEngine` under N concurrent requests.

    uv run python scripts/dev/bench_server.py [concurrency] [max_tokens]
                                                   (default 32, 32)

The README's "37.84 tok/s at 32 concurrent" was measured before the correctness
fixes in `docs/iterations/013` and `014`, on a model that emitted the wrong
token, so it needs re-taking. This drives the real engine through its async
`generate()`, which is what a served request goes through -- scheduling,
admission and all -- so it is lower than the raw `bench_batch.py` figure for the
same batch, and that gap is the engine's own overhead.

Every request gets a distinct prompt so prefix reuse does not quietly turn this
into a measurement of the cache.
"""
import asyncio
import os
import sys
import time
from pathlib import Path

from ttrunner_qwen38_flash_next.engine import GenerationRequest
from ttrunner_qwen38_flash_next.tt.engine import TTEngine

CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 32
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
# One round is not a measurement. `moe_chunk_sweep.py` published a 1.85x that was
# 1.32x because it timed a single unwarmed pass; the first round here pays for
# kernel compilation and trace capture, so it is discarded and the rest reported
# as a median (invariant 7).
GGUF = os.environ.get("TTRUNNER_GGUF_DIR", str(Path.home() / "models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS"))
CACHE = os.environ.get("TTRUNNER_TT_CACHE", str(Path.home() / "models/qwen38-tt-cache"))
TOKENIZER = os.environ.get(
    "TTRUNNER_TOKENIZER", str(Path.home() / "models/Qwen3.8-Flash-Next-tokenizer/tokenizer.json")
)


async def one(engine, text, out):
    ids = engine.encode(text)
    first = None
    n = 0
    async for ev in engine.generate(
        GenerationRequest(prompt_token_ids=ids, max_tokens=MAX_TOKENS, temperature=0.0)
    ):
        if ev.token_id >= 0:
            if first is None:
                first = time.perf_counter()
            n += 1
    out.append((first, n))


async def main():
    engine = TTEngine(
        cache_dir=CACHE, gguf_dir=GGUF, tokenizer_path=TOKENIZER,
        max_concurrency=CONC, max_seq_len=2048, use_trace=True,
    )
    try:
        # distinct prompts: prefix reuse across slots would measure the cache
        prompts = [
            f"Question {i}: describe what happens in step {i} of a rolling upgrade, "
            "covering the order of operations and what is verified before moving on.\n\nAnswer:"
            for i in range(CONC)
        ]
        print(f"RESULT concurrency {CONC}  max_tokens {MAX_TOKENS}  "
              f"{ROUNDS} rounds, first discarded", flush=True)
        e2e, gen = [], []
        for r in range(ROUNDS + 1):
            out: list = []
            t0 = time.perf_counter()
            await asyncio.gather(*(one(engine, p, out) for p in prompts))
            total = time.perf_counter() - t0
            tokens = sum(n for _, n in out)
            firsts = [f for f, _ in out if f]
            span = time.perf_counter() - min(firsts) if firsts else total
            tag = "warm" if r == 0 else f"round {r}"
            print(f"RESULT   {tag}: {tokens} tokens, {tokens / total:.2f} tok/s end to "
                  f"end, {tokens / span:.2f} tok/s generation-only", flush=True)
            if r:
                e2e.append(tokens / total)
                gen.append(tokens / span)
        e2e.sort()
        gen.sort()
        print(f"RESULT median {gen[len(gen) // 2]:.2f} tok/s sustained generation, "
              f"{e2e[len(e2e) // 2]:.2f} tok/s end to end", flush=True)
    finally:
        await engine.close()


asyncio.run(main())
