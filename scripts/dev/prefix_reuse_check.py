"""Does the engine really skip a chat history it already holds?

    uv run python scripts/dev/prefix_reuse_check.py [--chunked] [--trace]

Three turns of a conversation through `TTEngine` with one slot. Turn 2's prompt
is turn 1's prompt plus what turn 1 emitted, so the slot already holds it and
only the new tokens should be fed; turn 3 continues turn 2. A fourth request
with an unrelated prompt must *not* hit, or the bookkeeping is claiming a state
it does not have.

What is checked: `EngineStats.cached_prompt_tokens` accounts for exactly the
tokens that were reused, time to first token drops in proportion, and the
continuation is the same one a cold slot produces -- reuse that changes the
answer is not reuse.

Needs the whole engine, so ~2 min of weight loading before anything runs.
"""
import asyncio
import os
import time
from pathlib import Path

from ttrunner_qwen38_flash_next.engine import GenerationRequest
from ttrunner_qwen38_flash_next.tt.engine import TTEngine

GGUF = os.environ.get("TWTEST_GGUF_DIR", str(Path.home() / "models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS"))
CACHE = os.environ.get("TWTEST_TT_CACHE", str(Path.home() / "models/qwen38-tt-cache"))
TOKENIZER = os.environ.get(
    "TWTEST_TOKENIZER", str(Path.home() / "models/Qwen3.8-Flash-Next-tokenizer/tokenizer.json")
)
import sys

N = 6
CHUNKED = "--chunked" in sys.argv
TRACE = "--trace" in sys.argv


async def run(engine, ids, tag):
    t0 = time.perf_counter()
    first = None
    out = []
    async for ev in engine.generate(
        GenerationRequest(prompt_token_ids=ids, max_tokens=N, temperature=0.0)
    ):
        if ev.token_id >= 0:
            if first is None:
                first = time.perf_counter() - t0
            out.append(ev.token_id)
    print(f"RESULT {tag:22s} ttft {first:6.2f}s  tokens {out}  "
          f"cached {engine.stats.cached_prompt_tokens}", flush=True)
    return out, first


async def main():
    engine = TTEngine(
        cache_dir=CACHE, gguf_dir=GGUF, tokenizer_path=TOKENIZER,
        max_concurrency=1, max_seq_len=4096, use_trace=TRACE,
        chunked_prefill=CHUNKED,
    )
    print(f"RESULT chunked_prefill={CHUNKED} trace={TRACE}", flush=True)
    try:
        turn1 = engine.encode(
            "The Rosetta Stone is a granodiorite stele inscribed with three versions "
            "of a decree issued in Memphis in 196 BC. The top and middle texts are "
            "in Ancient Egyptian, using hieroglyphic and Demotic scripts, while the "
            "bottom is in Ancient Greek. Because the decree has only minor"
        )
        t1, ttft1 = await run(engine, turn1, "turn 1 (cold)")

        turn2 = turn1 + t1 + engine.encode(" and it was")
        before = engine.stats.cached_prompt_tokens
        t2, ttft2 = await run(engine, turn2, "turn 2 (continues 1)")
        reused2 = engine.stats.cached_prompt_tokens - before
        print(f"RESULT   reused {reused2} of {len(turn2)} prompt tokens "
              f"(expected {len(turn1) + len(t1) - 1})", flush=True)

        turn3 = turn2 + t2 + engine.encode(" found in")
        before = engine.stats.cached_prompt_tokens
        t3, ttft3 = await run(engine, turn3, "turn 3 (continues 2)")
        reused3 = engine.stats.cached_prompt_tokens - before
        print(f"RESULT   reused {reused3} of {len(turn3)} prompt tokens", flush=True)

        unrelated = engine.encode("A completely different sentence about weather and")
        before = engine.stats.cached_prompt_tokens
        await run(engine, unrelated, "unrelated (must miss)")
        print(f"RESULT   reused {engine.stats.cached_prompt_tokens - before} "
              f"(must be 0)", flush=True)

        # the answer must not depend on whether the state was reused
        cold, ttft_cold = await run(engine, turn2, "turn 2 again (cold)")
        print(f"RESULT turn2 warm == cold: {'YES' if cold == t2 else 'NO'}", flush=True)
        print(f"RESULT ttft warm {ttft2:.2f}s vs cold {ttft_cold:.2f}s", flush=True)
    finally:
        await engine.close()


asyncio.run(main())
