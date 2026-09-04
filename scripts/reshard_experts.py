"""Re-convert the MoE expert stacks onto the expert axis.

    uv run python scripts/reshard_experts.py [layer_regex]      default: all

The expert stacks were sharded on the *intermediate* axis, so every device held
all 512 experts at a quarter width and `sparse_matmul` wrote [1, 512, M, K] per
layer -- 84 MB to carry ~59 KB of selected output (handoff invariant 27).
Sharding on the expert axis gives each device 128 whole experts for the same
bytes, and a quarter of the output.

Rewrites the cache in place, entry by entry, so it needs a few GB of scratch
rather than a second copy of a 130 GB cache. It is resumable: `_localise` reads
each layer's expert count off its own weight, so a half-converted cache runs
correctly (the converted layers simply run faster).

Re-run `scripts/fuse_expert_gate_up.py` afterwards -- the fused gate|up tensor is
built from the per-device gate/up shards and has to follow them.
"""
import os
import sys
import time
from pathlib import Path

from ttrunner_qwen38_flash_next.tt.convert import convert

GGUF = os.environ.get("TTRUNNER_GGUF_DIR", str(Path.home() / "models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS"))
CACHE = os.environ.get("TTRUNNER_TT_CACHE", str(Path.home() / "models/qwen38-tt-cache"))
layers = sys.argv[1] if len(sys.argv) > 1 else r"\d+"
pattern = rf"^blk\.{layers}\.ffn_(gate|up|down)_exps\.weight$"

print(f"RESULT re-sharding {pattern}", flush=True)
t0 = time.perf_counter()
stats = convert(GGUF, CACHE, n_dev=4, only=pattern, force=True,
                progress=lambda *a: print("   ", *a, flush=True))
print(f"RESULT done in {time.perf_counter() - t0:.1f}s: {stats}", flush=True)
