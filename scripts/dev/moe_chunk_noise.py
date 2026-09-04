"""Is the `moe_chunk` cliff real, or is the pipeline just nondeterministic?

    uv run python scripts/dev/moe_chunk_noise.py [repeats]           (default 3)

The cap in `TTModel.prefill` was set from one `device_quality.py` run per
setting. Two runs at the *same* setting then disagreed (50.0 % and 53.1 % top-1
at moe_chunk=32), so that evidence cannot separate a real effect from run noise.

This compares the thing itself instead of a downstream score: prefill the same
prompt on a fresh state and read the final logits. Repeats at one `moe_chunk`
give the noise floor; differences across `moe_chunk` only mean something if they
clear it. `_MAX_MOE_CHUNK` is lifted here because the point is to re-derive it.
"""
import sys

import torch
import ttnn

from _device_model import open_model, synthetic_prompt

import ttrunner_qwen38_flash_next.tt.model as model_mod

REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SIZES = [16, 32, 64, 128]
model_mod._MAX_MOE_CHUNK = 128        # measuring is how the cap gets justified

mesh, cfg, m = open_model(max_seq_len=512)
prompt = synthetic_prompt(128)

runs: dict[tuple[int, int], torch.Tensor] = {}
for mc in SIZES:
    for r in range(REPEATS):
        st = m.new_state(batch=1)
        hidden = m.prefill(list(prompt), st, chunk=128, moe_chunk=mc)
        runs[(mc, r)] = m.logits(hidden)[0].float()
        print(f"RESULT prefilled moe_chunk={mc} repeat={r}", flush=True)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-9)) * 100.0


def top1(t):
    return int(t.argmax())


print(f"\nRESULT {'moe_chunk':>10} {'within-setting spread':>22} {'vs moe_chunk=16':>18} "
      f"{'argmax':>8}", flush=True)
ref = runs[(16, 0)]
for mc in SIZES:
    within = max(
        (rel(runs[(mc, i)], runs[(mc, j)]) for i in range(REPEATS) for j in range(REPEATS) if i < j),
        default=0.0,
    )
    across = rel(runs[(mc, 0)], ref)
    args = {top1(runs[(mc, r)]) for r in range(REPEATS)}
    print(f"RESULT {mc:10d} {within:21.4f}% {across:17.4f}% {str(sorted(args)):>8}", flush=True)

print("\nRESULT a real effect must clear the within-setting spread", flush=True)
ttnn.close_mesh_device(mesh)
