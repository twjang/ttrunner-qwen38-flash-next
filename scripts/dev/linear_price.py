"""What do the 460 `ttnn.linear` calls actually cost in the traced step?

    uv run python scripts/dev/linear_price.py [stub]

Handoff 45.29 puts 22.5 ms of the 32.65 ms traced step outside bytes and
dispatch, and the census says 460 linears carry 80 % of all bytes while running
at roughly a quarter of bandwidth. That is an inference from a budget
subtraction, and 45.14 closed `ksgemv` -- which targets exactly those calls --
as buying nothing. Both cannot be right.

The device profiler would settle it, but this wheel is not a Tracy build
(`TT_METAL_DEVICE_PROFILER requires a Tracy-enabled build`). So ablate instead:
replace every `fast_linear` with a cached zero tensor of the right shape and
time the step. The answer is wrong, the shapes are not, and the delta is what
the matmuls cost where they actually live -- no isolated-timing discount to
argue about (invariant 89).

Run it as `linear_price.py` for the baseline and `linear_price.py stub` for the
stubbed arm, in separate processes, so neither can contaminate the other's
program cache.
"""
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device_model import open_model                                # noqa: E402
import ttrunner_qwen38_flash_next.tt.ops as ops                     # noqa: E402

STUB = len(sys.argv) > 1 and sys.argv[1] == "stub"
# 512, like `step_op_census.py`: an eager step at 4096 does not finish in
# 700 s here, and the question is the linears' *share*, not the absolute ms.
SEQ = int(__import__("os").environ.get("TTRUNNER_MAX_SEQ", "512"))
mesh, cfg, m = open_model(max_seq_len=SEQ)
m.selection_active = False          # invariant 145: the engine's below-budget path

_real = ops.fast_linear
_cache: dict = {}


def stub_linear(x, w, **kw):
    """The output `fast_linear` would produce, without producing it."""
    # `list(...)` first: slicing a ttnn `Shape` raises TypeError, which is
    # the same trap as handoff 45.3.
    xs = [int(d) for d in x.shape]
    shape = tuple(xs[:-1]) + (int(w.shape[-1]),)
    dt = kw.get("dtype") or x.dtype
    key = (shape, str(dt))
    got = _cache.get(key)
    if got is None:
        # **Not zeros.** The router is a `fast_linear` too, and `moe_block`
        # selects experts by thresholding at the k-th largest probability --
        # which admits ties (moe.py:186). Zeroed logits make all 512 experts tie
        # at the threshold, so the expert path runs on every one of them and the
        # step never finishes: the first version of this probe timed out at 700 s
        # where the baseline stepped in one minute. A fixed random draw routes to
        # ~k experts like the real thing, which is what makes the arms
        # comparable. The answer is still wrong; only the shapes and the control
        # flow need to be right.
        torch.manual_seed(0)
        got = ttnn.from_torch(
            torch.randn(*[int(d) for d in shape]) * 0.05, dtype=dt,
            layout=ttnn.TILE_LAYOUT, device=x.device(),
            mesh_mapper=ttnn.ReplicateTensorToMesh(x.device()))
        _cache[key] = got
    return got


if STUB:
    ops.fast_linear = stub_linear
    import ttrunner_qwen38_flash_next.tt.model as mm
    import ttrunner_qwen38_flash_next.tt.moe as moe
    for mod in (mm, moe):
        if hasattr(mod, "fast_linear"):
            mod.fast_linear = stub_linear

# Fill `_cache` on an eager step *before* the decoder captures. Allocating a
# device buffer inside a trace capture is the hazard that has bitten this project
# repeatedly (45.25's eager-argument note), and the stub allocates on first use.
if STUB:
    warm = m.new_state(batch=1)
    m.step([1000], warm)
    ttnn.synchronize_device(mesh)
    print(f"RESULT stub cache primed: {len(_cache)} shapes", flush=True)
    del warm

from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder      # noqa: E402

state = m.new_state(batch=1)
dec = TracedDecoder(m, state)
dec.reset()
for _ in range(3):
    dec.step([1000])
ttnn.synchronize_device(mesh)
ts = []
for _ in range(21):
    t0 = time.perf_counter()
    dec.step([1000])
    ttnn.synchronize_device(mesh)
    ts.append(1000 * (time.perf_counter() - t0))
ts.sort()
print(f"RESULT {'linears STUBBED' if STUB else 'baseline':16s} "
      f"median {ts[len(ts) // 2]:7.2f} ms  min {ts[0]:7.2f}", flush=True)
dec.release()
ttnn.close_mesh_device(mesh)
