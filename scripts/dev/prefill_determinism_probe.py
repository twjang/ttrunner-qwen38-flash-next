"""Why does a 128-row prefill differ between processes but not within one?

    uv run python scripts/dev/prefill_determinism_probe.py [cold|warm]

`device_quality.py 250 --prefill 128` has returned 21.5 %, 25.2 % and 27.1 %
top-1 from the identical binary, stable within a batch of invocations and
shifting between them (handoff invariant 8). Within one process it is exact --
three prefills agree to 0.0000 % (`moe_chunk_noise.py`) -- so whatever varies is
fixed at process start.

`cold` prefills as the first thing the process does. `warm` runs eight decode
steps first, which compiles a different set of kernels and allocates before the
chunk path ever runs. If the two disagree, prior device work in the process
changes prefill's answer, which points at kernel selection or allocation rather
than at the arithmetic; if they agree, the cause is earlier still.

Prints a checksum of the final logits, so runs can be compared across processes.
"""
import sys

import torch
import ttnn

from _device_model import open_model, synthetic_prompt

MODE = sys.argv[1] if len(sys.argv) > 1 else "cold"
if MODE not in ("cold", "warm", "text"):
    raise SystemExit("mode must be cold, warm or text")

mesh, cfg, m = open_model(max_seq_len=512)
if MODE == "text":
    # The same tokens `device_quality.py` scores. Synthetic ids may simply lack
    # the near-ties in expert routing that a real passage has, and a tie is
    # where an otherwise invisible difference would become a different expert.
    import ast
    import pathlib as _pl

    from _device_model import tokenizer

    # Lift `TEXT` out of the source rather than importing it: that module runs
    # its whole measurement at import.
    src = ast.parse((_pl.Path(__file__).parent / "device_quality.py").read_text())
    text = next(
        ast.literal_eval(n.value) for n in src.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "TEXT"
    )
    prompt = tokenizer(cfg).encode(text)[:140]
else:
    prompt = synthetic_prompt(140)

if MODE == "warm":
    st = m.new_state(batch=1)
    for t in prompt[:8]:
        m.step([t], st)
    print("RESULT ran 8 decode steps first", flush=True)
    del st

st = m.new_state(batch=1)
hidden = m.prefill(prompt[:128], st)
lg = m.logits(hidden)[0].float()
print(f"RESULT mode {MODE}  argmax {int(lg.argmax())}  "
      f"sum {float(lg.sum()):.6f}  max {float(lg.max()):.6f}", flush=True)
top = lg.topk(5)
print(f"RESULT top5 {[int(i) for i in top.indices]} "
      f"{[round(float(v), 4) for v in top.values]}", flush=True)

# The prefill's own logits turned out identical across processes, so if anything
# varies it is the decode steps *after* one -- which is what
# `device_quality --prefill 128` actually scores.
toks = []
for i in range(8):
    h = m.step([prompt[128 + i]], st)
    l = m.logits(h)[0].float()
    toks.append((int(l.argmax()), round(float(l.sum()), 3)))
print(f"RESULT post-prefill decode {toks}", flush=True)
ttnn.close_mesh_device(mesh)
