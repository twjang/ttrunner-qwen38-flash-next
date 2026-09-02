"""Does `prefill` produce the same next token as feeding the prompt through `step`?

    uv run python scripts/dev/prefill_check.py [len ...]     (default 4 16 64 128)

Prints one RESULT line per length. The step path is the oracle: it is the path
verified against the CPU reference.
"""
import sys

import ttnn

from _device_model import host_row, open_model, synthetic_prompt

lengths = [int(x) for x in sys.argv[1:]] or [4, 16, 64, 128]
mesh, cfg, m = open_model()
for L in lengths:
    P = synthetic_prompt(L)
    stp = m.new_state(batch=1)
    hp = m.prefill(P, stp, moe_chunk=8)
    got = m.greedy_tokens(hp)[0]
    hp_h = host_row(mesh, hp)
    del stp, hp
    sts = m.new_state(batch=1)
    for t in P:
        hs = m.step([t], sts)
    ref = m.greedy_tokens(hs)[0]
    hs_h = host_row(mesh, hs)
    d = (hp_h - hs_h).abs().max().item()
    scale = max(hs_h.abs().max().item(), 1e-6)
    del sts, hs
    print(
        f"RESULT len={L:4d} prefill->{got:7d} step->{ref:7d} {'MATCH' if got == ref else 'DIFFER'}"
        f"   hidden rel {100 * d / scale:7.2f}%  (maxdiff {d:.3f} of {scale:.3f})", flush=True,
    )
ttnn.close_mesh_device(mesh)
