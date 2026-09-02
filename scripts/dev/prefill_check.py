"""Does `prefill` produce the same next token as feeding the prompt through `step`?

    uv run python scripts/dev/prefill_check.py [len ...] [--chunk C ...]
                                            (default 4 16 64 128, chunk 128)

`--chunk` sweeps the prefill step size. `prepare()` pads a short sequence up to
the op's 128-wide chunk, so a smaller step is well defined, and the op's error
grows with position inside a chunk -- so this says whether trading prefill
throughput for a shorter step actually buys accuracy.

Prints one RESULT line per length. The step path is the oracle: it is the path
verified against the CPU reference.
"""
import sys

import ttnn

from _device_model import host_row, open_model, synthetic_prompt

argv = sys.argv[1:]
chunks = [128]
if "--chunk" in argv:
    i = argv.index("--chunk")
    chunks = [int(x) for x in argv[i + 1:]]
    argv = argv[:i]
lengths = [int(x) for x in argv] or [4, 16, 64, 128]
mesh, cfg, m = open_model()
for L in lengths:
  for C in chunks:
    P = synthetic_prompt(L)
    stp = m.new_state(batch=1)
    hp = m.prefill(P, stp, chunk=min(C, max(32, L - L % 32)), moe_chunk=8)
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
        f"RESULT len={L:4d} chunk={C:4d} prefill->{got:7d} step->{ref:7d} "
        f"{'MATCH' if got == ref else 'DIFFER'}"
        f"   hidden rel {100 * d / scale:7.2f}%", flush=True,
    )
ttnn.close_mesh_device(mesh)
