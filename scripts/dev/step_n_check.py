"""Does `step_n` on k tokens reproduce `step` called k times?

    uv run python scripts/dev/step_n_check.py [k ...]         (default 1 2 4 8)

`step_n` rides the k tokens on the batch axis -- which a step is flat in, up to
64 rows -- and unrolls only the two pieces that cannot be batched: the DeltaNet
convolution, whose window is the three rows before it, and the recurrence, whose
state is the previous row's. It exists so a speculative verifier can score k
drafts for about the price of one step instead of k.

Correctness first: the hidden it produces at every one of the k positions must
match what the sequential path produces there, and the greedy tokens must agree.
Then the timing, against k sequential steps.
"""
import sys
import time

import torch
import ttnn

from _device_model import host_row, open_model, synthetic_prompt

KS = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8]
PRE = 8

mesh, cfg, m = open_model(max_seq_len=512)
prompt = synthetic_prompt(PRE + max(KS))

for k in KS:
    pre, draft = prompt[:PRE], prompt[PRE : PRE + k]

    st = m.new_state(batch=1)
    for t in pre:
        m.step([t], st)
    seq_h, seq_tok = [], []
    for t in draft:
        h = m.step([t], st)
        seq_h.append(host_row(mesh, h).reshape(-1))
        seq_tok.append(m.greedy_tokens(h)[0])
    seq_pos = list(st.positions)
    del st

    st = m.new_state(batch=1)
    for t in pre:
        m.step([t], st)
    t0 = time.perf_counter()
    hn = m.step_n(draft, st)
    ttnn.synchronize_device(mesh)
    dt = time.perf_counter() - t0
    got = host_row(mesh, hn)[0, 0]
    n_tok = m.greedy_tokens(hn)
    n_pos = list(st.positions)
    del st

    worst = max(
        (got[i] - seq_h[i]).abs().max().item() / max(seq_h[i].abs().max().item(), 1e-6)
        for i in range(k)
    )
    print(
        f"RESULT k={k:2d}  hidden worst rel {100 * worst:6.2f}%   "
        f"tokens {'MATCH' if n_tok[:k] == seq_tok else f'DIFFER {n_tok[:k]} vs {seq_tok}'}   "
        f"positions {'ok' if n_pos == seq_pos else f'{n_pos} vs {seq_pos}'}",
        flush=True,
    )
    print(f"RESULT k={k:2d}  step_n {1000 * dt:7.1f} ms", flush=True)
ttnn.close_mesh_device(mesh)
