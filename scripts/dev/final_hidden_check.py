"""Device `step` and device `prefill` against the CPU oracle, on the final hidden.

    uv run python scripts/dev/final_hidden_check.py [len ...]      (default 1 4 8)

Per-layer comparisons of the hyper-connection stream turned out to be a poor
metric: the stream is four redundant copies that the output mixer averages, so
individual streams can wander a long way while the mixed result -- the only
thing the LM head sees -- stays put. This compares the mixed result, plus the
token each path would emit.

The oracle is the reference's whole-prompt forward, which since the
`_indexer_mask` tail fix agrees with its own token-by-token path to 1e-4 %. So
a device path that is further from the oracle than the other device path is
worse, full stop, and the numbers are comparable across lengths.
"""
import sys

import torch
import ttnn

from _device_model import host_row, open_model, synthetic_prompt

from twtest.reference.cache import HybridCache
from twtest.reference.model import Qwen4ExpModel

lengths = [int(x) for x in sys.argv[1:]] or [1, 4, 8]
mesh, cfg, m = open_model()
ref = Qwen4ExpModel(cfg, m.host)

for L in lengths:
    P = synthetic_prompt(L)

    h_ref = ref.forward(torch.tensor([P]), HybridCache(cfg.num_layers))[0, -1].float()
    t_ref = int(ref.logits(h_ref).argmax())

    stp = m.new_state(batch=1)
    hp = m.prefill(P, stp)
    t_pre = m.greedy_tokens(hp)[0]
    h_pre = host_row(mesh, hp).reshape(-1)
    del stp, hp

    sts = m.new_state(batch=1)
    for t in P:
        hs = m.step([t], sts)
    t_step = m.greedy_tokens(hs)[0]
    h_step = host_row(mesh, hs).reshape(-1)
    del sts, hs

    scale = max(h_ref.abs().max().item(), 1e-6)
    d_pre = (h_pre - h_ref).abs().max().item()
    d_step = (h_step - h_ref).abs().max().item()
    print(
        f"RESULT len={L:4d}  step vs ref {100 * d_step / scale:7.2f}%   "
        f"prefill vs ref {100 * d_pre / scale:7.2f}%   scale {scale:.2f}",
        flush=True,
    )
    print(
        f"RESULT len={L:4d}  token  ref {t_ref:7d}  step {t_step:7d}  prefill {t_pre:7d}   "
        f"step {'ok' if t_step == t_ref else 'BAD'}  prefill {'ok' if t_pre == t_ref else 'BAD'}",
        flush=True,
    )
ttnn.close_mesh_device(mesh)
