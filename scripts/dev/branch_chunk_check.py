"""The chunked branches against the reference's own whole-sequence branches.

    uv run python scripts/dev/branch_chunk_check.py [layer ...] [--seq N]
                                                     (default 0 3, seq 8)

`deltanet_block_check.py` compares the chunked branch to the *decode* branch,
which only ever says whether the two device paths agree. This compares it to the
reference, which is what says whether either is right -- the same move that
turned up the head-expansion bug in the decode path.

The reference takes the whole sequence in one forward, so it runs
`chunk_gated_delta_rule` for DeltaNet and the sparse-attention mask for QSA:
exactly the algorithms the chunked device path is meant to reproduce, rather
than the recurrent forms decode uses.
"""
import sys

import torch
import ttnn

from _device_model import host_row, open_model

from twtest.reference.cache import HybridCache
from twtest.reference.model import Qwen4ExpModel
from twtest.tt.model import LayerState

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SEQ = 8
for i, a in enumerate(sys.argv):
    if a == "--seq":
        SEQ = int(sys.argv[i + 1])
        args = [x for x in args if x != sys.argv[i + 1]]
LAYERS = [int(x) for x in args] or [0, 3]

mesh, cfg, m = open_model()
ref = Qwen4ExpModel(cfg, m.host)

torch.manual_seed(0)
x = torch.randn(1, 1, SEQ, cfg.hidden_size) * 0.5
host_in = x.reshape(1, SEQ, cfg.hidden_size).float()

for LAYER in LAYERS:
    kind = "QSA" if cfg.is_full_attention(LAYER) else "DeltaNet"
    print(f"RESULT ---- layer {LAYER} ({kind}), seq {SEQ} ----", flush=True)
    cache = HybridCache(cfg.num_layers)
    if cfg.is_full_attention(LAYER):
        positions = cache.extend_positions(torch.arange(SEQ)[None])
        cos, sin = ref.rotary(positions)
        want = ref._full_attention(host_in, LAYER, cos, sin, cache, 0)
        got = m._attention_chunk(m.to_dev(x), LAYER, LayerState(), 0, SEQ)
    else:
        want = ref._linear_attention(host_in, LAYER, cache)
        got = m._linear_attention_chunk(m.to_dev(x), LAYER, LayerState(), SEQ)
    g = host_row(mesh, got)[0, 0]
    w = want[0].float()
    scale = max(w.abs().max().item(), 1e-9)
    for t in range(SEQ):
        d = (g[t] - w[t]).abs().max().item()
        print(f"RESULT pos {t:3d} rel {100 * d / scale:8.2f}%", flush=True)
    print(f"RESULT overall rel {100 * (g - w).abs().max().item() / scale:8.2f}%  "
          f"scale {scale:.4f}", flush=True)

ttnn.close_mesh_device(mesh)
