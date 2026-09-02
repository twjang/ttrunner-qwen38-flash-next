"""Compare `_attention_chunk` against `_attention_step` in isolation.

    uv run python scripts/dev/attn_block_check.py [layer] [seq]   (default 3, 8)

Whole-model bisection put the first prefill jump at layer 3 -- the first QSA
layer -- but could not separate the block from everything downstream of it.
This drives the two implementations directly, from the same random `mixed`
input and the same empty cache, so any disagreement is theirs alone. A random
input is deliberate: it makes a head-mapping mistake maximally visible, where
real activations could hide one behind correlated heads.

Position 0 is the one to read first: with an empty cache a query attends to
exactly one key, so the output collapses to `sigmoid(gate) * v` through the
output projection. Nothing about masking, history or the softmax can move it,
and rope is the identity at position 0. A disagreement there is a disagreement
about the projections themselves -- see `attn_pos0_oracle.py`.
"""
import sys

import torch
import ttnn

from _device_model import host_row, open_model

from twtest.tt.model import LayerState

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 8

mesh, cfg, m = open_model()
assert cfg.is_full_attention(LAYER), f"layer {LAYER} is not a QSA layer"
hd, n_q, n_kv = cfg.head_dim, cfg.num_attention_heads, cfg.num_kv_heads

torch.manual_seed(0)
mixed_t = torch.randn(1, 1, SEQ, cfg.hidden_size) * 0.5

st_c = LayerState()
out_c = host_row(mesh, m._attention_chunk(m.to_dev(mixed_t), LAYER, st_c, 0, SEQ))[0, 0]

st_s = LayerState()
rows = []
for i in range(SEQ):
    row = m.to_dev(mixed_t[:, :, i : i + 1, :].contiguous())
    rows.append(host_row(mesh, m._attention_step(row, LAYER, st_s, i))[0, 0, 0])
out_s = torch.stack(rows)

scale = out_s.abs().max().item()
for i in range(SEQ):
    d = (out_c[i] - out_s[i]).abs().max().item()
    print(f"RESULT pos {i:3d} maxdiff {d:9.4f}  rel {100 * d / scale:7.2f}%", flush=True)
print(f"RESULT scale {scale:.4f}", flush=True)

ttnn.close_mesh_device(mesh)
