"""Compare `_linear_attention_chunk` against `_linear_attention_step` in isolation.

    uv run python scripts/dev/deltanet_block_check.py [layer] [seq]  (default 0, 8)

The companion to `attn_block_check.py`. Whole-model bisection showed ~1.5 %
between prefill and decode already at layers 0-2, which are DeltaNet-only; this
says whether that is the block or the hyper-connection plumbing around it.

The chunked path fixes its chunk size at 128 internally
(`gated_delta_attn_seq`), so a `seq` below that exercises the padding as well.
"""
import sys

import torch
import ttnn

from _device_model import host_row, open_model

from twtest.tt.model import LayerState

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 8

mesh, cfg, m = open_model()
assert not cfg.is_full_attention(LAYER), f"layer {LAYER} is not a DeltaNet layer"

torch.manual_seed(0)
mixed_t = torch.randn(1, 1, SEQ, cfg.hidden_size) * 0.5

st_c = LayerState()
out_c = host_row(mesh, m._linear_attention_chunk(m.to_dev(mixed_t), LAYER, st_c, SEQ))[0, 0]

st_s = LayerState()
rows = []
for i in range(SEQ):
    row = m.to_dev(mixed_t[:, :, i : i + 1, :].contiguous())
    rows.append(host_row(mesh, m._linear_attention_step(row, LAYER, st_s))[0, 0, 0])
out_s = torch.stack(rows)

scale = out_s.abs().max().item()
for i in range(SEQ):
    d = (out_c[i] - out_s[i]).abs().max().item()
    print(f"RESULT pos {i:3d} maxdiff {d:9.4f}  rel {100 * d / scale:7.2f}%", flush=True)
print(f"RESULT scale {scale:.4f}", flush=True)
ttnn.close_mesh_device(mesh)
