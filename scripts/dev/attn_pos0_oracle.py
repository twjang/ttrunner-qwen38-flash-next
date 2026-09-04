"""What *is* the right QSA output at position 0? Compute it on the host.

    uv run python scripts/dev/attn_pos0_oracle.py [layer]        (default 3)

`attn_block_check.py` shows `_attention_chunk` and `_attention_step`
disagreeing by 93 % at position 0. One of them is wrong and the per-layer
comparisons cannot say which, so this closes the loop with a third opinion
computed in float32 from the GGUF weights.

At position 0 the attention has one key, so the softmax is 1 and the whole block
collapses to

    out = W_out @ (sigmoid(gate) * v[head_map])

with rope the identity. That is small enough to write out by hand, which also
lets us try both GQA conventions -- `repeat_interleave` (query head i reads K/V
head i // groups) and tiling (i % n_kv) -- and see which one the verified decode
path is actually using.
"""
import sys

import torch
import ttnn

from _device_model import host_row, open_model

from ttrunner_qwen38_flash_next.tt.model import LayerState

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 3
mesh, cfg, m = open_model()
assert cfg.is_full_attention(LAYER)
hd, n_q, n_kv = cfg.head_dim, cfg.num_attention_heads, cfg.num_kv_heads
groups = n_q // n_kv

torch.manual_seed(0)
mixed_t = torch.randn(1, 1, 1, cfg.hidden_size) * 0.5
x = mixed_t.reshape(1, cfg.hidden_size).float()


def w(name):
    return m.host.get(f"blk.{LAYER}.{name}").float()


def rms(t, weight):
    return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * weight


qg = (x @ w("attn_q.weight").T).view(n_q, hd * 2)
gate = qg[:, hd:].reshape(-1)
v = (x @ w("attn_v.weight").T).view(n_kv, hd)

out_w = w("attn_output.weight")
cands = {
    "repeat_interleave (i // groups)": torch.stack([v[i // groups] for i in range(n_q)]),
    "tile (i % n_kv)": torch.stack([v[i % n_kv] for i in range(n_q)]),
}
oracle = {
    name: ((heads.reshape(-1) * torch.sigmoid(gate)) @ out_w.T)
    for name, heads in cands.items()
}

st_c = LayerState()
got_c = host_row(mesh, m._attention_chunk(m.to_dev(mixed_t), LAYER, st_c, 0, 1))[0, 0, 0]
st_s = LayerState()
got_s = host_row(mesh, m._attention_step(m.to_dev(mixed_t), LAYER, st_s, 0))[0, 0, 0]

print(f"RESULT layer {LAYER} n_q={n_q} n_kv={n_kv} groups={groups}", flush=True)
for name, ref in oracle.items():
    scale = ref.abs().max().item()
    dc = (got_c - ref).abs().max().item()
    ds = (got_s - ref).abs().max().item()
    print(
        f"RESULT vs {name:32s} chunk {100 * dc / scale:7.2f}%  step {100 * ds / scale:7.2f}%",
        flush=True,
    )
print(
    f"RESULT chunk vs step {100 * (got_c - got_s).abs().max().item() / got_s.abs().max().item():.2f}%",
    flush=True,
)
ttnn.close_mesh_device(mesh)
