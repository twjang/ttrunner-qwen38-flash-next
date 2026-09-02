"""Walk `_attention_chunk`'s intermediates against a float32 host computation.

    uv run python scripts/dev/attn_chunk_trace.py [layer] [seq]     (default 3, 1)

`attn_pos0_oracle.py` narrowed the prefill bug to QSA at a single token with an
empty cache -- where the projections are literally the same code as the decode
path, rope is the identity and the softmax has one entry. So the fault is in
what happens between the projections and the output: the cache write, the
attention call, or the head layout coming out of it.

This inlines the block step by step and prints each intermediate's distance from
the same quantity computed in float32 from the GGUF weights. The first line that
is not at the bf16 floor (~1 %) is the operation that is wrong.
"""
import sys

import torch
import ttnn

from _device_model import host_row, open_model

from twtest.tt.ops import HIFI4
from twtest.tt.ops import rms_norm

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 1

mesh, cfg, m = open_model()
hd, n_q, n_kv = cfg.head_dim, cfg.num_attention_heads, cfg.num_kv_heads
groups = n_q // n_kv

torch.manual_seed(0)
mixed_t = torch.randn(1, 1, SEQ, cfg.hidden_size) * 0.5
x = mixed_t.reshape(SEQ, cfg.hidden_size).float()


def w(name):
    return m.host.get(f"blk.{LAYER}.{name}").float()


def hrms(t, weight):
    return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * weight


def report(tag, got, want):
    got = got.reshape(want.shape).float()
    scale = max(want.abs().max().item(), 1e-6)
    print(f"RESULT {tag:28s} rel {100 * (got - want).abs().max().item() / scale:8.2f}%", flush=True)


# -- host truth (position 0..SEQ-1, rope is identity only at 0; SEQ=1 default)
qg_h = (x @ w("attn_q.weight").T).view(SEQ, n_q, hd * 2)
q_h = hrms(qg_h[..., :hd], w("attn_q_norm.weight"))
gate_h = qg_h[..., hd:]
k_h = hrms((x @ w("attn_k.weight").T).view(SEQ, n_kv, hd), w("attn_k_norm.weight"))
v_h = (x @ w("attn_v.weight").T).view(SEQ, n_kv, hd)

# -- device, inlined from _attention_chunk ---------------------------------
mixed = m.to_dev(mixed_t)
qg = ttnn.linear(mixed, m.w.blk(LAYER, "attn_q.weight"), compute_kernel_config=HIFI4)
qg = ttnn.reshape(qg, (1, SEQ, n_q, hd * 2))
q = m._slice_last(qg, 0, hd)
gate = m._slice_last(qg, hd, hd * 2)
q = rms_norm(q, m.w.blk(LAYER, "attn_q_norm.weight"), cfg.rms_norm_eps)
k = ttnn.reshape(
    ttnn.linear(mixed, m.w.blk(LAYER, "attn_k.weight"), compute_kernel_config=HIFI4),
    (1, SEQ, n_kv, hd),
)
v = ttnn.reshape(
    ttnn.linear(mixed, m.w.blk(LAYER, "attn_v.weight"), compute_kernel_config=HIFI4),
    (1, SEQ, n_kv, hd),
)
k = rms_norm(k, m.w.blk(LAYER, "attn_k_norm.weight"), cfg.rms_norm_eps)

report("q after norm", host_row(mesh, q)[0], q_h)
report("gate", host_row(mesh, gate)[0], gate_h)
report("k after norm", host_row(mesh, k)[0], k_h)
report("v", host_row(mesh, v)[0], v_h)

cos_t, sin_t = m.rope(list(range(SEQ)))
cos = ttnn.permute(m.to_dev(cos_t, ttnn.float32), (0, 2, 1, 3))
sin = ttnn.permute(m.to_dev(sin_t, ttnn.float32), (0, 2, 1, 3))
qp = m._apply_rope_dev(ttnn.permute(q, (0, 2, 1, 3)), cos, sin)
kp = m._apply_rope_dev(ttnn.permute(k, (0, 2, 1, 3)), cos, sin)
report("q after rope [1,nq,s,hd]", host_row(mesh, qp)[0].permute(1, 0, 2), q_h)
report("k after rope [1,nkv,s,hd]", host_row(mesh, kp)[0].permute(1, 0, 2), k_h)

keys = ttnn.zeros((1, n_kv, m.max_seq_len, hd), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=m.mesh)
values = ttnn.zeros((1, n_kv, m.max_seq_len, hd), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=m.mesh)
ttnn.fill_cache(keys, kp, 0, update_idx=0)
ttnn.fill_cache(values, ttnn.permute(v, (0, 2, 1, 3)), 0, update_idx=0)
kc = ttnn.slice(keys, (0, 0, 0, 0), (1, n_kv, SEQ, hd))
vc = ttnn.slice(values, (0, 0, 0, 0), (1, n_kv, SEQ, hd))
report("k read back from cache", host_row(mesh, kc)[0].permute(1, 0, 2), k_h)
report("v read back from cache", host_row(mesh, vc)[0].permute(1, 0, 2), v_h)

kr = ttnn.repeat_interleave(kc, groups, dim=1)
vr = ttnn.repeat_interleave(vc, groups, dim=1)
want_vr = torch.stack([v_h[:, i // groups] for i in range(n_q)], dim=1)   # (SEQ, n_q, hd)
report("v after repeat_interleave", host_row(mesh, vr)[0].permute(1, 0, 2), want_vr)

qpos = torch.arange(SEQ).unsqueeze(-1)
kpos = torch.arange(SEQ).unsqueeze(0)
mask = torch.where(kpos <= qpos, 0.0, float("-inf")).reshape(1, 1, SEQ, SEQ)
out = ttnn.transformer.scaled_dot_product_attention(
    qp, kr, vr, attn_mask=m.to_dev(mask, ttnn.bfloat16), is_causal=False,
    scale=hd**-0.5, compute_kernel_config=HIFI4,
)
report("sdpa out [1,nq,s,hd]", host_row(mesh, out)[0].permute(1, 0, 2), want_vr)

flat = ttnn.reshape(ttnn.permute(out, (0, 2, 1, 3)), (1, 1, SEQ, n_q * hd))
report("sdpa out flattened", host_row(mesh, flat)[0, 0], want_vr.reshape(SEQ, n_q * hd))
gate_flat = ttnn.reshape(gate, (1, 1, SEQ, n_q * hd))
report("gate flattened", host_row(mesh, gate_flat)[0, 0], gate_h.reshape(SEQ, n_q * hd))

gated = ttnn.multiply(flat, ttnn.sigmoid(gate_flat))
want_gated = want_vr.reshape(SEQ, n_q * hd) * torch.sigmoid(gate_h.reshape(SEQ, n_q * hd))
report("gated", host_row(mesh, gated)[0, 0], want_gated)
final = ttnn.linear(gated, m.w.blk(LAYER, "attn_output.weight"), compute_kernel_config=HIFI4)
report("final", host_row(mesh, final)[0, 0], want_gated @ w("attn_output.weight").T)

ttnn.close_mesh_device(mesh)
