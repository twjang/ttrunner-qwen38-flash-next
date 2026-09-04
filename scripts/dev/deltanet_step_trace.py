"""Walk `_linear_attention_step` against the reference, sub-step by sub-step.

    uv run python scripts/dev/deltanet_step_trace.py [layer]      (default 0)

Bisecting layer 0 of the decode path put the whole gap in the DeltaNet branch:
41 % against the float32 reference on a single token, with the hyper-connection
mix before it at 1 % and the weights at their dtype floor. This drives the
block's intermediates side by side to say which sub-step.

The device is head-sharded, so a gathered tensor comes back as
[q0|k0|v0|q1|k1|v1|...] and has to be reassembled into the reference's global
[q|k|v] before anything can be compared.
"""
import sys

import torch
import torch.nn.functional as F
import ttnn

from _device_model import open_model

from ttrunner_qwen38_flash_next.reference.layers import (
    causal_conv1d_step,
    l2norm,
    recurrent_gated_delta_rule,
    rms_norm_gated,
)
from ttrunner_qwen38_flash_next.tt import linear_attn
from ttrunner_qwen38_flash_next.tt.model import LayerState
from ttrunner_qwen38_flash_next.tt.ops import HIFI4

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
mesh, cfg, m = open_model()
assert not cfg.is_full_attention(LAYER)

n_v, n_k, hd = cfg.linear_num_v_heads, cfg.linear_num_k_heads, cfg.linear_head_dim
nv_l, nk_l = m.n_v_local, m.n_k_local
kd_l, vd_l = m.key_dim_local, m.value_dim_local
n_dev = m.n_dev

torch.manual_seed(0)
mixed_t = torch.randn(1, 1, 1, cfg.hidden_size) * 0.5
x = mixed_t.reshape(1, 1, cfg.hidden_size).float()


def w(name):
    return m.host.get(f"blk.{LAYER}.{name}").float()


def gather(t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()


def regroup_qkv(flat):
    """[q0|k0|v0|q1|k1|v1|...] -> [q|k|v]."""
    per = 2 * kd_l + vd_l
    qs, ks, vs = [], [], []
    for d in range(n_dev):
        blk = flat[..., d * per : (d + 1) * per]
        qs.append(blk[..., :kd_l])
        ks.append(blk[..., kd_l : 2 * kd_l])
        vs.append(blk[..., 2 * kd_l :])
    return torch.cat(qs + ks + vs, dim=-1)


def report(tag, got, want):
    got, want = got.reshape(-1).float(), want.reshape(-1).float()
    scale = max(want.abs().max().item(), 1e-9)
    print(f"RESULT {tag:26s} rel {100 * (got - want).abs().max().item() / scale:8.2f}%  "
          f"scale {scale:.4f}", flush=True)


# -- host, float32, exactly the reference's order --------------------------
qkv_h = F.linear(x, w("attn_qkv.weight"))                       # (1,1,conv_dim)
z_h = F.linear(x, w("attn_gate.weight"))
b_h = F.linear(x, w("ssm_beta.weight"))
a_h = F.linear(x, w("ssm_alpha.weight"))
conv_w = w("ssm_conv1d.weight")
state0 = torch.zeros(1, cfg.conv_dim, cfg.conv_kernel - 1)
qkv_conv_h, _ = causal_conv1d_step(qkv_h.transpose(1, 2), state0, conv_w)
qkv_conv_h = qkv_conv_h.transpose(1, 2)                          # (1,1,conv_dim)
kd = cfg.linear_key_dim
q_h, k_h, v_h = torch.split(qkv_conv_h, [kd, kd, cfg.linear_value_dim], dim=-1)
q_h = q_h.reshape(1, 1, n_k, hd).repeat_interleave(n_v // n_k, dim=2)
k_h = k_h.reshape(1, 1, n_k, hd).repeat_interleave(n_v // n_k, dim=2)
v_h = v_h.reshape(1, 1, n_v, hd)
beta_h = b_h.sigmoid()
g_h = w("ssm_a") * F.softplus(a_h + w("ssm_dt.bias"))
out_h, _ = recurrent_gated_delta_rule(q_h, k_h, v_h, g_h, beta_h)
gated_h = rms_norm_gated(
    out_h.reshape(-1, hd), z_h.reshape(-1, hd), w("ssm_norm.weight"), cfg.rms_norm_eps, "sigmoid"
).reshape(1, 1, -1)
final_h = F.linear(gated_h, w("ssm_out.weight"))

# -- device, inlined from _linear_attention_step ---------------------------
st = LayerState()
mixed = m.to_dev(mixed_t)
qkv = ttnn.linear(mixed, m.w.blk(LAYER, "attn_qkv.weight"), compute_kernel_config=HIFI4)
z = ttnn.linear(mixed, m.w.blk(LAYER, "attn_gate.weight"), compute_kernel_config=HIFI4)
report("qkv projection", regroup_qkv(gather(qkv)), qkv_h)
report("z (output gate)", regroup_qkv(gather(z)) if False else gather(z), z_h)

qkv_col = ttnn.transpose(ttnn.permute(qkv, (0, 2, 1, 3)), -2, -1)
conv_out, st.conv = m._causal_conv_step(
    qkv_col, m.w.blk(LAYER, "ssm_conv1d.weight"), st.conv, m.conv_dim_local, 1, 0, LAYER
)
qkv2 = ttnn.permute(ttnn.transpose(conv_out, -2, -1), (0, 2, 1, 3))
report("qkv after conv", regroup_qkv(gather(qkv2)), qkv_conv_h)

a = ttnn.linear(mixed, m.w.blk(LAYER, "ssm_alpha.weight"), compute_kernel_config=HIFI4)
b = ttnn.linear(mixed, m.w.blk(LAYER, "ssm_beta.weight"), compute_kernel_config=HIFI4)
report("alpha", gather(a), a_h)
report("beta (pre-sigmoid)", gather(b), b_h)
g = ttnn.multiply(
    m.w.blk(LAYER, "ssm_a"),
    ttnn.softplus(ttnn.add(a, m.w.blk(LAYER, "ssm_dt.bias"))),
)
report("g", gather(g), g_h)

q = m._slice_last(qkv2, 0, kd_l)
k = m._slice_last(qkv2, kd_l, 2 * kd_l)
v = m._slice_last(qkv2, 2 * kd_l, 2 * kd_l + vd_l)
reps = nv_l // nk_l
q = ttnn.reshape(ttnn.repeat_interleave(ttnn.reshape(q, (1, 1, nk_l, hd)), reps, dim=2), (nv_l, 1, 1, hd))
k = ttnn.reshape(ttnn.repeat_interleave(ttnn.reshape(k, (1, 1, nk_l, hd)), reps, dim=2), (nv_l, 1, 1, hd))
v = ttnn.reshape(v, (nv_l, 1, 1, hd))
q = ttnn.multiply(m._l2norm(q), hd**-0.5)
k = m._l2norm(k)
# gather the per-device heads back into global head order
q_dev = ttnn.to_torch(q, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(n_dev * nv_l, hd)
k_dev = ttnn.to_torch(k, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(n_dev * nv_l, hd)
v_dev = ttnn.to_torch(v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(n_dev * nv_l, hd)
report("q (normed, scaled)", q_dev, l2norm(q_h.reshape(n_v, hd)) / hd**0.5)
report("k (normed)", k_dev, l2norm(k_h.reshape(n_v, hd)))
report("v", v_dev, v_h.reshape(n_v, hd))

g_exp = ttnn.reshape(ttnn.exp(g), (nv_l, 1, 1, 1))
beta = ttnn.reshape(ttnn.sigmoid(b), (nv_l, 1, 1, 1))
st.recurrent = ttnn.zeros((nv_l, 1, hd, hd), dtype=m.state_dtype, layout=ttnn.TILE_LAYOUT, device=mesh)
out = linear_attn.decode_step(q, k, v, g_exp, beta, st.recurrent)
out_dev = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(n_v, hd)
report("recurrence out", out_dev, out_h.reshape(n_v, hd))

out_r = ttnn.reshape(out, (1, 1, nv_l, hd))
z_heads = ttnn.reshape(z, (1, 1, nv_l, hd))
normed = ttnn.rms_norm(out_r, epsilon=cfg.rms_norm_eps,
                       weight=m.w.blk(LAYER, "ssm_norm.weight"), compute_kernel_config=HIFI4)
gated = ttnn.multiply(normed, ttnn.sigmoid(z_heads))
gated_dev = ttnn.to_torch(gated, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(n_v, hd)
report("after norm+gate", gated_dev, gated_h.reshape(n_v, hd))

gated2 = ttnn.reshape(gated, (1, 1, 1, vd_l))
final = m.all_reduce(ttnn.linear(gated2, m.w.blk(LAYER, "ssm_out.weight"), compute_kernel_config=HIFI4))
final_dev = ttnn.to_torch(final, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
report("ssm_out", final_dev, final_h)

ttnn.close_mesh_device(mesh)
