"""Where inside a layer does the device decode path leave the oracle?

    uv run python scripts/dev/layer_decode_bisect.py [layer ...]   (default 0 1 3)

The decode path disagrees with the float32 reference on real text, and the gap
appears within a single layer -- too early for accumulation. The weights check
out at their dtype floor, so the gap is in a block.

This drives a layer's sub-blocks side by side on a single token, device against
the reference's own methods, and prints each intermediate's distance. The first
one that is not at the bf16 floor is the answer. Pick one layer of each kind:
0 (DeltaNet), 1 (DeltaNet + PLE injection), 3 (sparse attention).
"""
import sys

import torch
import ttnn

from _device_model import host_row, open_model

from ttrunner_qwen38_flash_next.reference.cache import HybridCache
from ttrunner_qwen38_flash_next.reference.model import Qwen4ExpModel
from ttrunner_qwen38_flash_next.tt import moe
from ttrunner_qwen38_flash_next.tt.ops import gated_residual_mix, reinject

LAYERS = [int(x) for x in sys.argv[1:]] or [0, 1, 3]
mesh, cfg, m = open_model()
ref = Qwen4ExpModel(cfg, m.host)

TOKEN = 1000


def report(tag, got, want):
    got = got.reshape(-1).float()
    want = want.reshape(-1).float()
    scale = max(want.abs().max().item(), 1e-9)
    print(f"RESULT {tag:30s} rel {100 * (got - want).abs().max().item() / scale:8.2f}%  "
          f"scale {scale:.4f}", flush=True)


for LAYER in LAYERS:
    kind = "QSA" if cfg.is_full_attention(LAYER) else "DeltaNet"
    ple = " +PLE" if LAYER in m._ple_layers else ""
    print(f"RESULT ---- layer {LAYER} ({kind}{ple}) ----", flush=True)

    # Each layer is driven from a fresh state at position 0, so what it shows is
    # that layer's own error and not what the layers above it handed down.
    cache = HybridCache(cfg.num_layers)
    positions = cache.extend_positions(torch.tensor([[0]]))
    cos, sin = ref.rotary(positions)
    ids = torch.tensor([[TOKEN]])

    # -- host, float32 -----------------------------------------------------
    emb_h = m.host.get_rows("token_embd.weight", torch.tensor([TOKEN])).float()
    hid_h = emb_h.reshape(1, 1, cfg.hidden_size).repeat(1, 1, cfg.hc_count)

    # -- device ------------------------------------------------------------
    state = m.new_state(batch=1)
    st = state[LAYER]
    emb = m._input("embed", m.embed([TOKEN]), ttnn.bfloat16)
    hid = ttnn.repeat(emb, (1, 1, 1, cfg.hc_count))
    report("embedding", host_row(mesh, hid), hid_h)

    if LAYER in m._ple_layers:
        state.histories[0].append(TOKEN)
        ple_h = ref._ple(hid_h, ids, LAYER, cache)
        hid_h = hid_h + ple_h
        ple_d = m._ple_step(hid, LAYER, st, state)
        report("ple injection", host_row(mesh, ple_d), ple_h)
        hid = ttnn.add(hid, ple_d)
        report("after ple", host_row(mesh, hid), hid_h)

    mixed_h, hyper_h, inject_h = ref._gated_residual(hid_h, LAYER, "attn")
    mixed, inject = gated_residual_mix(
        hid, m.w.blk(LAYER, "hc_attn_norm.weight"), m.w.blk(LAYER, "hc_attn_down.weight"),
        m.w.blk(LAYER, "hc_attn_up.weight"), m.w.blk(LAYER, "hc_attn_inject.weight"),
        cfg.rms_norm_eps, cfg.hc_count, cfg.hidden_size,
    )
    report("hc attn mixed", host_row(mesh, mixed), mixed_h)
    report("hc attn inject", host_row(mesh, inject), inject_h)

    if cfg.is_full_attention(LAYER):
        branch_h = ref._full_attention(mixed_h, LAYER, cos, sin, cache, 0)
        branch = m._attention_step(mixed, LAYER, st, 0)
    else:
        branch_h = ref._linear_attention(mixed_h, LAYER, cache)
        branch = m._linear_attention_step(mixed, LAYER, st)
    report(f"{kind} branch", host_row(mesh, branch), branch_h)

    hid2_h = ref._reinject(hyper_h, branch_h, inject_h)
    hid2 = reinject(hid, branch, inject, cfg.hc_count)
    report("after attn reinject", host_row(mesh, hid2), hid2_h)

    mixed2_h, hyper2_h, inject2_h = ref._gated_residual(hid2_h, LAYER, "ffn")
    mixed2, inject2 = gated_residual_mix(
        hid2, m.w.blk(LAYER, "hc_ffn_norm.weight"), m.w.blk(LAYER, "hc_ffn_down.weight"),
        m.w.blk(LAYER, "hc_ffn_up.weight"), m.w.blk(LAYER, "hc_ffn_inject.weight"),
        cfg.rms_norm_eps, cfg.hc_count, cfg.hidden_size,
    )
    report("hc ffn mixed", host_row(mesh, mixed2), mixed2_h)

    ffn_h = ref._moe(mixed2_h, LAYER)
    gate_w = m.w.fused_gate_up(LAYER) if m.fuse_expert_gate_up else m.w.blk(LAYER, "ffn_gate_exps.weight")
    up_w = None if m.fuse_expert_gate_up else m.w.blk(LAYER, "ffn_up_exps.weight")
    routed = m.all_reduce(moe.moe_block(
        mixed2, m.w.blk(LAYER, "ffn_gate_inp.weight"), gate_w, up_w,
        m.w.blk(LAYER, "ffn_down_exps.weight"),
        cfg.num_experts_per_tok, cfg.num_experts, cfg.hidden_size, cfg.expert_intermediate,
    ))
    shared = moe.shared_expert(
        mixed2, m.w.blk(LAYER, "ffn_gate_shexp.weight"), m.w.blk(LAYER, "ffn_up_shexp.weight"),
        m.w.blk(LAYER, "ffn_down_shexp.weight"), m.w.blk(LAYER, "ffn_gate_inp_shexp.weight"),
    )
    ffn = ttnn.add(routed, shared)
    report("moe (routed+shared)", host_row(mesh, ffn), ffn_h)

    out_h = ref._reinject(hyper2_h, ffn_h, inject2_h)
    out = reinject(hid2, ffn, inject2, cfg.hc_count)
    report("layer output", host_row(mesh, out), out_h)
    del state

ttnn.close_mesh_device(mesh)
