"""`decode_step`'s split-matmul and broadcast-outer forms against the originals.

    uv run python scripts/dev/decode_step_check.py

Both changes alter the arithmetic's shape, not its meaning: two matmuls instead
of a stack and one, and a broadcast multiply instead of a matmul whose reduction
is one element long. Neither should move the answer more than bfloat16 already
does, and this checks that against float64 rather than against each other.
"""
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device_model import open_model                                # noqa: E402

import ttrunner_qwen38_flash_next.tt.linear_attn as la              # noqa: E402

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)
comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
torch.manual_seed(0)

hd, n_v = cfg.linear_head_dim, m.n_v_local
try:
    for label, BH in (("batch 1", n_v), ("batch 8", 8 * n_v)):
        q_t = torch.randn(BH, 1, 1, hd) * 0.1
        k_t = torch.randn(BH, 1, 1, hd) * 0.1
        v_t = torch.randn(BH, 1, 1, hd) * 0.3
        g_t = torch.rand(BH, 1, 1, 1) * 0.5 + 0.5
        b_t = torch.rand(BH, 1, 1, 1)
        s_t = torch.randn(BH, 1, hd, hd) * 0.05

        def dv(t, dtype=ttnn.bfloat16):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                   device=mesh, mesh_mapper=rep)

        def run(split: bool, bcast: bool):
            la._NO_SPLIT_KQ = not split
            la._NO_BCAST_OUTER = not bcast
            st = dv(s_t, ttnn.float32)
            out = la.decode_step(dv(q_t), dv(k_t), dv(v_t), dv(g_t), dv(b_t), st)
            ttnn.synchronize_device(mesh)
            return (ttnn.to_torch(out, mesh_composer=comp)[:BH].to(torch.float64),
                    ttnn.to_torch(st, mesh_composer=comp)[:BH].to(torch.float64))

        # float64 on the bfloat16-rounded operands, which is what the device saw.
        rq = ttnn.to_torch(dv(q_t), mesh_composer=comp)[:BH].to(torch.float64)
        rk = ttnn.to_torch(dv(k_t), mesh_composer=comp)[:BH].to(torch.float64)
        rv = ttnn.to_torch(dv(v_t), mesh_composer=comp)[:BH].to(torch.float64)
        rg = ttnn.to_torch(dv(g_t), mesh_composer=comp)[:BH].to(torch.float64)
        rb = ttnn.to_torch(dv(b_t), mesh_composer=comp)[:BH].to(torch.float64)
        rs = s_t.to(torch.float64)
        dec = rs * rg
        pred = rq.new_zeros(BH, 1, 1, hd)
        predk = torch.matmul(rk, dec)
        delta = (rv - predk) * rb
        new_state = dec + torch.matmul(rk.transpose(-2, -1), delta)
        qk = (rq * rk).sum(-1, keepdim=True)
        truth_out = torch.matmul(rq, dec) + qk * delta

        for tag, split, bcast in (("old  (concat + matmul outer)", False, False),
                                  ("new  (split + broadcast)", True, True)):
            o, st_after = run(split, bcast)
            so = max(truth_out.abs().max().item(), 1e-30)
            ss = max(new_state.abs().max().item(), 1e-30)
            print(f"RESULT {label} {tag:30s} out {(o - truth_out).abs().max().item() / so:.3e}"
                  f"   state {(st_after - new_state).abs().max().item() / ss:.3e}", flush=True)
    la._NO_SPLIT_KQ = False
    la._NO_BCAST_OUTER = False
finally:
    ttnn.close_mesh_device(mesh)
