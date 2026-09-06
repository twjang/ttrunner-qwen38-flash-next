"""The fused delta-rule recurrence against `decode_step`, and both against float64.

    uv run python scripts/dev/recur_check.py

`decode_step` is 2.95 ms a token (handoff 45.23) in seven launches that make five
passes over a 786 KB state. `ops.fused_recurrence` does it in one launch with one
read and one write, keeping `decayed` in L1.

Accuracy first and against **float64 on the operands the device actually saw**,
not against the ops it replaces -- the fusion changes the order of the sums, so
"differs" is not "worse". Both the output *and* the updated state are checked:
the state is the half a fused version is most likely to get wrong, and it is the
half that compounds over a sequence.
"""
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ttrunner_qwen38_flash_next.tt.ops as ops                      # noqa: E402
import ttrunner_qwen38_flash_next.tt.linear_attn as la               # noqa: E402

BH, DK, DV = 12, 128, 128
DT = ttnn.float32


def main() -> None:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    rep = ttnn.ReplicateTensorToMesh(mesh)
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    torch.manual_seed(0)
    try:
        def dev(t):
            return ttnn.from_torch(t, dtype=DT, layout=ttnn.TILE_LAYOUT,
                                   device=mesh, mesh_mapper=rep)

        q_t = torch.randn(BH, 1, 1, DK) * 0.1
        k_t = torch.randn(BH, 1, 1, DK) * 0.1
        v_t = torch.randn(BH, 1, 1, DV) * 0.1
        g_t = torch.rand(BH, 1, 1, 1) * 0.5 + 0.5
        b_t = torch.rand(BH, 1, 1, 1)
        s_t = torch.randn(BH, 1, DK, DV) * 0.1

        # float64 on exactly what the device holds, after DT rounding.
        rq, rk, rv, rg, rb, rs = (
            ttnn.to_torch(dev(x), mesh_composer=comp)[:BH].to(torch.float64)
            for x in (q_t, k_t, v_t, g_t, b_t, s_t))
        decayed = rs * rg
        predicted = (rk @ decayed)
        q_decayed = (rq @ decayed)
        delta = (rv - predicted) * rb
        qk = (rq * rk).sum(-1, keepdim=True)
        ref_out = q_decayed + qk * delta
        ref_state = decayed + rk.transpose(-2, -1) @ delta

        def run(fused):
            st = dev(s_t)
            q, k, v = dev(q_t), dev(k_t), dev(v_t)
            g, b = dev(g_t), dev(b_t)
            if fused:
                kt = ttnn.transpose(k, -2, -1)
                out = ttnn.from_torch(torch.zeros(BH, 1, 1, DV), dtype=DT,
                                      layout=ttnn.TILE_LAYOUT, device=mesh,
                                      mesh_mapper=rep)
                got = ops.fused_recurrence(st, q, k, kt, v, g, b, out)
                if got is None:
                    return None, None
            else:
                got = la.decode_step(q, k, v, g, b, st)
            return (ttnn.to_torch(got, mesh_composer=comp)[:BH].to(torch.float64),
                    ttnn.to_torch(st, mesh_composer=comp)[:BH].to(torch.float64))

        def err(a, b):
            return float((a - b).abs().max())

        # Each STAGE returns a different intermediate as `out`, so compare
        # against *that* intermediate rather than the final answer -- otherwise
        # a stage that is working looks broken and a stage that is broken looks
        # like an incomplete stage. STAGE 5 computes `out` in full, so it is the
        # first one whose error should be ~1e-3.
        import os as _os
        _stage = int(_os.environ.get("TT_RECUR_STAGE", "6"))
        _expect = {0: ("v", rv), 1: ("v", rv), 2: ("predicted", predicted),
                   3: ("qk broadcast", None), 4: ("delta", delta),
                   5: ("out", ref_out), 6: ("out", ref_out)}
        _name, _ref = _expect.get(_stage, ("out", ref_out))

        base_o, base_s = run(False)
        print(f"RESULT decode_step   out {err(base_o, ref_out):.3e}  "
              f"state {err(base_s, ref_state):.3e}", flush=True)

        fus_o, fus_s = run(True)
        if fus_o is None:
            print("RESULT fused recurrence declined -- see the warning above",
                  flush=True)
            return
        if _ref is not None:
            print(f"RESULT stage {_stage} out vs {_name}: {err(fus_o, _ref):.3e}",
                  flush=True)
        print(f"RESULT stage {_stage} state vs decayed: "
              f"{err(fus_s, decayed):.3e}   (stages < 6 return decayed)", flush=True)
        print(f"RESULT fused         out {err(fus_o, ref_out):.3e}  "
              f"state {err(fus_s, ref_state):.3e}", flush=True)
        print(f"RESULT verdict: {'as accurate' if err(fus_o, ref_out) <= 4 * err(base_o, ref_out) + 1e-6 and err(fus_s, ref_state) <= 4 * err(base_s, ref_state) + 1e-6 else 'WORSE -- do not ship'}",
              flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
