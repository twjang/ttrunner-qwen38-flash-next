"""The fused recurrence over a *sequence* of steps, against float64 in lockstep.

    uv run python scripts/dev/recur_seq_check.py [steps]      (default 64)

`recur_check.py` checks one step and says 1.09e-03, which looks fine. Handoff
45.27: that number is meaningless on its own. A recurrence feeds its own state
back, so a per-step error compounds, and the fused kernel that passed the
one-shot check drove the model from 73 % to 12.6 % next-token accuracy over 128
steps (invariant 142).

This runs both the op chain and the fused kernel forward from the same state,
advancing a float64 reference alongside, and prints the error against step
number. The op chain is the control: whatever it does is the compounding a
correct implementation still has, so the fused column is only damning where it
diverges from that.
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
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 64


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

        def back(t):
            return ttnn.to_torch(t, mesh_composer=comp)[:BH].to(torch.float64)

        s0 = torch.randn(BH, 1, DK, DV) * 0.1
        # The same operand sequence for every arm, rounded through the device so
        # the float64 reference sees exactly what the kernels see.
        seq = []
        for _ in range(STEPS):
            q = torch.randn(BH, 1, 1, DK) * 0.1
            k = torch.randn(BH, 1, 1, DK) * 0.1
            v = torch.randn(BH, 1, 1, DV) * 0.1
            g = torch.rand(BH, 1, 1, 1) * 0.5 + 0.5
            b = torch.rand(BH, 1, 1, 1)
            seq.append((q, k, v, g, b))

        ref_state = back(dev(s0))
        ref_outs = []
        for q, k, v, g, b in seq:
            rq, rk, rv, rg, rb = (back(dev(x)) for x in (q, k, v, g, b))
            decayed = ref_state * rg
            delta = (rv - rk @ decayed) * rb
            ref_outs.append(rq @ decayed + (rq * rk).sum(-1, keepdim=True) * delta)
            ref_state = decayed + rk.transpose(-2, -1) @ delta

        def arm(fused):
            # `decode_step` tries the fused path itself when TT_FUSED_RECUR=1,
            # so the control arm has to switch the module flag off -- otherwise
            # both arms run the same kernel and report identical error, which is
            # exactly what the first version of this script did.
            ops._NO_FUSED_RECUR = not fused
            st = dev(s0)
            errs = []
            for i, (q, k, v, g, b) in enumerate(seq):
                dq, dk, dv_, dg, db = (dev(x) for x in (q, k, v, g, b))
                if fused:
                    out = ttnn.from_torch(
                        torch.zeros(BH, 1, 1, DV), dtype=DT,
                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
                    got = ops.fused_recurrence(
                        st, dq, dk, ttnn.transpose(dk, -2, -1), dv_, dg, db, out)
                    if got is None:
                        return None
                else:
                    got = la.decode_step(dq, dk, dv_, dg, db, st)
                errs.append(float((back(got) - ref_outs[i]).abs().max()))
            return errs, back(st)

        base = arm(False)
        fus = arm(True)
        ops._NO_FUSED_RECUR = False
        if fus is None:
            print("RESULT fused recurrence declined -- see the warning above")
            return
        base_errs, base_state = base
        fus_errs, fus_state = fus

        print(f"RESULT steps {STEPS}   out error by step "
              f"(op chain -> fused)", flush=True)
        marks = sorted({0, 1, 2, 3, 7, 15, 31, min(63, STEPS - 1), STEPS - 1})
        for i in marks:
            if i < STEPS:
                print(f"  step {i:4d}   {base_errs[i]:.3e}   {fus_errs[i]:.3e}",
                      flush=True)
        print(f"RESULT final state error   op chain {float((base_state - ref_state).abs().max()):.3e}"
              f"   fused {float((fus_state - ref_state).abs().max()):.3e}", flush=True)
        grew = fus_errs[-1] / max(fus_errs[0], 1e-12)
        print(f"RESULT fused out error grew {grew:.1f}x from step 0 to {STEPS - 1}"
              f"  (op chain {base_errs[-1] / max(base_errs[0], 1e-12):.1f}x)", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
