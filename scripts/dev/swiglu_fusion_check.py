"""Can `ttnn.swiglu` replace the MoE's four-op SwiGLU chain, and which half is which?

    uv run python scripts/dev/swiglu_fusion_check.py

`expert_ffn` computes the gate/up projection as one fused matmul into
`[1, E, M, 2N]` and then spends four ops splitting and combining it:

    gate = slice(both, 0, n);  up = slice(both, n, 2n)
    hidden = multiply(silu(gate), up)

Measured at E=128, M=1: slice 29.26 us x2, silu 51.49, multiply 73.03 --
183 us a layer, **8.79 ms a token**, on tensors that are 97 % zeros. Op cost here
is fixed per call and driven by E, not by M (invariant 42), so collapsing four
calls into one is the whole saving.

`ttnn.swiglu` does exactly that shape of work, but its docstring says it applies
SiLU to the **second** half and multiplies by the first, where ours applies it to
the first. If that is right the halves have to be swapped somewhere, and the
cheapest place is the packed weight -- but only if the speed is worth it.

So: check the semantics against a float64 golden rather than trusting the prose,
and time it against the chain it would replace.
"""
import sys
import time

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                # noqa: E402

E, M, N = 128, 1, 640
REPS, ITERS = 100, 10

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)
torch.manual_seed(0)


def dev(t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                           mesh_mapper=rep)


def marginal(fn, label):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(REPS):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    best = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        best = min(best, time.perf_counter() - t0)
    ttnn.release_trace(mesh, tid)
    us = best / REPS * 1e6
    print(f"RESULT {label:44s} {us:8.2f} us/call", flush=True)
    return us


host = torch.randn(1, E, M, 2 * N) * 0.5
both = dev(host)


def chain():
    g = ttnn.slice(both, (0, 0, 0, 0), (1, E, M, N))
    u = ttnn.slice(both, (0, 0, 0, N), (1, E, M, 2 * N))
    return ttnn.multiply(ttnn.silu(g), u)


chain_us = marginal(chain, "slice+slice+silu+multiply (today)")
fused_us = marginal(lambda: ttnn.swiglu(both), "ttnn.swiglu (fused)")

# --- which half gets the SiLU? Decide it against float64, not the docstring. --
got = ttnn.to_torch(ttnn.swiglu(both), mesh_composer=m.compose)[0:1].to(torch.float64)
x = host.to(torch.float64)
first, second = x[..., :N], x[..., N:]
ours = torch.nn.functional.silu(first) * second          # silu(gate) * up
docs = first * torch.nn.functional.silu(second)          # what the docstring says

for label, want in (("silu(first)*second  (what expert_ffn does)", ours),
                    ("first*silu(second)  (what the docstring says)", docs)):
    err = (got - want).abs().max().item()
    scale = want.abs().max().item()
    print(f"RESULT {label:44s} max abs err {err:.3e} "
          f"(rel {err / max(scale, 1e-30):.2e}) {'<-- MATCH' if err < 0.05 * scale else ''}",
          flush=True)

print("RESULT ---", flush=True)
saved = (chain_us - fused_us) * 48 / 1e3
print(f"RESULT {chain_us:.1f} -> {fused_us:.1f} us a layer, so {saved:+.2f} ms a token "
      f"over 48 layers", flush=True)
print("RESULT verdict: " + ("worth fusing" if saved > 1.0 else
                            "not worth the weight re-pack"), flush=True)

ttnn.close_mesh_device(mesh)
