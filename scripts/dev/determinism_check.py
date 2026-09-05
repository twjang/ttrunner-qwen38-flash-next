"""Is one decode step reproducible? And if not, where does it stop being?

    uv run python scripts/dev/determinism_check.py

Two runs of batch-1 greedy generation on identical code produce completely
different text after the first token, and four runs of `device_quality.py 192`
spread over six top-1 tokens. Greedy decoding amplifies anything, so the question
is whether a *single* step is bit-reproducible -- and if it is not, whether the
cause is the collective (float addition is not associative and `ttnn.all_reduce`
sums in arrival order) or something worse, like a race or an uninitialised read.

Three levels, each from the same state:
  1. the same op twice on the same input -- the device itself
  2. `all_reduce` twice on the same input -- the collective alone
  3. a whole decode step twice, comparing the hidden state
"""
import sys

import torch
import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _device_model import open_model                                 # noqa: E402

mesh, cfg, m = open_model(max_seq_len=512)
rep = ttnn.ReplicateTensorToMesh(mesh)
comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
torch.manual_seed(0)


def host(t):
    return ttnn.to_torch(t, mesh_composer=comp)


# --- 1. a plain op ----------------------------------------------------------
a = ttnn.from_torch(torch.randn(1, 1, 32, 2560), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
w = ttnn.from_torch(torch.randn(1, 1, 2560, 2560) * 0.02, dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
r1, r2 = host(ttnn.linear(a, w)), host(ttnn.linear(a, w))
print(f"RESULT one matmul, twice: max abs diff {(r1 - r2).abs().max().item():.3e}",
      flush=True)

# --- 2. the collective ------------------------------------------------------
p = ttnn.from_torch(torch.randn(4, 1, 32, 2560) * 0.1, dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
diffs = []
base = None
for _ in range(8):
    got = host(ttnn.all_reduce(p, cluster_axis=1, topology=ttnn.Topology.Linear))
    if base is None:
        base = got
    else:
        diffs.append((got - base).abs().max().item())
print(f"RESULT all_reduce, 8 times: max abs diff across runs "
      f"{max(diffs) if diffs else 0.0:.3e}", flush=True)

# --- 3. the step, one token at a time, and with pieces removed --------------
import ttrunner_qwen38_flash_next.tt.moe as moe                       # noqa: E402
import ttrunner_qwen38_flash_next.tt.ops as ops_mod                   # noqa: E402


def spread(n_steps=1, runs=3):
    outs = []
    for _ in range(runs):
        st = m.new_state(batch=1)
        for i in range(n_steps):
            h = m.step([1000 + i], st)
        outs.append(host(h)[:1].clone())
    d = max((outs[i] - outs[0]).abs().max().item() for i in range(1, runs))
    return d, d / max(outs[0].abs().max().item(), 1e-30)


for n in (1, 2, 3):
    d, rel = spread(n)
    print(f"RESULT {n} step(s), 3 runs: max abs {d:.3e} (rel {rel:.3e})", flush=True)

real_block = moe.moe_block
moe.moe_block = lambda mixed, *a, **kw: mixed
d, rel = spread(3)
print(f"RESULT 3 steps, MoE stubbed:    max abs {d:.3e} (rel {rel:.3e})", flush=True)
moe.moe_block = real_block

real_shared = moe.shared_expert
moe.shared_expert = lambda mixed, *a, **kw: mixed
d, rel = spread(3)
print(f"RESULT 3 steps, shared stubbed: max abs {d:.3e} (rel {rel:.3e})", flush=True)
moe.shared_expert = real_shared

real_sel = moe.fused_router_select
moe.fused_router_select = lambda *a, **kw: None
d, rel = spread(3)
print(f"RESULT 3 steps, router kernel off: max abs {d:.3e} (rel {rel:.3e})", flush=True)
moe.fused_router_select = real_sel

real_gm = ops_mod.fused_gated_mean
ops_mod.fused_gated_mean = lambda mix, normed, hc, hidden, apply_sigmoid=False: (
    real_gm(mix, normed, hc, hidden, apply_sigmoid))
d, rel = spread(3)
print(f"RESULT 3 steps, gated_mean kernel on (control): max abs {d:.3e} "
      f"(rel {rel:.3e})", flush=True)

ttnn.close_mesh_device(mesh)
