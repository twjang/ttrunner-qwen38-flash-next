"""Cumulative ablation inside `gated_residual_mix`, the model's second-largest
component (7.80 ms of 35.6 by `cumulative_ablation.py`).

Each stub returns a bound buffer of the right shape, so the substitution costs
one launch of nothing rather than a fresh allocation.
"""
import sys, time, torch, ttnn
sys.path.insert(0, "scripts/dev")
from _device_model import open_model
import ttrunner_qwen38_flash_next.tt.ops as ops

ORDER = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] else []
mesh, cfg, m = open_model(max_seq_len=4096)
m.selection_active = False

_B = {}
def like(t, shape, dtype=None):
    key = (tuple(shape), str(dtype or t.dtype))
    b = _B.get(key)
    if b is None:
        b = ttnn.from_torch(torch.zeros(*shape), dtype=dtype or t.dtype,
                            layout=ttnn.TILE_LAYOUT, device=t.device(),
                            mesh_mapper=ttnn.ReplicateTensorToMesh(t.device()))
        _B[key] = b
    return b

STUBS = {
    "norm":      lambda: setattr(ops, "fused_group_norm",
                                 lambda x, w, eps, gs, g, key=None:
                                 (x, like(x, (1, 1, 1, gs)))),
    "allreduce": lambda: setattr(ops, "all_reduce", lambda t: t),
    "silu":      lambda: setattr(ops, "ew_slice_silu",
                                 lambda a, w, site=None: like(a, (1, 1, 1, w)))
                          if hasattr(ops, "ew_slice_silu") else None,
    "mean":      lambda: setattr(ops, "fused_gated_mean",
                                 lambda mix, normed, hc, H, apply_sigmoid=False:
                                 like(mix, (1, 1, 1, H))),
    "linears":   lambda: (setattr(ops, "linear_rows",
                                  lambda x, w, **k: like(x, (x.shape[0], x.shape[1],
                                                             x.shape[2], w.shape[-1]))),),
}
for name in ORDER:
    f = STUBS.get(name)
    if f is not None:
        f()

from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder
state = m.new_state(batch=1)
dec = TracedDecoder(m, state)
dec.reset()
for _ in range(3):
    dec.step([1000])
ttnn.synchronize_device(mesh)
ts = []
for _ in range(21):
    t0 = time.perf_counter(); dec.step([1000]); ttnn.synchronize_device(mesh)
    ts.append(1000 * (time.perf_counter() - t0))
ts.sort()
print(f"RESULT stripped[{','.join(ORDER) or '-':34s}] median {ts[len(ts)//2]:7.2f} ms "
      f"min {ts[0]:7.2f}", flush=True)
dec.release()
ttnn.close_mesh_device(mesh)
