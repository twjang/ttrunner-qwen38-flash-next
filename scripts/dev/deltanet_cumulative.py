"""Cumulative ablation inside the DeltaNet step."""
import sys, time, torch, ttnn
sys.path.insert(0, "scripts/dev")
from _device_model import open_model
import ttrunner_qwen38_flash_next.tt.ops as ops
import ttrunner_qwen38_flash_next.tt.model as mm
import ttrunner_qwen38_flash_next.tt.linear_attn as la

ORDER = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] else []
mesh, cfg, m = open_model(max_seq_len=4096)
m.selection_active = False

_BUF = {}
def like(t, shape, dtype=None):
    key = (tuple(shape), str(dtype or t.dtype))
    b = _BUF.get(key)
    if b is None:
        b = ttnn.from_torch(torch.zeros(*shape), dtype=dtype or t.dtype,
                            layout=ttnn.TILE_LAYOUT, device=t.device(),
                            mesh_mapper=ttnn.ReplicateTensorToMesh(t.device()))
        _BUF[key] = b
    return b

def stub_linear(x, w, **kw):
    return like(x, (x.shape[0], x.shape[1], x.shape[2], w.shape[-1]))

STUBS = {
    "matmuls":   lambda: (setattr(ops, "linear_rows", stub_linear),
                          setattr(mm, "linear_rows", stub_linear)),
    "conv":      lambda: setattr(mm.TTModel, "_causal_conv_step",
                                 lambda self, x, w, st, c, b=1, s=0, l=0: (x, st)),
    "heads":     lambda: setattr(ops, "fused_qkv_heads",
                                 lambda qkv, kd, vd, h, hd, eps, sq, sk, key=None:
                                 (like(qkv, (h, 1, 1, hd)),) * 3),
    "scalars":   lambda: setattr(ops, "fused_delta_scalars",
                                 lambda ab, dt, a, h, key=None:
                                 (like(ab, (h, 1, 1, 1)), like(ab, (h, 1, 1, 1)))),
    "recur":     lambda: setattr(la, "decode_step", lambda q, k, v, g, b, st: v),
    "tail":      lambda: setattr(ops, "fused_delta_tail",
                                 lambda o, z, w, h, hd, eps, key=None:
                                 like(o, (1, 1, 1, h * hd))),
    "allreduce": lambda: setattr(mm.TTModel, "all_reduce", lambda self, t: t),
}
for name in ORDER:
    STUBS[name]()

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
print(f"RESULT stripped[{','.join(ORDER) or '-':46s}] median {ts[len(ts)//2]:7.2f} ms "
      f"min {ts[0]:7.2f}", flush=True)
dec.release()
ttnn.close_mesh_device(mesh)
