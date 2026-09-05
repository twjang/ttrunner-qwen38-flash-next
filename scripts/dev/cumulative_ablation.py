"""Cumulative ablation: strip components one at a time and never put them back.

A single-part ablation measures a component *in the presence of everything else*,
and those overlap -- this session's single-part numbers summed to 26 of 36 ms
with no way to attribute the rest. Stripping cumulatively makes each step's delta
the component's marginal cost given what is already gone, and the last reading is
the floor: what the step costs with every named component replaced by a stub.
"""
import sys, time, ttnn
sys.path.insert(0, "scripts/dev")
from _device_model import open_model
import ttrunner_qwen38_flash_next.tt.ops as ops
import ttrunner_qwen38_flash_next.tt.moe as moe
import ttrunner_qwen38_flash_next.tt.model as mm
import ttrunner_qwen38_flash_next.tt.linear_attn as la

ORDER = sys.argv[1].split(",") if len(sys.argv) > 1 else []
mesh, cfg, m = open_model(max_seq_len=4096)
m.selection_active = False

def grm_stub(hyper, nw, dw, uw, iw, eps, hc, H):
    mixed = ttnn.slice(hyper, (0, 0, 0, 0), (hyper.shape[0], hyper.shape[1],
                                             hyper.shape[2], H))
    return mixed, (None if iw is None else hyper)

STUBS = {
    "moe":      lambda: setattr(moe, "moe_block", lambda mixed, *a, **k: mixed),
    "shared":   lambda: setattr(moe, "shared_expert", lambda mixed, *a, **k: mixed),
    "grm":      lambda: (setattr(ops, "gated_residual_mix", grm_stub),
                         setattr(mm, "gated_residual_mix", grm_stub)),
    "deltanet": lambda: setattr(mm.TTModel, "_linear_attention_step",
                                lambda self, mixed, *a, **k: mixed),
    "qsa":      lambda: setattr(mm.TTModel, "_attention_step",
                                lambda self, mixed, *a, **k: mixed),
    "reinject": lambda: setattr(ops, "reinject",
                                lambda hyper, branch, inject, hc, base=None: hyper),
    "ple":      lambda: setattr(mm.TTModel, "_ple_inject",
                                lambda self, hidden, *a, **k: hidden),
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
print(f"RESULT stripped[{','.join(ORDER) or '-':44s}] median {ts[len(ts)//2]:7.2f} ms "
      f"min {ts[0]:7.2f}", flush=True)
dec.release()
ttnn.close_mesh_device(mesh)
