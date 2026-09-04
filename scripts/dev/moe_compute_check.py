"""Wire one layer onto `ttnn.experimental.moe_compute` and see what it costs.

    uv run python scripts/dev/moe_compute_check.py

The MoE we hand-rolled on `sparse_matmul` materialises [1, E, M, K] per layer --
84 MB before the expert-axis reshard, 21 after -- to carry ~59 KB of selected
expert output (handoff invariant 27). ttnn ships a purpose-built pipeline that
never does: `all_to_all_dispatch_metadata` produces a *sparse* buffer of only the
tokens routed to this device's experts, and `moe_compute` runs gate/up/down plus
SwiGLU over it in one fused kernel on the Blackhole 8-core matmul ring.

This is the first wiring: pack one layer's weights with the reference packers,
run the op in `compute_only=True` mode (which skips the A2A combine, so the
model's existing all-reduce still applies), and compare against the current path
for both answer and time.

Shapes and layouts are learned by running rather than assumed -- the packers are
"executable documentation" per the module docstring, and the layout is derived
from (hidden_size, intermediate_size) in ways no summary here would get right.
"""
import time

import torch
import ttnn

E_TOTAL, K, N, TOPK = 512, 2560, 640, 10      # experts, hidden, intermediate, top-k
NDEV = 4
E_LOCAL = E_TOTAL // NDEV

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NDEV))
u = ttnn.experimental.moe_compute_utils
torch.manual_seed(0)

ring = u.effective_matmul_ring_size(mesh)
print(f"RESULT matmul ring size: {ring}", flush=True)
maps = u.get_weight_core_shard_maps(mesh, K, N)
print(f"RESULT shard maps: {type(maps)} -> "
      f"{[type(m).__name__ for m in maps] if isinstance(maps, tuple) else 'single'}", flush=True)

w0 = torch.randn(1, E_LOCAL, K, N) * 0.05     # gate, [L, E, K, N]
w1 = torch.randn(1, E_LOCAL, K, N) * 0.05     # up
w2 = torch.randn(1, E_LOCAL, N, K) * 0.05     # down, [L, E, N, K]

try:
    w0w1 = u.prepare_w0_w1_tensor_for_moe_compute(w0, w1, 1, E_LOCAL, K, N, maps[0])
    print(f"RESULT packed w0w1: {tuple(w0w1.shape)}", flush=True)
except Exception as exc:                                        # noqa: BLE001
    print(f"RESULT prepare_w0_w1 rejected: {type(exc).__name__}: "
          f"{str(exc).splitlines()[0][:200]}", flush=True)
    raise SystemExit(0)

try:
    w2p = u.prepare_w2_tensor_for_moe_compute(w2, 1, E_LOCAL, N, K, maps[1], maps[0])
    print(f"RESULT packed w2: {tuple(w2p.shape)}", flush=True)
except Exception as exc:                                        # noqa: BLE001
    print(f"RESULT prepare_w2 rejected: {type(exc).__name__}: "
          f"{str(exc).splitlines()[0][:200]}", flush=True)

ttnn.close_mesh_device(mesh)
