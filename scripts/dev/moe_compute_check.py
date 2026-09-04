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

w0_w1_map, w2_map, dram_cores = maps

w0w1 = u.prepare_w0_w1_tensor_for_moe_compute(w0, w1, 1, E_LOCAL, K, N, w0_w1_map)
w2p = u.prepare_w2_tensor_for_moe_compute(w2, 1, E_LOCAL, N, K, w2_map, w0_w1_map)
print(f"RESULT packed w0w1 {tuple(w0w1.shape)}  w2 {tuple(w2p.shape)}", flush=True)

mem = u.get_weight_mem_configs(1, E_LOCAL, K, N, w0_w1_map, w2_map, dram_cores)
print(f"RESULT weight mem configs: {len(mem) if hasattr(mem, '__len__') else mem}", flush=True)

rep = ttnn.ReplicateTensorToMesh(mesh)
shard1 = ttnn.ShardTensorToMesh(mesh, dim=0)          # per-core axis is already 8 == ring


def dev(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mapper=rep, mc=None):
    kw = {"memory_config": mc} if mc is not None else {}
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=mapper, **kw)


try:
    w0w1_d = dev(w0w1, ttnn.bfloat4_b, mc=mem[0] if hasattr(mem, "__getitem__") else None)
    w2_d = dev(w2p, ttnn.bfloat4_b, mc=mem[1] if hasattr(mem, "__getitem__") else None)
    print(f"RESULT weights on device: {tuple(w0w1_d.shape)} {tuple(w2_d.shape)}", flush=True)
except Exception as exc:                                        # noqa: BLE001
    print(f"RESULT weight upload rejected: {type(exc).__name__}: "
          f"{str(exc).splitlines()[0][:200]}", flush=True)
    ttnn.close_mesh_device(mesh)
    raise SystemExit(0)

# --- the four dispatch inputs, for a single decode token ---------------------
B, S = 1, 1
HEIGHT_SHARD = 1
# `Silu` is the standard SwiGLU the model uses -- silu(W0.x) * (W1.x) --
# applied between the W0/W1 and W2 projections. `SwiGluOai` is the OpenAI
# variant and is not what this checkpoint does.
_AF = ttnn.operations.ccl.MoEActivationFunction
ACT = getattr(_AF, 'Silu', None) or getattr(_AF, 'SILU', None) or list(
    v for k, v in vars(_AF).items() if not k.startswith('_'))[0]
tok = torch.randn(B, S, 1, K) * 0.1
sel = torch.randperm(E_TOTAL)[:TOPK].sort().values
idx = sel.reshape(1, 1, 1, TOPK).to(torch.int32)
sc = torch.softmax(torch.randn(1, 1, 1, TOPK), dim=-1).to(torch.bfloat16)
# expert e lives on device e // E_LOCAL -- exactly what the reshard produced
# The docstring says [1, 1, E, D] with experts on rows. The op asserts rank 2
# *and* `mapping_shape[0] == num_devices`, so it is [D, E]: row per device,
# one-hot over the experts resident on it. Expert e lives on device
# e // E_LOCAL, which is exactly what the expert-axis reshard produced.
mapping = torch.zeros(NDEV, E_TOTAL, dtype=torch.int32)
for e in range(E_TOTAL):
    mapping[e // E_LOCAL, e] = 1

rm = ttnn.ROW_MAJOR_LAYOUT
try:
    d_in = dev(tok, ttnn.bfloat16, rm)
    d_idx = dev(idx, ttnn.uint16, rm)
    d_sc = dev(sc, ttnn.bfloat16, rm)
    d_map = dev(mapping, ttnn.uint16, rm)
    # cluster_axis is required (the op asserts without it) and is the 4-device
    # axis of the (1, 4) mesh -- the same one the model's all_reduce uses.
    width_dim = u.auto_output_width_shard_dim(K, matmul_ring_size=ring)
    drain = ttnn.experimental.get_moe_tilize_drain_core(mesh, HEIGHT_SHARD, width_dim, K)
    print(f"RESULT width_shard_dim={width_dim} drain_core={drain}", flush=True)
    # the helper hands back a CoreCoord; the op wants a plain (x, y)
    drain_xy = (drain.x, drain.y) if hasattr(drain, "x") else tuple(drain)
    out = ttnn.experimental.all_to_all_dispatch_metadata(
        d_in, d_idx, d_sc, d_map, cluster_axis=1, drain_sync_tilizer_core=drain_xy,
    )
    print(f"RESULT dispatch -> {len(out)} tensors: "
          f"{[tuple(t.shape) for t in out]}", flush=True)
except Exception as exc:                                        # noqa: BLE001
    msg = str(exc) or repr(exc)
    print(f"RESULT dispatch rejected: {type(exc).__name__}: {msg[:600]}", flush=True)
    ttnn.close_mesh_device(mesh)
    raise SystemExit(0)

# --- the fused expert compute ------------------------------------------------
tok_d, idx_d, sc_d = out
try:
    res = ttnn.experimental.moe_compute(
        tok_d, idx_d, sc_d, d_map, w0w1_d, w2_d,
        layer_id=0,
        output_height_shard_dim=HEIGHT_SHARD,
        intermediate_size=N,
        activation_type=ACT,
        compute_only=True,
    )
    shapes = [tuple(t.shape) for t in res] if isinstance(res, (list, tuple)) else [tuple(res.shape)]
    print(f"RESULT moe_compute -> {len(shapes)} tensors: {shapes}", flush=True)

    # What actually breaks is narrower than "calling it twice". After the first
    # `moe_compute` returns, the *next device operation of any kind* hangs -- the
    # run below never reached even a `deallocate`, spinning at ~120 % CPU with the
    # log frozen. So the question is whether the first call leaves the device in a
    # state nothing else survives, and the cheapest probe is a trivial op.
    print("RESULT probing: a plain add after moe_compute", flush=True)
    t0 = time.perf_counter()
    try:
        probe = ttnn.add(d_in, d_in)
        ttnn.synchronize_device(mesh)
        print(f"RESULT plain add after moe_compute: OK in "
              f"{1000 * (time.perf_counter() - t0):.1f} ms", flush=True)
    except Exception as exc:                                    # noqa: BLE001
        print(f"RESULT plain add rejected: {type(exc).__name__}: "
              f"{(str(exc) or repr(exc))[:300]}", flush=True)

    print("RESULT   (current sparse_matmul path at E=128 measured 2.78 ms a layer)",
          flush=True)
except Exception as exc:                                        # noqa: BLE001
    msg = str(exc) or repr(exc)
    import traceback
    print(f"RESULT moe_compute rejected: {type(exc).__name__}: {msg[:400]}", flush=True)
    sig = [l for l in traceback.format_exc().splitlines() if "moe_compute(tilize_input" in l]
    print("RESULT SIG " + (sig[0][:1500] if sig else "n/a"), flush=True)

ttnn.close_mesh_device(mesh)
