"""Wire one layer onto `ttnn.experimental.moe_compute`, the way the op supports.

    uv run python scripts/dev/moe_compute_check.py

The MoE we hand-rolled on `sparse_matmul` materialises [1, E, M, K] per layer --
84 MB before the expert-axis reshard, 21 after -- to carry ~59 KB of selected
expert output (handoff invariant 27). `moe_compute` runs gate/up/down plus the
activation in one fused kernel over a *sparse* buffer that holds only the tokens
routed to this device's experts, so it never materialises that.

The first attempt fed it from `all_to_all_dispatch_metadata` and hung the card.
That was not our wiring: all-to-all is non-functional on Blackhole upstream
(tt-metal#27859, #30030), and ttnn's asynchrony hid it -- reading a shape does
not synchronise, so the first call only looked like it worked and the failure
surfaced at the next op, even a bare `ttnn.add`. See handoff §4c, invariant 28.

So this follows the path the op actually supports on one card, which is what the
official `test_moe_compute_single_card.py` exercises: build the four inputs
locally and pass `cluster_axis=None` with no topology, links, mux or semaphore.
That fits us rather than fighting us. The 512 experts are already sharded across
the four cards (`Shard.EXPERT`, 128 local each) with an all-reduce behind them,
so there is nothing for a device-to-device dispatch to do -- each card needs only
its own tokens against its own experts.

The milestone here is narrow and worth stating plainly: run the op repeatedly
without wedging the device, and find out what a layer costs. Numerics come after.
"""
import time

import torch
import ttnn

E_TOTAL, K, N, TOPK = 512, 2560, 640, 10      # experts, hidden, intermediate, top-k
NDEV = 4
E_LOCAL = E_TOTAL // NDEV
M = 32                                        # tokens; 32 == one tile row, as upstream tests use
ROUNDS = 5

# No `set_fabric_config` here, deliberately. Fabric is for the CCL paths, and
# every one of those hangs on this box.
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NDEV))
u = ttnn.experimental.moe_compute_utils
torch.manual_seed(0)


def bail(stage, exc):
    print(f"RESULT {stage} rejected: {type(exc).__name__}: "
          f"{(str(exc) or repr(exc)).splitlines()[0][:240]}", flush=True)
    ttnn.close_mesh_device(mesh)
    raise SystemExit(0)


ring = u.effective_matmul_ring_size(mesh)
HEIGHT_SHARD = 4                              # what the upstream single-card tests use
width_dim = u.auto_output_width_shard_dim(K, matmul_ring_size=ring)
print(f"RESULT ring={ring} height_shard={HEIGHT_SHARD} width_shard={width_dim}", flush=True)

# --- weights ----------------------------------------------------------------
w0 = torch.randn(1, E_LOCAL, K, N) * 0.05     # gate, [L, E, K, N]
w1 = torch.randn(1, E_LOCAL, K, N) * 0.05     # up
w2 = torch.randn(1, E_LOCAL, N, K) * 0.05     # down, [L, E, N, K]

w0_w1_map, w2_map, dram_cores = u.get_weight_core_shard_maps(mesh, K, N)
w0w1 = u.prepare_w0_w1_tensor_for_moe_compute(w0, w1, 1, E_LOCAL, K, N, w0_w1_map)
w2p = u.prepare_w2_tensor_for_moe_compute(w2, 1, E_LOCAL, N, K, w2_map, w0_w1_map)
w0w1_mc, w2_mc, _, _ = u.get_weight_mem_configs(1, E_LOCAL, K, N, w0_w1_map, w2_map, dram_cores)
print(f"RESULT packed w0w1 {tuple(w0w1.shape)}  w2 {tuple(w2p.shape)}", flush=True)

rep = ttnn.ReplicateTensorToMesh(mesh)
shard0 = ttnn.ShardTensorToMesh(mesh, dim=0)

try:
    w0w1_d = ttnn.from_torch(w0w1, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                             device=mesh, memory_config=w0w1_mc, mesh_mapper=rep)
    w2_d = ttnn.from_torch(w2p, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                           device=mesh, memory_config=w2_mc, mesh_mapper=rep)
except Exception as exc:                                        # noqa: BLE001
    bail("weight upload", exc)
print("RESULT weights uploaded", flush=True)

# --- the four inputs, built locally ------------------------------------------
# `gen_sparse_buffer_and_indices` upstream exists to "simulate the output from
# all_to_all_dispatch", which is exactly the part we are replacing: a token sits
# in device d's slice only when one of its top-k experts lives on d.
tokens = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
idx = torch.stack([torch.randperm(E_TOTAL)[:TOPK] for _ in range(M)]).to(torch.int64)
scores = torch.softmax(torch.randn(M, TOPK), dim=-1)

owner = idx // E_LOCAL                                          # [M, TOPK] -> device
sparse = torch.zeros(NDEV, M, K, dtype=torch.bfloat16)
for d in range(NDEV):
    hit = (owner == d).any(dim=-1)                              # [M]
    sparse[d][hit] = tokens[hit]

mapping = (torch.arange(E_TOTAL) // E_LOCAL).to(torch.uint16)
mapping = mapping.unsqueeze(0).repeat(NDEV, 1)                  # [D, E], replicated

drain = ttnn.experimental.get_moe_tilize_drain_core(mesh, HEIGHT_SHARD, width_dim, K)
drain_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(drain.x, drain.y),
                                              ttnn.CoreCoord(drain.x, drain.y))})


def on_drain(shape, dtype):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(drain_crs, list(shape), ttnn.ShardOrientation.ROW_MAJOR))


try:
    d_in = ttnn.from_torch(sparse, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG,
                           mesh_mapper=shard0)
    d_map = ttnn.from_torch(mapping, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG,
                            mesh_mapper=rep)
    d_idx = ttnn.from_torch(idx.to(torch.uint16).unsqueeze(0).repeat(NDEV, 1, 1),
                            dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                            memory_config=on_drain((M, TOPK), ttnn.uint16),
                            mesh_mapper=shard0)
    d_sc = ttnn.from_torch(scores.to(torch.bfloat16).unsqueeze(0).repeat(NDEV, 1, 1),
                           dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                           memory_config=on_drain((M, TOPK), ttnn.bfloat16),
                           mesh_mapper=shard0)
except Exception as exc:                                        # noqa: BLE001
    bail("input upload", exc)
print("RESULT inputs uploaded", flush=True)


def run():
    return ttnn.experimental.moe_compute(
        d_in, d_idx, d_sc, d_map, w0w1_d, w2_d,
        layer_id=0,
        output_height_shard_dim=HEIGHT_SHARD,
        intermediate_size=N,
        # All five must be None on the local path; anything else pulls in CCL.
        cluster_axis=None, topology=None, num_links=None,
        mux_core_range_set=None, optional_cross_device_semaphore=None,
        activation_type=ttnn.operations.ccl.MoEActivationFunction.SILU,
        compute_only=True,
    )


# Slots 3 and 4 share a backing buffer -- releasing 4 releases both, and freeing
# 3 as well is a double free.
FREE = (0, 1, 2, 4)

for r in range(ROUNDS):
    t0 = time.perf_counter()
    try:
        out = run()
        ttnn.synchronize_device(mesh)                           # the real checkpoint
    except Exception as exc:                                    # noqa: BLE001
        bail(f"round {r}", exc)
    ms = 1000 * (time.perf_counter() - t0)
    if r == 0:
        print(f"RESULT {len(out)} outputs: {[tuple(t.shape) for t in out]}", flush=True)
    print(f"RESULT round {r}: {ms:.2f} ms", flush=True)
    for i in FREE:
        ttnn.deallocate(out[i])

print(f"RESULT survived {ROUNDS} rounds without wedging the device", flush=True)
ttnn.close_mesh_device(mesh)
