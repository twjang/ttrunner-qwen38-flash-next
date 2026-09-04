"""What 48 layers of `moe_compute` cost inside a trace, staged from DRAM.

    uv run python scripts/dev/moe_compute_traced.py

Every number in §4c.1 and §4c.3 was eager, with a `synchronize_device` per call.
Decode is one traced replay, so those numbers do not price the thing we would
actually ship. This is the measurement that does, and it is the one that decides
whether re-packing the 130 GB weight cache into the `moe_compute` layout is worth
doing:

  - all 48 layers in a single trace, the way the decoder is captured
  - inputs staged DRAM -> L1 per layer and freed after, because the reference
    says L1 is too tight to hold them for even one invocation (§4c.4)
  - FullCcl, since that is the only path whose output we can consume (§4c.3)

Weights are one layer's, reused with `layer_id=0` for all 48 calls. The real
thing packs all 48 into one tensor, but that is 67 GB of host memory to build a
benchmark with; the per-call DRAM traffic is the same either way, since reading
the same layer 48 times still reads it from DRAM 48 times.
"""
import time

import torch
import ttnn

E_TOTAL, K, N, TOPK = 512, 2560, 640, 10
NDEV, LAYERS = 4, 48
E_LOCAL = E_TOTAL // NDEV
M = 32                                        # decode pads its single row up to a tile
HEIGHT_SHARD = 4
MUX_RANGE = ((1, 1), (3, 3))                  # upstream default; feeds three placement helpers

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NDEV), trace_region_size=256 << 20)
u = ttnn.experimental.moe_compute_utils
torch.manual_seed(0)

ring = u.effective_matmul_ring_size(mesh)
width_dim = u.auto_output_width_shard_dim(K, matmul_ring_size=ring)
mux = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*MUX_RANGE[0]),
                                        ttnn.CoreCoord(*MUX_RANGE[1]))])

# --- weights ----------------------------------------------------------------
w0 = torch.randn(1, E_LOCAL, K, N) * 0.05
w1 = torch.randn(1, E_LOCAL, K, N) * 0.05
w2 = torch.randn(1, E_LOCAL, N, K) * 0.05
w0_w1_map, w2_map, dram_cores = u.get_weight_core_shard_maps(mesh, K, N)
w0w1 = u.prepare_w0_w1_tensor_for_moe_compute(w0, w1, 1, E_LOCAL, K, N, w0_w1_map)
w2p = u.prepare_w2_tensor_for_moe_compute(w2, 1, E_LOCAL, N, K, w2_map, w0_w1_map)
w0w1_mc, w2_mc, _, _ = u.get_weight_mem_configs(1, E_LOCAL, K, N, w0_w1_map, w2_map, dram_cores)

rep = ttnn.ReplicateTensorToMesh(mesh)
shard0 = ttnn.ShardTensorToMesh(mesh, dim=0)
w0w1_d = ttnn.from_torch(w0w1, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                         device=mesh, memory_config=w0w1_mc, mesh_mapper=rep)
w2_d = ttnn.from_torch(w2p, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                       device=mesh, memory_config=w2_mc, mesh_mapper=rep)
print("RESULT weights on device (bfloat4_b, as served)", flush=True)

# --- placement: mux first, then everything that depends on it ----------------
drain = ttnn.experimental.get_moe_tilize_drain_core(
    mesh, HEIGHT_SHARD, width_dim, K, mux_core_range_set=mux)
drain_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(drain.x, drain.y),
                                              ttnn.CoreCoord(drain.x, drain.y))})
combine_cores = ttnn.experimental.get_moe_combine_cores(
    mesh, HEIGHT_SHARD, width_dim, K, mux_core_range_set=mux)
sem = ttnn.create_global_semaphore(
    mesh, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in combine_cores]), 0)
print(f"RESULT placement: drain=({drain.x},{drain.y}) "
      f"combine_cores={len(combine_cores)}", flush=True)


def sharded(shape):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(drain_crs, list(shape), ttnn.ShardOrientation.ROW_MAJOR))


IDX_MC, SC_MC = sharded((M, TOPK)), sharded((M, TOPK))

# --- inputs, resident in DRAM ------------------------------------------------
tokens = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
idx = torch.stack([torch.randperm(E_TOTAL)[:TOPK] for _ in range(M)]).to(torch.int64)
scores = torch.softmax(torch.randn(M, TOPK), dim=-1)
owner = idx // E_LOCAL
sparse = torch.zeros(NDEV, M, K, dtype=torch.bfloat16)
for d in range(NDEV):
    sparse[d][(owner == d).any(dim=-1)] = tokens[(owner == d).any(dim=-1)]

dram = ttnn.DRAM_MEMORY_CONFIG
in_dram = ttnn.from_torch(sparse, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                          device=mesh, memory_config=dram, mesh_mapper=shard0)
idx_dram = ttnn.from_torch(idx.to(torch.uint16).unsqueeze(0).repeat(NDEV, 1, 1),
                           dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                           memory_config=dram, mesh_mapper=shard0)
sc_dram = ttnn.from_torch(scores.to(torch.bfloat16).unsqueeze(0).repeat(NDEV, 1, 1),
                          dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                          memory_config=dram, mesh_mapper=shard0)
d_map = ttnn.from_torch((torch.arange(E_TOTAL) // E_LOCAL).to(torch.uint16)
                        .unsqueeze(0).repeat(NDEV, 1),
                        dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                        memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=rep)
out_t = ttnn.from_torch(torch.zeros(TOPK, M, K, dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
print("RESULT inputs resident in DRAM", flush=True)


def one_layer():
    """Stage from DRAM, run, free -- what a 48-layer model has to do per layer."""
    s = ttnn.to_memory_config(in_dram, memory_config=ttnn.L1_MEMORY_CONFIG)
    i = ttnn.to_memory_config(idx_dram, memory_config=IDX_MC)
    c = ttnn.to_memory_config(sc_dram, memory_config=SC_MC)
    outs = ttnn.experimental.moe_compute(
        s, i, c, d_map, w0w1_d, w2_d,
        layer_id=0, output_height_shard_dim=HEIGHT_SHARD, intermediate_size=N,
        cluster_axis=1, topology=ttnn.Topology.Linear, num_links=1,
        mux_core_range_set=mux, optional_output_tensor=out_t,
        optional_cross_device_semaphore=sem,
        activation_type=ttnn.operations.ccl.MoEActivationFunction.SILU)
    ttnn.deallocate(s)
    ttnn.deallocate(i)
    ttnn.deallocate(c)
    # Slot 3 shares slot 4's buffer, so it is never freed on its own (§4c.2).
    for j in (0, 1, 2, 4):
        ttnn.deallocate(outs[j])
    return outs[5]


def step():
    for _ in range(LAYERS):
        one_layer()


# --- eager first, for the comparison, then traced ----------------------------
step()                                                          # JIT all programs
ttnn.synchronize_device(mesh)
best_eager = float("inf")
for _ in range(3):
    t0 = time.perf_counter()
    step()
    ttnn.synchronize_device(mesh)
    best_eager = min(best_eager, 1000 * (time.perf_counter() - t0))
print(f"RESULT eager  {LAYERS} layers: {best_eager:.1f} ms "
      f"({best_eager / LAYERS:.2f} ms/layer)", flush=True)

tid = ttnn.begin_trace_capture(mesh, cq_id=0)
step()
ttnn.end_trace_capture(mesh, tid, cq_id=0)
print("RESULT trace captured", flush=True)

ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
best = float("inf")
for _ in range(5):
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    best = min(best, 1000 * (time.perf_counter() - t0))
print(f"RESULT traced {LAYERS} layers: {best:.1f} ms "
      f"({best / LAYERS:.2f} ms/layer)", flush=True)
print(f"RESULT trace vs eager: {best_eager / best:.2f}x", flush=True)
print(f"RESULT for reference, the whole traced decode step today is 109.2 ms",
      flush=True)

ttnn.release_trace(mesh, tid)
ttnn.close_mesh_device(mesh)
