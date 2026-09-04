"""What `moe_compute` costs at the row counts we actually run, and whether it is right.

    uv run python scripts/dev/moe_compute_sweep.py

`moe_compute_check.py` established that the op runs repeatedly on the local path
(handoff §4c). Two questions decide whether it replaces `moe_block`:

  1. Cost at our row counts -- decode is M=1, prefill runs 512-row chunks. The
     1.5 ms measured at M=32 says nothing about either end.
  2. Whether the answer is right, and specifically whether `compute_only=False`
     ("FullLocal": the fused combine, which upstream runs without CCL and so is
     safe here) can replace our `_combine` matmul as well as `expert_ffn`.

For (2) the combine output is [k, tokens, hidden] -- one slot per selected
expert, not yet summed -- so the device-local MoE output is its sum over k, and
the existing all-reduce then sums across the four cards. A device owns only some
of a token's k experts, so this only works if the slots it does not own are
zero rather than garbage; that is the thing to check, not assume.

Weights are bfloat16 here, not the bfloat4_b we serve, so that a mismatch means
a wiring bug rather than quantisation.

One M per process, driven by argv, because an unsupported row count does not
raise -- M=1 aborts inside `CircularBufferConfig::set_page_size` in the workload
factory and takes the interpreter with it, so a single in-process loop loses
every measurement after the first bad one.

    uv run python scripts/dev/moe_compute_sweep.py 32        # cost at M=32
    uv run python scripts/dev/moe_compute_sweep.py combine   # fused-combine check
"""
import sys
import time

import torch
import ttnn

E_TOTAL, K, N, TOPK = 512, 2560, 640, 10
NDEV = 4
E_LOCAL = E_TOTAL // NDEV
HEIGHT_SHARD = 4
ROUNDS = 4

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NDEV))
u = ttnn.experimental.moe_compute_utils
torch.manual_seed(0)

ring = u.effective_matmul_ring_size(mesh)
width_dim = u.auto_output_width_shard_dim(K, matmul_ring_size=ring)

w0 = torch.randn(1, E_LOCAL, K, N) * 0.05
w1 = torch.randn(1, E_LOCAL, K, N) * 0.05
w2 = torch.randn(1, E_LOCAL, N, K) * 0.05

w0_w1_map, w2_map, dram_cores = u.get_weight_core_shard_maps(mesh, K, N)
w0w1 = u.prepare_w0_w1_tensor_for_moe_compute(w0, w1, 1, E_LOCAL, K, N, w0_w1_map)
w2p = u.prepare_w2_tensor_for_moe_compute(w2, 1, E_LOCAL, N, K, w2_map, w0_w1_map)
w0w1_mc, w2_mc, _, _ = u.get_weight_mem_configs(1, E_LOCAL, K, N, w0_w1_map, w2_map, dram_cores)

rep = ttnn.ReplicateTensorToMesh(mesh)
shard0 = ttnn.ShardTensorToMesh(mesh, dim=0)

w0w1_d = ttnn.from_torch(w0w1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=mesh, memory_config=w0w1_mc, mesh_mapper=rep)
w2_d = ttnn.from_torch(w2p, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                       device=mesh, memory_config=w2_mc, mesh_mapper=rep)

drain = ttnn.experimental.get_moe_tilize_drain_core(mesh, HEIGHT_SHARD, width_dim, K)
drain_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(drain.x, drain.y),
                                              ttnn.CoreCoord(drain.x, drain.y))})
mapping = (torch.arange(E_TOTAL) // E_LOCAL).to(torch.uint16).unsqueeze(0).repeat(NDEV, 1)
d_map = ttnn.from_torch(mapping, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=rep)


def on_drain(shape):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(drain_crs, list(shape), ttnn.ShardOrientation.ROW_MAJOR))


def build(m):
    """The four inputs for `m` tokens, plus the host-side routing for the golden."""
    tokens = torch.randn(m, K, dtype=torch.bfloat16) * 0.1
    idx = torch.stack([torch.randperm(E_TOTAL)[:TOPK] for _ in range(m)]).to(torch.int64)
    scores = torch.softmax(torch.randn(m, TOPK), dim=-1)

    owner = idx // E_LOCAL
    sparse = torch.zeros(NDEV, m, K, dtype=torch.bfloat16)
    for d in range(NDEV):
        sparse[d][(owner == d).any(dim=-1)] = tokens[(owner == d).any(dim=-1)]

    d_in = ttnn.from_torch(sparse, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=shard0)
    d_idx = ttnn.from_torch(idx.to(torch.uint16).unsqueeze(0).repeat(NDEV, 1, 1),
                            dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                            memory_config=on_drain((m, TOPK)), mesh_mapper=shard0)
    d_sc = ttnn.from_torch(scores.to(torch.bfloat16).unsqueeze(0).repeat(NDEV, 1, 1),
                           dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                           memory_config=on_drain((m, TOPK)), mesh_mapper=shard0)
    return (d_in, d_idx, d_sc), tokens, idx, scores


def call(inputs, compute_only=True, out_tensor=None):
    return ttnn.experimental.moe_compute(
        inputs[0], inputs[1], inputs[2], d_map, w0w1_d, w2_d,
        layer_id=0, output_height_shard_dim=HEIGHT_SHARD, intermediate_size=N,
        cluster_axis=None, topology=None, num_links=None,
        mux_core_range_set=None, optional_cross_device_semaphore=None,
        optional_output_tensor=out_tensor,
        activation_type=ttnn.operations.ccl.MoEActivationFunction.SILU,
        compute_only=compute_only)


arg = sys.argv[1] if len(sys.argv) > 1 else "combine"

# --- 1. cost at the row counts we run ---------------------------------------
if arg != "combine":
    m = int(arg)
    inputs, *_ = build(m)
    call(inputs)                                                # JIT
    ttnn.synchronize_device(mesh)
    best = float("inf")
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        out = call(inputs)
        ttnn.synchronize_device(mesh)
        best = min(best, 1000 * (time.perf_counter() - t0))
        for i in (0, 1, 2, 4):
            ttnn.deallocate(out[i])
    print(f"RESULT M={m:4d}: {best:.2f} ms/layer", flush=True)
    ttnn.close_mesh_device(mesh)
    raise SystemExit(0)

# --- 2. is the fused combine right, and can it replace ours? -----------------
print("RESULT --- fused combine (compute_only=False) ---", flush=True)
M = 32
inputs, tokens, idx, scores = build(M)

out_t = ttnn.from_torch(torch.zeros(TOPK, M, K, dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh, mesh_mapper=rep)
outs = call(inputs, compute_only=False, out_tensor=out_t)
ttnn.synchronize_device(mesh)
print(f"RESULT {len(outs)} outputs; combine slot = {tuple(outs[-1].shape)}", flush=True)

# Device-local partial = sum over the k axis; the mesh all-reduce sums the rest.
shards = [ttnn.to_torch(t).to(torch.float64) for t in ttnn.get_device_tensors(outs[-1])]
got = sum(s.sum(dim=0) for s in shards)                         # [M, K]

x = tokens.to(torch.float64)
W0, W1, W2 = (w.to(torch.float64) for w in (w0[0], w1[0], w2[0]))
want = torch.zeros(M, K, dtype=torch.float64)
for t in range(M):
    for j in range(TOPK):
        el = int(idx[t, j]) % E_LOCAL
        g = x[t] @ W0[el]
        h = (g * torch.sigmoid(g)) * (x[t] @ W1[el])
        want[t] += float(scores[t, j]) * (h @ W2[el])

err = (got - want).abs().max().item()
scale = want.abs().max().item()
print(f"RESULT combine max abs err {err:.4e} (values up to {scale:.4e}, "
      f"rel {err / max(scale, 1e-30):.3e})", flush=True)
print("RESULT verdict: " + ("sum-over-k is the MoE output" if err < 0.05 * scale else
                            "MISMATCH -- unowned slots are not zero, or the combine "
                            "contract differs"), flush=True)

ttnn.close_mesh_device(mesh)
