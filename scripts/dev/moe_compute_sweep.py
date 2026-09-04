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

# Fabric is only needed by the FullCcl path, and must be set before the mesh
# opens. The local paths deliberately run without it.
if len(sys.argv) > 1 and sys.argv[1] == "ccl":
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NDEV))
u = ttnn.experimental.moe_compute_utils
torch.manual_seed(0)

ring = u.effective_matmul_ring_size(mesh)
width_dim = u.auto_output_width_shard_dim(K, matmul_ring_size=ring)

w0 = (torch.randn(1, E_LOCAL, K, N) * 0.05).to(torch.bfloat16)
w1 = (torch.randn(1, E_LOCAL, K, N) * 0.05).to(torch.bfloat16)
w2 = (torch.randn(1, E_LOCAL, N, K) * 0.05).to(torch.bfloat16)

rep = ttnn.ReplicateTensorToMesh(mesh)
shard0 = ttnn.ShardTensorToMesh(mesh, dim=0)

# The packers in `moe_compute_utils` are, as that module says of itself,
# "executable specifications" -- reference implementations for tests. The
# production path is the on-device prepare ops plus a host quantise, and using
# the spec instead is what made the combine output garbage.
def _raw(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=mesh,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)


_w0, _w1, _w2 = _raw(w0), _raw(w1), _raw(w2)
_w0w1_prep = ttnn.experimental.prepare_w0_w1_tensor_for_moe_compute(
    _w0, _w1, L=1, E=E_LOCAL, K=K, N=N)
ttnn.deallocate(_w0)
ttnn.deallocate(_w1)
w0w1_host = ttnn.experimental.quantize_weights_via_host(
    _w0w1_prep, dtype=ttnn.bfloat4_b, memory_config=None)
ttnn.deallocate(_w0w1_prep)

_w2_prep = ttnn.experimental.prepare_w2_tensor_for_moe_compute(
    _w2, L=1, E=E_LOCAL, N=N, K=K)
ttnn.deallocate(_w2)
w2_host = ttnn.experimental.quantize_weights_via_host(
    _w2_prep, dtype=ttnn.bfloat4_b, memory_config=None)
ttnn.deallocate(_w2_prep)

wmc = ttnn.experimental.get_weight_mem_configs(
    mesh, num_layers=1, experts_per_device=E_LOCAL,
    hidden_size=K, intermediate_size=N, has_bias=False)
w0w1_d = ttnn.to_device(w0w1_host, mesh, memory_config=wmc.w0_w1)
w2_d = ttnn.to_device(w2_host, mesh, memory_config=wmc.w2)
print(f"RESULT weights prepared on device: {tuple(w0w1_d.shape)} {tuple(w2_d.shape)}",
      flush=True)

# FullCcl carries the combine over fabric through mux workers, which need cores
# of their own -- "Not enough mux cores! Needed: 1 ... Available: 0" is what
# passing None gets you. ((1,1),(3,3)) is the upstream default. It feeds three
# placement helpers, so it has to be decided before any of them.
MUX = (ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))])
       if len(sys.argv) > 1 and sys.argv[1] == "ccl" else None)

drain = ttnn.experimental.get_moe_tilize_drain_core(
    mesh, HEIGHT_SHARD, width_dim, K,
    **({"mux_core_range_set": MUX} if MUX is not None else {}))
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


def call(inputs, compute_only=True, out_tensor=None, ccl=None):
    """`ccl` carries the FullCcl options, or None for the purely local path.

    Topology is forced to Linear rather than left to `get_usable_topology()`,
    which marks any tensor spanning a full mesh row as Ring. Four p150a cards
    are physically a line, so the auto-detected Ring sends traffic across a wrap
    edge that does not exist -- which is what hung the first attempt, since
    `all_to_all_dispatch_metadata` takes no topology argument at all.
    """
    return ttnn.experimental.moe_compute(
        inputs[0], inputs[1], inputs[2], d_map, w0w1_d, w2_d,
        layer_id=0, output_height_shard_dim=HEIGHT_SHARD, intermediate_size=N,
        cluster_axis=None if ccl is None else 1,
        topology=None if ccl is None else ttnn.Topology.Linear,
        num_links=None if ccl is None else 1,
        mux_core_range_set=None if ccl is None else MUX,
        optional_cross_device_semaphore=ccl,
        optional_output_tensor=out_tensor,
        activation_type=ttnn.operations.ccl.MoEActivationFunction.SILU,
        compute_only=compute_only)


arg = sys.argv[1] if len(sys.argv) > 1 else "combine"

# --- 3. FullCcl: the fused combine over fabric, fed by locally-built inputs ---
# Never tried in this combination. The earlier full-mode attempt hung, but it
# also ran `all_to_all_dispatch_metadata` in the same round, and that is the op
# with no topology argument. `ttnn.all_reduce(cluster_axis=1, Linear)` runs in
# every layer of the live model, so fabric itself works on this box.
if arg == "ccl":
    M = 32
    inputs, tokens, idx, scores = build(M)
    # On the combine cores, not the whole grid -- that is where the barrier is
    # waited on (§4c.4).
    combine_cores = ttnn.experimental.get_moe_combine_cores(
        mesh, HEIGHT_SHARD, width_dim, K, mux_core_range_set=MUX)
    sem = ttnn.create_global_semaphore(
        mesh, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in combine_cores]), 0)
    print(f"RESULT semaphore on {len(combine_cores)} combine cores", flush=True)
    out_t = ttnn.from_torch(torch.zeros(TOPK, M, K, dtype=torch.bfloat16),
                            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
    print(f"RESULT combine output allocated {tuple(out_t.shape)}", flush=True)
    t0 = time.perf_counter()
    outs = call(inputs, compute_only=False, out_tensor=out_t, ccl=sem)
    ttnn.synchronize_device(mesh)
    print(f"RESULT FullCcl ran: {len(outs)} outputs in "
          f"{1000 * (time.perf_counter() - t0):.1f} ms (incl. JIT)", flush=True)
    print(f"RESULT combine slot = {tuple(outs[-1].shape)}", flush=True)
    best = float("inf")
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        call(inputs, compute_only=False, out_tensor=out_t, ccl=sem)
        ttnn.synchronize_device(mesh)
        best = min(best, 1000 * (time.perf_counter() - t0))
    print(f"RESULT FullCcl steady state: {best:.2f} ms/layer", flush=True)

    # The combine output is token-sharded: device d holds its own slice of the
    # tokens, already summed across every expert the CCL stage gathered. So
    # concatenating on the token axis and summing over k should *be* the MoE
    # output -- no all-reduce, unlike the path we have now.
    full = ttnn.to_torch(outs[-1], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
    f64 = full.to(torch.float64)

    # Unpopulated (k, token) slots are not zeroed by the op -- upstream's
    # validator carries an `output_data_map` for exactly this reason -- so find
    # out which slots carry data before summing anything.
    bad = ~torch.isfinite(f64)
    print(f"RESULT non-finite: {bad.sum().item()} of {bad.numel()} elements; "
          f"per-k rows affected {[int(bad[k].any(dim=-1).sum()) for k in range(TOPK)]}",
          flush=True)
    rowbad = bad.any(dim=-1)                                    # [k, M]
    print(f"RESULT tokens with any bad slot: "
          f"{int(rowbad.any(dim=0).sum())} of {M}", flush=True)
    # Slot k holds the raw output of the token's k-th selected expert, with **no
    # score applied** -- `compute_matmul_golden` upstream takes no scores at all,
    # and the combine golden assigns `contrib` unweighted. The router weighting
    # is the model's job, so apply it here.
    clean = torch.where(rowbad.unsqueeze(-1), torch.zeros_like(f64), f64)
    w = scores.to(torch.float64).transpose(0, 1).unsqueeze(-1)  # [k, M, 1]
    got = (clean * w).sum(dim=0)

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
    num = (got * want).sum().item()
    den = (got.norm() * want.norm()).item()
    print(f"RESULT concat shape {tuple(full.shape)} -> summed {tuple(got.shape)}", flush=True)
    print(f"RESULT max abs err {err:.4e} (values up to {scale:.4e}, "
          f"rel {err / max(scale, 1e-30):.3e}), cosine {num / max(den, 1e-30):.6f}",
          flush=True)
    ttnn.close_mesh_device(mesh)
    raise SystemExit(0)

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
