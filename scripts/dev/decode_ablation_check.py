"""Where does the *traced* decode step's 236 ms actually go?

    uv run python scripts/dev/decode_ablation_check.py <part> [prefill_tokens]

With a token count it ablates one *chunked prefill* of that many tokens instead
of the traced step. Prefill is dispatch-bound where the traced step is not
(invariants 19 and 21), so the two answer different questions -- but "which
component" is worth knowing on both, and only op counts were known for prefill.

    parts: none moe shared allreduce attn reinject ple
           expertffn combine permute routing topk<N>
           qsa deltanet sdpa kvupdate gateup downproj
           prefill-only seams: applyexperts chunkattn chunkqsa chunkdeltanet
                               routechunk gdas prepare grm reinjectp sharedp

Note that the decode seams do nothing to a prefill: it takes the route/
apply_experts split and `_attention_chunk`, not `moe_block` and
`_attention_step`. Ablating `moe` on a prefill "saves" 3.5 ms of 922 for exactly
that reason -- a stub that is never called measures nothing, and the giveaway is
a component that appears to cost nothing at all.

Invariant 19 says the eager prefill path is dispatch-bound, and 5.7 says the
traced step is emphatically not -- removing 97 ops moved it 236.1 -> 236.0 ms.
So the step's time is real device work, and speeding it up means knowing which
work. Op counts cannot answer that: they price dispatches, and dispatches are
what the trace already removed.

Ablation on the traced path measures it directly. Each `part` replaces one
component with an identity of the same shape -- *zero* extra ops, so the delta
against `none` is that component's device time and nothing else. The output is
wrong by construction; this is a stopwatch, not a check.

One ablation per process. Capturing a second trace and alternating replays hangs
this build (`ttnn_bug_report/`), and releasing between captures is not a path
worth trusting for a measurement.
"""
import sys
import time

import ttnn

import ttrunner_qwen38_flash_next.tt.moe as moe
import ttrunner_qwen38_flash_next.tt.model as model_mod
from _device_model import open_model

PART = sys.argv[1] if len(sys.argv) > 1 else "none"
ITERS = 25

mesh, cfg, m = open_model(max_seq_len=4096 if len(sys.argv) <= 2 else 512)

# Each stand-in returns a tensor of the shape the real component returns, built
# from an input that already has it, so the substitution costs nothing at all.
if PART == "moe":
    moe.moe_block = lambda mixed, *a, **kw: mixed
elif PART == "shared":
    moe.shared_expert = lambda mixed, *a, **kw: mixed
elif PART == "allreduce":
    m.all_reduce = lambda x: x
elif PART == "attn":
    model_mod.TTModel._attention_step = lambda self, mixed, *a, **kw: mixed
    model_mod.TTModel._linear_attention_step = lambda self, mixed, *a, **kw: mixed
elif PART == "noselect":
    # Not an ablation of a component but of a *regime*: the selection is exact
    # below indexer_budget, where topk returns every visible block, so this is
    # what a sequence under 2048 tokens could legitimately run. The block cache
    # is still maintained, which is the part that has to stay.
    m.selection_active = False
elif PART == "idxtopk":
    # Only the indexer's topk: k=512 out of 1024 blocks, i.e. half a sort, and
    # it runs in all twelve QSA layers. Substituted by a constant of the shape
    # it returns, so the delta is the op alone.
    _topk_real, _topk_const = ttnn.topk, {}

    def _topk(t, k, dim=-1, *a, **kw):
        if k != m.indexer_topk:
            return _topk_real(t, k, dim=dim, *a, **kw)
        sig = (tuple(t.shape), k)
        if sig not in _topk_const:
            import torch as _t
            shape = list(t.shape)
            shape[dim] = k
            _topk_const[sig] = (
                _topk_real(t, k, dim=dim, *a, **kw)[0],
                m.to_dev(_t.zeros(shape, dtype=_t.float32), ttnn.uint16),
            )
        return _topk_const[sig]

    ttnn.topk = _topk
elif PART == "idxscatter":
    # Only the scatter that paints the selection into a max_seq_len-wide row.
    # The comment at the call site prices it at 0.3-0.7 ms; twelve layers would
    # make that 3.6-8.4 ms of the 36.8.
    _scatter_real = ttnn.scatter
    ttnn.scatter = lambda base, dim, idx, src, *a, **kw: base
elif PART == "indexer":
    # QSA's sparse selection only. Dropping it makes the read plain causal, so
    # `is_causal` flips back on its own and no mask is built -- which is the
    # point: the mask is [B, n_q, max_seq_len] and is rebuilt every layer.
    m.use_indexer = False
elif PART == "qsa":
    # the 12 full-attention layers only
    model_mod.TTModel._attention_step = lambda self, mixed, *a, **kw: mixed
elif PART == "deltanet":
    # the 36 gated-DeltaNet layers only
    model_mod.TTModel._linear_attention_step = lambda self, mixed, *a, **kw: mixed
elif PART == "reinject":
    # reinject(hidden, branch, inject, hc) -> [.., hc*hidden]; `hidden` already is
    model_mod.reinject = lambda hidden, branch, inject, hc: hidden
elif PART == "ple":
    model_mod.TTModel._ple_step = lambda self, hidden, *a, **kw: ttnn.zeros_like(hidden)
elif PART == "expertffn":
    # keep the routing, drop only the expert arithmetic: [1, E, M, K] of zeros
    # one cached buffer, not a fresh zeros per layer: allocating 48 of these
    # inside the capture is its own cost and would be counted as the FFN's
    _ffn_stub = {}

    def _stub_ffn(x, gw, uw, dw, sp, nnz, E, K, I):
        key = (E, x.shape[-2], K)
        if key not in _ffn_stub:
            import torch
            _ffn_stub[key] = ttnn.from_torch(
                torch.zeros(1, E, x.shape[-2], K), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        return _ffn_stub[key]

    moe.expert_ffn = _stub_ffn
elif PART in ("combine", "routing", "permute"):
    # Split moe_block's non-FFN half. `combine` drops the weighted sum over the
    # expert axis -- [1, 512, 1, 2560], which TILE_LAYOUT pads to 32 rows, so it
    # is 84 MB a layer rather than 2.6. `routing` keeps the combine but feeds it
    # a constant gate, dropping the router linear/softmax/topk/threshold chain.
    from ttrunner_qwen38_flash_next.tt.moe import expert_ffn as _ef, HIFI4 as _H
    _cache = {}

    def _patched(x, router_w, gate_w, up_w, down_w, top_k, E, K, I):
        m = x.shape[-2]
        if PART == "routing":
            key = ("w", E, m)
            if key not in _cache:
                import torch
                _cache[key] = ttnn.from_torch(
                    torch.full((1, 1, m, E), 1.0 / top_k), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                _cache[("s", E, m)] = ttnn.to_layout(ttnn.from_torch(
                    torch.ones(1, 1, 1, E), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)), ttnn.ROW_MAJOR_LAYOUT)
            weights, sparsity = _cache[key], _cache[("s", E, m)]
        else:
            logits = ttnn.linear(x, router_w, compute_kernel_config=_H)
            probs = ttnn.softmax(logits, dim=-1, compute_kernel_config=_H)
            values, _ = ttnn.topk(probs, k=top_k, dim=-1, largest=True, sorted=True)
            v = list(values.shape)
            threshold = ttnn.slice(values, (0, 0, 0, top_k - 1), (v[0], v[1], v[2], top_k))
            keep = ttnn.ge(probs, threshold, dtype=ttnn.bfloat16)
            kept = ttnn.multiply(probs, keep)
            weights = ttnn.divide(kept, ttnn.sum(kept, dim=-1, keepdim=True))
            sparsity = ttnn.max(keep, dim=-2, keepdim=True)
            sparsity = ttnn.to_layout(ttnn.typecast(sparsity, ttnn.bfloat16),
                                      ttnn.ROW_MAJOR_LAYOUT)
        per_expert = _ef(x, gate_w, up_w, down_w, sparsity, None, E, K, I)
        if PART == "combine":
            # take one expert's slab instead of the weighted sum over all E
            return ttnn.reshape(ttnn.slice(per_expert, (0, 0, 0, 0), (1, 1, m, K)),
                                (1, 1, m, K))
        if PART == "permute":
            key = ("g", E, m)
            if key not in _cache:
                import torch
                _cache[key] = ttnn.from_torch(
                    torch.full((1, E, m, 1), 1.0 / top_k), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            gate_per_expert = _cache[key]
        else:
            gate_per_expert = ttnn.permute(weights, (0, 3, 2, 1))
        return ttnn.sum(ttnn.multiply(per_expert, gate_per_expert), dim=1, keepdim=True)

    moe.moe_block = _patched
elif PART == "applyexperts":
    # the whole MoE compute half of a prefill chunk (routing kept)
    moe.apply_experts = lambda x, *a, **kw: x
elif PART == "routechunk":
    _rc = {}

    def _stub_route(x, router_w, top_k):
        key = tuple(x.shape)
        if key not in _rc:
            import torch
            E = router_w.shape[-1]
            m = x.shape[-2]
            w = ttnn.from_torch(torch.full((1, 1, m, E), 1.0 / top_k),
                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            _rc[key] = (w, w)
        return _rc[key]

    moe.route = _stub_route
elif PART in ("grm", "reinjectp", "sharedp"):
    # Seams that are shared by both paths, stubbed the shape-learning way so the
    # first call per shape pays and nothing allocates in the timed region. These
    # cover the two thirds of a prefill chunk that the component ablations above
    # do not account for.
    _seen = {}

    def _stub(real):
        def wrapper(*a, **kw):
            key = tuple(tuple(t.shape) for t in a if hasattr(t, "shape"))
            if key not in _seen:
                out = real(*a, **kw)
                items = out if isinstance(out, tuple) else (out,)
                z = tuple(ttnn.zeros_like(t) if hasattr(t, "shape") else t for t in items)
                _seen[key] = z if isinstance(out, tuple) else z[0]
            return _seen[key]
        return wrapper

    import ttrunner_qwen38_flash_next.tt.ops as ops_mod
    if PART == "grm":
        ops_mod.gated_residual_mix = _stub(ops_mod.gated_residual_mix)
        model_mod.gated_residual_mix = ops_mod.gated_residual_mix
    elif PART == "reinjectp":
        ops_mod.reinject = _stub(ops_mod.reinject)
        model_mod.reinject = ops_mod.reinject
    else:
        moe.shared_expert = _stub(moe.shared_expert)
elif PART in ("gdas", "prepare"):
    # Split the DeltaNet chunk into the fused ttnn op and the preparation this
    # project wrote. The stub learns the real output shapes from one call and
    # then returns cached zeros, so only the first invocation per shape costs
    # anything and no allocation happens in the timed region.
    _learn = {}

    def _shape_stub(real):
        def wrapper(*a, **kw):
            key = tuple(tuple(t.shape) for t in a if hasattr(t, "shape"))
            if key not in _learn:
                out = real(*a, **kw)
                items = out if isinstance(out, tuple) else (out,)
                zeros = tuple(ttnn.zeros_like(t) if hasattr(t, "shape") else t
                              for t in items)
                _learn[key] = zeros if isinstance(out, tuple) else zeros[0]
            return _learn[key]
        return wrapper

    if PART == "gdas":
        ttnn.transformer.gated_delta_attn_seq = _shape_stub(
            ttnn.transformer.gated_delta_attn_seq)
    else:
        import ttrunner_qwen38_flash_next.tt.deltanet as dn
        dn.prepare_device = _shape_stub(dn.prepare_device)
        if getattr(model_mod, "prepare_device", None) is not None:
            model_mod.prepare_device = dn.prepare_device
elif PART == "chunkqsa":
    model_mod.TTModel._attention_chunk = lambda self, mixed, *a, **kw: mixed
elif PART == "chunkdeltanet":
    model_mod.TTModel._linear_attention_chunk = lambda self, mixed, *a, **kw: mixed
elif PART == "chunkattn":
    model_mod.TTModel._attention_chunk = lambda self, mixed, *a, **kw: mixed
    model_mod.TTModel._linear_attention_chunk = lambda self, mixed, *a, **kw: mixed
elif PART == "sdpa":
    # the attention itself, not the projections around it
    _z = {}

    def _stub_sdpa(q, *a, **kw):
        key = tuple(q.shape)
        if key not in _z:
            import torch
            _z[key] = ttnn.from_torch(
                torch.zeros(*q.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        return _z[key]

    ttnn.transformer.paged_scaled_dot_product_attention_decode = _stub_sdpa
elif PART == "kvupdate":
    ttnn.experimental.paged_update_cache = lambda *a, **kw: None
elif PART in ("gateup", "downproj"):
    # split expert_ffn's two matmuls. `gateup` stubs the fused gate|up and keeps
    # the down projection; `downproj` the reverse. The stub is one cached buffer,
    # so neither measures an allocation.
    from ttrunner_qwen38_flash_next.tt.moe import sparse_program_config as _spc, HIFI4 as _H2
    _buf = {}

    def _zeros(shape):
        if shape not in _buf:
            import torch
            _buf[shape] = ttnn.from_torch(
                torch.zeros(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        return _buf[shape]

    def _split_ffn(x, gate_w, up_w, down_w, sparsity, nnz, E, K, I):
        m, k_in = x.shape[-2], x.shape[-1]
        n = gate_w.shape[-1] // 2
        kw2 = {"sparsity": sparsity, "nnz": nnz, "is_input_a_sparse": True,
               "is_input_b_sparse": True}
        if PART == "gateup":
            hidden = _zeros((1, E, m, n))
        else:
            both = ttnn.reshape(ttnn.sparse_matmul(
                x, gate_w, program_config=_spc(m, k_in, 2 * n),
                compute_kernel_config=_H2, sparsity=sparsity, nnz=nnz,
                is_input_a_sparse=False, is_input_b_sparse=True), (1, E, m, 2 * n))
            gate = ttnn.slice(both, (0, 0, 0, 0), (1, E, m, n))
            up = ttnn.slice(both, (0, 0, 0, n), (1, E, m, 2 * n))
            hidden = ttnn.multiply(ttnn.silu(gate), up)
        if PART == "downproj":
            return _zeros((1, E, m, K))
        return ttnn.sparse_matmul(
            hidden, down_w, program_config=_spc(m, hidden.shape[-1], K),
            compute_kernel_config=_H2, **kw2)

    moe.expert_ffn = _split_ffn
elif PART.startswith("topk"):
    # Does sparse_matmul actually skip the unselected experts? If the step time
    # is flat in top_k it is reading all 512 regardless, which is the whole
    # question for decode.
    cfg.num_experts_per_tok = int(PART[4:])
elif PART != "none":
    raise SystemExit(f"unknown part {PART}")

PREFILL = int(sys.argv[2]) if len(sys.argv) > 2 else 0

if PREFILL:
    prompt = [1000] * PREFILL

    def work():
        st = m.new_state(batch=1)
        m.prefill(prompt, st)
else:
    from ttrunner_qwen38_flash_next.tt.traced import TracedDecoder  # noqa: E402

    state = m.new_state(batch=1)
    dec = TracedDecoder(m, state)
    dec.reset()

    def work():
        dec.step([1000])

ITERS = 7 if PREFILL else ITERS
for _ in range(2 if PREFILL else 5):
    work()
ttnn.synchronize_device(mesh)

samples = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    work()
    ttnn.synchronize_device(mesh)
    samples.append(1000 * (time.perf_counter() - t0))
samples.sort()
med = samples[len(samples) // 2]
what = f"prefill{PREFILL}" if PREFILL else "step"
print(f"RESULT {what} ablate={PART:10s} median {med:7.2f} ms  min {samples[0]:7.2f}  "
      f"max {samples[-1]:7.2f}", flush=True)
