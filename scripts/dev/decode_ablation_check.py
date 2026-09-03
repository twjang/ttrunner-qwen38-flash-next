"""Where does the *traced* decode step's 236 ms actually go?

    uv run python scripts/dev/decode_ablation_check.py <part>     (default none)

    parts: none moe shared allreduce attn reinject ple
           expertffn combine permute routing topk<N>
           qsa deltanet

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

import twtest.tt.moe as moe
import twtest.tt.model as model_mod
from _device_model import open_model

PART = sys.argv[1] if len(sys.argv) > 1 else "none"
ITERS = 25

mesh, cfg, m = open_model(max_seq_len=4096)

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
    from twtest.tt.moe import expert_ffn as _ef, HIFI4 as _H
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
elif PART.startswith("topk"):
    # Does sparse_matmul actually skip the unselected experts? If the step time
    # is flat in top_k it is reading all 512 regardless, which is the whole
    # question for decode.
    cfg.num_experts_per_tok = int(PART[4:])
elif PART != "none":
    raise SystemExit(f"unknown part {PART}")

from twtest.tt.traced import TracedDecoder  # noqa: E402

state = m.new_state(batch=1)
dec = TracedDecoder(m, state)
dec.reset()
for _ in range(5):
    dec.step([1000])
ttnn.synchronize_device(mesh)

samples = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    dec.step([1000])
    ttnn.synchronize_device(mesh)
    samples.append(1000 * (time.perf_counter() - t0))
samples.sort()
med = samples[len(samples) // 2]
print(f"RESULT ablate={PART:10s} median {med:7.2f} ms  min {samples[0]:7.2f}  "
      f"max {samples[-1]:7.2f}", flush=True)
