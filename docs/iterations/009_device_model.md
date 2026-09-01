# 009 — The device model

Date: 2026-09-01
Status: written and reviewed; device validation pending the weight conversion.

---

## Shape of the forward

```
hidden [1, 1, B, 10240]          replicated on all 4 devices
  |
  +- PLE (layer 1 only): n-gram rows gathered on host, 16 per sequence
  +- hyper-connection read gate  -> mixed [1, 1, B, 2560]
  |    +- 36 layers: Gated DeltaNet   (recurrent, no collective)
  |    +- 12 layers: QSA attention    (sdpa_decode, no collective)
  +- hyper-connection re-inject
  +- hyper-connection read gate
  |    +- MoE: 10 of 512 routed experts + 1 shared  -> ALL-REDUCE
  +- hyper-connection re-inject
  |
final hyper-connection mixer -> LM head (vocab-sharded) -> all-gather
```

One all-reduce per layer and one all-gather per step: 49 collectives, against
~100 for a fully head-sharded layout (iteration 008).

---

## Observation 1 — decode wants a different algorithm than prefill

`gated_delta_attn_seq` needs eight prepared tensors, including a unit-diagonal
triangular system and its per-32×32-block inverses. Preparing those on the host
would mean 36 PCIe round-trips per token — far more expensive than the work they
replace.

For a single token the delta rule is just its recurrence:

```
state <- state * exp(g)
delta  = (v - k @ state) * beta
state <- state + kᵀ delta
out    = q @ state
```

Three batched matmuls over the (batch × heads) axis plus elementwise ops, all on
device. `gated_delta_attn_seq` stays for chunked prefill, where the preparation
amortises over 128 tokens instead of one.

---

## Observation 2 — static shapes, chosen for a reason

The K/V cache is preallocated to `max_seq_len` and written in place by
`ttnn.experimental.paged_update_cache(..., update_idxs_tensor=pos)`; attention is
`scaled_dot_product_attention_decode(..., cur_pos_tensor=pos)`. Both take the
position as a **tensor**, so no shape in the step depends on how far decoding has
got.

The obvious alternative — growing the cache with `ttnn.concat` — changes shapes
every token, which reallocates and re-dispatches each step and rules out trace
capture entirely.

---

## Observation 3 — four bugs found by review, before ever running

Reading the model against the reference caught things a shape check would not:

1. **`ffn_gate_inp_shexp` is a 1-D tensor that is a linear weight.** It is the
   shared expert's sigmoid gate, `2560 -> 1`. The generic 1-D rule laid it out as
   a row `(1,1,1,2560)`; as a linear weight it must be a column `(1,1,2560,1)`.
   Every *other* 1-D tensor here (norm gammas, `ssm_a`, `dt_bias`) is applied
   elementwise and does want the row form. Test: `test_shared_expert_gate_is_a_column`.

2. **Head tiling must happen inside each sequence.** V heads are stored tiled over
   K heads, so Q/K tile to 48 heads. Flattening `(batch, head)` first and then
   repeating gives row `j % (batch·16)`, which is right for batch 1 and wrong for
   every larger batch — it pairs sequence *b*'s query with another sequence's
   value. Correct order: reshape to `(batch, 1, n_k, hd)`, repeat on the head
   axis, then flatten. Test: `test_head_tiling_order_is_per_sequence`.

3. **`ttnn.repeat` takes a plain sequence**, not a `ttnn.Shape`; five call sites
   were relying on an implicit conversion.

4. **The PLE gate's `clamp_min(1e-6)`** was missing from the signed square root.
   Small, but free to match exactly — and `ttnn.clamp` does have a float overload
   (only its tensor overload shows in the first line of the error).

All 66 referenced `ttnn.*` names were checked to exist, and the overload lists for
`slice`, `ge`, `clamp`, `maximum` and `repeat` were read rather than assumed —
`slice` and `ge` each have both a tensor and a sequence/scalar form, so the
tuple-based calls are fine.

---

## Observation 4 — the MoE never touches the host

It runs 48 times per token, so a host round-trip inside it would dominate.

- `nnz=None` lets `sparse_matmul` count non-zeros on device. An explicit count
  would require reading the top-k indices back, and a wrong count **hangs the
  device** rather than erroring.
- The routing mask is `probs >= kth_largest`, not a scatter of indices, so no
  index tensor is ever materialised. Verified equal to a `topk` reference
  including renormalisation.
- The sparsity union is `max` over the token axis; the per-expert gate is a
  `permute` of the same weights to `[1, E, M, 1]`.

---

## Observation 5 — batching is the optimisation that does not need tracing

Decode here is dispatch-bound, not FLOP-bound: a single MoE layer measures
1.79 ms for one token and 3.21 ms for eight, and an all-reduce costs 0.23 ms on
an 8 KB payload. Two levers follow.

**Batching** is implemented: every activation carries a batch axis in the `M`
position, `sdpa_decode`/`paged_update_cache` take per-sequence positions, the
DeltaNet state is `[B·48, 1, 128, 128]`, and the MoE runs with `M = B`. Note the
MoE's broadcast formulation is exactly optimal at `M = 1` (it computes precisely
the 10 selected experts) and wastes work as `M` grows, because the union of
experts across the batch approaches 512; at these sizes the matmuls are small
enough that dispatch still dominates, so the trade is favourable — but a token
permutation becomes worthwhile for large prefill chunks.

**Trace capture** (`begin/end_trace_capture`, `execute_trace`) would remove the
dispatch cost outright, and the static-shape design above is a precondition for
it. Per-step inputs already route through `TTModel._input`, which writes into
persistent bound buffers when `self.bound` is set. What is *not* yet done: every
state update must also be in place. Today `decode_step` returns a new state
tensor and the conv/PLE windows are rebuilt by `slice`, so each replay would
allocate — the remaining work is to pass `output_tensor=` through those ops. It is
deliberately left until the model is validated, rather than debugging numerics
and trace semantics at the same time.

---

## Files

```
tt/model.py       48-layer forward, batched decode, static-shape attention
tt/moe.py         on-device routing + sparse_matmul expert FFN + shared expert
tt/linear_attn.py DeltaNet decode recurrence
tt/deltanet.py    chunked-prefill preparation for gated_delta_attn_seq
tt/ops.py         grouped RMS norm, hyper-connection gate, re-inject
tt/weights.py     .tensorbin loader (replicate vs shard per manifest)
tt/convert.py     GGUF -> .tensorbin, device layouts, shard dims
tt/blockfloat.py  bit-exact bfp8/bfp4 quantiser
tt/engine.py      async engine: one device thread, interleaved sequences
tt/bench.py       per-section and end-to-end timing
tt/plan.py        residency + precision policy
```

## Next
Validate against the CPU reference (auto-runs when conversion completes), then
benchmark and optimise from the profile.
