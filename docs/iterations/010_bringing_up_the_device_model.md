# 010 — Bringing up the device model

Date: 2026-09-01

---

## The weight cache

93.22 GB written across 1222 device tensors; the two gather tensors
(`per_layer_token_embd`, `token_embd`) are correctly absent, staying in the GGUF
mmap.

```
manifest: 1222 tensors, total 93.22 GB, per-device 26.65 GB
device tensors needed: 1222, missing: 0
host-resident (correctly skipped): per_layer_token_embd.weight, token_embd.weight
```

**26.65 GB per device against the 26.34 GB the plan predicted** — the residency
model in `tt/plan.py` was accurate to ~1 %, so the budget arithmetic in
iterations 005/008 holds against the real artefact.

---

## Observation 1 — the converter was doing the quantisation twice

The first run held ~18.6 MB/s and projected ~85 minutes. Profiling by arithmetic
rather than by guess: at ~33 s per expert tensor of 838 M elements, the numpy
block-float pass (39 Melem/s ≈ 21 s) was the largest single term.

And it was **redundant**. The converter called
`blockfloat.round_trip(t, "bfloat4_b")` and then
`ttnn.from_torch(t, dtype=bfloat4_b)`, which quantises to the same grid. The
numpy quantiser exists to *simulate* device precision inside the CPU reference
(iteration 008) — it was never needed to feed the converter. Since `round_trip`
was verified elementwise identical to ttnn's own encoder, pre-rounding changed
nothing except runtime.

**Result** — 18.6 → ~25 MB/s, and because the converter skips tensors already in
the manifest, restarting resumed from 40 GB rather than starting over.

---

## Observation 2 — the run finished and then crashed on its own summary

The converter reached 1220/1224 and then died with:

```
AttributeError: 'ConvertStats' object has no attribute 'device_bytes'
```

`ConvertStats.device_bytes` had been renamed `bytes_written` in the rewrite, and
only the *final summary print* still used the old name. So all 93 GB were
written correctly, the exit status said failure, and the watcher — which keyed
off a `DONE` line — concluded the conversion had failed and skipped validation.

Worth naming as a class of bug: a reporting statement that runs exactly once, at
the end, is invisible to every earlier iteration and turns a completed job into
an apparent failure. Verifying against the artefact (manifest vs the GGUF tensor
list) rather than the exit code is what settled it.

---

## Observation 3 — a refactor broke my own test harness

Generalising the model to batched decode moved token histories from
`LayerState.ple_tokens` (per layer) to `TTState.histories` (per sequence), which
is the correct home: only the PLE layer reads them and they are a property of
the sequence.

The validation harness, written before that change, still did
`for st in tt_state.layers: st.ple_tokens.append(tok)`. It would have failed at
the first PLE layer with an `AttributeError` after loading 26 GB of weights.

**Remedy** — caught by re-reading the harness against the current API rather
than trusting it, then locked with `tests/test_tt_state.py`, which asserts among
other things that `LayerState` does *not* carry a `ple_tokens` field.

---

## Observation 4 — the shexp gate needed re-converting

`ffn_gate_inp_shexp` is 1-D but is a linear weight (`2560 -> 1`), so it needs a
column layout (iteration 009). The fix landed mid-conversion, so 48 tensors had
already been written as rows. They were deleted, dropped from the manifest, and
re-converted — 48 tensors, manifest back to 1222.

Note the byte count is identical either way (`328048` bytes: a 2560-long vector
tile-padded to 32 in the other dimension), so file size could not have revealed
the error. Only the semantics differ.

---

## Observation 5 — the DeltaNet half worked first time; attention took five contracts

The first validation run got through the embedding and three linear-attention
layers immediately:

```
stage               max_rel    rms_rel
embed               0.00146    0.00147
L00.lin             0.01218    0.01446
L01.lin             0.01282    0.02094      <- includes the PLE n-gram layer
L02.lin             0.00858    0.01966
```

So the DeltaNet decode recurrence, the hyper-connection read gate and re-inject
(including the 3-op reformulation), the MoE with on-device routing, and the PLE
n-gram gather were all correct on hardware without a single numerical fix.

Layer 3 — the first full-attention layer — then surfaced five successive
requirements of the K/V-cache ops. **None is visible in the Python signature**;
each came from reading the device op's `validate()` in the bundled tt-metal
source:

| # | requirement | what the code had |
|---|---|---|
| 1 | `paged_update_cache` input is `[1, batch, n_kv, hd]` (asserts `dim0 == 1`, `dim1 == cache.shape[0]`) | reshaped to `[batch, n_kv, 1, hd]` |
| 2 | index tensor must be `DataType::INT32` | `uint32` |
| 3 | `paged_update_cache` **unconditionally** requires L1 height-sharded input | interleaved |
| 4 | plain `update_cache` takes interleaved but wants `dim1 == cache.shape[1]` — the *head* count, not the batch — and a matching dtype | `[1, batch, n_kv, hd]`, dtype promoted to f32 by the f32 rope factors |
| 5 | `sdpa_decode`'s default `k_chunk_size` overflows L1 | default |

(5) is worth quoting because the message is precise and the cause is not:

```
Statically allocated circular buffers grow to 1917696 B
which is beyond max L1 size of 1572864 B
```

24 query heads at head_dim 256 simply do not fit at the default chunk. Fixed by
an explicit `SDPAProgramConfig(k_chunk_size=128, grid 8x8)`, now a tunable model
attribute so the benchmark sweep can own it.

**Consequence of (3):** a captured trace needs the position as a *tensor*, since
an int is baked into the recorded program. Tracing the 12 attention layers is
therefore gated on sharding k/v into L1 first; the 36 DeltaNet layers have no
such constraint. The interleaved int path was taken for correctness and the
dependency recorded, rather than debugging numerics and trace semantics at once.

With all five resolved, layer 3 validates:

```
hc mixed    max=0.01580 rms=0.00950
hc inject   max=0.00076 rms=0.00073
branch      max=0.02176 rms=0.01829     <- QSA attention
reinject+hc max=0.02291 rms=0.02587
moe total   max=0.09424 rms=0.10709
layer out   max=0.02354 rms=0.02471
```

---

## Observation 6 — the MoE's 10.7 % is not what it looked like

Both sides use bit-identical bfp4 expert weights (the reference runs `quant_sim`),
so 10.7 % is far too large for arithmetic precision. The obvious suspect was a
top-k boundary swap: the device feeds the f32 router bf16 activations, so a logit
near the cut could reorder, and losing one of ten experts would shift the output
by roughly a tenth — suspiciously close to what was measured.

**That hypothesis was wrong.** Checking directly rather than accepting the
coincidence:

```
ref experts: [133, 137, 221, 283, 314, 336, 385, 400, 489, 495]
dev experts: [133, 137, 221, 283, 314, 336, 385, 400, 489, 495]
identical expert set: True
```

The routing is identical, weights included. The real chain is amplification:

```
hc gate            0.95 %
attention branch   1.83 %
MoE input          2.59 %
MoE output        10.71 %      <- silu(gate) * up is a product of two perturbed terms
layer output       2.47 %      <- MoE is scaled by the injection weights
```

This router is unusually flat — the top-10 probabilities are all ≈0.004 out of
512 experts, with a boundary gap of 2.6e-4 — so a few-percent input shift moves
the normalised weights measurably even when the *selection* is stable.

The question this leaves is not whether 2.5 % per layer is acceptable in
isolation, but whether it compounds across 48 layers or stays bounded. Only the
full walk answers that.
