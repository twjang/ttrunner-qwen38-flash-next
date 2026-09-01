# 004 — Validating the reference engine against llama.cpp

Date: 2026-09-01

## Goal
Prove deliverable (2) correct, rather than merely plausible.

---

## Observation 1 — the first run produced fluent-looking garbage

`"The capital of France is"` → top token `' ~>'` at logit 3.76. Note the shape
of the failure: no crash, no NaN, no shape error, and a *low, flat* logit
distribution. Every tensor had the right shape; the model was simply wrong.

There is no way to debug that by reading code. What is needed is a ground truth
for the intermediate values.

**Remedy** — `llama-eval-callback` runs the same GGUF through llama.cpp and
prints every graph node with its name, shape and sum. 7 950 tensors, of which
4 294 carry meaningful names (`hc_norm-0`, `q_conv-0`, `gate-0`,
`ffn_moe_logits-0`, ...). That is an op-by-op oracle for the exact checkpoint.

Comparing sums of the same quantities from my engine localised the first
divergence immediately.

---

## Observation 2 — `ssm_a` is not `A_log`

```
alpha-0        got=      -54.39925 exp=      -54.64651  OK
a_softplus-0   got=      130.82144 exp=      130.18947  OK
gate-0         got=      -69.96994 exp=    -1816.61755  **26x off**
```

Both inputs matched, the output was off by 26×, so the fault was the constant
between them. Upstream computes `g = -exp(A_log) * softplus(a + dt_bias)`. But
the stored tensor ranges over −158..−0.028 — already negative, so it cannot be
a log. The converter:

```python
if name.endswith(".A_log"):
    data_torch = -torch.exp(data_torch)
```

The GGUF stores `A = -exp(A_log)` directly. The correct expression is
`g = ssm_a * softplus(a + dt_bias)`, giving −1819.85 against llama.cpp's
−1816.62.

---

## Observation 3 — the DeltaNet output gate is sigmoid, not SiLU

```
attn_output-0    got=     7.34023 exp=     7.28363  OK
z-0              got= -17201.58203 exp= -16956.03125  OK
final_output-0   got=  -321.69165 exp=   -36.13839  **8x off**
```

Again both inputs fine, the combining step wrong. `RMSNormGated` applies
`ACT2FN[config.output_gate_type or config.hidden_act]`, and this model sets
`"output_gate_type": "sigmoid"` — overriding the `silu` that `hidden_act`
implies.

The trap: **the converter never writes `output_gate_type` to the GGUF.** It
cannot be recovered from the checkpoint, so llama.cpp hardcodes it and any
reimplementation that reads only the GGUF will silently inherit SiLU. It is now
pinned in `Qwen4ExpConfig` with a comment saying why it cannot be read back.

With both fixed: `' Paris'` at logit **16.80** — confident and correct.

### On tolerances
Post-matmul quantities differ by 0.5–1.5 % even when correct, because llama.cpp
quantises *activations* to Q8_0 and does integer dot products while this engine
dequantises to f32 and uses f32 matmul. Only order-of-magnitude divergences are
bugs; the comparison threshold has to allow for this or every matmul looks broken.

---

## Observation 4 — prefill was right and decode was broken

Greedy generation gave `' Paris'` then token 0 (`'!'`) forever. Prefill correct,
incremental decode not.

**Remedy** — a self-consistency test that needs no external oracle: prefilling
`[p0..p5]` in one pass must produce the same last-position state as prefilling
`[p0..p4]` and then decoding `p5`. Per-layer, the decode path went `nan` at
layer 0's DeltaNet.

Isolating that one layer:

```
prefill out finite: True
recurrent_state: finite=False  nan=16384  (= 1 head x 128 x 128)
```

The chunk **output** was finite and matched llama.cpp; only the carried
**state** was poisoned, in exactly one of 48 heads. So prefill-only testing
could never have caught it.

**Cause** — layer 0 has `A` as low as −158, so within a 64-token chunk
`cum_decay` reaches ≈ −10⁴ and `exp(cum_decay)` underflows to exactly 0. I formed
the position-to-chunk-end decay as a ratio of exponentials:

```python
chunk_decay[..., -1, None] / chunk_decay        # 0/0 -> NaN
```

Upstream does the subtraction in log space, which is bounded above by 1 and
cannot underflow to 0/0:

```python
key = key * (cum_decay[..., -1:] - cum_decay).exp()
```

**Result** — rewritten to hoist both decays out of the loop in log space. With
`decay_scale=200` the chunked and recurrent forms agree to 3.7e-06 and the state
stays finite.

---

## Result: exact agreement with llama.cpp

Greedy, 14 tokens, same GGUF:

```
MINE     : 'The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is'
LLAMA.CPP: 'The capital of France is Paris. The capital of Germany is Berlin. The capital of'
```

Token for token identical over llama.cpp's full 12-token output, then
continuing sensibly. **Deliverable (2) is validated.**

Speed: ~105 s prefill (5 tokens), ~12 s/token decode, 40 threads. Slow by
design — every selected expert is dequantised from IQ3_S on the fly.

---

## Observation 5 — a row cache makes decode usable

Decode re-selects largely the same experts step after step, but `get_rows` was
re-dequantising them every time. Added an LRU keyed by `(tensor, row)`.

**Result** — 11 115 hits / 13 910 misses over a 14-token generation, decode
28 s → ~12 s per token.

---

## Regression tests

`tests/` — 21 tests, all passing:

- `test_quants.py` — every format dequantises bit-exactly against `gguf`.
- `test_delta_rule.py` — chunked vs recurrent agreement across
  `seq ∈ {1,7,64,100}` × `decay ∈ {0.5, 160}`; an explicit guard that extreme
  decay leaves the carried state finite; and prefill-then-decode equals full
  prefill.

## Next
Iteration 005: the ttnn engine. Initial hardware survey is in that iteration —
notably `ttnn.transformer.gated_delta_attn_seq` already exists.
