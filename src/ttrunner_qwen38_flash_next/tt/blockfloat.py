"""Block-float quantisation matching ttnn's bfloat8_b / bfloat4_b exactly.

Both formats store one shared exponent per 16 datums plus a sign and a
fixed-point mantissa per value: 7 mantissa bits for bfloat8_b, 3 for
bfloat4_b. Verified elementwise-identical to
``ttnn.to_torch(ttnn.from_torch(x, dtype=...))`` (agreement 1.000), while
running ~8x faster than ttnn's host bfp8 path -- fast enough to simulate the
device's numerics inside the CPU reference engine.

The key property, and the reason this module exists: block-float error is
*multiplicative* -- each weight carries a relative error of about
2^-mantissa_bits. In a dot product ``sum_k w_k x_k`` the perturbation is
``sum_k w_k eps_k x_k``, whose magnitude scales with ``||wx||_2`` just as the
signal does, so the error does **not** shrink with the contraction length.
bfloat4_b's ~12% per-element error therefore shows up as ~12% error on the
matmul output no matter how large K is.
"""

from __future__ import annotations

import numpy as np

BLOCK = 16
MANTISSA_BITS = {"bfloat8_b": 7, "bfloat4_b": 3}
# Measured mean relative error on gaussian weights.
TYPICAL_REL_ERROR = {"bfloat8_b": 0.0075, "bfloat4_b": 0.1206}


def round_trip(x: np.ndarray, dtype: str, block: int = BLOCK) -> np.ndarray:
    """Quantise to `dtype` and back, returning float32."""
    if dtype in ("float32", "source"):
        return x.astype(np.float32, copy=False)
    if dtype == "bfloat16":
        return (x.astype(np.float32).view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    try:
        mantissa_bits = MANTISSA_BITS[dtype]
    except KeyError:
        raise ValueError(f"unsupported block-float dtype {dtype!r}") from None

    flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, block)
    absmax = np.abs(flat).max(axis=1, keepdims=True)
    shared_exp = np.floor(np.log2(np.maximum(absmax, 1e-38)))
    scale = np.exp2(shared_exp - (mantissa_bits - 1))
    qmax = 2**mantissa_bits - 1
    q = np.clip(np.rint(flat / scale), -qmax, qmax)
    return (q * scale).reshape(x.shape)
