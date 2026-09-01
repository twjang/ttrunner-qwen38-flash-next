"""Vectorised numpy dequantisation for the GGML block formats.

Only the formats present in the UD-IQ4_XS checkpoint are implemented:
F32, F16, BF16, Q8_0, Q4_K, Q6_K, IQ4_NL, IQ4_XS and IQ3_S. Each routine takes
the raw little-endian block bytes and returns a flat float32 array; the block
layouts follow ggml-common.h and the arithmetic follows ggml-quants.c exactly.
"""

from __future__ import annotations

import numpy as np

from ._codebooks import IQ3S_GRID, KVALUES_IQ4NL
from .reader import TYPE_TRAITS, GGMLType

# kmask_iq2xs: one sign bit per value within a group of eight.
_SIGN_BITS = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)


def _blocks(raw: memoryview, ggml_type: GGMLType, n_elements: int) -> np.ndarray:
    """Raw bytes -> (n_blocks, type_size) uint8 view."""
    block_size, type_size = TYPE_TRAITS[ggml_type]
    n_blocks = n_elements // block_size
    arr = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * type_size)
    return arr.reshape(n_blocks, type_size)


def _f16(b: np.ndarray) -> np.ndarray:
    """Two uint8 columns -> float32 scalar per block."""
    return b.copy().view(np.float16).astype(np.float32)


def _dequant_q8_0(raw: memoryview, n: int) -> np.ndarray:
    b = _blocks(raw, GGMLType.Q8_0, n)
    d = _f16(b[:, 0:2])  # (nb, 1)
    qs = b[:, 2:34].view(np.int8).astype(np.float32)
    return (d * qs).reshape(-1)


def _dequant_q6_K(raw: memoryview, n: int) -> np.ndarray:
    b = _blocks(raw, GGMLType.Q6_K, n)
    nb = b.shape[0]
    ql_all = b[:, 0:128]
    qh_all = b[:, 128:192]
    sc_all = b[:, 192:208].view(np.int8).astype(np.float32)
    d = _f16(b[:, 208:210])

    out = np.empty((nb, 256), dtype=np.float32)
    for half in (0, 1):
        ql = ql_all[:, half * 64 : half * 64 + 64].astype(np.int16)
        qh = qh_all[:, half * 32 : half * 32 + 32].astype(np.int16)
        sc = sc_all[:, half * 8 : half * 8 + 8]
        base = half * 128
        # is = l // 16 over the 32-wide l axis
        is_idx = np.arange(32) // 16  # (32,)
        lo, hi = ql[:, 0:32], ql[:, 32:64]
        q = [
            (lo & 0xF) | ((qh >> 0 & 3) << 4),
            (hi & 0xF) | ((qh >> 2 & 3) << 4),
            (lo >> 4) | ((qh >> 4 & 3) << 4),
            (hi >> 4) | ((qh >> 6 & 3) << 4),
        ]
        for k in range(4):
            scale = sc[:, is_idx + 2 * k]  # (nb, 32)
            out[:, base + 32 * k : base + 32 * k + 32] = d * scale * (q[k] - 32).astype(np.float32)
    return out.reshape(-1)


def _q4k_scale_min(scales: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray]:
    """get_scale_min_k4 from ggml-quants.c, vectorised over blocks."""
    q = scales.astype(np.uint16)
    if j < 4:
        return (q[:, j] & 63), (q[:, j + 4] & 63)
    d = (q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4)
    m = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
    return d, m


def _dequant_q4_K(raw: memoryview, n: int) -> np.ndarray:
    b = _blocks(raw, GGMLType.Q4_K, n)
    nb = b.shape[0]
    d = _f16(b[:, 0:2])
    dmin = _f16(b[:, 2:4])
    scales = b[:, 4:16]
    qs = b[:, 16:144]

    out = np.empty((nb, 256), dtype=np.float32)
    for i in range(4):
        q = qs[:, i * 32 : (i + 1) * 32].astype(np.int16)
        sc, m = _q4k_scale_min(scales, 2 * i)
        d1 = d * sc[:, None].astype(np.float32)
        m1 = dmin * m[:, None].astype(np.float32)
        sc, m = _q4k_scale_min(scales, 2 * i + 1)
        d2 = d * sc[:, None].astype(np.float32)
        m2 = dmin * m[:, None].astype(np.float32)
        out[:, i * 64 : i * 64 + 32] = d1 * (q & 0xF).astype(np.float32) - m1
        out[:, i * 64 + 32 : i * 64 + 64] = d2 * (q >> 4).astype(np.float32) - m2
    return out.reshape(-1)


def _dequant_iq4_nl(raw: memoryview, n: int) -> np.ndarray:
    b = _blocks(raw, GGMLType.IQ4_NL, n)
    nb = b.shape[0]
    d = _f16(b[:, 0:2])
    qs = b[:, 2:18]
    out = np.empty((nb, 32), dtype=np.float32)
    out[:, 0:16] = KVALUES_IQ4NL[qs & 0xF]
    out[:, 16:32] = KVALUES_IQ4NL[qs >> 4]
    return (out * d).reshape(-1)


def _dequant_iq4_xs(raw: memoryview, n: int) -> np.ndarray:
    b = _blocks(raw, GGMLType.IQ4_XS, n)
    nb = b.shape[0]
    d = _f16(b[:, 0:2])
    scales_h = b[:, 2:4].copy().view(np.uint16).astype(np.uint32)  # (nb, 1)
    scales_l = b[:, 4:8].astype(np.uint32)
    qs = b[:, 8:136]

    out = np.empty((nb, 256), dtype=np.float32)
    for ib in range(8):
        ls = ((scales_l[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (
            ((scales_h[:, 0] >> (2 * ib)) & 3) << 4
        )
        dl = d[:, 0] * (ls.astype(np.float32) - 32.0)  # (nb,)
        q = qs[:, ib * 16 : (ib + 1) * 16]
        out[:, ib * 32 : ib * 32 + 16] = KVALUES_IQ4NL[q & 0xF] * dl[:, None]
        out[:, ib * 32 + 16 : ib * 32 + 32] = KVALUES_IQ4NL[q >> 4] * dl[:, None]
    return out.reshape(-1)


def _dequant_iq3_s(raw: memoryview, n: int) -> np.ndarray:
    b = _blocks(raw, GGMLType.IQ3_S, n)
    nb = b.shape[0]
    d = _f16(b[:, 0:2])[:, 0]  # (nb,)
    qs = b[:, 2:66].astype(np.int32)
    qh = b[:, 66:74].astype(np.int32)
    signs = b[:, 74:106]
    scales = b[:, 106:110].astype(np.uint16)

    out = np.empty((nb, 256), dtype=np.float32)
    for ib32 in range(8):
        # Two 4-bit scales are packed per scales byte.
        sc = (scales[:, ib32 // 2] >> (4 * (ib32 % 2))) & 0xF
        db = d * (1.0 + 2.0 * sc.astype(np.float32))  # (nb,)

        q = qs[:, ib32 * 8 : ib32 * 8 + 8]  # (nb, 8)
        h = qh[:, ib32][:, None]  # (nb, 1)
        sg = signs[:, ib32 * 4 : ib32 * 4 + 4]  # (nb, 4)

        l = np.arange(4)
        # The 9th index bit for each of the eight grid lookups comes from qh.
        idx1 = q[:, 2 * l] | ((h << (8 - 2 * l)) & 256)
        idx2 = q[:, 2 * l + 1] | ((h << (7 - 2 * l)) & 256)
        g1 = IQ3S_GRID[idx1].astype(np.float32)  # (nb, 4, 4)
        g2 = IQ3S_GRID[idx2].astype(np.float32)

        s1 = np.where((sg[:, :, None] & _SIGN_BITS[0:4]) != 0, -1.0, 1.0)
        s2 = np.where((sg[:, :, None] & _SIGN_BITS[4:8]) != 0, -1.0, 1.0)

        chunk = np.empty((nb, 4, 8), dtype=np.float32)
        chunk[:, :, 0:4] = g1 * s1
        chunk[:, :, 4:8] = g2 * s2
        out[:, ib32 * 32 : ib32 * 32 + 32] = chunk.reshape(nb, 32) * db[:, None]
    return out.reshape(-1)


def _dequant_q5_0(raw: memoryview, n: int) -> np.ndarray:
    b = _blocks(raw, GGMLType.Q5_0, n)
    nb = b.shape[0]
    d = _f16(b[:, 0:2])
    qh = b[:, 2:6].copy().view(np.uint32)  # (nb, 1) -- the 5th bit of each value
    qs = b[:, 6:22].astype(np.int32)  # (nb, 16)

    j = np.arange(16)
    # low nibbles take bits 0..15 of qh, high nibbles take bits 16..31
    xh_lo = ((qh >> j) << 4) & 0x10
    xh_hi = (qh >> (j + 12)) & 0x10
    out = np.empty((nb, 32), dtype=np.float32)
    out[:, 0:16] = ((qs & 0x0F) | xh_lo) - 16
    out[:, 16:32] = ((qs >> 4) | xh_hi) - 16
    return (out * d).reshape(-1)


def _dequant_f32(raw: memoryview, n: int) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32, count=n).astype(np.float32, copy=True)


def _dequant_f16(raw: memoryview, n: int) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float16, count=n).astype(np.float32)


def _dequant_bf16(raw: memoryview, n: int) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16, count=n).astype(np.uint32)
    return (u16 << 16).view(np.float32)


_DEQUANT = {
    GGMLType.F32: _dequant_f32,
    GGMLType.F16: _dequant_f16,
    GGMLType.BF16: _dequant_bf16,
    GGMLType.Q8_0: _dequant_q8_0,
    GGMLType.Q5_0: _dequant_q5_0,
    GGMLType.Q4_K: _dequant_q4_K,
    GGMLType.Q6_K: _dequant_q6_K,
    GGMLType.IQ4_NL: _dequant_iq4_nl,
    GGMLType.IQ4_XS: _dequant_iq4_xs,
    GGMLType.IQ3_S: _dequant_iq3_s,
}

SUPPORTED_TYPES = frozenset(_DEQUANT)


def dequantize(raw: memoryview, ggml_type: GGMLType, n_elements: int) -> np.ndarray:
    """Dequantise `n_elements` values of `ggml_type` from `raw` to float32."""
    try:
        fn = _DEQUANT[ggml_type]
    except KeyError:
        raise NotImplementedError(
            f"{ggml_type.name} is not implemented; this build supports "
            f"{sorted(t.name for t in SUPPORTED_TYPES)}"
        ) from None
    out = fn(raw, n_elements)
    if out.shape[0] != n_elements:
        raise AssertionError(f"{ggml_type.name}: produced {out.shape[0]} of {n_elements} values")
    return out
