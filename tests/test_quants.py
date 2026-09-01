"""Dequantisation must stay bit-exact against the reference `gguf` package."""

from __future__ import annotations

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
from gguf.quants import dequantize as gguf_dequantize  # noqa: E402

from twtest.gguf.quants import SUPPORTED_TYPES, dequantize  # noqa: E402
from twtest.gguf.reader import TYPE_TRAITS, GGMLType  # noqa: E402

# Formats we can synthesise random blocks for (all of them: the bytes are
# opaque to the dequantiser, so random bytes exercise every code path).
ROUND_TRIP = sorted(SUPPORTED_TYPES, key=lambda t: t.name)


@pytest.mark.parametrize("ggml_type", ROUND_TRIP, ids=lambda t: t.name)
def test_matches_gguf_reference(ggml_type: GGMLType) -> None:
    block_size, type_size = TYPE_TRAITS[ggml_type]
    n_blocks = 64
    rng = np.random.default_rng(1234 + int(ggml_type))
    raw = rng.integers(0, 256, size=n_blocks * type_size, dtype=np.uint8)

    if ggml_type in (GGMLType.F32, GGMLType.F16, GGMLType.BF16):
        # random bytes can be NaN/Inf for float types; use real values instead
        values = rng.standard_normal(n_blocks * block_size).astype(np.float32)
        dtype = {GGMLType.F32: np.float32, GGMLType.F16: np.float16}.get(ggml_type)
        if dtype is not None:
            raw = values.astype(dtype).view(np.uint8)
        else:  # bf16
            raw = (values.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)

    n = n_blocks * block_size
    mine = dequantize(memoryview(raw.tobytes()), ggml_type, n)
    theirs = gguf_dequantize(
        np.frombuffer(raw.tobytes(), dtype=np.uint8), gguf.GGMLQuantizationType(int(ggml_type))
    ).reshape(-1)[:n]

    finite = np.isfinite(theirs)
    assert np.array_equal(mine[finite], theirs[finite]), f"{ggml_type.name} diverges from gguf"


def test_unsupported_type_raises() -> None:
    with pytest.raises(NotImplementedError, match="IQ1_S"):
        dequantize(memoryview(b"\x00" * 50), GGMLType.IQ1_S, 256)
