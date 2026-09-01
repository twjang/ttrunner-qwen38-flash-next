"""Fusing the experts' gate and up projections into one sparse_matmul.

The MoE's cost is a per-call floor rather than the work it discards: the block
costs 4.497 ms at M=1, where the expert union is exactly top_k and nothing is
wasted, against 13.4 ms at M=64 where 23x is wasted. That is why chunking M was
monotonically worse, and why the lever is issuing fewer sparse_matmuls.

The trap is how the fused tensor is built. `ttnn.concat` on two bfloat4_b
tensors requantises them (0.0547 max error) and the generated tokens change --
measured, diverging at the second token. Dequantising to float and quantising
the concatenation once is exact (0.0), because the per-device width 160 is a
whole number of 16-element blocks so the shared exponents align.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("ttnn")

from twtest.tt import moe  # noqa: E402
from twtest.tt.model import TTModel  # noqa: E402
from twtest.tt.weights import TTWeights  # noqa: E402


def test_fused_path_does_not_concat_quantised_weights_on_device() -> None:
    src = inspect.getsource(TTWeights.fused_gate_up)
    assert "ttnn.concat" not in src, (
        "fused_gate_up must load the prebuilt tensor: concatenating the quantised "
        "halves on device requantises bfloat4_b and changes the output"
    )
    assert "fuse_expert_gate_up.py" in src, "the error should name the builder script"


def test_expert_ffn_accepts_a_fused_weight() -> None:
    src = inspect.getsource(moe.expert_ffn)
    assert "if up_w is None:" in src, "expert_ffn lost its fused branch"
    # the fused branch must issue one sparse_matmul where the split path issues two
    fused_branch = src.split("if up_w is None:")[1].split("    else:")[0]
    # count calls, not the word in comments: one for gate|up where the split path
    # issues two, plus the shared down projection outside the branch
    assert fused_branch.count("ttnn.sparse_matmul(") == 1, "fused branch must issue one call"
    assert src.count("ttnn.sparse_matmul(") == 4, "expected fused(1) + split(2) + down(1)"


def test_fusion_is_opt_in_on_the_model() -> None:
    """It changes nothing numerically, but it needs weights that may not exist."""
    src = inspect.getsource(TTModel.__init__)
    assert "self.fuse_expert_gate_up" in src
