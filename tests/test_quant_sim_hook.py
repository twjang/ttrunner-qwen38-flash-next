"""`quant_sim` has to reach every path a weight can arrive by.

The hook exists so the CPU reference can reproduce the device's block-float
numerics (`docs/iterations/008` chose the precision policy with it). It was
applied on the bulk paths but not on the per-row one -- and per-row is how the
MoE fetches experts, one row per selected expert, which are the only bfloat4_b
tensors in the model. A simulation run through that hole left the most
aggressively quantised weights exact and reported the policy as cheaper than it
is.
"""

from __future__ import annotations

import inspect

from ttrunner_qwen38_flash_next.reference.weights import WeightStore


def test_every_dequantisation_path_applies_quant_sim() -> None:
    for fn in (WeightStore.get, WeightStore.get_rows, WeightStore.get_row_range):
        src = inspect.getsource(fn)
        assert "self.quant_sim(name, flat)" in src, (
            f"{fn.__name__} returns weights without passing them through quant_sim, "
            "so a simulation of device precision silently skips them"
        )


def test_the_row_path_is_covered_not_just_the_bulk_one() -> None:
    """`get_rows` has two branches -- a bulk dequantise above
    BULK_ROW_THRESHOLD and a per-row loop below it. Both return weights."""
    src = inspect.getsource(WeightStore.get_rows)
    assert src.count("self.quant_sim(name, flat)") >= 2, (
        "get_rows has a bulk branch and a per-row branch; both need the hook"
    )
