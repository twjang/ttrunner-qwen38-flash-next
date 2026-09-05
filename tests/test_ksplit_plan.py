"""The k-split's work plan must cover every output tile.

`ksplit_linear` floored its output-tile count -- `w.shape[-1] // 32` -- so a
weight narrower than one tile got `nt == 0`, its plan loop produced no work
items, and the untouched zero output buffer came back through `ttnn.sum` as
zeros. The guard did not catch it: `groups` is computed with `max(nt, 1)` and
stayed comfortably above the `groups >= 2` test, so the function returned a
tensor rather than None and the caller's fallback never ran.

That shipped, and every DeltaNet layer ran with `a = b = 0` until it was found.
These pin the arithmetic rather than the source line, so a rewrite has to keep
the property and not the spelling.
"""
from ttrunner_qwen38_flash_next.tt import ops


def _plan_size(k: int, n: int, n_cores: int = 110) -> tuple[int, int]:
    """(groups, work items) for a [k, n] weight, through the real tile count."""
    kt = k // 32
    nt = ops.output_tiles(n)
    groups = max(1, min(kt, n_cores // nt))
    return groups, groups * nt


def test_a_sub_tile_output_still_gets_work() -> None:
    """The shapes that broke it: fused ssm_alpha|ssm_beta is 24 wide, and the
    shared expert's sigmoid gate is 1."""
    for n in (1, 4, 12, 24, 31):
        groups, items = _plan_size(2560, n)
        assert items > 0, f"[2560, {n}] produced an empty plan"
        assert groups >= 2, f"[2560, {n}] should be worth splitting"


def test_a_ragged_output_covers_its_last_partial_tile() -> None:
    for n, want_nt in ((33, 2), (63, 2), (64, 2), (65, 3), (352, 11)):
        groups, items = _plan_size(2560, n)
        assert items == groups * want_nt, f"[2560, {n}] wants {want_nt} tiles"


def test_whole_tile_shapes_are_unchanged() -> None:
    """The router and the fused down|inject must keep the plan they had."""
    for n in (512, 352, 1280, 2560):
        assert _plan_size(2560, n) == (max(1, min(80, 110 // (n // 32))),
                                       max(1, min(80, 110 // (n // 32))) * (n // 32))


def test_output_tiles_rounds_up() -> None:
    """The property itself, on the function the split actually calls."""
    assert ops.output_tiles(1) == 1        # the shared expert's sigmoid gate
    assert ops.output_tiles(24) == 1       # fused ssm_alpha|ssm_beta
    assert ops.output_tiles(31) == 1
    assert ops.output_tiles(32) == 1
    assert ops.output_tiles(33) == 2
    assert ops.output_tiles(352) == 11     # the fused down|inject
    assert ops.output_tiles(512) == 16     # the router
    assert ops.output_tiles(0) == 1        # never zero, whatever it is asked
