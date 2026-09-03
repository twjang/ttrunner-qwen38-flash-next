"""The dev harnesses must still match the code they drive.

`docs/HANDOFF.md` cites harnesses as evidence for numbers and for correctness
claims, and two of them had silently stopped working: `indexer_select_check.py`
called `_indexer_select` with its pre-`q_cos` signature and raised `TypeError`
after minutes of setup, while `traced_step_n_check.py` hangs on the device. A
citation that no longer runs still reads as evidence, which is worse than no
citation at all.

The device half of that cannot be caught here. The API half can, and this is
where it gets caught -- at `pytest -q`, in milliseconds, rather than by someone
quoting a number a year from now.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV = ROOT / "scripts" / "dev"


@pytest.mark.skipif(not DEV.is_dir(), reason="dev harnesses not present")
def test_dev_harnesses_match_the_model_and_engine_api() -> None:
    sys.path.insert(0, str(DEV))
    try:
        from harness_api_check import check
    finally:
        sys.path.remove(str(DEV))
    problems = check(DEV)
    assert not problems, "harnesses have drifted from the API:\n  " + "\n  ".join(problems)
