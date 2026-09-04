"""Do the dev harnesses still match the code they drive? Static, no device.

    uv run python scripts/dev/harness_api_check.py

Two harnesses cited as evidence in `docs/HANDOFF.md` had silently stopped
working: `indexer_select_check.py` called `_indexer_select` with the pre-`q_cos`
signature and raised `TypeError` after minutes of setup, and
`traced_step_n_check.py` hangs. A citation that no longer runs still reads as
evidence, which is worse than no citation, so this catches the statically
detectable half -- names and arities -- with no device and no model load.

`tests/test_harness_api.py` runs it, so drift fails the suite rather than
waiting to be discovered by someone quoting the number.
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import sys


def _targets():
    from ttrunner_qwen38_flash_next.tt.engine import TTEngine
    from ttrunner_qwen38_flash_next.tt.model import TTModel

    return {"TTModel": TTModel, "TTEngine": TTEngine}, TTModel, TTEngine


def _arity(fn):
    """(required positional excluding self, max positional or None for *args)."""
    sig = inspect.signature(fn)
    pos = [p for p in sig.parameters.values()
           if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name != "self"]
    star = any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values())
    return sum(1 for p in pos if p.default is p.empty), None if star else len(pos)


def _bindings(tree, ttmodel, ttengine):
    """Names bound to a TTModel/TTEngine in this file.

    Resolved rather than guessed: several harnesses call `model.forward(...)` on
    the float32 *reference*, which has no business being checked against
    `TTModel`.
    """
    out: dict[str, type] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        fn = node.value.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        cls = None
        if name == "open_model":            # (mesh, cfg, model)
            cls = ttmodel
        elif name == "TTModel":
            cls = ttmodel
        elif name == "TTEngine":
            cls = ttengine
        if cls is None:
            continue
        for target in node.targets:
            names = target.elts if isinstance(target, ast.Tuple) else [target]
            if name == "open_model" and len(names) == 3:
                names = names[-1:]          # only the model of (mesh, cfg, model)
            for n in names:
                if isinstance(n, ast.Name):
                    out[n.id] = cls
    return out


def check(root: pathlib.Path = pathlib.Path("scripts/dev")) -> list[str]:
    classes, ttmodel, ttengine = _targets()
    problems: list[str] = []
    for path in sorted(root.glob("*.py")):
        if path.name == "harness_api_check.py":
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:
            problems.append(f"{path.name}: does not parse -- {exc}")
            continue
        bound = _bindings(tree, ttmodel, ttengine)
        bound.update(classes)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            owner = node.func.value
            if not (isinstance(owner, ast.Name) and owner.id in bound):
                continue
            cls, attr = bound[owner.id], node.func.attr
            fn = getattr(cls, attr, None)
            if fn is None or not callable(fn):
                problems.append(
                    f"{path.name}:{node.lineno}  {owner.id}.{attr} does not exist on "
                    f"{cls.__name__}")
                continue
            if any(isinstance(a, ast.Starred) for a in node.args):
                continue
            lo, hi = _arity(fn)
            n_pos = len(node.args)
            n_kw = len({k.arg for k in node.keywords if k.arg})
            if n_pos + n_kw < lo:
                problems.append(
                    f"{path.name}:{node.lineno}  {owner.id}.{attr}(...) passes {n_pos} "
                    f"positional + {n_kw} keyword, needs {lo}")
            elif hi is not None and n_pos > hi:
                problems.append(
                    f"{path.name}:{node.lineno}  {owner.id}.{attr}(...) passes {n_pos} "
                    f"positional, takes at most {hi}")
    return problems


def main() -> int:
    problems = check()
    for p in problems:
        print(f"RESULT {p}", flush=True)
    n = len(list(pathlib.Path("scripts/dev").glob("*.py")))
    print(f"RESULT {n} harnesses, {len(problems)} problem(s)", flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.path.insert(0, "src")
    raise SystemExit(main())
