"""Per-component and end-to-end timing for the device model.

Decode on this stack is expected to be *dispatch*-bound rather than FLOP-bound:
a single MoE layer measured 1.79 ms for one token and only 3.21 ms for eight, and
one all-reduce costs 0.23 ms on an 8 KB payload. This harness attributes the step
time so optimisation targets the part that actually dominates.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field


@dataclass
class Timing:
    name: str
    samples: list[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        self.samples.append(seconds)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples) * 1e3 if self.samples else float("nan")

    @property
    def min_ms(self) -> float:
        return min(self.samples) * 1e3 if self.samples else float("nan")


class Profiler:
    """Times device sections, synchronising so the numbers mean something."""

    def __init__(self, mesh, enabled: bool = True):
        self.mesh = mesh
        self.enabled = enabled
        self.sections: dict[str, Timing] = {}

    def section(self, name: str):
        return _Section(self, name)

    def record(self, name: str, seconds: float) -> None:
        self.sections.setdefault(name, Timing(name)).add(seconds)

    def report(self, total_name: str = "step") -> str:
        rows = sorted(self.sections.values(), key=lambda t: -t.median_ms)
        total = self.sections.get(total_name)
        lines = [f"{'section':28s} {'median ms':>10s} {'min ms':>9s} {'share':>7s} {'n':>5s}"]
        for t in rows:
            share = f"{100 * t.median_ms / total.median_ms:6.1f}%" if total and total.median_ms else "     -"
            lines.append(f"{t.name:28s} {t.median_ms:10.3f} {t.min_ms:9.3f} {share:>7s} {len(t.samples):5d}")
        return "\n".join(lines)


class _Section:
    __slots__ = ("prof", "name", "t0")

    def __init__(self, prof: Profiler, name: str):
        self.prof = prof
        self.name = name

    def __enter__(self):
        if self.prof.enabled:
            import ttnn

            ttnn.synchronize_device(self.prof.mesh)
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.prof.enabled:
            import ttnn

            ttnn.synchronize_device(self.prof.mesh)
            self.prof.record(self.name, time.perf_counter() - self.t0)
        return False


def benchmark_step(model, state, token_id: int, iters: int = 25, warmup: int = 5) -> dict[str, float]:
    """End-to-end decode step timing (median over `iters`).

    Defaults are deliberately generous: with 2 warmup steps and 3 samples this
    reported 107 tok/s where a 5-warmup, 25-sample run measured 86 -- a 26%
    optimistic error from JIT'd kernels and allocator warm-up landing inside the
    measured window.
    """
    import ttnn

    for _ in range(warmup):
        model.step(token_id, state)
    ttnn.synchronize_device(model.mesh)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        hidden = model.step(token_id, state)
        ttnn.synchronize_device(model.mesh)
        samples.append(time.perf_counter() - t0)
    median = statistics.median(samples)
    return {
        "median_ms": median * 1e3,
        "min_ms": min(samples) * 1e3,
        "tokens_per_s": 1.0 / median,
    }
