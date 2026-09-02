"""ttnn primitives for qwen4exp, each mirroring a function in reference/layers.py.

Every op here is checked numerically against the CPU reference (see
tests/test_tt_ops.py), which is the oracle validated against llama.cpp in
iteration 004.

Two ttnn facts shape this file:

* ``ttnn.rms_norm`` normalises over the last dimension only and its ``weight``
  must match that dimension. The hyper-connection norm needs per-2560 groups of
  a 10240-wide tensor with a distinct 10240-wide gamma, so it is expressed as a
  weightless grouped normalise followed by a full-width multiply.
* ``ttnn.moe`` is not a MoE layer (it returns expert-zero routing weights). The
  expert FFN is built on ``ttnn.sparse_matmul``, which skips the experts a
  sparsity mask zeroes out.
"""

from __future__ import annotations

import ttnn

# Blackhole prefers fp32 accumulation for anything that feeds a norm or a
# softmax; the extra dest registers are cheap next to a silent precision loss.
HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


def grouped_rms_norm(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    eps: float,
    group_size: int,
    groups: int,
) -> ttnn.Tensor:
    """RMS-normalise each `group_size` slice independently, then scale by `weight`.

    x:      [1, B, S, groups*group_size]
    weight: [1, 1, 1, groups*group_size]
    """
    shape = x.shape
    width = shape[-1]
    assert width == groups * group_size, f"{width} != {groups}*{group_size}"

    # Fold the group axis into rows so the normalised axis is the last one.
    folded = ttnn.reshape(x, (shape[0], shape[1], shape[2] * groups, group_size))
    normed = ttnn.rms_norm(folded, epsilon=eps, compute_kernel_config=HIFI4)
    normed = ttnn.reshape(normed, shape)
    return ttnn.multiply(normed, weight)


def reshape_to(x: ttnn.Tensor, shape) -> ttnn.Tensor:
    """`ttnn.reshape`, skipped when the tensor already has that shape.

    Not free to call for nothing: a reshape costs ~36 us on device even when it
    only changes rank and no data moves, roughly half an arithmetic op of the same
    size. At batch 1 the hyper-connection mean asks for a shape the tensor already
    has, 97 times per token -- 98 of the step's 914 reshapes.
    """
    return x if list(x.shape) == list(shape) else ttnn.reshape(x, shape)


def rms_norm(x: ttnn.Tensor, weight: ttnn.Tensor, eps: float) -> ttnn.Tensor:
    return ttnn.rms_norm(x, epsilon=eps, weight=weight, compute_kernel_config=HIFI4)


def swiglu(gate: ttnn.Tensor, up: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.multiply(ttnn.silu(gate), up)


def gated_residual_mix(
    hyper: ttnn.Tensor,
    norm_w: ttnn.Tensor,
    down_w: ttnn.Tensor,
    up_w: ttnn.Tensor,
    inject_w: ttnn.Tensor | None,
    eps: float,
    hc_count: int,
    hidden_size: int,
):
    """The hyper-connection read gate.

    Returns (mixed [.., hidden_size], normed [.., hc*hidden], inject or None).
    Mirrors Qwen4ExpModel._gated_residual.
    """
    normed = grouped_rms_norm(hyper, norm_w, eps, hidden_size, hc_count)

    # down_w and inject_w carry the 1/hc_count factor already (folded at
    # conversion; exact, since 1/4 only shifts the block-float exponent)
    mix = ttnn.silu(ttnn.linear(normed, down_w, compute_kernel_config=HIFI4))
    mix = ttnn.sigmoid(ttnn.linear(mix, up_w, compute_kernel_config=HIFI4))
    gated = ttnn.multiply(mix, normed)

    # Mean over the hc_count streams. The flattened layout is stream-major, so a
    # single reshape to [.., M, hc, hidden] exposes the stream axis directly --
    # no intermediate reshape needed.
    shape = list(gated.shape)
    rows = shape[1] * shape[2]
    per_stream = ttnn.reshape(gated, (shape[0], rows, hc_count, hidden_size))
    # `mean` rather than `sum` then a scalar multiply: one op instead of two,
    # 97 times a step. A/B'd rather than assumed, because op count is a poor
    # proxy for cost here -- see `reinject`, where a three-op form ran 9.5x
    # slower than the nine-op one it replaced.
    #
    #     eager    498.7 -> 469.1 ms   (-5.9 %)
    #     traced   236.1 -> 236.0 ms   (unchanged)
    #
    # 29.6 ms for 97 ops is 0.30 ms apiece, which is a host dispatch. A trace
    # replays with one dispatch, so it saves nothing there -- fewer launches is
    # an *eager*-path optimisation, and the traced step is not dispatch-bound.
    averaged = ttnn.mean(per_stream, dim=-2, keepdim=True)
    mixed = reshape_to(averaged, (shape[0], shape[1], shape[2], hidden_size))

    inject = None
    if inject_w is not None:
        inject = ttnn.linear(normed, inject_w, compute_kernel_config=HIFI4)
        inject = ttnn.multiply(ttnn.sigmoid(inject), 2.0)
        # Hand it back channel-major, [.., hc, M, 1], so `reinject` can broadcast
        # against the branch without materialising anything. The permute is on a
        # tensor of M*hc elements, so it is free.
        inject = ttnn.permute(inject, (0, 3, 2, 1))
    return mixed, inject


def reinject(hyper: ttnn.Tensor, branch: ttnn.Tensor, inject: ttnn.Tensor, hc_count: int) -> ttnn.Tensor:
    """hyper + (branch outer-product inject), flattened back to hc_count*hidden.

    branch: [.., M, hidden];  inject: [.., hc_count, M, 1] (channel-major)

    Measured on device, per call at M=1 -- op *count* is a poor proxy for cost:

        slice each scalar, scale, concat   9 ops    1.010 ms
        repeat + repeat_interleave         3 ops    9.634 ms
        broadcast multiply + permute       3 ops    0.372 ms
        (ttnn.repeat_interleave alone      1 op    14.439 ms)

    `repeat_interleave` is pathological here, so the three-op form built around
    it was 9.5x *slower* than the nine-op form it replaced. Broadcasting a
    [.., hc, M, 1] injection against a [.., 1, M, hidden] branch materialises
    nothing and is fastest. This runs 96 times per token, where it had been 61 %
    of the whole decode step.
    """
    hidden = branch.shape[-1]
    m = branch.shape[-2]
    prod = ttnn.multiply(inject, branch)                       # [.., hc, M, hidden]
    prod = ttnn.reshape(ttnn.permute(prod, (0, 2, 1, 3)), (1, 1, m, hc_count * hidden))
    return ttnn.add(hyper, prod)
