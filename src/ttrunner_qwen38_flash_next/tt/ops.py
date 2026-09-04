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

import struct
from pathlib import Path

import torch
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


ROW_GROUP = 128


def linear_rows(x, w, max_rows: int = ROW_GROUP, **kw):
    """`ttnn.linear`, but never on more than `max_rows` rows at a time.

    A row's result is not always independent of how many rows travel with it:
    the op picks its blocking from the shape, and five of the ten dense shapes
    this model uses change above 128 rows (`row_count_stability_check.py`). The
    change is small and, on every shape measured, slightly *closer* to the exact
    float64 product -- but it is a change, and the prefill chunk is 512 wide, so
    without this a wider chunk would silently rewrite every prefill's arithmetic.

    Grouping at 128 keeps each row in exactly the company it kept when the chunk
    was 128, which is what makes a wide chunk bit-identical to a narrow one. The
    slices and the concat cost a few dispatches; a chunk is 512 rows, so it is
    four groups, not many.

    Below the threshold this is `ttnn.linear` with one extra Python comparison.
    """
    m = x.shape[-2]
    if m <= max_rows:
        return ttnn.linear(x, w, **kw)
    shape = list(x.shape)
    outs = []
    for lo in range(0, m, max_rows):
        hi = min(lo + max_rows, m)
        part = ttnn.slice(x, (0, 0, lo, 0), (shape[0], shape[1], hi, shape[3]))
        outs.append(ttnn.linear(part, w, **kw))
    return ttnn.concat(outs, dim=-2)


# --- fused gate-and-average --------------------------------------------------
#
# The tail of `gated_residual_mix` was nine ttnn ops: multiply(mix, normed) over
# a 10240-wide pair, four tile-aligned slices to pull out the hc streams, three
# adds and a scale. They round-trip ~11 MB a call between them, and each pays
# ~5.5 us of fixed per-op cost whatever it touches (invariant 42) -- ~50 us
# before any data moves, against a `generic_op` launch measured at 8.1.
#
# Fused, one output tile reads eight input tiles and writes one, and the four
# products and three adds live in the destination registers:
# **45.62 -> 8.12 us a call, 5.62x, 3.64 ms a token**
# (`scripts/stage4_gated_mean.py`). The fused form also accumulates in fp32
# registers where the chain wrote bfloat16 between every step, so if anything it
# is the better-conditioned of the two.
_KDIR = Path(__file__).resolve().parents[3] / "scripts" / "kernels"
_GM_OUT: dict = {}


def _gated_mean_output(src, hidden_size: int):
    """The persistent output, one per (device, shape, dtype).

    `generic_op` needs its output pre-allocated, and allocating inside a trace
    capture corrupts the replay. One buffer serves every call because each is
    consumed before the next runs and a trace replays in order.
    """
    key = (id(src.device()), src.shape[1], src.shape[2], hidden_size, str(src.dtype))
    out = _GM_OUT.get(key)
    if out is None:
        out = ttnn.from_torch(
            torch.zeros(1, src.shape[1], src.shape[2], hidden_size),
            dtype=src.dtype, layout=ttnn.TILE_LAYOUT, device=src.device(),
            mesh_mapper=ttnn.ReplicateTensorToMesh(src.device()),
        )
        _GM_OUT[key] = out
    return out


def _gated_mean_program(a, b, out, hc_count: int):
    """Descriptors for this call's buffer addresses.

    Rebuilt per call because `a` and `b` are fresh allocations and their
    addresses are runtime args. The cost is host-side and paid at trace capture,
    not at replay.
    """
    grid = a.device().compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
    )
    acc = {}
    for tag, t in (("a", a), ("b", b), ("o", out)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError("fused gated-mean wants interleaved tensors")
        acc[tag] = ct
    tile_bytes = acc["a"][1]

    nt_out = out.shape[-1] // 32
    rows = max(out.shape[2] // 32, 1) * out.shape[1]
    total = rows * nt_out
    work = [((total * c) // len(cores), (total * (c + 1)) // len(cores))
            for c in range(len(cores))]

    cbs = [ttnn.CBDescriptor(
        total_size=(hc_count + 1) * tile_bytes, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=a.dtype, page_size=tile_bytes)])
        for i in (0, 1)]
    cbs.append(ttnn.CBDescriptor(
        total_size=2 * tile_bytes, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=2, data_format=out.dtype, page_size=tile_bytes)]))

    inv_bits = struct.unpack("<I", struct.pack("<f", 1.0 / hc_count))[0]

    def kern(name, ct, args, cfg):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfg)

    return ttnn.ProgramDescriptor(
        kernels=[
            kern("gated_mean_reader.cpp",
                 [nt_out, hc_count, tile_bytes] + acc["a"] + acc["b"],
                 [[a.buffer_address(), b.buffer_address(), lo, hi] for lo, hi in work],
                 ttnn.ReaderConfigDescriptor()),
            kern("gated_mean_compute.cpp", [hc_count, inv_bits],
                 [[hi - lo] for lo, hi in work], ttnn.ComputeConfigDescriptor()),
            kern("gated_mean_writer.cpp", [tile_bytes] + acc["o"],
                 [[out.buffer_address(), lo, hi] for lo, hi in work],
                 ttnn.WriterConfigDescriptor()),
        ],
        semaphores=[], cbs=cbs)


def fused_gated_mean(mix, normed, hc_count: int, hidden_size: int):
    """(1/hc) * sum_h mix_h * normed_h, in one pass. Falls back to the ops."""
    try:
        out = _gated_mean_output(mix, hidden_size)
        ttnn.generic_op([mix, normed, out], _gated_mean_program(mix, normed, out, hc_count))
        return out
    except Exception:                                               # noqa: BLE001
        gated = ttnn.multiply(mix, normed)
        parts = [
            ttnn.slice(gated, (0, 0, 0, h * hidden_size),
                       (gated.shape[0], gated.shape[1], gated.shape[2],
                        (h + 1) * hidden_size))
            for h in range(hc_count)
        ]
        total = parts[0]
        for nxt in parts[1:]:
            total = ttnn.add(total, nxt)
        return ttnn.multiply(total, 1.0 / hc_count)


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
    #
    # `down_w` is Shard.ROW: its 10240 of reduction is split across the four
    # devices, so this matmul returns a *partial* sum and has to be completed
    # before anything nonlinear touches it. The all_reduce is therefore inside
    # the silu, not outside -- sum-then-silu and silu-then-sum are not the same
    # function, and getting that backwards would be silently wrong rather than
    # loud. See plan.py for why `down` is split and `up` is not: replicated it
    # reads 3.48 MB at 0.1209 ms against 0.0445 for the split plus the
    # collective, 11.61 -> 4.27 ms a token over its 96 calls.
    #
    # No model state is needed for the collective -- `TTModel.all_reduce` is this
    # one line -- so it lives here and the nine call sites stay as they were.
    # A ROW shard splits the weight's reduction axis, so the activation has to be
    # split the same way -- `normed` is replicated 10240 wide and this device's
    # `down_w` is only [2560, 320]. `mesh_partition` is the inverse of
    # all_gather: device d keeps columns [d*2560, (d+1)*2560).
    part = linear_rows(
        ttnn.mesh_partition(normed, dim=-1), down_w, compute_kernel_config=HIFI4
    )
    mix = ttnn.silu(ttnn.all_reduce(part, cluster_axis=1, topology=ttnn.Topology.Linear))
    mix = ttnn.sigmoid(linear_rows(mix, up_w, compute_kernel_config=HIFI4))
    gated = ttnn.multiply(mix, normed)

    # Nine ttnn ops -- the multiply, four slices, three adds and the scale --
    # collapse into one kernel pass. See `fused_gated_mean` for why, and for the
    # fallback if `generic_op` is unavailable.
    mixed = fused_gated_mean(mix, normed, hc_count, hidden_size)

    inject = None
    if inject_w is not None:
        inject = linear_rows(normed, inject_w, compute_kernel_config=HIFI4)
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
