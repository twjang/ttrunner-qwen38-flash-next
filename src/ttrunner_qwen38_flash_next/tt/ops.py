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


# --- a GEMV whose reduction is split across cores ----------------------------
#
# `ttnn.linear` at M=1 runs at a quarter of bandwidth when its output is narrow,
# because N decides how many output tiles exist and so how many cores get work
# (invariant 38). Giving the idle cores a slice of the *reduction* instead --
# core (g, nt) accumulates only K-tiles [kt_lo, kt_hi) and writes a partial,
# which one `ttnn.sum` folds -- measures **2.12x** on `hc_down` and 1.49x on the
# router, and loses on shapes whose output already fills the grid.
#
# It is also *more accurate* than the op it replaces: against float64 the kernel
# is 9.71e-03 where `ttnn.linear` is 1.29e-02, because splitting the reduction is
# a pairwise summation and better conditioned than one long serial accumulation.
# The two differ from each other by 1.73e-02, which is what nearly got this
# discarded -- divergence from `ttnn.linear` is not error (invariant 57).
_KSPLIT: dict = {}
_KS_KDIR = Path(__file__).resolve().parents[3] / "scripts" / "kernels"


def _ksplit_build(a, w, out, groups: int, kt: int, nt: int):
    grid = a.device().compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    acc = {}
    for tag, t in (("a", a), ("w", w), ("o", out)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"{tag}: the k-split GEMV wants interleaved tensors")
        acc[tag] = ct

    plan = []
    for g in range(groups):
        lo, hi = (kt * g) // groups, (kt * (g + 1)) // groups
        for n in range(nt):
            plan.append((lo, hi, n, g))
    if len(plan) > len(cores):
        raise RuntimeError(f"{len(plan)} work items over {len(cores)} cores")
    while len(plan) < len(cores):
        plan.append((0, 0, 0, 0))                     # idle core

    cbs = [
        ttnn.CBDescriptor(total_size=4 * acc["a"][1], core_ranges=crs,
                          format_descriptors=[ttnn.CBFormatDescriptor(
                              buffer_index=0, data_format=a.dtype, page_size=acc["a"][1])]),
        ttnn.CBDescriptor(total_size=4 * acc["w"][1], core_ranges=crs,
                          format_descriptors=[ttnn.CBFormatDescriptor(
                              buffer_index=1, data_format=w.dtype, page_size=acc["w"][1])]),
        ttnn.CBDescriptor(total_size=2 * acc["o"][1], core_ranges=crs,
                          format_descriptors=[ttnn.CBFormatDescriptor(
                              buffer_index=16, data_format=out.dtype, page_size=acc["o"][1])]),
    ]

    def kern(name, ct, args, cfg):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KS_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfg)

    aa, wa, oa = a.buffer_address(), w.buffer_address(), out.buffer_address()
    return ttnn.ProgramDescriptor(
        kernels=[
            kern("ksplit_reader.cpp", [kt, nt] + acc["a"] + acc["w"],
                 [[aa, wa, lo, hi, n] for lo, hi, n, _ in plan],
                 ttnn.ReaderConfigDescriptor()),
            kern("ksplit_compute.cpp", [], [[hi - lo] for lo, hi, _, _ in plan],
                 ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)),
            kern("ksplit_writer.cpp", [nt] + acc["o"],
                 [[oa, g, n] for _, _, n, g in plan], ttnn.WriterConfigDescriptor()),
        ],
        semaphores=[], cbs=cbs)


def ksplit_linear(x, w):
    """`x @ w` with the reduction split across cores, or `ttnn.linear` if it would not pay.

    Only worth it while the output cannot fill the grid on its own: at
    `[1536, 2560]` the split measured 0.79x and less accurate, so the guard is
    that the split has to buy at least two reduction groups.
    """
    dev = x.device()
    kt, nt = w.shape[-2] // 32, w.shape[-1] // 32
    grid = dev.compute_with_storage_grid_size()
    n_cores = grid.x * grid.y
    groups = max(1, min(kt, n_cores // max(nt, 1)))
    if groups < 2 or x.shape[-2] > 32:
        return linear_rows(x, w, compute_kernel_config=HIFI4)

    key = (id(w), x.shape[-2], groups)
    got = _KSPLIT.get(key)
    if got is None:
        out = ttnn.from_torch(
            torch.zeros(1, groups, x.shape[-2], w.shape[-1]), dtype=x.dtype,
            layout=ttnn.TILE_LAYOUT, device=dev,
            mesh_mapper=ttnn.ReplicateTensorToMesh(dev))
        got = (out, None)
        _KSPLIT[key] = got
    out = got[0]
    ttnn.generic_op([x, w, out], _ksplit_build(x, w, out, groups, kt, nt))
    return ttnn.sum(out, dim=1, keepdim=True)

# --- down and inject as one matmul -------------------------------------------
#
# `inject_w` is [10240, 4]: four output columns, one tile, and the narrowest
# shape in the model. Invariant 38 says achieved bandwidth tracks the output
# width, and this is where that bites hardest -- 80.35 us a call against 0.4 us
# of bytes, **200x**, and it runs 96 times a token for 7.71 ms
# (`scripts/dev/route_cost.py`'s sibling, hc_inject).
#
# It takes the same `normed` that `down_w` does, so the two are one matmul with a
# wider output. Concatenating them costs 320 -> 352 columns (four real, the rest
# padding to a tile boundary) and the pair then costs about what `down` alone
# did. `inject_w` is mesh-partitioned to match `down_w`'s row shard, so its
# result is a partial sum too -- which is fine, because the all_reduce that
# `down` already needs comes before anything nonlinear touches either.
_DOWNINJ: dict = {}


def down_inject_weight(down_w, inject_w, hidden_span: int):
    """`[K_local, span + 32]` with inject in the first four of the tail columns.

    Built once per layer at first use and cached: it is a device concat of two
    tensors that never change, so paying it every token would be the same
    mistake the fusion is fixing.
    """
    key = (id(down_w), id(inject_w))
    got = _DOWNINJ.get(key)
    if got is None:
        local = ttnn.mesh_partition(inject_w, dim=-2)      # match down_w's row shard
        n_inj = local.shape[-1]
        if n_inj < 32:
            local = ttnn.pad(local, [(0, 0), (0, 0), (0, 0), (0, 32 - n_inj)], 0.0)
        got = ttnn.concat([down_w, ttnn.typecast(local, down_w.dtype)], dim=-1)
        _DOWNINJ[key] = got
    return got

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
    span = down_w.shape[-1]
    local = ttnn.mesh_partition(normed, dim=-1)
    if inject_w is None:
        part = linear_rows(local, down_w, compute_kernel_config=HIFI4)
    else:
        # One matmul for both: see `down_inject_weight`.
        part = ksplit_linear(local, down_inject_weight(down_w, inject_w, span))
    whole = ttnn.all_reduce(part, cluster_axis=1, topology=ttnn.Topology.Linear)
    mix = ttnn.silu(
        whole if inject_w is None
        else ttnn.slice(whole, (0, 0, 0, 0),
                        (whole.shape[0], whole.shape[1], whole.shape[2], span))
    )
    mix = ttnn.sigmoid(linear_rows(mix, up_w, compute_kernel_config=HIFI4))
    gated = ttnn.multiply(mix, normed)

    # Nine ttnn ops -- the multiply, four slices, three adds and the scale --
    # collapse into one kernel pass. See `fused_gated_mean` for why, and for the
    # fallback if `generic_op` is unavailable.
    mixed = fused_gated_mean(mix, normed, hc_count, hidden_size)

    inject = None
    if inject_w is not None:
        # Already computed, in the tail of the fused matmul above.
        n_inj = inject_w.shape[-1]
        inject = ttnn.slice(whole, (0, 0, 0, span),
                            (whole.shape[0], whole.shape[1], whole.shape[2], span + n_inj))
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
