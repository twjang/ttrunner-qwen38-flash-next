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

import os
import struct
import sys
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
    normed = _sharded_rms_norm(folded, eps)
    if normed is None:
        normed = ttnn.rms_norm(folded, epsilon=eps, compute_kernel_config=HIFI4)
    normed = ttnn.reshape(normed, shape)
    return ttnn.multiply(normed, weight)


# `ttnn.rms_norm` on its default program config is **18.0 us** for the 80-tile
# row this normalises, three to four times what a wide elementwise op costs. Its
# sharded config is 5.6 us -- and *four times more accurate*, 4.96e-03 from
# float64 against the default's 2.09e-02 (`rms_norm_config.py`), which is the
# same shape as the matmul's program config (handoff 19): the default is not a
# neutral choice, it is a slow and slightly worse one.
#
# The sharding round trip costs ~2.1 us each way and is counted in the total.
_RMSN_CFG: dict = {}
_NO_SHARDED_RMSNORM = bool(os.environ.get("TT_NO_SHARDED_RMSNORM"))


def _sharded_rms_norm(x, eps: float):
    """`rms_norm` through the sharded config, or None to let the caller use the op."""
    if _NO_SHARDED_RMSNORM:
        return None
    # **One row-tile only.** The sharded config's `block_h` is the row-tile
    # count, and the reduction it performs depends on it -- so a batch whose
    # folded stream is more than 32 rows normalises differently from a single
    # sequence, and `batch_equivalence_check.py` went 32/32 -> 8/32. The default
    # op is row-count independent. Restricting this to one row-tile keeps decode
    # (four rows at batch 1) on the fast path and everything wider on the exact
    # one; the two still differ from each other, which is the price and is why
    # `TT_NO_SHARDED_RMSNORM=1` exists.
    if x.shape[-2] > _TILE:
        return None
    key = (tuple(x.shape), str(x.dtype), id(x.device()))
    plan = _RMSN_CFG.get(key)
    if plan is False:
        return None
    try:
        if plan is None:
            grid = x.device().compute_with_storage_grid_size()
            nt = output_tiles(x.shape[-1])
            rows = max(x.shape[-2], _TILE)
            plan = False
            # Eight cores at ten tiles each was the fastest that builds; twenty
            # and beyond are rejected by the op, and fewer than four gives up
            # most of the speed.
            for n_cores in (8, 10, 4, 5, 2):
                if nt % n_cores or n_cores > grid.x * grid.y:
                    continue
                bw = nt // n_cores
                sub = next((sw for sw in (4, 2, 1) if bw % sw == 0), 1)
                crs = ttnn.num_cores_to_corerangeset(n_cores, grid, True)
                spec = ttnn.ShardSpec(crs, [rows, x.shape[-1] // n_cores],
                                      ttnn.ShardOrientation.ROW_MAJOR)
                mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                                       ttnn.BufferType.L1, spec)
                # `block_h` is the shard's row-tile count, so it comes from the
                # *physical* height -- the same number `rows` is built from. Taken
                # from `x.shape[-2]` it is 1 for any stream whose shard is more
                # than one tile tall, and every candidate is then rejected.
                pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=grid, subblock_w=sub,
                    block_h=rows // _TILE, block_w=bw, inplace=False)
                try:
                    xs = ttnn.to_memory_config(x, mc)
                    ttnn.rms_norm(xs, epsilon=eps, program_config=pc,
                                  memory_config=mc)
                except Exception:                                   # noqa: BLE001
                    # Per candidate, not per shape: the op rejects some core
                    # counts outright, and one rejection used to abandon the
                    # whole search -- including the counts that do build.
                    continue
                plan = (mc, pc)
                break
            _RMSN_CFG[key] = plan
            if plan is False:
                return None
        mc, pc = plan
        xs = ttnn.to_memory_config(x, mc)
        out = ttnn.rms_norm(xs, epsilon=eps, program_config=pc, memory_config=mc)
        return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
    except Exception:                                               # noqa: BLE001
        _RMSN_CFG[key] = False
        return None


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


# --- the decode matmul's program config --------------------------------------
#
# `ttnn.linear` with no program config picks one itself, and at M=1 the one it
# picks reads **two** k-tiles a block. Three things were ruled out before this
# was found: the time does not depend on the weight's dtype ([2560, 3200] is
# 61.4 us at bfloat4_b, bfloat8_b *and* bfloat16, `matmul_dtype_width.py`), it
# does not depend on MathFidelity (HiFi4 through LoFi within 2 %,
# `matmul_fidelity.py`), and it tracks the length of the K loop at ~0.65 us a
# k-tile across every shape in the census. That is read latency, not throughput,
# and `in0_block_w` is how many k-tiles a core pulls before it computes.
#
# Raising it to eight (`matmul_program_config.py`):
#
#     MoE gate|up   [2560, 3200]   61.35 -> 20.19 us   3.04x
#     attn_output   [6144, 2560]  123.55 -> 47.32      2.61x
#     shexp gate|up [2560, 1280]   40.54 -> 15.82      2.56x
#     MoE down      [1600, 2560]   34.55 -> 17.48      1.98x   (block 50)
#
# The error moves because the reduction is blocked differently, and it moves in
# both directions -- against float64 the gate|up is *better* at 8 (1.37e-02
# against 1.45e-02) and the shared expert worse (2.60e-02 against 1.28e-02).
# Blocking the whole K at once is worse on both counts (4.69e-02 and slower), so
# eight is a real optimum and not just the largest value that fits.
_MM_CFG: dict = {}
_NO_MM_CFG = bool(os.environ.get("TT_NO_MM_CONFIG"))


def decode_matmul_config(x, w):
    """A 1D-multicast config for an M <= 32 matmul, or None to let ttnn choose.

    Cached on the shapes, because building one is host work and the model asks
    for the same dozen shapes 484 times a token.
    """
    if _NO_MM_CFG:
        return None
    key = (tuple(x.shape), tuple(w.shape), id(x.device()))
    if key in _MM_CFG:
        return _MM_CFG[key]

    cfg = None
    try:
        m, kdim = x.shape[-2], x.shape[-1]
        if m <= _TILE and len(w.shape) == 4 and w.shape[-2] == kdim:
            grid = x.device().compute_with_storage_grid_size()
            n_cores = grid.x * grid.y
            kt = kdim // _TILE
            nt = output_tiles(w.shape[-1])
            per_core_n = (nt + n_cores - 1) // n_cores
            # `in0_block_w` has to divide the K tiles exactly, so take the
            # largest divisor up to ten. Ten and eight are level on kt=80
            # (3.03x and 3.04x) and ten wins on kt=50 (2.19x against nothing,
            # since neither 8 nor 4 divides it); above ten it gets slower *and*
            # less accurate -- 16 is 2.48x on kt=192 where 8 is 2.61x, at 3.18e-02
            # against 2.06e-02.
            blk = max((b for b in range(2, 11) if kt % b == 0), default=1)
            sub_w = next((sw for sw in (4, 2, 1) if per_core_n % sw == 0), 1)
            # Two conditions, both measured rather than reasoned.
            #
            # `per_core_n <= 2`: with more output tiles than that a core already
            # has enough work queued to hide the read latency, and the config
            # only gets in the way. attn_q|gate [2560, 12288] is 384 tiles over
            # 110 cores, runs at 93 % of bandwidth by itself, and every setting
            # tried made it *slower*; attn_qkv [2560, 4608] is 144 tiles and
            # gains 1.43x.
            #
            # `kt >= 16`: a short reduction has little latency to hide. hc_up
            # [320, 10240] is ten k-tiles and moves 1.03x at best.
            if blk > 2 and per_core_n <= 2 and kt >= 16:
                cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=grid,
                    in0_block_w=blk,
                    out_subblock_h=1,
                    out_subblock_w=sub_w,
                    per_core_M=1,
                    per_core_N=per_core_n,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
    except Exception:                                               # noqa: BLE001
        cfg = None
    _MM_CFG[key] = cfg
    return cfg


def fast_linear(x, w, **kw):
    """`ttnn.linear` with the decode program config, falling back to the op.

    The fallback is not decoration: a config that a shape rejects raises at
    dispatch, and the shapes this model uses are not all alike.
    """
    if "program_config" not in kw:
        pc = decode_matmul_config(x, w)
        if pc is not None:
            try:
                return ttnn.linear(x, w, program_config=pc, **kw)
            except Exception:                                       # noqa: BLE001
                _MM_CFG[(tuple(x.shape), tuple(w.shape), id(x.device()))] = None
    return ttnn.linear(x, w, **kw)


# --- rotary embedding, in one pass -------------------------------------------
#
# `_apply_rope_dev` is eleven ttnn ops -- six slices, a negate, two concats, two
# multiplies and an add -- on tensors of eight tiles or fewer, three dozen times
# a token. At 5.8 us an op whatever its shape (invariant 66) that is ~64 us a
# call and about 3 ms of a 49 ms step, to rotate 64 of 256 channels.
#
# It collapses because **rope_dim is 64 and half is 32, which is exactly one
# tile**: the rotation `[-second, first]` is a swap of two whole tiles rather
# than a shuffle inside one, so every output tile is either a fused pair of
# multiplies or a straight copy.
_ROPE_OUT: dict = {}
_ROPE_FELL_BACK = False
_NO_FUSED_ROPE = bool(os.environ.get("TT_NO_FUSED_ROPE"))


def _rope_program(x, cos, sin, out, nt, nt_rope):
    grid = x.device().compute_with_storage_grid_size()
    n_tiles = 1
    for d in list(out.shape)[:-2]:
        n_tiles *= d
    n_tiles *= max(1, (out.shape[-2] + _TILE - 1) // _TILE) * nt
    n = min(n_tiles, grid.x * grid.y)
    cols = min(n, grid.x)
    rows = (n + grid.x - 1) // grid.x
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])
    cores = [ttnn.CoreCoord(cx, cy) for cy in range(rows) for cx in range(cols)]

    acc = {}
    for tag, t in (("x", x), ("c", cos), ("s", sin), ("o", out)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"fused rope: {tag} must be interleaved")
        acc[tag] = ct
    if acc["o"][1] != acc["x"][1] or acc["c"][1] != acc["s"][1]:
        raise RuntimeError("fused rope: x and out share a dtype, cos and sin share one")

    work = [((n_tiles * i) // len(cores), (n_tiles * (i + 1)) // len(cores))
            for i in range(len(cores))]
    cbs = [ttnn.CBDescriptor(
        total_size=2 * acc["x"][1], core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=x.dtype, page_size=acc["x"][1])])
        for i in (0, 1, 4)]
    cbs += [ttnn.CBDescriptor(
        total_size=2 * acc["c"][1], core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=cos.dtype, page_size=acc["c"][1])])
        for i in (2, 3)]

    def kern(name, ct, args, cfgd):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfgd)

    return ttnn.ProgramDescriptor(kernels=[
        kern("rope_reader.cpp", [nt, nt_rope] + acc["x"] + acc["c"] + acc["s"],
             [[x.buffer_address(), cos.buffer_address(), sin.buffer_address(), lo, hi]
              for lo, hi in work], ttnn.ReaderConfigDescriptor()),
        kern("rope_compute.cpp", [nt, nt_rope], [[lo, hi] for lo, hi in work],
             ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        kern("rope_writer.cpp", acc["o"],
             [[out.buffer_address(), lo, hi] for lo, hi in work],
             ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs)


_DSCAL_OUT: dict = {}
_DSCAL_FELL_BACK = False
_NO_FUSED_DELTA_SCALARS = bool(os.environ.get("TT_NO_FUSED_DELTA_SCALARS"))

# ttnn.softplus's own defaults, as float bits: softplus(x) = log(1 + exp(x)),
# with anything above the threshold passed through unchanged.
_SP_BETA = struct.unpack("<I", struct.pack("<f", 1.0))[0]
_SP_THRESH = struct.unpack("<I", struct.pack("<f", 20.0))[0]


def _delta_scalars_program(ab, dt, a_decay, g_out, b_out, heads):
    dev = ab.device()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cores = [ttnn.CoreCoord(0, 0)]

    acc = []
    for t in (ab, dt, a_decay, g_out, b_out):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError("fused delta scalars: every operand must be interleaved")
        acc.append(ct)
    page = acc[0][1]
    if acc[1][1] != page or acc[2][1] != page:
        raise RuntimeError("fused delta scalars: the three inputs must share a page size")

    elem = {ttnn.bfloat16: 2, ttnn.float32: 4}.get(ab.dtype)
    if elem is None:
        raise RuntimeError(f"fused delta scalars: no element size for {ab.dtype}")
    face = 16 * 16 * elem

    cbs = [ttnn.CBDescriptor(
        total_size=page, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=ab.dtype, page_size=page)])
        for i in range(3)]
    cbs.append(ttnn.CBDescriptor(
        total_size=2 * page, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=3, data_format=ab.dtype, page_size=page)]))
    # 64 bytes a value, both outputs, plus a 64-byte alignment slack.
    cbs.append(ttnn.CBDescriptor(
        total_size=2 * heads * 64 + 64, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=4, data_format=ttnn.uint32, page_size=64)]))

    def kern(name, ct, args, cfgd):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfgd)

    return ttnn.ProgramDescriptor(kernels=[
        kern("delta_scalars_reader.cpp", acc[0] + acc[1] + acc[2],
             [[ab.buffer_address(), dt.buffer_address(), a_decay.buffer_address()]],
             ttnn.ReaderConfigDescriptor()),
        kern("delta_scalars_compute.cpp", [_SP_BETA, _SP_BETA, _SP_THRESH], [[]],
             ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        kern("delta_scalars_writer.cpp",
             [heads, elem, face, page] + acc[3] + acc[4],
             [[g_out.buffer_address(), b_out.buffer_address()]],
             ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs)


def fused_delta_scalars(both_ab, dt, a_decay, heads: int, key=None):
    """`(exp(A * softplus(a + dt)), sigmoid(b))` as [heads, 1, 1, 1], or None.

    Replaces two slices, an add, a softplus, a multiply, an exp, a sigmoid and
    two reshapes -- nine launches a layer to turn 24 numbers into 24 numbers.
    Both halves live in one tile, so the whole thing is one core reading three
    pages.
    """
    global _DSCAL_FELL_BACK
    if _NO_FUSED_DELTA_SCALARS:
        return None
    try:
        why = None
        if list(both_ab.shape) != [1, 1, 1, 2 * heads]:
            why = f"both_ab is {list(both_ab.shape)}, expected [1, 1, 1, {2 * heads}]"
        elif 2 * heads > _TILE:
            why = f"{2 * heads} columns do not fit one tile"
        elif list(dt.shape) != [1, 1, 1, heads] or list(a_decay.shape) != [1, 1, 1, heads]:
            why = f"dt {list(dt.shape)} and A {list(a_decay.shape)} must be [1, 1, 1, {heads}]"
        elif dt.dtype != both_ab.dtype or a_decay.dtype != both_ab.dtype:
            why = (f"dtypes differ: both_ab {both_ab.dtype}, dt {dt.dtype}, "
                   f"A {a_decay.dtype}")
        if why is not None:
            if not _DSCAL_FELL_BACK:
                _DSCAL_FELL_BACK = True
                import warnings
                warnings.warn(f"fused delta scalars declined, using the ops: {why}",
                              RuntimeWarning, stacklevel=2)
            return None

        okey = (key, id(both_ab.device()), str(both_ab.dtype), heads)
        pair = _DSCAL_OUT.get(okey)
        if pair is None:
            pair = tuple(
                ttnn.from_torch(
                    torch.zeros(heads, 1, 1, 1), dtype=both_ab.dtype,
                    layout=ttnn.TILE_LAYOUT, device=both_ab.device(),
                    mesh_mapper=ttnn.ReplicateTensorToMesh(both_ab.device()))
                for _ in range(2))
            _DSCAL_OUT[okey] = pair
        g_out, b_out = pair
        ttnn.generic_op(
            [both_ab, dt, a_decay, g_out, b_out],
            _delta_scalars_program(both_ab, dt, a_decay, g_out, b_out, heads))
        return g_out, b_out
    except Exception as exc:                                        # noqa: BLE001
        if not _DSCAL_FELL_BACK:
            _DSCAL_FELL_BACK = True
            import warnings
            warnings.warn(f"fused delta scalars unavailable, using the ops: "
                          f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        return None


_CONV_OUT: dict = {}
_CONV_FELL_BACK = False
_NO_FUSED_CONV = bool(os.environ.get("TT_NO_FUSED_CONV"))


def _conv_step_program(x, state, taps, out):
    grid = x.device().compute_with_storage_grid_size()
    n_tiles = _tile_count(x)
    n = min(n_tiles, grid.x * grid.y)
    cols = min(n, grid.x)
    rows = (n + grid.x - 1) // grid.x
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])
    cores = [ttnn.CoreCoord(cx, cy) for cy in range(rows) for cx in range(cols)]

    tensors = [x] + list(state) + list(taps)
    acc = []
    for t in tensors + [out]:
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError("fused conv: every operand must be interleaved")
        acc.append(ct)
    page = acc[0][1]
    if any(a[1] != page for a in acc):
        raise RuntimeError("fused conv: every operand must share one page size")

    work = [((n_tiles * i) // len(cores), (n_tiles * (i + 1)) // len(cores))
            for i in range(len(cores))]
    # Nine buffers of one tile each, double-buffered: eight inputs and the
    # output. At 2 KB a bfloat16 tile that is 36 KB of L1 a core.
    cbs = [ttnn.CBDescriptor(
        total_size=2 * page, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=x.dtype, page_size=page)])
        for i in range(9)]

    def kern(name, ct, args, cfgd):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfgd)

    addrs = [t.buffer_address() for t in tensors]
    read_ct = [page]
    for a in acc[:-1]:
        read_ct += a
    return ttnn.ProgramDescriptor(kernels=[
        kern("conv_step_reader.cpp", read_ct,
             [addrs + [lo, hi] for lo, hi in work], ttnn.ReaderConfigDescriptor()),
        kern("conv_step_compute.cpp", [], [[lo, hi] for lo, hi in work],
             ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        kern("conv_step_writer.cpp", acc[-1],
             [[out.buffer_address(), lo, hi] for lo, hi in work],
             ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs)


def fused_conv_step(x, state, taps, key=None):
    """`silu(sum_j taps[j] * hist[j])` and the ring shift, in one launch, or None.

    `state` is [newest, middle, oldest] and `taps` is oldest-first, which is the
    pairing `_causal_conv_step` uses (age = k-1-tap). The ring is advanced **in
    place** by the reader, so this returns only the output.

    Replaces eleven wide ops a layer -- four multiplies, three adds, three ring
    copies and a silu -- with one launch whose bytes are ~9 us a layer.
    """
    global _CONV_FELL_BACK
    if _NO_FUSED_CONV:
        return None
    try:
        # Loudly, not silently. This declined for a whole measurement round
        # because the conv weight is float32 and the stream is bfloat16, and a
        # quiet `return None` made the A/B compare the op path with itself
        # (invariant 71).
        why = None
        if len(state) != 3 or len(taps) != 4:
            why = f"expected 3 history columns and 4 taps, got {len(state)} and {len(taps)}"
        shape = list(x.shape)
        if why is None and any(list(t.shape) != shape for t in list(state) + list(taps)):
            why = ("operand shapes differ: "
                   + ", ".join(str(list(t.shape)) for t in [x] + list(state) + list(taps)))
        if why is None and any(t.dtype != x.dtype for t in list(state) + list(taps)):
            why = ("operand dtypes differ: "
                   + ", ".join(str(t.dtype) for t in [x] + list(state) + list(taps)))
        if why is not None:
            if not _CONV_FELL_BACK:
                _CONV_FELL_BACK = True
                import warnings
                warnings.warn(f"fused conv declined, using the ops: {why}",
                              RuntimeWarning, stacklevel=2)
            return None
        okey = (key, id(x.device()), tuple(shape), str(x.dtype))
        out = _CONV_OUT.get(okey)
        if out is None:
            out = ttnn.from_torch(
                torch.zeros(*shape), dtype=x.dtype, layout=ttnn.TILE_LAYOUT,
                device=x.device(), mesh_mapper=ttnn.ReplicateTensorToMesh(x.device()))
            _CONV_OUT[okey] = out
        ttnn.generic_op([x] + list(state) + list(taps) + [out],
                        _conv_step_program(x, state, taps, out))
        return out
    except Exception as exc:                                        # noqa: BLE001
        if not _CONV_FELL_BACK:
            _CONV_FELL_BACK = True
            import warnings
            warnings.warn(f"fused conv unavailable, using the ops: "
                          f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        return None


def fused_rope(x, cos_full, sin_full, rope_dim: int):
    """`x` with its first `rope_dim` channels rotated, in one launch, or None.

    `cos_full` and `sin_full` must be row-expanded to a whole tile: the SFPU
    multiplies whole tiles and cannot broadcast a row, and the tables are
    [.., 1, rope_dim] as the ops path wants them. Expanding costs nothing on
    device -- a one-row tensor already occupies a 32-row tile.
    """
    global _ROPE_FELL_BACK
    if _NO_FUSED_ROPE:
        return None
    try:
        hd = x.shape[-1]
        nt, nt_rope = hd // _TILE, rope_dim // _TILE
        if hd % _TILE or rope_dim % _TILE or nt_rope != 2 or nt < 2:
            return None
        # The tables must share `x`'s dtype. The compute kernel configures its
        # unpacker once, from the activation's circular buffer, so a float32
        # table read through it comes back as garbage -- and only in the two
        # rotated tiles, which is exactly the failure that is easiest to miss
        # (`rope_kernel_check.py` measured 1.96e+36 there and nothing wrong in
        # the six passthrough tiles).
        if cos_full.dtype != x.dtype or sin_full.dtype != x.dtype:
            return None
        key = (id(x.device()), tuple(x.shape), str(x.dtype))
        out = _ROPE_OUT.get(key)
        if out is None:
            out = ttnn.from_torch(
                torch.zeros(*x.shape), dtype=x.dtype, layout=ttnn.TILE_LAYOUT,
                device=x.device(), mesh_mapper=ttnn.ReplicateTensorToMesh(x.device()))
            _ROPE_OUT[key] = out
        ttnn.generic_op([x, cos_full, sin_full, out],
                        _rope_program(x, cos_full, sin_full, out, nt, nt_rope))
        return out
    except Exception as exc:                                        # noqa: BLE001
        if not _ROPE_FELL_BACK:
            _ROPE_FELL_BACK = True
            import warnings
            warnings.warn(f"fused rope unavailable, using the ops: "
                          f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        return None


# --- right-sized elementwise and slice ----------------------------------------
#
# `dispatch_floor.py`: a launch is ~1.8 us plus 0.036 us a core, so one core is
# 2.06 and a hundred and ten is 5.81 -- and ttnn's elementwise ops take the whole
# grid whatever the tensor. `step_op_census.py` buckets the step's calls by their
# largest operand and finds **over half of them at 64 tiles or fewer**, about
# 10 ms of a 49 ms step spent starting cores with nothing to do.
#
# These run the same arithmetic in a `generic_op` whose core range is
# `min(tiles, grid)`. Measured against `ttnn.multiply` on one tile: 2.01 us
# against 5.87, **2.9x**, and identical against float64 (`small_ew_probe.py`).
#
# The output buffer is per **call site**, not per shape. `generic_op` cannot
# allocate under trace capture, so it has to be persistent, and a shape-keyed
# buffer would alias two live results of the same shape. A site produces one
# value per layer and it is consumed before the layer ends, which is the same
# property `_SWIGLU_OUT` and `_GM_OUT` already rely on. All of this Python runs
# at capture time only -- the replay is recorded commands -- so looking the
# caller up in the stack costs nothing at run time.
_EW_OUT: dict = {}
# Set TT_EW_STATS=1 to have `ew_stats()` report what was actually launched: how
# many cores each converted call used, which is the whole point of the exercise.
EW_STATS: dict = {}
_EW_FELL_BACK = False
_NO_SMALL_EW = bool(os.environ.get("TT_NO_SMALL_EW"))

# Above this many tiles the full grid is the right answer and there is nothing
# to win: at 64 tiles the right-sized launch is 4.38 us against 5.95 (1.36x) and
# at 320 they are equal.
_EW_MAX_TILES = 64

EW_MUL, EW_ADD, EW_SUB, EW_SIGMOID, EW_SILU, EW_COPY, EW_MULS, EW_SIG_MULS = range(8)


def _tile_count(t) -> int:
    n = 1
    sh = list(t.shape)
    for d in sh[:-2]:
        n *= d
    return n * max(1, (sh[-2] + _TILE - 1) // _TILE) * output_tiles(sh[-1])


def _ew_site(depth: int = 2):
    f = sys._getframe(depth)
    return (f.f_code.co_filename, f.f_lineno)


def _ew_out(site, like, shape=None):
    shape = tuple(shape or like.shape)
    key = (site, shape, str(like.dtype), id(like.device()))
    out = _EW_OUT.get(key)
    if out is None:
        out = ttnn.from_torch(
            torch.zeros(*shape), dtype=like.dtype, layout=ttnn.TILE_LAYOUT,
            device=like.device(), mesh_mapper=ttnn.ReplicateTensorToMesh(like.device()))
        _EW_OUT[key] = out
    return out


def _ew_program(a, b, out, op, scalar, in_base, in_nt):
    dev = a.device()
    grid = dev.compute_with_storage_grid_size()
    n_tiles = _tile_count(out)
    n = min(n_tiles, grid.x * grid.y)
    cols = min(n, grid.x)
    rows = (n + grid.x - 1) // grid.x
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])
    cores = [ttnn.CoreCoord(cx, cy) for cy in range(rows) for cx in range(cols)]
    binary = op <= EW_SUB
    acc = {}
    for tag, t in (("a", a), ("b", b if binary else a), ("o", out)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"small_ew: {tag} must be interleaved")
        acc[tag] = ct
    if acc["o"][1] != acc["a"][1] or (binary and acc["b"][1] != acc["a"][1]):
        raise RuntimeError("small_ew wants one dtype throughout")
    work = [((n_tiles * i) // len(cores), (n_tiles * (i + 1)) // len(cores))
            for i in range(len(cores))]
    cbs = [ttnn.CBDescriptor(
        total_size=2 * acc["a"][1], core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=a.dtype, page_size=acc["a"][1])])
        for i in (0, 1, 2)]

    def kern(name, ct, args, cfgd):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfgd)

    return ttnn.ProgramDescriptor(kernels=[
        kern("small_ew_reader.cpp",
             [int(binary), acc["a"][1], in_base, in_nt, output_tiles(out.shape[-1])]
             + acc["a"] + acc["b"],
             [[a.buffer_address(), (b if binary else a).buffer_address(), lo, hi]
              for lo, hi in work], ttnn.ReaderConfigDescriptor()),
        kern("small_ew_compute.cpp", [op, scalar], [[hi - lo] for lo, hi in work],
             ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        kern("small_ew_writer.cpp", acc["o"],
             [[out.buffer_address(), lo, hi] for lo, hi in work],
             ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs)


def _small_ew(op, a, b=None, scalar=0, out_shape=None, in_base=0, in_nt=0,
              site=None):
    """One elementwise op (or a tile-aligned slice) on only the cores it needs.

    Returns None when it declines -- too big to gain, a dtype it does not
    handle, or an output that would alias an input -- so the caller keeps its
    own kwargs and its own fallback.
    """
    global _EW_FELL_BACK
    if _NO_SMALL_EW:
        return None
    try:
        if a.dtype not in (ttnn.bfloat16, ttnn.float32):
            return None
        if b is not None and (b.dtype != a.dtype or tuple(b.shape) != tuple(a.shape)):
            return None
        shape = tuple(out_shape or a.shape)
        if _tile_count(a) > _EW_MAX_TILES:
            return None
        out = _ew_out(site or _ew_site(3), a, shape)
        # An accumulator (`acc = add(acc, x)`) would read and write one buffer.
        if out is a or out is b:
            return None
        prog = _ew_program(a, b, out, op, scalar, in_base,
                           in_nt or output_tiles(a.shape[-1]))
        if os.environ.get("TT_EW_STATS"):
            n = min(_tile_count(out),
                    a.device().compute_with_storage_grid_size().x
                    * a.device().compute_with_storage_grid_size().y)
            EW_STATS[(op, n)] = EW_STATS.get((op, n), 0) + 1
        ttnn.generic_op([a, b if b is not None else a, out], prog)
        return out
    except Exception as exc:                                        # noqa: BLE001
        if not _EW_FELL_BACK:
            _EW_FELL_BACK = True
            import warnings
            warnings.warn(f"small elementwise unavailable, using the ops: "
                          f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        return None


# `site` on every helper: a wrapper's own line is the same for all of its
# callers, so without it two live results of one wrapper share a buffer. That is
# not hypothetical -- `_l2norm` calls `ew_scale` on one line for both q and k,
# and aliasing them took the model's top-1 from 73 % to 0.5 %.
def ew_mul(a, b, site=None):
    got = _small_ew(EW_MUL, a, b, site=site or _ew_site())
    return got if got is not None else ttnn.multiply(a, b)


def ew_add(a, b, site=None):
    got = _small_ew(EW_ADD, a, b, site=site or _ew_site())
    return got if got is not None else ttnn.add(a, b)


def ew_sigmoid(a, site=None):
    got = _small_ew(EW_SIGMOID, a, site=site or _ew_site())
    return got if got is not None else ttnn.sigmoid(a)


def ew_silu(a, site=None):
    got = _small_ew(EW_SILU, a, site=site or _ew_site())
    return got if got is not None else ttnn.silu(a)


def ew_scale(a, scalar: float, site=None):
    bits = struct.unpack("<I", struct.pack("<f", float(scalar)))[0]
    got = _small_ew(EW_MULS, a, scalar=bits, site=(site or _ew_site(), bits))
    return got if got is not None else ttnn.multiply(a, scalar)


def ew_sigmoid_scale(a, scalar: float, site=None):
    """2*sigmoid(x) and friends -- two ops for the price of one launch."""
    bits = struct.unpack("<I", struct.pack("<f", float(scalar)))[0]
    got = _small_ew(EW_SIG_MULS, a, scalar=bits, site=(site or _ew_site(), bits))
    if got is not None:
        return got
    return ttnn.multiply(ttnn.sigmoid(a), scalar)


def ew_slice_last(x, start: int, stop: int, site=None):
    """`x[..., start:stop]` when both bounds are tile-aligned: a page copy.

    `site` because this is usually reached through a wrapper, and a wrapper's own
    line is the same for every caller -- which would give two live slices of one
    tensor the same output buffer. The bounds go in the key too, so two slices on
    one source line still get their own.
    """
    if start % _TILE or stop % _TILE or stop <= start:
        s = list(x.shape)
        return ttnn.slice(x, (0, 0, 0, start), (s[0], s[1], s[2], stop))
    shape = list(x.shape)
    shape[-1] = stop - start
    got = _small_ew(EW_COPY, x, out_shape=shape, in_base=start // _TILE,
                    in_nt=output_tiles(x.shape[-1]),
                    site=(site or _ew_site(), start, stop))
    if got is not None:
        return got
    s = list(x.shape)
    return ttnn.slice(x, (0, 0, 0, start), (s[0], s[1], s[2], stop))


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
        # Deliberately *not* routed through `ksplit_linear` here. Applying the
        # split to every narrow linear in the model measured **82.25 -> 83.68 ms**
        # and moved NLL from 0.666 to 0.691: a `groups >= 2` guard is too loose,
        # and the shapes with only two or three reduction groups pay the kernel
        # launch and the `ttnn.sum` without earning them back. It is a per-shape
        # decision, made at the call site where it has been measured.
        return fast_linear(x, w, **kw)
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


def _gated_mean_program(a, b, out, hc_count: int, sigmoid_a: bool = False):
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
            kern("gated_mean_compute.cpp", [hc_count, inv_bits, int(sigmoid_a)],
                 [[hi - lo] for lo, hi in work], ttnn.ComputeConfigDescriptor()),
            kern("gated_mean_writer.cpp", [tile_bytes] + acc["o"],
                 [[out.buffer_address(), lo, hi] for lo, hi in work],
                 ttnn.WriterConfigDescriptor()),
        ],
        semaphores=[], cbs=cbs)


def fused_gated_mean(mix, normed, hc_count: int, hidden_size: int,
                     apply_sigmoid: bool = False):
    """(1/hc) * sum_h sigmoid?(mix_h) * normed_h, in one pass.

    `apply_sigmoid` folds the caller's read gate in. The kernel already holds the
    tile in a register to multiply it, so the sigmoid is free there where outside
    it was an op on a 10240-wide stream, 96 times a token.
    """
    try:
        out = _gated_mean_output(mix, hidden_size)
        ttnn.generic_op(
            [mix, normed, out],
            _gated_mean_program(mix, normed, out, hc_count, apply_sigmoid))
        return out
    except Exception:                                               # noqa: BLE001
        if apply_sigmoid:
            mix = ttnn.sigmoid(mix)
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
# The fused reinject is **off**. It is correct -- bit-exact against the ops in
# the sense that matters, and *closer* to float64 at every shape once the
# destination registers accumulate in fp32 (4.06e-03 against 4.35e-03 at M=1,
# 3.54e-03 against 6.11e-03 on the raw form) -- and it is slower: 51.74/51.44 ms
# a token against 50.79/50.79 for the eight ttnn ops it replaces, both pairs
# agreeing.
#
# That was true, and the cause was one word of it: "caching it does not help
# because the circular buffer's slots alternate". The SFPU multiplies whole
# tiles, so the per-row injection scalar has to be spread across one first, and
# that spread is 1024 scalar writes. The reader caches it and keys the cache on
# the slot address -- which a *double-buffered* CB alternates on every push, so
# the cache never hit and the spread ran on all 320 output tiles instead of on
# four. Giving the broadcast buffer one page instead of two makes the address
# constant and the cache work.
#
# The same A/B, before and after that one line: **-0.50 ms -> +0.58 ms**. The
# kernel is also nearer float64 than the ops at every M it was checked at
# (`reinject_kernel_check.py`).
#
# `TT_NO_FUSED_REINJECT=1` turns it off.
_NO_FUSED_REINJECT = bool(os.environ.get("TT_NO_FUSED_REINJECT"))
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
    n_active = len(plan)
    while len(plan) < len(cores):
        plan.append((0, 0, 0, 0))                     # idle core: writes nothing

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
                 [[oa, g, n, int(i < n_active)]
                  for i, (_, _, n, g) in enumerate(plan)],
                 ttnn.WriterConfigDescriptor()),
        ],
        semaphores=[], cbs=cbs)


_TILE = 32


def output_tiles(width: int) -> int:
    """Tiles needed to cover `width` columns -- rounded **up**, never floored.

    This existed as `width // 32` inline and cost the model a silent
    correctness bug: the fused `ssm_alpha|ssm_beta` weight is 24 columns, so the
    count was zero, `_ksplit_build`'s plan loop produced no work items, and the
    untouched output buffer came back as zeros through `ttnn.sum`. Every
    DeltaNet layer then ran with `a = b = 0` -- `beta = sigmoid(0) = 0.5`,
    constant -- and nothing raised, because the `groups >= 2` guard is computed
    from `max(nt, 1)` and stayed well above the threshold.

    It is a function so that a test can hold it rather than a source line.
    """
    return max(1, (width + _TILE - 1) // _TILE)


def ksplit_linear(x, w):
    """`x @ w` with the reduction split across cores, or None if it would not pay.

    Only worth it while the output cannot fill the grid on its own: at
    `[1536, 2560]` the split measured 0.79x and *less* accurate, so the guard is
    that the split has to buy at least two reduction groups. Returns None rather
    than falling back itself, so the caller keeps its own kwargs.
    """
    dev = x.device()
    kt = w.shape[-2] // 32
    nt = output_tiles(w.shape[-1])
    grid = dev.compute_with_storage_grid_size()
    n_cores = grid.x * grid.y
    groups = max(1, min(kt, n_cores // nt))
    if groups < 2 or x.shape[-2] > 32:
        return None

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
        # Measured here: 2.12x and more accurate than `ttnn.linear` (invariant
        # 57). `ksplit_linear` returns None if the split would not pay.
        fused_w = down_inject_weight(down_w, inject_w, span)
        part = ksplit_linear(local, fused_w)
        if part is None:
            part = linear_rows(local, fused_w, compute_kernel_config=HIFI4)
    whole = ttnn.all_reduce(part, cluster_axis=1, topology=ttnn.Topology.Linear)
    mix = ttnn.silu(
        whole if inject_w is None
        else ttnn.slice(whole, (0, 0, 0, 0),
                        (whole.shape[0], whole.shape[1], whole.shape[2], span))
    )
    # No sigmoid here: fused_gated_mean is its only reader and folds it in.
    mix = linear_rows(mix, up_w, compute_kernel_config=HIFI4)
    # Nine ttnn ops -- the multiply, four slices, three adds and the scale --
    # collapse into one kernel pass. See `fused_gated_mean` for why, and for the
    # fallback if `generic_op` is unavailable.
    #
    # (A `gated = ttnn.multiply(mix, normed)` used to sit here, left behind when
    # the fused kernel took over the work: a 10240-wide multiply on a tile padded
    # from one row to thirty-two, 96 times a token, whose result nothing read.)
    mixed = fused_gated_mean(mix, normed, hc_count, hidden_size, apply_sigmoid=True)

    inject = None
    if inject_w is not None:
        # Handed over **raw**, as (stream, first column). It is already computed,
        # in the tail of the fused matmul above, and `reinject` reads column
        # `span + h` of it directly -- so the slice, the sigmoid, the multiply
        # and the permute that used to carve it out here are gone. Four ops on a
        # stream four numbers wide, 96 times a token, and the kernel that
        # consumes them was reading that tile anyway.
        inject = (whole, span)
    return mixed, inject


_REINJECT_OUT: dict = {}
_REINJECT_FELL_BACK = False


def _reinject_output(hyper):
    """The persistent output, one per (device, shape, dtype).

    `generic_op` needs its output pre-allocated and allocating inside a trace
    capture corrupts the replay, so this is made once and reused by all 96 calls
    -- safe because each call's result is consumed by the next layer before the
    one after it runs, and a trace replays in order.
    """
    key = (id(hyper.device()), tuple(hyper.shape), str(hyper.dtype))
    out = _REINJECT_OUT.get(key)
    if out is None:
        out = ttnn.from_torch(
            torch.zeros(*hyper.shape), dtype=hyper.dtype, layout=ttnn.TILE_LAYOUT,
            device=hyper.device(), mesh_mapper=ttnn.ReplicateTensorToMesh(hyper.device()))
        _REINJECT_OUT[key] = out
    return out


def _reinject_program(hyper, branch, inject, out, hc_count: int, inj_base=None):
    grid = hyper.device().compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    acc = {}
    for tag, t in (("h", hyper), ("b", branch), ("i", inject), ("o", out)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"fused reinject: {tag} must be interleaved")
        acc[tag] = ct
    tile_bytes = acc["h"][1]
    if acc["b"][1] != tile_bytes or acc["o"][1] != tile_bytes:
        raise RuntimeError("fused reinject wants one dtype for hyper, branch and out")
    if hyper.dtype == ttnn.bfloat16:
        bf16 = 1
    elif hyper.dtype == ttnn.float32:
        bf16 = 0
    else:
        raise RuntimeError(f"fused reinject: dtype {hyper.dtype} unsupported")

    hidden = branch.shape[-1]
    nt_h = hidden // _TILE
    mt = max(branch.shape[-2] // _TILE, 1)
    if hyper.shape[-1] != hc_count * hidden:
        raise RuntimeError("fused reinject: hyper is not hc_count * hidden wide")
    total = mt * hc_count * nt_h
    work = [((total * c) // len(cores), (total * (c + 1)) // len(cores))
            for c in range(len(cores))]

    # `inj_base` is set when `inject` is the raw gate stream rather than the
    # [1, hc, M, 1] tensor the ttnn ops used to build; the kernel then reads
    # column `inj_base + h` of it and applies the 2*sigmoid itself.
    raw = inj_base is not None
    inj_nt = output_tiles(inject.shape[-1]) if raw else 1
    if raw and inject.shape[-1] < inj_base + hc_count:
        raise RuntimeError("fused reinject: the gate stream is too narrow")

    cbs = [
        ttnn.CBDescriptor(
            # **One** page for the broadcast tile (index 1), two for the rest.
            # The reader caches the spread scalar across work items, and its
            # cache key includes the slot address -- so a double-buffered CB,
            # whose write pointer alternates, invalidated the cache on every
            # single item and ran 1024 scalar writes a tile instead of one in
            # eighty. That is the whole of why this kernel measured slower than
            # the ops it replaces.
            total_size=(1 if i == 1 else 2) * tile_bytes, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=i, data_format=hyper.dtype, page_size=tile_bytes)])
        for i in (0, 1, 2, 4)
    ] + [
        ttnn.CBDescriptor(
            total_size=2 * acc["i"][1], core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=3, data_format=inject.dtype, page_size=acc["i"][1])]),
    ]

    def kern(name, ct, args, cfg):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, a) for c, a in zip(cores, args)], config=cfg)

    return ttnn.ProgramDescriptor(
        kernels=[
            kern("reinject_reader.cpp",
                 [nt_h, hc_count, mt, tile_bytes, acc["i"][1], bf16,
                  int(raw), int(inj_base or 0), inj_nt]
                 + acc["h"] + acc["b"] + acc["i"],
                 [[hyper.buffer_address(), branch.buffer_address(),
                   inject.buffer_address(), lo, hi] for lo, hi in work],
                 ttnn.ReaderConfigDescriptor()),
            # fp32 in the destination registers. Without it the product and
            # the sum each round to bfloat16 on the way through DST, which is
            # one rounding more than the ops it replaces do -- measured 7.98e-03
            # against float64 where the ops are 4.35e-03.
            kern("reinject_compute.cpp", [int(raw)], [[hi - lo] for lo, hi in work],
                 ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
            kern("reinject_writer.cpp", acc["o"],
                 [[out.buffer_address(), lo, hi] for lo, hi in work],
                 ttnn.WriterConfigDescriptor()),
        ],
        semaphores=[], cbs=cbs)


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
    # Worth ~0.4 ms a token, not the ~1.9 the byte count predicted: three
    # on/off pairs measured 59.93/58.36/58.74 against 59.80/59.10/59.29. Two of
    # the three favour the kernel and the mean is 0.39 ms, which is also the
    # first honest look at this rig's between-run drift -- about 1 ms, so a
    # single pair cannot resolve anything smaller.
    # `gated_residual_mix` hands over the gate stream raw, as (tensor, column),
    # so that the four ops that used to carve the injection out of it -- a
    # slice, a sigmoid, a multiply and a permute, on a stream four numbers wide,
    # 96 times a token -- happen inside the kernel that was already reading that
    # tile.
    base = None
    if isinstance(inject, tuple):
        inject, base = inject

    if not _NO_FUSED_REINJECT:
        try:
            out = _reinject_output(hyper)
            ttnn.generic_op(
                [hyper, branch, inject, out],
                _reinject_program(hyper, branch, inject, out, hc_count, base))
            return out
        except Exception as exc:                                # noqa: BLE001
            # Not silent. A `pass` here would leave the ops running while every
            # measurement claimed the kernel -- which is the exact shape of the
            # bug this project already recorded once, in `moe_block`.
            global _REINJECT_FELL_BACK
            if not _REINJECT_FELL_BACK:
                _REINJECT_FELL_BACK = True
                import warnings
                warnings.warn(
                    f"fused reinject unavailable, using the ops: "
                    f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)

    hidden = branch.shape[-1]
    m = branch.shape[-2]
    if base is not None:
        w = list(inject.shape)
        inject = ttnn.slice(inject, (0, 0, 0, base), (w[0], w[1], w[2], base + hc_count))
        inject = ttnn.permute(ttnn.multiply(ttnn.sigmoid(inject), 2.0), (0, 3, 2, 1))
    prod = ttnn.multiply(inject, branch)                       # [.., hc, M, hidden]
    prod = ttnn.reshape(ttnn.permute(prod, (0, 2, 1, 3)), (1, 1, m, hc_count * hidden))
    return ttnn.add(hyper, prod)
