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


# `num_links` on the collective, which nothing had ever swept. On a 1x4 mesh the
# default picks one link; three is measurably better once the tensor is wide, and
# bit-identical (the reduction order is the same):
#
#   [1,1,1,2560]  default 38.88 us   2 links 34.60   **3 links 32.77**   4 links 38.91
#   [1,1,1,352]   default 12.79      2 links 13.37     3 links 12.87     4 links 12.76
#
# Below the grid it is already at its floor -- a 352-wide reduce is 12.8 us of
# fixed cost for 0.7 KB -- so the extra links only add setup.
# Every fused kernel below falls back to the plain ops when its guard declines,
# and warns **once per process** -- so a second decline elsewhere is silent. A
# guard that declines by accident is expensive and invisible: a `shape[:3]` that
# ttnn's Shape does not support made `fused_delta_scalars` throw and take the
# nine-op path on all 36 layers, which measured +2.25 ms and looked like noise
# until the log was read. TT_STRICT_KERNELS turns every decline into a raise, so
# a check run says which kernel and why.
_STRICT_KERNELS = bool(os.environ.get("TT_STRICT_KERNELS"))
if _STRICT_KERNELS:
    # Not every decline in this file routes through `_declined` -- some warn
    # inline -- so promote the warning itself. That covers the ones this helper
    # does not, and any added later.
    import warnings as _warnings
    _warnings.simplefilter("error", RuntimeWarning)


def _declined(what: str, why: str) -> None:
    """Warn, or raise under TT_STRICT_KERNELS. Callers still return None."""
    if _STRICT_KERNELS:
        raise RuntimeError(f"{what} declined and TT_STRICT_KERNELS is set: {why}")
    import warnings
    warnings.warn(f"{what}, using the ops: {why}", RuntimeWarning, stacklevel=3)


_AR_LINKS = int(os.environ.get("TT_AR_LINKS", "3"))
_NO_AR_LINKS = bool(os.environ.get("TT_NO_AR_LINKS"))
_AR_MIN_TILES = 16


# The 12.8 -> 32.8 us step above is an **algorithm switch, not hops and not
# bytes.** `ttnn.all_reduce` is a dispatcher: `all_reduce_async` looks for a
# scatter dim, and at 352 wide (11 tiles) nothing divides by four, so it takes
# the *composite* path -- one line all_gather and a local sum. At 2560 (80
# tiles) 80 % 4 == 0, so it takes the *native* path, which is
# `reduce_scatter_minimal_async` **then** `all_gather_async`: two fabric
# collectives where the composite runs one. That is the whole width step.
#
# So the 84 wide reduces a token can be put back on the one-collective path
# without a kernel: gather the four partials onto the batch axis and add them
# here. It moves the sum out of `reduce_scatter` and into a local reduction, so
# the reduction *order* changes -- which is the gate handoff 35.1 set for
# `Topology.Ring` and it applies unchanged: `device_quality.py` against the
# Linear control, not a speed measurement.
# On. Five paired runs at -0.45 ms (handoff 45.5), determinism 0.000e+00 with it,
# and `device_quality.py 192` moves one token in 191 either way -- 72.8 -> 72.3
# top-1, 91.1 -> 90.6 top-5, NLL 1.219 -> 1.217 -- against invariant 70's noise
# floor of six tokens. `TT_NO_AR_COMPOSITE` restores the native path.
_AR_COMPOSITE = not os.environ.get("TT_NO_AR_COMPOSITE")


def all_reduce(t):
    """`ttnn.all_reduce` over the mesh's one axis, with the link count that wins."""
    wide = _tile_count(t) >= _AR_MIN_TILES
    if _AR_COMPOSITE and wide and len(t.shape) == 4 and int(t.shape[1]) == 1:
        # dim=1 so the four partials arrive as four *rows of tiles*, which is
        # the [1, G, 32, N] shape `fused_group_sum` already folds for the
        # k-split -- no reshape, and the fold is on the cores the output needs.
        g = ttnn.all_gather(t, dim=1, cluster_axis=1,
                            topology=ttnn.Topology.Linear, num_links=_AR_LINKS)
        folded = fused_group_sum(g, key=("arcomp", int(t.shape[-1]), str(t.dtype)))
        return folded if folded is not None else ttnn.sum(g, dim=1, keepdim=True)
    if _NO_AR_LINKS or not wide:
        return ttnn.all_reduce(t, cluster_axis=1, topology=ttnn.Topology.Linear)
    return ttnn.all_reduce(t, cluster_axis=1, topology=ttnn.Topology.Linear,
                           num_links=_AR_LINKS)


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

    # The cast here rather than at the call sites: the norm weights are stored
    # float32 and `fused_group_norm` needs the activation's dtype, so a caller
    # that did not know silently got the six-op path. The PLE's three norms did
    # exactly that for as long as the fused kernel has existed -- and only the
    # *first* decline warns, so fixing the one in `gated_residual_mix` hid them.
    weight = as_dtype(weight, x.dtype)
    fused = fused_group_norm(x, weight, eps, group_size, groups,
                             key=("grn", id(weight)))
    if fused is not None:
        return fused[0]

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


_DOUT_OUT: dict = {}
_DOUT_FELL_BACK = False
_NO_FUSED_DELTA_OUT = bool(os.environ.get("TT_NO_FUSED_DELTA_OUT"))


def _delta_out_program(ins, delta, out, tph, heads):
    dev = ins[0].device()
    grid = dev.compute_with_storage_grid_size()
    n = max(1, min(heads, grid.x * grid.y))
    cols = min(n, grid.x)
    rows = (n + grid.x - 1) // grid.x
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])
    cores = [ttnn.CoreCoord(cx, cy) for cy in range(rows) for cx in range(cols)]

    acc = []
    for t in list(ins) + [delta, out]:
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError("fused delta out: every operand must be interleaved")
        acc.append(ct)
    page = acc[0][1]
    if any(a[1] != page for a in acc):
        raise RuntimeError("fused delta out: every operand must share a page size")

    work = [((heads * i) // len(cores), (heads * (i + 1)) // len(cores))
            for i in range(len(cores))]
    sizes = {0: tph, 1: tph, 2: 2, 3: tph, 4: tph, 5: tph,
             6: 2, 7: 2, 8: 2 * tph, 9: 2 * tph, 10: 2, 11: 2}
    cbs = [ttnn.CBDescriptor(
        total_size=n_ * page, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=ins[0].dtype, page_size=page)])
        for i, n_ in sizes.items()]

    def kern(name, ct, args, cfgd):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfgd)

    addrs = [t.buffer_address() for t in ins]
    read_ct = [tph, page]
    for a in acc[:6]:
        read_ct += a
    return ttnn.ProgramDescriptor(kernels=[
        kern("delta_out_reader.cpp", read_ct,
             [addrs + [lo, hi] for lo, hi in work], ttnn.ReaderConfigDescriptor()),
        kern("delta_out_compute.cpp", [tph], [[lo, hi] for lo, hi in work],
             ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        kern("delta_out_writer.cpp", [tph] + acc[6] + acc[7],
             [[delta.buffer_address(), out.buffer_address(), lo, hi]
              for lo, hi in work], ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs)


def fused_delta_out(v, predicted, beta, q, k, q_decayed, heads, head_dim, key=None):
    """The delta rule's tail in one launch: `(delta, out)`. Or None.

    Replaces a subtract, two multiplies, a reduction and two more multiplies --
    six launches a layer on [BH, 1, 1, Dv] tensors -- with one core a head.
    Written as `out = q_decayed + (qk*beta)*(v - predicted)` so the difference is
    computed once and `delta` has a single consumer.
    """
    global _DOUT_FELL_BACK
    if _NO_FUSED_DELTA_OUT:
        return None
    try:
        why = None
        want = [heads, 1, 1, head_dim]
        for name, t in (("v", v), ("predicted", predicted), ("q", q), ("k", k),
                        ("q_decayed", q_decayed)):
            if list(t.shape) != want or t.dtype != v.dtype:
                why = f"{name} is {list(t.shape)} {t.dtype}, expected {want} {v.dtype}"
                break
        if why is None and (list(beta.shape) != [heads, 1, 1, 1] or beta.dtype != v.dtype):
            why = f"beta is {list(beta.shape)} {beta.dtype}"
        if why is None and head_dim % _TILE:
            why = f"head_dim {head_dim} is not a whole number of tiles"
        if why is not None:
            if not _DOUT_FELL_BACK:
                _DOUT_FELL_BACK = True
                _declined("fused delta out", f"{why}")
            return None

        okey = (key, id(v.device()), heads, head_dim, str(v.dtype))
        bufs = _DOUT_OUT.get(okey)
        if bufs is None:
            bufs = tuple(
                ttnn.from_torch(torch.zeros(heads, 1, 1, head_dim), dtype=v.dtype,
                                layout=ttnn.TILE_LAYOUT, device=v.device(),
                                mesh_mapper=ttnn.ReplicateTensorToMesh(v.device()))
                for _ in range(2))
            _DOUT_OUT[okey] = bufs
        delta, out = bufs
        ins = [v, predicted, beta, q, k, q_decayed]
        ttnn.generic_op(ins + [delta, out],
                        _delta_out_program(ins, delta, out, head_dim // _TILE, heads))
        return delta, out
    except Exception as exc:                                        # noqa: BLE001
        if not _DOUT_FELL_BACK:
            _DOUT_FELL_BACK = True
            import warnings
            warnings.warn(f"fused delta out unavailable, using the ops: "
                          f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        return None


_DTAIL_OUT: dict = {}
_DTAIL_FELL_BACK = False
_NO_FUSED_DELTA_TAIL = bool(os.environ.get("TT_NO_FUSED_DELTA_TAIL"))


def _delta_tail_program(o, z, w, out, tph, heads, recip_d, eps):
    grid = o.device().compute_with_storage_grid_size()
    n = min(heads, grid.x * grid.y)
    cols = min(n, grid.x)
    rows = (n + grid.x - 1) // grid.x
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])
    cores = [ttnn.CoreCoord(cx, cy) for cy in range(rows) for cx in range(cols)]

    acc = []
    for t in (o, z, w, out):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError("fused delta tail: every operand must be interleaved")
        acc.append(ct)
    page = acc[0][1]
    if any(a[1] != page for a in acc):
        raise RuntimeError("fused delta tail: every operand must share a page size")

    work = [((heads * i) // len(cores), (heads * (i + 1)) // len(cores))
            for i in range(len(cores))]
    sizes = {0: tph, 1: tph, 2: tph, 3: 2, 4: 2 * tph, 5: 2}
    cbs = [ttnn.CBDescriptor(
        total_size=sizes[i] * page, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=o.dtype, page_size=page)])
        for i in range(6)]

    def kern(name, ct, args, cfgd):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfgd)

    return ttnn.ProgramDescriptor(kernels=[
        kern("delta_tail_reader.cpp", [tph, page] + acc[0] + acc[1] + acc[2],
             [[o.buffer_address(), z.buffer_address(), w.buffer_address(), lo, hi]
              for lo, hi in work], ttnn.ReaderConfigDescriptor()),
        kern("delta_tail_compute.cpp", [tph, recip_d, eps],
             [[lo, hi] for lo, hi in work],
             ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        kern("delta_tail_writer.cpp", [tph] + acc[3],
             [[out.buffer_address(), lo, hi] for lo, hi in work],
             ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs)


def fused_delta_tail(o, z, w, heads: int, head_dim: int, eps: float, key=None):
    """`rms_norm(o) * w * sigmoid(z)`, flattened, in one launch. Or None.

    Replaces two reshapes, an `rms_norm`, a sigmoid, a multiply and a reshape --
    six launches a layer, 28.73 us. The reshapes are there only because
    `ttnn.rms_norm` reduces over the last axis; a kernel that does its own
    reduction leaves every mapping the identity.
    """
    global _DTAIL_FELL_BACK
    if _NO_FUSED_DELTA_TAIL:
        return None
    try:
        why = None
        if list(o.shape) != [heads, 1, 1, head_dim]:
            why = f"o is {list(o.shape)}, expected [{heads}, 1, 1, {head_dim}]"
        elif list(z.shape) != [1, 1, 1, heads * head_dim]:
            why = f"z is {list(z.shape)}, expected [1, 1, 1, {heads * head_dim}]"
        elif list(w.shape) != [1, 1, 1, head_dim]:
            why = f"weight is {list(w.shape)}, expected [1, 1, 1, {head_dim}]"
        elif head_dim % _TILE:
            why = f"head_dim {head_dim} is not a whole number of tiles"
        elif z.dtype != o.dtype or w.dtype != o.dtype:
            why = f"dtypes differ: o {o.dtype}, z {z.dtype}, weight {w.dtype}"
        if why is not None:
            if not _DTAIL_FELL_BACK:
                _DTAIL_FELL_BACK = True
                _declined("fused delta tail", f"{why}")
            return None

        okey = (key, id(o.device()), heads, head_dim, str(o.dtype))
        out = _DTAIL_OUT.get(okey)
        if out is None:
            out = ttnn.from_torch(
                torch.zeros(1, 1, 1, heads * head_dim), dtype=o.dtype,
                layout=ttnn.TILE_LAYOUT, device=o.device(),
                mesh_mapper=ttnn.ReplicateTensorToMesh(o.device()))
            _DTAIL_OUT[okey] = out
        bits = lambda f: struct.unpack("<I", struct.pack("<f", float(f)))[0]  # noqa: E731
        ttnn.generic_op([o, z, w, out],
                        _delta_tail_program(o, z, w, out, head_dim // _TILE, heads,
                                            bits(1.0 / head_dim), bits(eps)))
        return out
    except Exception as exc:                                        # noqa: BLE001
        if not _DTAIL_FELL_BACK:
            _DTAIL_FELL_BACK = True
            import warnings
            warnings.warn(f"fused delta tail unavailable, using the ops: "
                          f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        return None


_GNORM_OUT: dict = {}
_GNORM_FELL_BACK = False
_NO_FUSED_GROUP_NORM = bool(os.environ.get("TT_NO_FUSED_GROUP_NORM"))


_CAST_ONCE: dict = {}


def as_dtype(t, dtype):
    """`t` in `dtype`, converted once and kept.

    A compute kernel configures its unpacker from one circular buffer, so a
    float32 weight read through a bfloat16 one is garbage (invariant 76) -- and
    a guard that only declines makes the A/B compare a path with itself, which
    has now happened four times in this project.
    """
    if t.dtype == dtype:
        return t
    key = (id(t), str(dtype))
    hit = _CAST_ONCE.get(key)
    if hit is None:
        hit = ttnn.typecast(t, dtype)
        _CAST_ONCE[key] = hit
    return hit


def _gnorm_devid(device):
    """Which mesh device is running, as a tensor.

    `generic_op` broadcasts one program to the whole mesh, so runtime args are
    identical everywhere. A tensor sharded on dim 0 is not -- the same trick
    `moe._router_devid` uses. Sixteen uint32 so the page clears DRAM's 64 B
    alignment.
    """
    key = ("gnorm_devid", id(device))
    got = _GNORM_OUT.get(key)
    if got is None:
        n = device.get_num_devices()
        got = ttnn.from_torch(
            torch.arange(n, dtype=torch.int32).reshape(n, 1, 1, 1)
            .expand(n, 1, 1, 16).contiguous(),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0))
        _GNORM_OUT[key] = got
    return got


# Partials a group in the RMS reduction. The reduction is SFPU-bound, not
# bandwidth-bound: one core over a whole 80-tile group is 240 tile operations and
# measured 31 us on four cores. Ten partials a group is 24 each.
# Swept against the ops it replaces (27.85 us): 4 -> 23.09, 5 -> 25.61,
# 8 -> 21.66, 10 -> 22.56, 16 -> 21.45, 20 -> 22.03.
_GNORM_PARTS = int(os.environ.get("TT_GNORM_PARTS", "16"))


def _gnorm_programs(x, w, part, scale, out, local, devid, nt_g, groups, recip, eps,
                    parts):
    dev = x.device()
    grid = dev.compute_with_storage_grid_size()
    acc = {}
    for tag, t in (("x", x), ("w", w), ("p", part), ("s", scale), ("o", out),
                   ("l", local), ("d", devid)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"fused group norm: {tag} must be interleaved")
        acc[tag] = ct
    page = acc["x"][1]
    for tag in ("w", "p", "s", "o", "l"):
        if acc[tag][1] != page:
            raise RuntimeError("fused group norm: every tile operand shares a page size")

    def kern(name, ct, args, cfgd, crs, cores):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfgd)

    def core_set(n):
        n = max(1, min(n, grid.x * grid.y))
        cols = min(n, grid.x)
        rows = (n + grid.x - 1) // grid.x
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(
            ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])
        return crs, [ttnn.CoreCoord(cx, cy) for cy in range(rows) for cx in range(cols)]

    SN, SD = 1, 2                      # the reader takes half of every run

    def reduction(runs, src, src_acc, dst, dst_acc, square, cap):
        """One reduction pass: `runs` is [(out_page, tile_lo, tile_len), ...]."""
        crs, cores = core_set(len(runs))
        args = []
        for i, c in enumerate(cores):
            if i < len(runs):
                o, lo, ln = runs[i]
                args.append((o, o + 1, lo, ln))
            else:
                args.append((0, 0, 0, 0))
        half = max(1, (cap * SN + SD - 1) // SD)
        cbs = [ttnn.CBDescriptor(
            total_size=n * page, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=i, data_format=x.dtype, page_size=page)])
            for i, n in ((0, half), (1, max(1, cap - half + 1)), (2, 2))]
        return ttnn.ProgramDescriptor(kernels=[
            kern("group_rms_reader.cpp", [page, SN, SD] + src_acc,
                 [[src.buffer_address(), a[0], a[1], a[2], a[3]] for a in args],
                 ttnn.ReaderConfigDescriptor(), crs, cores),
            kern("group_rms_compute.cpp", [square, recip, eps, SN, SD],
                 [[a[0], a[1], a[3]] for a in args],
                 ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True), crs, cores),
            kern("group_rms_writer.cpp", [page, SN, SD] + dst_acc + src_acc,
                 [[dst.buffer_address(), src.buffer_address(), a[0], a[1], a[2], a[3]]
                  for a in args],
                 ttnn.WriterConfigDescriptor(), crs, cores),
        ], semaphores=[], cbs=cbs)

    # --- the reduction, in one launch ----------------------------------------
    #
    # Each group's `parts` cores compute a partial and the one at part 0 gathers:
    # the others write their tile straight into its fold buffer -- the same
    # circular buffer at the same L1 offset on every core -- and signal. Two
    # launches until the multicast work made the handshake routine.
    runs_a, cap_a = [], 0
    for g in range(groups):
        for j in range(parts):
            lo = g * nt_g + (nt_g * j) // parts
            hi = g * nt_g + (nt_g * (j + 1)) // parts
            runs_a.append((g * parts + j, lo, hi - lo))
            cap_a = max(cap_a, hi - lo)

    crs_a, cores_a = core_set(len(runs_a))
    d0 = dev.get_devices()[0] if hasattr(dev, "get_devices") else dev
    half = max(1, (cap_a * SN + SD - 1) // SD)
    cbs_a = [ttnn.CBDescriptor(
        total_size=n_ * page, core_ranges=crs_a,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=x.dtype, page_size=page)])
        for i, n_ in ((0, half), (1, max(1, cap_a - half + 1)), (2, 2), (3, parts))]

    r_args, c_args, w_args = [], [], []
    for i, c in enumerate(cores_a):
        if i < len(runs_a):
            o, lo, ln = runs_a[i]
            g, j = divmod(i, parts)
            gp = d0.worker_core_from_logical_core(cores_a[g * parts])
            r_args.append([x.buffer_address(), o, o + 1, lo, ln])
            c_args.append([o, o + 1, ln, j])
            w_args.append([scale.buffer_address(), x.buffer_address(), o, o + 1,
                           lo, ln, j, gp.x, gp.y, g])
        else:
            r_args.append([0, 0, 0, 0, 0])
            c_args.append([0, 0, 0, 1])
            w_args.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    prog_a = ttnn.ProgramDescriptor(kernels=[
        kern("group_rms_reader.cpp", [page, SN, SD] + acc["x"], r_args,
             ttnn.ReaderConfigDescriptor(), crs_a, cores_a),
        kern("group_rms_compute.cpp", [1, recip, eps, SN, SD, 1, parts], c_args,
             ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True), crs_a, cores_a),
        kern("group_rms_writer.cpp", [page, SN, SD, 1, parts] + acc["s"] + acc["x"],
             w_args, ttnn.WriterConfigDescriptor(), crs_a, cores_a),
    ], semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=crs_a,
                                            initial_value=0)], cbs=cbs_a)

    # --- pass 3: scale, weight, and this device's own group -------------------
    total = groups * nt_g
    crs_b, cores_b = core_set(total)
    work_b = [((total * i) // len(cores_b), (total * (i + 1)) // len(cores_b))
              for i in range(len(cores_b))]
    cbs_b = [ttnn.CBDescriptor(
        total_size=n * page, core_ranges=crs_b,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=x.dtype, page_size=page)])
        # index 2 is the scale: **one** page, so the reader's cache key is stable
        for i, n in ((0, 2), (1, 2), (2, 1), (3, 2), (4, 2))]
    cbs_b.append(ttnn.CBDescriptor(
        total_size=64 * ((acc["d"][1] + 64 + 63) // 64), core_ranges=crs_b,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=5, data_format=ttnn.uint32, page_size=64)]))
    prog_b = ttnn.ProgramDescriptor(kernels=[
        kern("group_scale_reader.cpp", [nt_g, page] + acc["x"] + acc["w"] + acc["s"],
             [[x.buffer_address(), w.buffer_address(), scale.buffer_address(), lo, hi]
              for lo, hi in work_b], ttnn.ReaderConfigDescriptor(), crs_b, cores_b),
        kern("group_scale_compute.cpp", [], [[lo, hi] for lo, hi in work_b],
             ttnn.ComputeConfigDescriptor(), crs_b, cores_b),
        kern("group_scale_writer.cpp", [nt_g, 1] + acc["o"] + acc["l"] + acc["d"],
             [[out.buffer_address(), local.buffer_address(), devid.buffer_address(),
               lo, hi] for lo, hi in work_b],
             ttnn.WriterConfigDescriptor(), crs_b, cores_b),
    ], semaphores=[], cbs=cbs_b)
    return prog_a, prog_b


# Rows of the core grid one group owns in the single-launch form. A group's
# cores must be a rectangle for the scale multicast, and whole grid rows are the
# rectangle that needs no bookkeeping. Two rows a group over four groups is 88 of
# 110 cores, ~4 tiles each and a 22-tile fold.
_GNORM1_ROWS = int(os.environ.get("TT_GNORM1_ROWS", "2"))
_NO_GNORM1 = bool(os.environ.get("TT_NO_GNORM1"))


def _gnorm1_program(x, w, out, local, devid, nt_g, groups, recip, eps):
    """The whole grouped norm in ONE launch.

    Two launches existed because pass 3 cannot start until pass 1 has written the
    scale -- which, inside one launch, a semaphore says just as well. Each core
    owns a run of tiles from exactly one group and reads them **once**: they stay
    in L1 across both phases, so the stream crosses DRAM once instead of twice.
    """
    dev = x.device()
    grid = dev.compute_with_storage_grid_size()
    if grid.y // groups < 1:
        raise RuntimeError(f"single-launch group norm: {groups} groups over "
                           f"{grid.y} grid rows")
    rows = max(1, min(_GNORM1_ROWS, grid.y // groups))
    while rows > 1 and rows * grid.x > nt_g:
        rows -= 1
    cpg = rows * grid.x                       # cores a group, all of them active
    if cpg > nt_g:
        raise RuntimeError(f"single-launch group norm: {cpg} cores over {nt_g} tiles")

    acc = {}
    for tag, t in (("x", x), ("w", w), ("o", out), ("l", local), ("d", devid)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"single-launch group norm: {tag} must be interleaved")
        acc[tag] = ct
    page = acc["x"][1]
    for tag in ("w", "o", "l"):
        if acc[tag][1] != page:
            raise RuntimeError("single-launch group norm: one page size for the tiles")

    crs = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, groups * rows - 1))])
    d0 = dev.get_devices()[0] if hasattr(dev, "get_devices") else dev

    cores, r_args, c_args, w_args, cap = [], [], [], [], 0
    for g in range(groups):
        y0, y1 = g * rows, g * rows + rows - 1
        p0 = d0.worker_core_from_logical_core(ttnn.CoreCoord(0, y0))
        p1 = d0.worker_core_from_logical_core(ttnn.CoreCoord(grid.x - 1, y1))
        gp = d0.worker_core_from_logical_core(ttnn.CoreCoord(0, y0))
        for j in range(cpg):
            cy, cx = divmod(j, grid.x)
            cores.append(ttnn.CoreCoord(cx, y0 + cy))
            lo = g * nt_g + (nt_g * j) // cpg
            hi = g * nt_g + (nt_g * (j + 1)) // cpg
            cap = max(cap, hi - lo)
            r_args.append([x.buffer_address(), w.buffer_address(), lo, hi - lo,
                           cpg, int(j == 0), gp.x, gp.y,
                           p0.x, p0.y, p1.x, p1.y, j])
            c_args.append([hi - lo, int(j == 0), cpg])
            w_args.append([out.buffer_address(), local.buffer_address(),
                           devid.buffer_address(), lo, hi - lo, g])

    def kern(name, ct, args, cfgd):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfgd)

    # index 4 is the scale: **one** page, so its address is the same on every
    # core and stable across the push -- which is what the multicast targets.
    cbs = [ttnn.CBDescriptor(
        total_size=n * page, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=x.dtype, page_size=page)])
        for i, n in ((0, cap), (1, cap), (2, 1), (3, cpg), (4, 1), (5, 2), (6, 2),
                     (8, 1))]
    cbs.append(ttnn.CBDescriptor(
        total_size=64 * ((acc["d"][1] + 64 + 63) // 64), core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=7, data_format=ttnn.uint32, page_size=64)]))

    return ttnn.ProgramDescriptor(kernels=[
        kern("gnorm1_reader.cpp", [page, cpg] + acc["x"] + acc["w"], r_args,
             ttnn.ReaderConfigDescriptor()),
        kern("gnorm1_compute.cpp", [recip, eps], c_args,
             ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        kern("gnorm1_writer.cpp", [nt_g, 1] + acc["o"] + acc["l"] + acc["d"],
             w_args, ttnn.WriterConfigDescriptor()),
    ], semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0)
                   for i in (0, 1)], cbs=cbs)


def fused_group_norm(x, weight, eps: float, group_size: int, groups: int, key=None):
    """`grouped_rms_norm` and `mesh_partition` in two launches. Or None.

    Returns `(normed, local)` where `local` is this device's own group -- the
    same tensor `mesh_partition(normed, dim=-1)` produces, written by the same
    kernel that is already producing those tiles.

    Six launches become two: the reshapes only existed because `ttnn.rms_norm`
    reduces over the last axis, and a kernel doing its own reduction leaves the
    groups where they are.
    """
    global _GNORM_FELL_BACK
    if _NO_FUSED_GROUP_NORM:
        return None
    try:
        why = None
        shape = list(x.shape)
        if shape[:3] != [1, 1, 1]:
            why = f"x is {shape}; this path is M = 1 only"
        elif shape[3] != groups * group_size:
            why = f"x is {shape[3]} wide, expected {groups} x {group_size}"
        elif group_size % _TILE:
            why = f"group_size {group_size} is not a whole number of tiles"
        elif list(weight.shape) != shape or weight.dtype != x.dtype:
            why = (f"weight is {list(weight.shape)} {weight.dtype}, expected "
                   f"{shape} {x.dtype}")
        if why is not None:
            if not _GNORM_FELL_BACK:
                _GNORM_FELL_BACK = True
                _declined("fused group norm", f"{why}")
            return None

        dev = x.device()
        okey = (key, id(dev), tuple(shape), str(x.dtype), groups)
        bufs = _GNORM_OUT.get(okey)
        if bufs is None:
            def mk(sh):
                return ttnn.from_torch(
                    torch.zeros(*sh), dtype=x.dtype, layout=ttnn.TILE_LAYOUT,
                    device=dev, mesh_mapper=ttnn.ReplicateTensorToMesh(dev))
            bufs = (mk((groups * _GNORM_PARTS, 1, _TILE, _TILE)),
                    mk((groups, 1, _TILE, _TILE)), mk(shape),
                    mk((1, 1, 1, group_size)))
            _GNORM_OUT[okey] = bufs
        part, scale, out, local = bufs
        devid = _gnorm_devid(dev)
        bits = lambda f: struct.unpack("<I", struct.pack("<f", float(f)))[0]  # noqa: E731
        if not _NO_GNORM1:
            p1 = _gnorm1_program(x, weight, out, local, devid,
                                 group_size // _TILE, groups,
                                 bits(1.0 / group_size), bits(eps))
            ttnn.generic_op([x, weight, out, local, devid], p1)
            return out, local
        pa, pb = _gnorm_programs(x, weight, part, scale, out, local, devid,
                                 group_size // _TILE, groups,
                                 bits(1.0 / group_size), bits(eps), _GNORM_PARTS)
        ttnn.generic_op([x, scale], pa)
        ttnn.generic_op([x, weight, scale, out, local, devid], pb)
        return out, local
    except Exception as exc:                                        # noqa: BLE001
        if not _GNORM_FELL_BACK:
            _GNORM_FELL_BACK = True
            import warnings
            warnings.warn(f"fused group norm unavailable, using the ops: "
                          f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        return None


_QKVH_OUT: dict = {}
_QKVH_FELL_BACK = False
_NO_FUSED_QKV_HEADS = bool(os.environ.get("TT_NO_FUSED_QKV_HEADS"))


def _qkv_heads_program(qkv, q_out, k_out, v_out, tph, kt, heads, eps, sq, sk):
    grid = qkv.device().compute_with_storage_grid_size()
    n = min(heads, grid.x * grid.y)
    cols = min(n, grid.x)
    rows = (n + grid.x - 1) // grid.x
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])
    cores = [ttnn.CoreCoord(cx, cy) for cy in range(rows) for cx in range(cols)]

    acc = []
    for t in (qkv, v_out, q_out, k_out):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError("fused qkv heads: every operand must be interleaved")
        acc.append(ct)
    page = acc[0][1]
    if any(a[1] != page for a in acc):
        raise RuntimeError("fused qkv heads: every operand must share a page size")

    work = [((heads * i) // len(cores), (heads * (i + 1)) // len(cores))
            for i in range(len(cores))]
    # All five buffers carry the activation's dtype: the packer is configured
    # once and the kernel switches between the SFPU and the broadcast FPU path
    # without reconfiguring it (invariant 76 in the other direction).
    sizes = {0: tph, 1: tph, 2: 2, 3: 2 * tph, 4: 1}
    cbs = [ttnn.CBDescriptor(
        total_size=sizes[i] * page, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=qkv.dtype, page_size=page)])
        for i in range(5)]

    def kern(name, ct, args, cfgd):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfgd)

    return ttnn.ProgramDescriptor(kernels=[
        kern("qkv_heads_reader.cpp", [0, kt, 2 * kt, tph, page] + acc[0] + acc[1],
             [[qkv.buffer_address(), v_out.buffer_address(), lo, hi] for lo, hi in work],
             ttnn.ReaderConfigDescriptor()),
        kern("qkv_heads_compute.cpp", [tph, eps, sq, sk],
             [[lo, hi] for lo, hi in work],
             ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        kern("qkv_heads_writer.cpp", [tph] + acc[2] + acc[3],
             [[q_out.buffer_address(), k_out.buffer_address(), lo, hi] for lo, hi in work],
             ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs)


def fused_qkv_heads(qkv, key_dim: int, value_dim: int, heads: int, head_dim: int,
                    eps: float, scale_q: float, scale_k: float, key=None):
    """Split qkv into per-head q, k, v and l2-normalise q and k. Or None.

    Replaces three slices, three reshapes, two `rms_norm`s and two scales -- ten
    launches a layer, 42.23 us. At M = 1 the slices are tile-aligned and the
    reshapes are the identity page mapping, so the split is a page copy and only
    the two norms are arithmetic.
    """
    global _QKVH_FELL_BACK
    if _NO_FUSED_QKV_HEADS:
        return None
    try:
        why = None
        shape = list(qkv.shape)
        if shape[:3] != [1, 1, 1]:
            why = f"qkv is {shape}; this path is M = 1 only"
        elif shape[3] != 2 * key_dim + value_dim:
            why = f"qkv is {shape[3]} wide, expected {2 * key_dim + value_dim}"
        elif key_dim % _TILE or head_dim % _TILE or key_dim != heads * head_dim:
            why = (f"key_dim {key_dim}, head_dim {head_dim}, heads {heads} are not "
                   "a tile-aligned head split")
        elif value_dim != heads * head_dim:
            why = f"value_dim {value_dim} != heads {heads} x head_dim {head_dim}"
        if why is not None:
            if not _QKVH_FELL_BACK:
                _QKVH_FELL_BACK = True
                _declined("fused qkv heads", f"{why}")
            return None

        okey = (key, id(qkv.device()), tuple(shape), str(qkv.dtype))
        outs = _QKVH_OUT.get(okey)
        if outs is None:
            outs = tuple(
                ttnn.from_torch(
                    torch.zeros(heads, 1, 1, head_dim), dtype=qkv.dtype,
                    layout=ttnn.TILE_LAYOUT, device=qkv.device(),
                    mesh_mapper=ttnn.ReplicateTensorToMesh(qkv.device()))
                for _ in range(3))
            _QKVH_OUT[okey] = outs
        q_out, k_out, v_out = outs
        bits = lambda f: struct.unpack("<I", struct.pack("<f", float(f)))[0]  # noqa: E731
        ttnn.generic_op(
            [qkv, q_out, k_out, v_out],
            _qkv_heads_program(qkv, q_out, k_out, v_out, head_dim // _TILE,
                               key_dim // _TILE, heads, bits(eps),
                               bits(scale_q), bits(scale_k)))
        return q_out, k_out, v_out
    except Exception as exc:                                        # noqa: BLE001
        if not _QKVH_FELL_BACK:
            _QKVH_FELL_BACK = True
            import warnings
            warnings.warn(f"fused qkv heads unavailable, using the ops: "
                          f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        return None


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
        # `>=` rather than `==`: TILE layout pads a 24-wide row out to 32
        # columns anyway, so a caller that declares the padding (to get the
        # weight past `ksgemv`'s whole-tile guard) hands this the *same physical
        # tile*. The kernel reads columns 0..heads and heads..2*heads by index,
        # so anything past 2*heads is untouched either way.
        sh = list(both_ab.shape)
        if len(sh) != 4 or sh[:3] != [1, 1, 1] or int(sh[-1]) < 2 * heads:
            why = f"both_ab is {list(both_ab.shape)}, expected [1, 1, 1, >= {2 * heads}]"
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
                _declined("fused delta scalars", f"{why}")
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
                _declined("fused conv", f"{why}")
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
# The k-split is off. It won when `ttnn.linear` ran its default program
# config; `fast_linear` picking `in0_block_w` took the matmuls from 18.30
# to 9.45 ms a token and took the k-split's advantage with it. Re-measured
# at all three sites it is now slower *and* four times less accurate:
#
#   router      30.59 us / 1.37e-02   against linear_rows 19.50 / 2.79e-03
#   shexp gate  24.00 / 9.12e-04                           9.73 / 9.12e-04
#   down|inject 17.24 / 1.03e-02                          12.21 / 2.74e-03
#
# `TT_KSPLIT=1` puts it back.
_NO_KSPLIT = os.environ.get("TT_KSPLIT", "0") != "1"
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
    deep = max(1, max(hi - lo for lo, hi, _, _ in plan))

    cbs = [
        # A core's whole share of the reduction, so its reads go out together.
        ttnn.CBDescriptor(total_size=deep * acc["a"][1], core_ranges=crs,
                          format_descriptors=[ttnn.CBFormatDescriptor(
                              buffer_index=0, data_format=a.dtype, page_size=acc["a"][1])]),
        ttnn.CBDescriptor(total_size=deep * acc["w"][1], core_ranges=crs,
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
            kern("ksplit_reader.cpp",
                 [kt, nt, acc["a"][1], acc["w"][1]] + acc["a"] + acc["w"],
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


_KS_SUM_OUT: dict = {}
_KS_SUM_FELL_BACK = False
_NO_FUSED_KSPLIT_SUM = bool(os.environ.get("TT_NO_FUSED_KSPLIT_SUM"))


def _ksplit_sum_program(parts, out, groups: int, stride: int, n_out: int):
    grid = parts.device().compute_with_storage_grid_size()
    n = min(n_out, grid.x * grid.y)
    cols = min(n, grid.x)
    rows = (n + grid.x - 1) // grid.x
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])
    cores = [ttnn.CoreCoord(cx, cy) for cy in range(rows) for cx in range(cols)]

    acc = []
    for t in (parts, out):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError("ksplit sum: both operands must be interleaved")
        acc.append(ct)
    page = acc[0][1]
    if acc[1][1] != page:
        raise RuntimeError("ksplit sum: partials and output must share a page size")

    work = [((n_out * i) // len(cores), (n_out * (i + 1)) // len(cores))
            for i in range(len(cores))]
    # The input buffer holds all G partials at once, so the reader can issue G
    # reads against one barrier instead of G round trips.
    cbs = [ttnn.CBDescriptor(
        total_size=(groups if i == 0 else 2) * page, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=parts.dtype, page_size=page)])
        for i in (0, 1)]

    def kern(name, ct, args, cfgd):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfgd)

    return ttnn.ProgramDescriptor(kernels=[
        kern("ksplit_sum_reader.cpp", [groups, stride, page] + acc[0],
             [[parts.buffer_address(), lo, hi] for lo, hi in work],
             ttnn.ReaderConfigDescriptor()),
        kern("ksplit_sum_compute.cpp", [groups], [[lo, hi] for lo, hi in work],
             ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        kern("ksplit_sum_writer.cpp", acc[1],
             [[out.buffer_address(), lo, hi] for lo, hi in work],
             ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs)


def fused_group_sum(parts, key):
    """`sum(parts, dim=1, keepdim=True)` on the cores the output needs, or None.

    `ttnn.sum` takes the whole grid to reduce at most a hundred and ten tiles to
    ten: **6.51 us**, 193 times a token, where `dispatch_floor.py` prices ten
    cores at 2.46. The reduction is also kept in fp32 dest here, where the op
    rounds to the tensor's dtype as it accumulates.
    """
    global _KS_SUM_FELL_BACK
    if _NO_FUSED_KSPLIT_SUM:
        return None
    try:
        shape = list(parts.shape)
        groups = shape[1]
        mt = max(1, (shape[-2] + _TILE - 1) // _TILE)
        nt = output_tiles(shape[-1])
        stride = mt * nt
        # Only where `ttnn.sum` stops being cheap. Measured on the three shapes
        # the model reduces: at 110 input tiles it is 6.68 us against the
        # kernel's 4.15, but at 96 and at 30 it is 3.37 and 3.15 against 4.43 and
        # 3.91 -- already at the floor, with nothing for a right-sized launch to
        # take back. The grid is the crossover, which is what it should be.
        if groups * stride < parts.device().compute_with_storage_grid_size().x \
                * parts.device().compute_with_storage_grid_size().y:
            return None
        okey = (key, id(parts.device()), tuple(shape), str(parts.dtype))
        out = _KS_SUM_OUT.get(okey)
        if out is None:
            out = ttnn.from_torch(
                torch.zeros(shape[0], 1, shape[2], shape[3]), dtype=parts.dtype,
                layout=ttnn.TILE_LAYOUT, device=parts.device(),
                mesh_mapper=ttnn.ReplicateTensorToMesh(parts.device()))
            _KS_SUM_OUT[okey] = out
        ttnn.generic_op([parts, out],
                        _ksplit_sum_program(parts, out, groups, stride, stride))
        return out
    except Exception as exc:                                        # noqa: BLE001
        if not _KS_SUM_FELL_BACK:
            _KS_SUM_FELL_BACK = True
            import warnings
            warnings.warn(f"fused ksplit sum unavailable, using ttnn.sum: "
                          f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        return None


def ksplit_linear(x, w):
    """`x @ w` with the reduction split across cores, or None if it would not pay.

    Only worth it while the output cannot fill the grid on its own: at
    `[1536, 2560]` the split measured 0.79x and *less* accurate, so the guard is
    that the split has to buy at least two reduction groups. Returns None rather
    than falling back itself, so the caller keeps its own kwargs.
    """
    if _NO_KSPLIT:
        return None
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
    reduced = fused_group_sum(out, key=key)
    return reduced if reduced is not None else ttnn.sum(out, dim=1, keepdim=True)


_KSG_OUT: dict = {}
_KSG_FELL_BACK = False
_NO_KSGEMV = bool(os.environ.get("TT_NO_KSGEMV"))
# Bisection handles, off in every normal run: TT_KSG_NOMCAST makes every core
# read its own activation tiles instead of taking the group head's multicast,
# and TT_KSG_NOFOLD has each group write its own partial and skips the
# cross-core handshake. A hang that survives both is in the matmul.
_KSG_NOMCAST = bool(os.environ.get("TT_KSG_NOMCAST"))
# The partials go to DRAM and `fused_group_sum` adds them in its own launch.
#
# TT_KSG_FOLD=1 folds them inside the kernel instead -- every group writes its
# partial into the gatherer's L1, a semaphore counts them, a second compute
# window adds them -- and that is 10.29 -> 8.91 us on [2560, 352], bit-identical
# (3.291e-03 against float64 either way), about +0.35 ms a token. It is off
# because it hangs **in the model** and nowhere else: ninety-six distinct
# fold programs with an all_reduce between them capture and replay fine
# (`ksg_trace.py 96 ar`), and twenty-five reps of one program in a trace are
# fine, but the 48-layer step hangs with no device program event after the
# first. Whatever the interaction is, it is not the fold's own handshake, and
# 0.35 ms did not justify more 25-minute cycles to find it.
_KSG_FOLD = os.environ.get("TT_KSG_FOLD", "0") == "1"
# `ksplit_linear` and `ksgemv` split the same reduction; the difference is that
# `ksgemv` has the group's head **multicast** the activation while `ksplit_linear`
# leaves every core to read it. At M = 1 that is not a detail -- a group of
# sixteen cores each reading the same 80-tile activation moves as many bytes as
# the weight slice they are there to multiply. This flag moves the three
# remaining `ksplit_linear`/`fast_linear` call sites (the MoE router, the shared
# expert's gate|up, and ssm_alpha|beta) onto `ksgemv`. Off until measured.
# A comma list of sites rather than a boolean, because `TT_KSG_WIDE=1` hangs in
# **traced replay** five times out of five while every one of its three shapes
# runs clean outside a trace (`ksgemv_check.py`, 45.8). That is trace-specific and
# needs bisecting per site: `TT_KSG_WIDE=router`, `=shexp`, `=ab`, or any comma
# combination. `1` still means all three.
_KSG_WIDE_SITES = frozenset(
    s.strip() for s in os.environ.get("TT_KSG_WIDE", "").split(",") if s.strip())
if "1" in _KSG_WIDE_SITES:
    _KSG_WIDE_SITES = frozenset({"router", "shexp", "ab", "qkv", "indexer"})


def ksg_wide(site: str) -> bool:
    """Is the k-split wired at this call site? See `_KSG_WIDE_SITES`."""
    return site in _KSG_WIDE_SITES


_KSG_WIDE = bool(_KSG_WIDE_SITES)
_KSG_SEM = int(os.environ.get("TT_KSG_SEM", "2"))
_KSG_ROWS = int(os.environ.get("TT_KSG_ROWS", "0"))
_KSG_PARTIAL = os.environ.get("TT_KSG_PARTIAL") == "1"


def _ksgemv_plan(grid, kt, nt):
    """Cores as (k-group, output tile), each group a rectangle. Or None.

    A group has to be a rectangle because the head multicasts the activation to
    it. Two shapes of rectangle cover everything the model runs: when `nt` fits
    inside the grid width, `nt` consecutive cores of one row -- so a 1-tile
    output gets one group a core and eleven groups a row -- and otherwise whole
    rows, with the cores past `nt` in the rectangle for the handshake only.
    """
    # `grid.x % nt == 0`, restored. Dropping it packed `grid.x // nt` groups into
    # a row, which uses more cores per row -- and leaves the *rest of the grid*
    # idle, which is fatal until the idle-core bug is fixed in the kernel: at
    # nt = 4 it turned 10 groups x 11 cores (all 110) into 20 x 4 (80), and the
    # indexer then hung where it had been fine. Handoff 45.10.
    if _KSG_ROWS == 0 and nt <= grid.x and grid.x % nt == 0:
        per_row = grid.x // nt
        cores_pg = nt
        groups = min(kt, per_row * grid.y)

        def where(g, j):
            return ttnn.CoreCoord((g % per_row) * nt + j, g // per_row)
    else:
        rows_pg = max(_KSG_ROWS, -(-nt // grid.x))
        cores_pg = rows_pg * grid.x
        groups = min(kt, grid.y // rows_pg)

        def where(g, j):
            return ttnn.CoreCoord(j % grid.x, g * rows_pg + j // grid.x)

    if groups < 2:
        return None
    # **Every core, or none of them.** A core outside every group runs the kernel
    # with all-zero runtime args, so its `is_head` is 0, it takes the reader's
    # non-head path and increments the `ready` semaphore of whatever sits at
    # (0, 0) -- group 0's head, which then multicasts before its real members
    # have armed. Measured: plans covering all 110 cores run (hc_down 10 x 11,
    # router and qkv 5 x 22); plans leaving cores idle hang four times out of
    # four (shexp 2 x 44 = 88, ssm_ab 80 x 1 = 80, indexer 20 x 4 = 80).
    #
    # Declining is the conservative half of the fix -- the caller falls back to
    # `ttnn.linear`, which is slower but correct. The real fix is a "not in any
    # group" runtime flag and an early return in the three kernels, keeping one
    # whole-grid core range; a range *per group* is not it, because tt-metal
    # allocates CBs and semaphores per range and the multicast then writes to
    # offsets the members do not share. That variant hung the shipped hc_down
    # path five runs out of five.
    # `TT_KSG_PARTIAL=1` lifts this. The reader now has an `in_group` guard, so a
    # partial plan *should* be safe -- this is the flag that tests whether it is.
    if not _KSG_PARTIAL and groups * cores_pg != grid.x * grid.y:
        return None
    return cores_pg, groups, where


def _ksgemv_program(a, w, out, kt, nt, plan):
    dev = a.device()
    grid = dev.compute_with_storage_grid_size()
    cores_pg, groups, where = plan
    acc = {}
    for tag, t in (("a", a), ("w", w), ("o", out)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"k-split gemv: {tag} must be interleaved")
        acc[tag] = ct

    # The whole grid, one range -- **not** one range a group. Restricting the
    # core ranges to the cores with work is the right idea and it is how the
    # idle-core bug below has to be fixed, but a ten-range CoreRangeSet made the
    # *shipped* hc_down path hang five runs out of five where one whole-grid
    # range had always been fine: tt-metal allocates circular buffers and
    # semaphores per core range, so a group's members no longer share the L1
    # offsets that the head's `noc_async_write_multicast_loopback_src` assumes.
    #
    # The bug this was trying to fix is real and is described in handoff 45.10:
    # a core outside every group gets all-zero runtime args, so `is_head` is 0,
    # it takes the reader's non-head path and increments the `ready` semaphore of
    # whatever sits at (0, 0) -- group 0's head. The fix has to be an explicit
    # "not in any group" runtime flag and an early return in the three kernels,
    # keeping one core range. Until that lands, only plans that cover all 110
    # cores are safe, which is what `_ksgemv_plan` must therefore produce.
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    all_cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    d0 = dev.get_devices()[0] if hasattr(dev, "get_devices") else dev

    idle_r = [0] * 14        # trailing 0 = not in any group
    idle_c = [0, 0, 0, groups]
    idle_w = [0] * 8
    r_args = {c: idle_r for c in all_cores}
    c_args = {c: idle_c for c in all_cores}
    w_args = {c: idle_w for c in all_cores}
    cap = 0
    for g in range(groups):
        lo, hi = (kt * g) // groups, (kt * (g + 1)) // groups
        cap = max(cap, hi - lo)
        head = where(g, 0)
        ph = d0.worker_core_from_logical_core(head)
        c0 = d0.worker_core_from_logical_core(head)
        c1 = d0.worker_core_from_logical_core(where(g, cores_pg - 1))
        for j in range(cores_pg):
            c = where(g, j)
            active = int(j < nt)
            gath = d0.worker_core_from_logical_core(where(0, j))
            r_args[c] = [a.buffer_address(), w.buffer_address(), lo, hi - lo, j,
                         active, int(j == 0), c0.x, c0.y, c1.x, c1.y, ph.x, ph.y,
                         1]                       # in_group
            c_args[c] = [hi - lo, active, int(g == 0 and active), groups]
            w_args[c] = [out.buffer_address(), j, active, int(g == 0 and active),
                         groups, g, gath.x, gath.y]

    def kern(name, ct, args, cfgd):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, args[c]) for c in all_cores], config=cfgd)

    # Not float32: packing into a Float32 circular buffer hangs the card here,
    # and the launch this replaces wrote bfloat16 partials to DRAM anyway.
    part_dt, part_page = a.dtype, acc["a"][1]
    cbs = [
        ttnn.CBDescriptor(
            total_size=cap * acc["a"][1], core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=0, data_format=a.dtype, page_size=acc["a"][1])]),
        ttnn.CBDescriptor(
            total_size=cap * acc["w"][1], core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=1, data_format=w.dtype, page_size=acc["w"][1])]),
        ttnn.CBDescriptor(
            total_size=part_page, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=2, data_format=part_dt, page_size=part_page)]),
        ttnn.CBDescriptor(
            total_size=groups * part_page, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=3, data_format=part_dt, page_size=part_page)]),
        ttnn.CBDescriptor(
            total_size=2 * acc["o"][1], core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=16, data_format=out.dtype, page_size=acc["o"][1])]),
    ]
    return ttnn.ProgramDescriptor(kernels=[
        kern("ksgemv_reader.cpp",
             [nt, acc["a"][1], acc["w"][1], 1 if _KSG_NOMCAST else cores_pg]
             + acc["a"] + acc["w"],
             r_args, ttnn.ReaderConfigDescriptor()),
        kern("ksgemv_compute.cpp", [1 if _KSG_FOLD else 0], c_args,
             ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                          fp32_dest_acc_en=True)),
        kern("ksgemv_writer.cpp",
             [part_page, 1 if _KSG_FOLD else 0, _KSG_SEM, nt] + acc["o"], w_args,
             ttnn.WriterConfigDescriptor()),
    ], semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0)
                   for i in range(max(2, _KSG_SEM + 1))], cbs=cbs)


def ksgemv(x, w, key=None):
    """`x @ w` with the reduction split across cores. Or None.

    `ttnn.linear` at M = 1 gives each output tile to one core, so a narrow output
    leaves the grid idle: `[2560, 352]` is eleven tiles and runs at 19.6 % of
    bandwidth, ninety-six times a token. This splits the reduction instead, folds
    the partials in the kernel rather than with a `ttnn.sum`, and has the group's
    head multicast the activation -- which at M = 1 is not a detail, since a
    padded activation tile duplicated across a group is larger than the weight.
    """
    global _KSG_FELL_BACK
    if _NO_KSGEMV:
        return None
    try:
        why = None
        if len(x.shape) != 4 or x.shape[0] != 1 or x.shape[1] != 1:
            why = f"x is {list(x.shape)}; this path is [1, 1, M, K]"
        elif x.shape[-2] > _TILE:
            why = f"x has {x.shape[-2]} rows; this path is one tile of rows"
        elif int(x.shape[-1]) != int(w.shape[-2]):
            why = f"x is {x.shape[-1]} wide, w reduces {w.shape[-2]}"
        elif int(w.shape[-2]) % _TILE or int(w.shape[-1]) % _TILE:
            why = f"w is {list(w.shape)}; both axes must be whole tiles"
        if why is not None:
            if not _KSG_FELL_BACK:
                _KSG_FELL_BACK = True
                _declined("k-split gemv", f"{why}")
            return None

        dev = x.device()
        kt, nt = int(w.shape[-2]) // _TILE, int(w.shape[-1]) // _TILE
        plan = _ksgemv_plan(dev.compute_with_storage_grid_size(), kt, nt)
        if plan is None:
            return None                # the output already fills the grid

        groups = plan[1]
        okey = (key, id(dev), int(w.shape[-1]), str(x.dtype), groups,
                int(x.shape[-2]))
        out = _KSG_OUT.get(okey)
        if out is None:
            out = ttnn.from_torch(
                torch.zeros(1, 1 if _KSG_FOLD else groups, int(x.shape[-2]),
                            int(w.shape[-1])),
                dtype=x.dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                mesh_mapper=ttnn.ReplicateTensorToMesh(dev))
            _KSG_OUT[okey] = out
        ttnn.generic_op([x, w, out], _ksgemv_program(x, w, out, kt, nt, plan))
        if _KSG_FOLD:
            return out
        reduced = fused_group_sum(out, key=okey)
        return reduced if reduced is not None else ttnn.sum(out, dim=1, keepdim=True)
    except Exception as exc:                                        # noqa: BLE001
        if not _KSG_FELL_BACK:
            _KSG_FELL_BACK = True
            import warnings
            warnings.warn(f"k-split gemv unavailable, using ttnn.linear: "
                          f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        return None


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

_OUTER_OUT: dict = {}
_OUTER_FELL_BACK = False
# Off: the outer product needs a column broadcast and a row broadcast, and a
# kernel that uses both hangs even with a full `init_bcast` before each --
# each alone runs (TT_OUTER_STAGE 1 and 4). Set TT_FUSED_OUTER_ADD=1 to try
# it again if a single-broadcast formulation is found.
_NO_FUSED_OUTER_ADD = os.environ.get("TT_FUSED_OUTER_ADD", "0") != "1"
# Bisection handle: 1 stops after the column broadcast, 2 after the row one.
_OUTER_STAGE = int(os.environ.get("TT_OUTER_STAGE", "3"))


def _outer_ones(dev, dtype):
    """A tile of ones, once per (device, dtype).

    The outer product needs kt broadcast down the columns and delta across the
    rows, and the FPU does one broadcast at a time, so kt's single column is
    turned into a full tile against this first.
    """
    key = ("ones", id(dev), str(dtype))
    got = _OUTER_OUT.get(key)
    if got is None:
        got = ttnn.from_torch(torch.ones(1, 1, _TILE, _TILE), dtype=dtype,
                              layout=ttnn.TILE_LAYOUT, device=dev,
                              mesh_mapper=ttnn.ReplicateTensorToMesh(dev))
        _OUTER_OUT[key] = got
    return got


def _outer_add_program(decayed, kt, delta, ones, state, dkt, dvt):
    dev = decayed.device()
    grid = dev.compute_with_storage_grid_size()
    acc = {}
    for tag, t in (("d", decayed), ("k", kt), ("v", delta), ("n", ones), ("s", state)):
        ct = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if len(ct) != 2:
            raise RuntimeError(f"fused outer add: {tag} must be interleaved")
        acc[tag] = ct
    if acc["k"][1] != acc["n"][1] or acc["k"][1] != acc["v"][1]:
        raise RuntimeError("fused outer add: kt, delta and the ones tile share a page")
    if acc["d"][1] != acc["s"][1]:
        raise RuntimeError("fused outer add: decayed and state share a page")

    total = int(state.shape[0]) * dkt * dvt
    n = max(1, min(total, grid.x * grid.y))
    cols = min(n, grid.x)
    rows = (n + grid.x - 1) // grid.x
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])
    cores = [ttnn.CoreCoord(cx, cy) for cy in range(rows) for cx in range(cols)]
    work = [((total * i) // len(cores), (total * (i + 1)) // len(cores))
            for i in range(len(cores))]

    def kern(name, ct, args, cfgd):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct,
            runtime_args=[(c, v) for c, v in zip(cores, args)], config=cfgd)

    small, big = acc["k"][1], acc["d"][1]
    cbs = [ttnn.CBDescriptor(
        total_size=sz * pg, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(
            buffer_index=i, data_format=dt, page_size=pg)])
        for i, sz, pg, dt in ((0, 2, big, decayed.dtype), (1, 2, small, kt.dtype),
                              (2, 2, small, delta.dtype), (3, 1, small, ones.dtype),
                              (4, 2, small, kt.dtype), (5, 2, big, decayed.dtype),
                              (16, 2, big, state.dtype))]

    r_args = [[decayed.buffer_address(), kt.buffer_address(), delta.buffer_address(),
               ones.buffer_address(), lo, hi - lo] for lo, hi in work]
    return ttnn.ProgramDescriptor(kernels=[
        kern("outer_add_reader.cpp",
             [big, dkt, dvt] + acc["d"] + acc["k"] + acc["v"] + acc["n"], r_args,
             ttnn.ReaderConfigDescriptor()),
        kern("outer_add_compute.cpp", [_OUTER_STAGE], [[hi - lo] for lo, hi in work],
             ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        kern("outer_add_writer.cpp", [big] + acc["s"],
             [[state.buffer_address(), lo, hi - lo] for lo, hi in work],
             ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs)


def fused_outer_add(decayed, kt, delta, state, key=None):
    """`state = decayed + kt (x) delta` in one launch. True, or False if declined.

    Today `ttnn.multiply(kt, delta)` materialises a full state-sized tensor --
    786 KB at batch 1 -- that the following `add` reads once and nothing else
    ever looks at. Thirty-six layers a token is 57 MB of DRAM whose only job is
    to carry a value between two launches. This is invariant 101's own
    prescription: fuse to stop moving the same bytes twice.

    No semaphores and no cross-core anything, so unlike the k-split's fold this
    cannot hang the card.
    """
    global _OUTER_FELL_BACK
    if _NO_FUSED_OUTER_ADD:
        return False
    try:
        why = None
        ds, ks, vs = list(decayed.shape), list(kt.shape), list(delta.shape)
        if ds != list(state.shape):
            why = f"state is {list(state.shape)}, decayed {ds}"
        elif len(ds) != 4 or ds[1] != 1:
            why = f"decayed is {ds}; this path is [BH, 1, Dk, Dv]"
        elif ks[:2] != ds[:2] or ks[2] != ds[2] or ks[3] != 1:
            why = f"kt is {ks}, expected {ds[:3] + [1]}"
        elif vs[:2] != ds[:2] or vs[2] != 1 or vs[3] != ds[3]:
            why = f"delta is {vs}, expected {ds[:2] + [1, ds[3]]}"
        elif kt.dtype != delta.dtype:
            why = f"kt is {kt.dtype} and delta {delta.dtype}; one broadcast format"
        elif ds[2] % _TILE or ds[3] % _TILE:
            why = f"state is {ds}; Dk and Dv must be whole tiles"
        if why is not None:
            if not _OUTER_FELL_BACK:
                _OUTER_FELL_BACK = True
                _declined("fused outer add", f"{why}")
            return False
        dev = decayed.device()
        ttnn.generic_op(
            [decayed, kt, delta, _outer_ones(dev, kt.dtype), state],
            _outer_add_program(decayed, kt, delta, _outer_ones(dev, kt.dtype),
                               state, ds[2] // _TILE, ds[3] // _TILE))
        return True
    except Exception as exc:                                        # noqa: BLE001
        if not _OUTER_FELL_BACK:
            _OUTER_FELL_BACK = True
            import warnings
            warnings.warn(f"fused outer add unavailable, using the ops: "
                          f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        return False


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
    # Seven launches -- the norm's six and the partition behind it -- become
    # three. The reshapes existed only because `ttnn.rms_norm` reduces over the
    # last axis, and the partition hands device d exactly the group this kernel
    # is already writing. 27.85 us to 21.45 (`group_norm_check.py`).
    fused = fused_group_norm(hyper, as_dtype(norm_w, hyper.dtype), eps,
                             hidden_size, hc_count, key=("grm", id(norm_w)))
    normed, local_pre = fused if fused is not None else (None, None)
    if normed is None:
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
    local = local_pre if local_pre is not None else ttnn.mesh_partition(normed, dim=-1)
    if inject_w is None:
        part = linear_rows(local, down_w, compute_kernel_config=HIFI4)
    else:
        # One matmul for both: see `down_inject_weight`.
        # Measured here: 2.12x and more accurate than `ttnn.linear` (invariant
        # 57). `ksplit_linear` returns None if the split would not pay.
        fused_w = down_inject_weight(down_w, inject_w, span)
        # [2560, 352] is eleven output tiles, so `ttnn.linear` runs it on eleven
        # of a hundred and ten cores -- 19.6 % of bandwidth, and the worst line
        # in the linear census. `ksgemv` splits the reduction instead.
        part = ksgemv(local, fused_w, key=("grm_down", id(fused_w)))
        if part is None:
            part = ksplit_linear(local, fused_w)
        if part is None:
            part = linear_rows(local, fused_w, compute_kernel_config=HIFI4)
    whole = all_reduce(part)
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
