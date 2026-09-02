"""Gated DeltaNet on device via ttnn.transformer.gated_delta_attn_seq.

The op implements the expensive half of the chunked gated delta rule: blocked
forward substitution against a unit-diagonal lower-triangular system, plus the
sequential inter-chunk state scan. Everything up to that -- the decays, the
intra-chunk attention, and the triangular system itself -- is prepared here.

Hard constraints from the device op (validated in its device_operation.cpp):
chunk_size == key_dim == val_dim == 128. Qwen3.8-Flash-Next uses
linear_key_head_dim == linear_value_head_dim == 128, so only the chunk size is a
choice, and it is forced to 128.

The prep mirrors reference/layers.py::chunk_gated_delta_rule exactly, including
forming every decay by subtraction in log space rather than as a ratio of
exponentials (see iteration 004 -- the ratio is 0/0 for the head with A = -158).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..reference.layers import l2norm

CHUNK = 128
BLOCK = 32  # L_inv holds one inverse per 32x32 diagonal block


def _merge_mask(size: int, s: int) -> torch.Tensor:
    """The lower-left quadrant of every `s`-sized diagonal block."""
    idx = torch.arange(size)
    row, col = idx[:, None], idx[None, :]
    return ((row // s) == (col // s)) & ((row % s) >= s // 2) & ((col % s) < s // 2)


def block_diag_inverse(unit_lower: torch.Tensor, block: int) -> torch.Tensor:
    """Invert every `block`-sized diagonal block of a unit lower triangular matrix.

    The result is block diagonal at `block` -- entries outside those blocks are
    exactly zero -- so each block's inverse reads straight off the diagonal, and
    all of them are produced in one pass rather than one per block.

    Blocked, not a Neumann series. Doubling the block size from 1 up to `block`,
    write the matrix at each level as ``D + C``: ``D`` the diagonal blocks, whose
    inverse is already in hand, and ``C`` the newly included lower-left
    quadrants. ``C`` maps the high half of a block to the low half, so
    ``(D^-1 C)^2 == 0`` and

        (D + C)^-1 == D^-1 - D^-1 C D^-1

    holds *exactly*, in two matmuls per level. Nothing grows: every intermediate
    is the inverse of a unit-triangular submatrix, bounded by the answer.

    That last property is the reason this replaced the telescoping Neumann sum
    ``sum_k (-N)^k``. The sum is algebraically correct and unstable on real
    data: with off-diagonal entries near 1 -- head 7 of layer 0 reaches 0.983 --
    ``N^16`` for a 32x32 block grows to ~1e8 before nilpotency cancels it back
    to an answer bounded by 1. Float32 survived that at 0.27 absolute error,
    quietly; on device, where a matmul carries ~1e-3 relative error rather than
    ~1e-7, the same cancellation left 6209.

    Still all matmuls and adds, so it ports to ttnn unchanged -- which is what
    lets the whole `gated_delta_attn_seq` preparation run on device instead of
    round-tripping ~30 MB per layer per chunk over PCIe. See
    `block_diag_inverse_device`.
    """
    size = unit_lower.shape[-1]
    if block & (block - 1) or size % block:
        raise ValueError(f"block must be a power of two dividing {size}, got {block}")
    eye = torch.eye(size, dtype=unit_lower.dtype, device=unit_lower.device)
    inv = eye.expand_as(unit_lower).clone()
    s = 2
    while s <= block:
        mask = _merge_mask(size, s).to(unit_lower.dtype).to(unit_lower.device)
        inv = inv - inv @ (unit_lower * mask) @ inv
        s *= 2
    return inv


def block_inverse(unit_lower: torch.Tensor) -> torch.Tensor:
    """Invert a unit-diagonal lower-triangular matrix without `linalg.inv`."""
    size = unit_lower.shape[-1]
    if size & (size - 1):
        raise ValueError(f"block size must be a power of two, got {size}")
    return block_diag_inverse(unit_lower, size)


def prepare(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the eight device inputs from (q, k, v, g, beta).

    q/k: [B, S, H, 128]; v: [B, S, H, 128]; g/beta: [B, S, H]
    Returns tensors shaped [BH, NC, ...] as the op expects.
    """
    batch, seq, heads, k_dim = key.shape
    v_dim = value.shape[-1]
    assert k_dim == v_dim == CHUNK, f"device op requires dims of {CHUNK}, got {k_dim}/{v_dim}"

    q, k, v, b, decay = (
        x.transpose(1, 2).to(torch.float32).contiguous() for x in (query, key, value, beta, g)
    )
    q = l2norm(q, dim=-1)
    k = l2norm(k, dim=-1)
    q = q * (k_dim**-0.5)

    pad = (CHUNK - seq % CHUNK) % CHUNK
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        b, decay = (F.pad(x, (0, pad)) for x in (b, decay))
    total = seq + pad
    n_chunks = total // CHUNK

    v_beta = v * b.unsqueeze(-1)
    k_beta = k * b.unsqueeze(-1)
    q, k, k_beta, v_beta = (
        x.reshape(batch, heads, n_chunks, CHUNK, x.shape[-1]) for x in (q, k, k_beta, v_beta)
    )
    decay = decay.reshape(batch, heads, n_chunks, CHUNK)
    cum_decay = decay.cumsum(dim=3)

    strict_upper = torch.ones(CHUNK, CHUNK, dtype=torch.bool).triu(1)
    pairwise = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).masked_fill(strict_upper, float("-inf")).exp()

    w = (k_beta @ k.transpose(-1, -2)) * pairwise
    eye = torch.eye(CHUNK, dtype=torch.bool)
    # L is unit-diagonal lower triangular: solving L @ X = RHS reproduces the
    # forward substitution the reference does explicitly.
    strictly_lower = w.masked_fill(strict_upper | eye, 0.0)
    l_unit = torch.eye(CHUNK, dtype=torch.float32) + strictly_lower

    intra_attn = (q @ k.transpose(-1, -2)) * pairwise
    k_bd_sc = k_beta * cum_decay.exp().unsqueeze(-1)
    q_decay = q * cum_decay.exp().unsqueeze(-1)
    k_decay = k * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    dl_exp = cum_decay[..., -1].exp()[..., None, None]

    # Per-32x32 diagonal block inverse, packed as [.., CHUNK, 32]: rows
    # 32b..32b+31 hold the inverse of diagonal block b.
    diag_inv = block_diag_inverse(l_unit, BLOCK)
    l_inv = torch.cat(
        [
            diag_inv[..., b * BLOCK : (b + 1) * BLOCK, b * BLOCK : (b + 1) * BLOCK]
            for b in range(CHUNK // BLOCK)
        ],
        dim=-2,
    )

    def flat(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(batch * heads, *t.shape[2:]).contiguous()

    return {
        "L_unit": flat(l_unit),
        "v_beta_sc": flat(v_beta),
        "k_bd_sc": flat(k_bd_sc),
        "intra_attn": flat(intra_attn),
        "q_decay": flat(q_decay),
        "k_decay_t": flat(k_decay.transpose(-1, -2).contiguous()),
        "dl_exp": flat(dl_exp),
        "L_inv": flat(l_inv),
        "_meta": torch.tensor([batch, heads, n_chunks, seq, pad]),
    }


# --------------------------------------------------------------------------
# on-device preparation
# --------------------------------------------------------------------------
# `prepare` above round-trips ~30 MB per layer per 128-token chunk over PCIe,
# and -- worse for tracing -- does it in host code, which a ttnn trace capture
# cannot see. The functions below are the same arithmetic in ttnn ops, working
# in the op's own [H, NC, C, D] layout so nothing has to be gathered: each
# device prepares its own heads in place.

_CONSTS: dict[tuple[int, int, int], dict[str, "ttnn.Tensor"]] = {}


def _consts(mesh, size: int, heads: int, n_chunks: int) -> dict:
    """Masks and identities for a chunk width, built once per (mesh, shape).

    Broadcasting a [1, 1, C, C] mask against [H, NC, C, C] is not something
    every ttnn binary op does, so these are materialised at full rank. They are
    a few MB and live for the process.
    """
    import ttnn

    key = (id(mesh), size, heads * n_chunks)
    hit = _CONSTS.get(key)
    if hit is not None:
        return hit
    rows = torch.arange(size)
    strict_upper = rows[None, :] > rows[:, None]
    strict_lower = rows[None, :] < rows[:, None]

    def dev(t):
        t = t.to(torch.float32).reshape(1, 1, size, size).expand(heads, n_chunks, size, size)
        return ttnn.from_torch(
            t.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh
        )

    # -1e30 rather than -inf: added before the exp, it underflows to exactly
    # zero for any finite operand, and unlike -inf it cannot produce a nan if
    # the unmasked difference is itself infinite.
    out = {
        "neg_upper": dev(torch.where(strict_upper, -1e30, 0.0)),
        "keep_lower": dev(strict_lower.float()),
        "eye": dev(torch.eye(size)),
    }
    s = 2
    while s <= BLOCK:
        out[f"merge{s}"] = dev(_merge_mask(size, s).float())
        s *= 2
    _CONSTS[key] = out
    return out


def block_diag_inverse_device(unit_lower, consts, block: int):
    """`block_diag_inverse` in ttnn ops. See that function for the derivation.

    Runs on the full CHUNK-wide matrix, so one pass produces all four 32x32
    diagonal inverses at once -- ten matmuls total, where inverting each block
    separately would have dispatched forty.
    """
    import ttnn

    kern = _HIFI4()
    inv = consts["eye"]
    s = 2
    while s <= block:
        c = ttnn.multiply(unit_lower, consts[f"merge{s}"])
        step = ttnn.matmul(ttnn.matmul(inv, c, compute_kernel_config=kern), inv,
                           compute_kernel_config=kern)
        inv = ttnn.subtract(inv, step)
        s *= 2
    return inv


def _HIFI4():
    import ttnn

    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def _l2norm_device(x, eps: float = 1e-6):
    import ttnn

    sq = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
    return ttnn.multiply(x, ttnn.rsqrt(ttnn.add(sq, eps)))


def prepare_device(query, key, value, g, beta, *, mesh):
    """Build the eight device inputs without leaving the device.

    q/k/v: [H, NC, C, 128] float32; g/beta: [H, NC, C, 1]. H is this device's
    own head count and beta is already through its sigmoid, matching what
    `prepare` receives after the host gather. Returns the same eight names,
    already shaped as the op expects, so no host padding step is needed: the
    caller pads the sequence to a multiple of CHUNK when it lays out the chunks.
    """
    import ttnn

    heads, n_chunks, size, k_dim = key.shape
    assert k_dim == value.shape[-1] == CHUNK, f"device op requires dims of {CHUNK}, got {k_dim}"
    assert size == CHUNK, f"chunk width must be {CHUNK}, got {size}"
    kern = _HIFI4()
    c = _consts(mesh, CHUNK, heads, n_chunks)

    q = ttnn.multiply(_l2norm_device(query), CHUNK**-0.5)
    k = _l2norm_device(key)

    v_beta = ttnn.multiply(value, beta)
    k_beta = ttnn.multiply(k, beta)

    cum = ttnn.cumsum(g, dim=-2)                                  # [H, NC, C, 1]
    # pairwise[i, j] = exp(cum[i] - cum[j]) for j <= i, else 0. Masked in log
    # space before the exp, as the reference does.
    diff = ttnn.subtract(cum, ttnn.transpose(cum, -2, -1))         # [H, NC, C, C]
    pairwise = ttnn.exp(ttnn.add(diff, c["neg_upper"]))

    k_t = ttnn.transpose(k, -2, -1)
    w = ttnn.multiply(ttnn.matmul(k_beta, k_t, compute_kernel_config=kern), pairwise)
    l_unit = ttnn.add(c["eye"], ttnn.multiply(w, c["keep_lower"]))
    intra_attn = ttnn.multiply(ttnn.matmul(q, k_t, compute_kernel_config=kern), pairwise)

    cum_exp = ttnn.exp(cum)
    last = ttnn.slice(cum, (0, 0, CHUNK - 1, 0), (heads, n_chunks, CHUNK, 1))
    k_bd_sc = ttnn.multiply(k_beta, cum_exp)
    q_decay = ttnn.multiply(q, cum_exp)
    k_decay = ttnn.multiply(k, ttnn.exp(ttnn.subtract(last, cum)))
    dl_exp = ttnn.exp(last)

    diag_inv = block_diag_inverse_device(l_unit, c, BLOCK)
    blocks = [
        ttnn.slice(
            diag_inv, (0, 0, b * BLOCK, b * BLOCK), (heads, n_chunks, (b + 1) * BLOCK, (b + 1) * BLOCK)
        )
        for b in range(CHUNK // BLOCK)
    ]

    return {
        "L_unit": l_unit,
        "v_beta_sc": v_beta,
        "k_bd_sc": k_bd_sc,
        "intra_attn": intra_attn,
        "q_decay": q_decay,
        "k_decay_t": ttnn.transpose(k_decay, -2, -1),
        "dl_exp": dl_exp,
        "L_inv": ttnn.concat(blocks, dim=-2),
    }
