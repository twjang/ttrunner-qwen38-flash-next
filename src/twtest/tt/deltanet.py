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


def block_inverse(unit_lower: torch.Tensor) -> torch.Tensor:
    """Invert a unit-diagonal lower-triangular matrix without `linalg.inv`.

    For strictly lower triangular ``N`` of size ``C`` we have ``N**C == 0``, so

        (I + N)^-1 = sum_{k=0}^{C-1} (-N)^k

    and that geometric sum telescopes into ``log2(C)`` squarings:

        sum_{k=0}^{2^m-1} M^k = prod_{i=0}^{m-1} (I + M^(2^i))

    For C = 32 that is 5 squarings and reproduces `torch.linalg.inv` to ~4e-7.
    The point is that every step is a matmul and an add, so the same routine
    translates directly to ttnn -- which is what lets the whole
    `gated_delta_attn_seq` preparation run on device instead of round-tripping
    ~1.1 GB per 128-token chunk over PCIe.
    """
    size = unit_lower.shape[-1]
    if size & (size - 1):
        raise ValueError(f"block size must be a power of two, got {size}")
    eye = torch.eye(size, dtype=unit_lower.dtype, device=unit_lower.device)
    eye = eye.expand_as(unit_lower)
    acc = eye.clone()
    power = -(unit_lower - eye)          # M = -N, N strictly lower
    for _ in range(size.bit_length() - 1):
        acc = acc + power @ acc
        power = power @ power
    return acc


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
    n_blocks = CHUNK // BLOCK
    blocks = []
    for blk in range(n_blocks):
        sl = slice(blk * BLOCK, (blk + 1) * BLOCK)
        blocks.append(block_inverse(l_unit[..., sl, sl]))
    l_inv = torch.cat(blocks, dim=-2)

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
