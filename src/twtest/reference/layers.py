"""Plain-PyTorch primitives for the qwen4exp reference engine.

Deliberately unoptimised: every operation is written the obvious way so it can
serve as the numerical reference the ttnn engine is checked against.

Three conventions differ from the upstream HF implementation because
llama.cpp's converter bakes them into the checkpoint:

1. Norm weights already include the ``+1`` (converter: ``data_torch + 1`` for
   every ``*norm.weight`` except ``linear_attn.norm.weight``, which upstream
   applies without the offset anyway). So every norm here is a plain
   ``normed * weight``.
2. DeltaNet value heads are stored in *tiled* order ``[v0k0, v0k1, ...]``
   rather than HF's grouped ``[k0v0, k0v1, ...]``, so the query/key heads are
   expanded with ``repeat`` (tile), never ``repeat_interleave``.
3. The QSA indexer's single ``index_qk_proj`` is split into separate
   ``indexer.q_proj`` / ``indexer.k_proj`` tensors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, group_size: int | None = None
) -> torch.Tensor:
    """RMS norm; `group_size` normalises each residual stream independently."""
    dtype = x.dtype
    x = x.float()
    if group_size is not None:
        x = x.reshape(*x.shape[:-1], -1, group_size)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        x = x.flatten(-2)
    else:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dtype)


def rms_norm_gated(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float, activation: str = "sigmoid"
) -> torch.Tensor:
    """RMS norm followed by a gate -- the DeltaNet output norm.

    Qwen3.8-Flash-Next sets ``output_gate_type: "sigmoid"``, which differs from
    the ``hidden_act`` (silu) that this norm would otherwise inherit. The GGUF
    does not record the field at all, so it cannot be read back from the
    checkpoint and is pinned here instead.
    """
    dtype = x.dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = weight.float() * h
    g = torch.sigmoid(gate.float()) if activation == "sigmoid" else F.silu(gate.float())
    return (h * g).to(dtype)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


# --------------------------------------------------------------------------
# rotary embeddings (partial + interleaved mrope)
# --------------------------------------------------------------------------


class RotaryEmbedding:
    """Interleaved multimodal RoPE.

    `position_ids` is (3, batch, seq) for the temporal/height/width axes; for
    text-only input all three rows are identical. Only the first `rope_dim`
    dimensions of each head are rotated (partial_rotary_factor 0.25).
    """

    def __init__(self, rope_dim: int, theta: float, mrope_section: list[int], device=None):
        self.rope_dim = rope_dim
        self.mrope_section = mrope_section
        inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float, device=device) / rope_dim))
        self.inv_freq = inv_freq

    def _interleave(self, freqs: torch.Tensor) -> torch.Tensor:
        """(3, batch, seq, rope_dim/2) -> (batch, seq, rope_dim/2).

        Reorganises [TTT..HHH..WWW] into [THWTHW..TT] so that frequency order
        stays continuous across the three axes.
        """
        out = freqs[0].clone()
        for axis, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[axis] * 3
            idx = slice(offset, length, 3)
            out[..., idx] = freqs[axis][..., idx]
        return out

    def __call__(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        inv = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        pos = position_ids[:, :, None, :].float()
        freqs = (inv @ pos).transpose(2, 3)  # (3, batch, seq, rope_dim/2)
        freqs = self._interleave(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)  # (batch, seq, rope_dim)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """Rotate the leading `cos.shape[-1]` dims of `x`, leaving the rest alone."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    x_rot = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([x_rot, x_pass], dim=-1)


# --------------------------------------------------------------------------
# causal depthwise convolution
# --------------------------------------------------------------------------


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, activation: bool = True) -> torch.Tensor:
    """Depthwise causal conv over (batch, channels, seq)."""
    channels = x.shape[1]
    pad = weight.shape[-1] - 1
    out = F.conv1d(x, weight.unsqueeze(1), padding=pad, groups=channels)[:, :, : x.shape[-1]]
    return F.silu(out) if activation else out


def causal_conv1d_step(x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, activation: bool = True):
    """Single-step conv against a rolling state; returns (output, new_state)."""
    channels = x.shape[1]
    full = torch.cat([state, x], dim=-1)
    new_state = full[:, :, -state.shape[-1] :].clone()
    out = F.conv1d(full, weight.unsqueeze(1), padding=0, groups=channels)[:, :, -x.shape[-1] :]
    return (F.silu(out) if activation else out), new_state


# --------------------------------------------------------------------------
# gated delta rule
# --------------------------------------------------------------------------


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-at-a-time gated delta rule.

    query/key: (batch, seq, n_v_heads, k_dim); value: (batch, seq, n_v_heads, v_dim)
    g/beta:    (batch, seq, n_v_heads);  state: (batch, n_v_heads, k_dim, v_dim)
    """
    dtype = query.dtype
    q, k, v, b, decay = (
        x.transpose(1, 2).to(torch.float32).contiguous() for x in (query, key, value, beta, g)
    )
    q = l2norm(q, dim=-1)
    k = l2norm(k, dim=-1)
    q = q / (q.shape[-1] ** 0.5)

    batch, n_heads, seq, k_dim = k.shape
    v_dim = v.shape[-1]
    state = (
        torch.zeros(batch, n_heads, k_dim, v_dim, dtype=torch.float32, device=v.device)
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    out = torch.zeros_like(v)
    for t in range(seq):
        q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]
        state = state * decay[:, :, t].exp()[..., None, None]
        # read what the state currently predicts for k_t, and correct it
        predicted = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - predicted) * b[:, :, t].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    return out.transpose(1, 2).contiguous().to(dtype), state


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked gated delta rule -- same result as the recurrent form, but the
    within-chunk work becomes matmuls instead of a Python loop over tokens."""
    dtype = query.dtype
    q, k, v, b, decay = (
        x.transpose(1, 2).to(torch.float32).contiguous() for x in (query, key, value, beta, g)
    )
    q = l2norm(q, dim=-1)
    k = l2norm(k, dim=-1)
    q = q * (q.shape[-1] ** -0.5)

    batch, n_heads, seq, k_dim = k.shape
    v_dim = v.shape[-1]

    pad = (chunk_size - seq % chunk_size) % chunk_size
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        b, decay = (F.pad(x, (0, pad)) for x in (b, decay))
    total = seq + pad
    n_chunks = total // chunk_size

    v_beta = v * b.unsqueeze(-1)
    k_beta = k * b.unsqueeze(-1)
    q, k, k_beta, v_beta = (
        x.reshape(batch, n_heads, n_chunks, chunk_size, x.shape[-1]) for x in (q, k, k_beta, v_beta)
    )
    decay = decay.reshape(batch, n_heads, n_chunks, chunk_size)

    strict_upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).triu(1)
    cum_decay = decay.cumsum(dim=3)

    # pairwise_decay[..., i, j] = decay accumulated from j to i within a chunk
    pairwise = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).masked_fill(strict_upper, float("-inf")).exp()

    ut = (k_beta @ k.transpose(-1, -2)) * pairwise
    intra_attn = (q @ k.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # Invert (I + strictly-lower ut) by forward substitution.
    ut = -ut.masked_fill(~strict_upper.T | torch.eye(chunk_size, dtype=torch.bool, device=q.device), 0)
    for i in range(1, chunk_size):
        ut[..., i, :i] = ut[..., i, :i] + (ut[..., i, :, None].clone() * ut[..., :, :i].clone()).sum(-2)
    ut = ut + torch.eye(chunk_size, dtype=ut.dtype, device=q.device)

    k_cumdecay = ut @ decayed_k_beta
    u = ut @ v_beta

    state = (
        torch.zeros(batch, n_heads, k_dim, v_dim, dtype=torch.float32, device=v.device)
        if initial_state is None
        else initial_state.to(torch.float32)
    )

    # Decays are applied once, up front, and every exponent is formed by
    # subtraction in log space. Some heads have A as low as -158, so
    # exp(cum_decay) underflows to exactly 0 within a chunk; forming the
    # position-to-chunk-end decay as a ratio exp(a)/exp(b) then yields 0/0 = NaN,
    # which silently poisons the carried recurrent state (the chunk *output* is
    # unaffected, so it only shows up on the next decode step).
    q_decayed = q * cum_decay.exp().unsqueeze(-1)
    k_decayed = k * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_end_decay = cum_decay[..., -1].exp()[..., None, None]

    out = torch.zeros_like(v.reshape(batch, n_heads, n_chunks, chunk_size, v_dim))
    for c in range(n_chunks):
        v_new = u[:, :, c] - k_cumdecay[:, :, c] @ state
        # contribution of the carried state, decayed to each position
        attn_inter = q_decayed[:, :, c] @ state
        out[:, :, c] = attn_inter + intra_attn[:, :, c] @ v_new
        state = state * chunk_end_decay[:, :, c] + k_decayed[:, :, c].transpose(-1, -2) @ v_new

    out = out.reshape(batch, n_heads, total, v_dim)[:, :, :seq]
    return out.transpose(1, 2).contiguous().to(dtype), state
