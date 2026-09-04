"""Per-sequence decoding state for the hybrid architecture.

A qwen4exp sequence carries four different kinds of state:

* linear-attention layers: a depthwise conv window and a recurrent matrix state
* full-attention layers: the usual K/V cache, plus the QSA indexer's own raw
  key cache (the indexer keys are pooled and RoPE'd at *block* granularity, so
  they must be kept unrotated and unpooled)
* the PLE layer: the last ``ngram_size - 1`` token ids and a dilated conv window
* the model as a whole: the full 3-D position ids, because the indexer needs
  the positions of *all* keys, not just the current ones
"""

from __future__ import annotations

import torch


class LayerCache:
    __slots__ = ("conv_state", "recurrent_state", "keys", "values", "indexer_keys", "ple_conv_state", "ple_tokens")

    def __init__(self) -> None:
        self.conv_state: torch.Tensor | None = None       # (B, conv_dim, kernel-1)
        self.recurrent_state: torch.Tensor | None = None  # (B, n_v_heads, k_dim, v_dim)
        self.keys: torch.Tensor | None = None             # (B, n_kv_heads, S, head_dim)
        self.values: torch.Tensor | None = None
        self.indexer_keys: torch.Tensor | None = None     # (B, S, indexer_head_dim), pre-norm/pre-rope
        self.ple_conv_state: torch.Tensor | None = None   # (B, hc_hidden, state_len)
        self.ple_tokens: torch.Tensor | None = None       # (B, ngram_size-1)


class HybridCache:
    def __init__(self, num_layers: int) -> None:
        self.layers = [LayerCache() for _ in range(num_layers)]
        self.position_ids: torch.Tensor | None = None  # (3, B, S_total)

    def __getitem__(self, idx: int) -> LayerCache:
        return self.layers[idx]

    @property
    def seq_len(self) -> int:
        return 0 if self.position_ids is None else self.position_ids.shape[-1]

    def extend_positions(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Append new positions and return the full history (3, B, S_total)."""
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        if self.position_ids is None:
            self.position_ids = position_ids
        else:
            self.position_ids = torch.cat([self.position_ids, position_ids], dim=-1)
        return self.position_ids

    def reset(self) -> None:
        for layer in self.layers:
            layer.__init__()  # type: ignore[misc]
        self.position_ids = None
