"""Sampling loop for the reference engine."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from .cache import HybridCache
from .model import Qwen4ExpModel


@dataclass(slots=True)
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()


def sample(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator | None) -> int:
    """logits: (vocab,)"""
    if params.temperature <= 0:
        return int(logits.argmax())

    logits = logits.float() / params.temperature
    if params.top_k > 0:
        k = min(params.top_k, logits.shape[-1])
        values, indices = torch.topk(logits, k)
    else:
        values, indices = torch.sort(logits, descending=True)

    probs = torch.softmax(values, dim=-1)
    if 0 < params.top_p < 1:
        cumulative = probs.cumsum(dim=-1)
        # keep the first token that crosses the threshold
        keep = cumulative - probs <= params.top_p
        keep[0] = True
        probs = torch.where(keep, probs, torch.zeros_like(probs))
        probs = probs / probs.sum()

    choice = torch.multinomial(probs, 1, generator=generator)
    return int(indices[choice])


def generate(
    model: Qwen4ExpModel,
    prompt_ids: list[int],
    params: SamplingParams | None = None,
) -> Iterator[int]:
    """Yield generated token ids one at a time."""
    params = params or SamplingParams()
    generator = None
    if params.seed is not None:
        generator = torch.Generator().manual_seed(params.seed)

    cache = HybridCache(model.config.num_layers)
    ids = torch.tensor([prompt_ids], dtype=torch.long)

    hidden = model.forward(ids, cache)
    logits = model.logits(hidden[:, -1])[0]

    for _ in range(params.max_tokens):
        token = sample(logits, params, generator)
        yield token
        if token in params.stop_token_ids:
            return
        ids = torch.tensor([[token]], dtype=torch.long)
        hidden = model.forward(ids, cache)
        logits = model.logits(hidden[:, -1])[0]
