"""Convenience entry point: open a checkpoint directory as a running model."""

from __future__ import annotations

from pathlib import Path

from ..gguf.reader import GGUFModel
from .config import Qwen4ExpConfig
from .model import Qwen4ExpModel
from .tokenizer import Qwen4ExpTokenizer
from .weights import WeightStore


def load(
    model_dir: str | Path,
    tokenizer_path: str | Path | None = None,
    cache_bytes: int = 24 << 30,
    quant_sim=None,
) -> tuple[Qwen4ExpModel, Qwen4ExpTokenizer | None, Qwen4ExpConfig]:
    gguf = GGUFModel.from_dir(model_dir)
    config = Qwen4ExpConfig.from_gguf(gguf.metadata)
    config.validate()
    store = WeightStore(gguf, cache_bytes=cache_bytes, quant_sim=quant_sim)
    model = Qwen4ExpModel(config, store)

    tokenizer = None
    if tokenizer_path is not None:
        eos = [config.eos_token_id] if config.eos_token_id is not None else []
        tokenizer = Qwen4ExpTokenizer(tokenizer_path, config.chat_template, eos)
    return model, tokenizer, config
