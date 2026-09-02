"""Shared bring-up for the device harnesses: open the mesh, load the model.

Paths come from the environment so nothing about a machine is hard-coded:

    TWTEST_GGUF_DIR   directory with the UD-IQ4_XS shards   (default ~/models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS)
    TWTEST_TT_CACHE   the converted device-weight cache      (default ~/models/qwen38-tt-cache)
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import ttnn

from twtest.gguf.reader import GGUFModel
from twtest.reference.config import Qwen4ExpConfig
from twtest.reference.weights import WeightStore
from twtest.tt.model import TTModel
from twtest.tt.weights import TTWeights

GGUF_DIR = os.environ.get(
    "TWTEST_GGUF_DIR", str(Path.home() / "models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS")
)
TT_CACHE = os.environ.get("TWTEST_TT_CACHE", str(Path.home() / "models/qwen38-tt-cache"))


def open_model(max_seq_len: int = 512, preload: bool = True, trace_region_bytes: int | None = None,
               num_command_queues: int | None = None):
    """Returns (mesh, cfg, model). Preloading everything but the split expert
    halves takes ~1.5 min and makes the first step's timing honest.

    `trace_region_bytes` mirrors `TTEngine`, which always passes one (128 MB,
    scaled by the number of capture widths). Leaving it None takes ttnn's
    default, which is what every harness here did while the engine's captures
    hung -- so it is a difference worth being able to reproduce, not a knob.
    """
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    kw = {} if trace_region_bytes is None else {"trace_region_size": trace_region_bytes}
    if num_command_queues is not None:
        # Two queues let two traces be issued independently, which is the usual
        # tt-metal answer to "these two captures interfere".
        kw["num_command_queues"] = num_command_queues
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), **kw)
    gguf = GGUFModel.from_dir(GGUF_DIR)
    cfg = Qwen4ExpConfig.from_gguf(gguf.metadata)
    host = WeightStore(gguf, cache_bytes=2 << 30, row_cache_bytes=8 << 30)
    w = TTWeights(TT_CACHE, mesh)
    model = TTModel(cfg, w, host, mesh, max_seq_len=max_seq_len, traceable_kv=True)
    model.fuse_expert_gate_up = "blk.0.ffn_gateup_exps.weight" in w
    if preload:
        for name in w.entries:
            # the split halves are superseded by the fused tensor; loading both
            # is ~11 GB per device over budget
            if "ffn_gate_exps" in name or "ffn_up_exps" in name:
                continue
            w.get(name)
        ttnn.synchronize_device(mesh)
    return mesh, cfg, model


def host_row(mesh, t: ttnn.Tensor) -> torch.Tensor:
    """Device 0's copy of a replicated tensor, as float32."""
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1].float().clone()


def synthetic_prompt(n: int) -> list[int]:
    """Deterministic token ids that avoid special tokens. Not real text."""
    return [1000 + ((i * 37) % 5000) for i in range(n)]


def tokenizer(cfg):
    """The real tokenizer, from TWTEST_TOKENIZER (default ~/models/Qwen3.8-Flash-Next-tokenizer/tokenizer.json)."""
    from twtest.reference.tokenizer import Qwen4ExpTokenizer

    path = os.environ.get(
        "TWTEST_TOKENIZER",
        str(Path.home() / "models/Qwen3.8-Flash-Next-tokenizer/tokenizer.json"),
    )
    return Qwen4ExpTokenizer(
        path, cfg.chat_template, [t for t in (cfg.eos_token_id, 248044) if t is not None]
    )
