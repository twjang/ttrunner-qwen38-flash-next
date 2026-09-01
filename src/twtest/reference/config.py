"""Model configuration, derived from GGUF metadata.

Everything is read from the checkpoint rather than hardcoded, so the same code
loads any qwen4exp GGUF. Where the GGUF and the upstream HF config express the
same quantity differently, the GGUF spelling wins and the mapping is noted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Qwen4ExpConfig:
    # -- core ------------------------------------------------------------
    hidden_size: int
    num_layers: int
    vocab_size: int
    rms_norm_eps: float
    context_length: int

    # -- full attention (QSA) layers -------------------------------------
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    rope_dim: int  # partial rotary: only the first rope_dim dims are rotated
    mrope_section: list[int]
    full_attention_interval: int

    # -- QSA indexer ------------------------------------------------------
    indexer_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int

    # -- linear attention (Gated DeltaNet) layers -------------------------
    linear_num_v_heads: int
    linear_num_k_heads: int
    linear_head_dim: int
    conv_kernel: int

    # -- MoE ---------------------------------------------------------------
    num_experts: int
    num_experts_per_tok: int
    expert_intermediate: int
    shared_expert_intermediate: int

    # -- gated residual / hyper-connections -------------------------------
    hc_count: int
    hc_lowrank: int

    # -- PLE / n-gram embedding -------------------------------------------
    ple_layers: list[int]
    ngram_size: int
    heads_per_ngram: int
    ple_conv_kernel: int
    ple_head_dim: int
    ple_eos_token_id: int
    ngram_layer_multipliers: list[int]
    ngram_head_offsets: list[int]
    ngram_head_vocab_sizes: list[int]

    # Not recorded in the GGUF; Qwen3.8-Flash-Next's config.json sets "sigmoid",
    # which overrides the silu that hidden_act would otherwise imply.
    output_gate_type: str = "sigmoid"

    # -- tokenizer ---------------------------------------------------------
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    chat_template: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    # -- derived -----------------------------------------------------------

    @property
    def hc_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_k_heads * self.linear_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_v_heads * self.linear_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.linear_key_dim + self.linear_value_dim

    @property
    def ngram_heads(self) -> int:
        """One head set per n-gram order from 2..ngram_size."""
        return (self.ngram_size - 1) * self.heads_per_ngram

    def is_full_attention(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self.full_attention_interval == 0

    def layer_type(self, layer_idx: int) -> str:
        return "full_attention" if self.is_full_attention(layer_idx) else "linear_attention"

    @classmethod
    def from_gguf(cls, metadata: dict[str, Any]) -> Qwen4ExpConfig:
        arch = metadata.get("general.architecture")
        if arch != "qwen4exp":
            raise ValueError(f"expected a qwen4exp checkpoint, got architecture {arch!r}")

        def m(key: str, default: Any = None) -> Any:
            full = f"qwen4exp.{key}"
            if full not in metadata and default is None:
                raise KeyError(f"missing required GGUF key {full!r}")
            return metadata.get(full, default)

        # GGUF stores four mrope sections (the last is padding); HF uses three.
        sections = [int(x) for x in m("rope.dimension_sections")]
        while sections and sections[-1] == 0:
            sections.pop()

        return cls(
            hidden_size=int(m("embedding_length")),
            num_layers=int(m("block_count")),
            vocab_size=len(metadata["tokenizer.ggml.tokens"]),
            rms_norm_eps=float(m("attention.layer_norm_rms_epsilon")),
            context_length=int(m("context_length")),
            num_attention_heads=int(m("attention.head_count")),
            num_kv_heads=int(m("attention.head_count_kv")),
            head_dim=int(m("attention.key_length")),
            rope_theta=float(m("rope.freq_base")),
            rope_dim=int(m("rope.dimension_count")),
            mrope_section=sections,
            full_attention_interval=int(m("full_attention_interval")),
            indexer_heads=int(m("attention.indexer.head_count")),
            indexer_head_dim=int(m("attention.indexer.key_length")),
            indexer_budget=int(m("attention.indexer.top_k")),
            # GGUF records one compress ratio per layer; the non-zero entries
            # all belong to full-attention layers and are identical.
            indexer_compress_ratio=int(max(m("attention.compress_ratios"))),
            linear_num_v_heads=int(m("ssm.time_step_rank")),
            linear_num_k_heads=int(m("ssm.group_count")),
            linear_head_dim=int(m("ssm.state_size")),
            conv_kernel=int(m("ssm.conv_kernel")),
            num_experts=int(m("expert_count")),
            num_experts_per_tok=int(m("expert_used_count")),
            expert_intermediate=int(m("expert_feed_forward_length")),
            shared_expert_intermediate=int(m("expert_shared_feed_forward_length")),
            hc_count=int(m("hyper_connection.count")),
            hc_lowrank=int(m("hyper_connection.low_rank")),
            ple_layers=[int(x) for x in m("ple.layers")],
            ngram_size=int(m("ple.ngram_size")),
            heads_per_ngram=int(m("ple.heads_per_ngram")),
            ple_conv_kernel=int(m("ple.conv_kernel")),
            ple_head_dim=int(m("embedding_length_per_layer_input")),
            ple_eos_token_id=int(m("ple.eos_token_id")),
            ngram_layer_multipliers=[int(x) for x in m("ple.layer_multipliers")],
            ngram_head_offsets=[int(x) for x in m("ple.head_offsets")],
            ngram_head_vocab_sizes=[int(x) for x in m("ple.head_vocab_sizes")],
            bos_token_id=metadata.get("tokenizer.ggml.bos_token_id"),
            eos_token_id=metadata.get("tokenizer.ggml.eos_token_id"),
            pad_token_id=metadata.get("tokenizer.ggml.padding_token_id"),
            chat_template=metadata.get("tokenizer.chat_template"),
        )

    def validate(self) -> None:
        """Cross-check the derived quantities that the weight shapes depend on."""
        problems = []
        if self.conv_dim != 10240:
            problems.append(f"conv_dim {self.conv_dim} != 10240 (attn_qkv row count)")
        if self.hc_hidden_size != 10240:
            problems.append(f"hc_hidden_size {self.hc_hidden_size} != 10240")
        if self.ngram_heads * self.ple_head_dim != self.hidden_size:
            problems.append(
                f"{self.ngram_heads} n-gram heads x {self.ple_head_dim} != hidden {self.hidden_size}"
            )
        if len(self.ngram_head_offsets) != self.ngram_heads:
            problems.append(
                f"{len(self.ngram_head_offsets)} head offsets for {self.ngram_heads} heads"
            )
        n_full = sum(self.is_full_attention(i) for i in range(self.num_layers))
        if n_full != self.num_layers // self.full_attention_interval:
            problems.append(f"{n_full} full-attention layers derived")
        if problems:
            raise ValueError("config inconsistent with architecture:\n  " + "\n  ".join(problems))
