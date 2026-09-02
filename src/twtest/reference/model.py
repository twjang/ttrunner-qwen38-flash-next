"""Plain PyTorch CPU reference implementation of qwen4exp (Qwen3.8-Flash-Next).

Weights are pulled from a GGUF checkpoint through `WeightStore`. Dense weights
are dequantised once and cached; the MoE experts and the 51 B-parameter n-gram
table are far too large for that, so those are dequantised per selected row on
each forward pass. That makes this engine slow -- which is fine, its job is to
be obviously correct and serve as the oracle for the ttnn engine.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .cache import HybridCache
from .config import Qwen4ExpConfig
from .layers import (
    RotaryEmbedding,
    apply_rotary,
    causal_conv1d,
    causal_conv1d_step,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
    rms_norm,
    rms_norm_gated,
)
from .weights import WeightStore


class Qwen4ExpModel:
    def __init__(self, config: Qwen4ExpConfig, store: WeightStore):
        self.config = config
        self.store = store
        self.rotary = RotaryEmbedding(config.rope_dim, config.rope_theta, config.mrope_section)
        self._ple_layers = {idx: n for n, idx in enumerate(config.ple_layers)}
        # Debug hook, called as probe(layer, hidden) after every layer. Mirrors
        # `TTModel.probe` exactly (same call site, same tensor) so the two can
        # be diffed layer by layer -- see `scripts/dev/decode_vs_reference.py`.
        self.probe = None

    # -- weight helpers ---------------------------------------------------

    def w(self, name: str) -> torch.Tensor:
        return self.store.get(name)

    def bw(self, layer: int, suffix: str) -> torch.Tensor:
        return self.store.get(f"blk.{layer}.{suffix}")

    # -- gated residual (hyper-connections) -------------------------------

    def _gated_residual(
        self, hyper: torch.Tensor, layer: int | None, kind: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Mix the `hc_count` residual streams down to one hidden vector.

        Returns (mixed, hyper_input_normed_source, injection_weights). For the
        final mixer there is no injection, so the third element is None.
        """
        cfg = self.config
        prefix = "output_hc" if layer is None else f"blk.{layer}.hc_{kind}"
        norm_w = self.w(f"{prefix}_norm.weight")
        down_w = self.w(f"{prefix}_down.weight")
        up_w = self.w(f"{prefix}_up.weight")

        normed = rms_norm(hyper, norm_w, cfg.rms_norm_eps, group_size=cfg.hidden_size)
        mix = F.silu(F.linear(normed, down_w) / cfg.hc_count)
        mix = torch.sigmoid(F.linear(mix, up_w))
        mix = mix.unflatten(-1, (cfg.hc_count, cfg.hidden_size))
        mixed = (mix * normed.unflatten(-1, (cfg.hc_count, cfg.hidden_size))).mean(dim=-2)

        if layer is None:
            return mixed, hyper, None
        inject_w = self.w(f"{prefix}_inject.weight")
        inject = 2 * torch.sigmoid(F.linear(normed, inject_w) / cfg.hc_count)
        return mixed, hyper, inject

    @staticmethod
    def _reinject(hyper: torch.Tensor, branch_out: torch.Tensor, inject: torch.Tensor) -> torch.Tensor:
        return hyper + (branch_out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)

    # -- PLE / n-gram embedding -------------------------------------------

    def _ngram_ids(self, token_ids: torch.Tensor, cache: HybridCache | None, layer: int) -> torch.Tensor:
        """Hash each token's trailing n-grams into `ngram_heads` table rows."""
        cfg = self.config
        context_len = cfg.ngram_size - 1
        eos = cfg.ple_eos_token_id
        batch, seq = token_ids.shape

        if cache is not None and cache[layer].ple_tokens is not None:
            previous = cache[layer].ple_tokens
        else:
            previous = token_ids.new_full((batch, context_len), eos)
        history = torch.cat([previous, token_ids], dim=-1)
        if cache is not None:
            cache[layer].ple_tokens = history[:, -context_len:].clone()

        # An n-gram never reaches across an EOS: positions before the segment
        # start read EOS instead of the previous document's tokens.
        positions = torch.arange(history.shape[1], device=history.device)
        eos_pos = torch.where(history == eos, positions, torch.full_like(positions, -1))
        prev_eos = torch.cummax(eos_pos, dim=1).values
        prev_eos = torch.cat([eos_pos.new_full((batch, 1), -1), prev_eos[:, :-1]], dim=1)
        pos_in_segment = positions.unsqueeze(0) - (prev_eos + 1)

        shifted = []
        for shift in range(cfg.ngram_size):
            if shift == 0:
                shifted.append(history)
                continue
            src = positions - shift
            gathered = history.gather(1, src.clamp_min(0).unsqueeze(0).expand(batch, -1))
            valid = (pos_in_segment >= shift) & (src.unsqueeze(0) >= 0)
            shifted.append(torch.where(valid, gathered, torch.full_like(history, eos)))

        multipliers = cfg.ngram_layer_multipliers
        vocab = torch.tensor(cfg.ngram_head_vocab_sizes, dtype=torch.int64)
        offsets = torch.tensor(cfg.ngram_head_offsets, dtype=torch.int64)

        blocks = []
        for order in range(2, cfg.ngram_size + 1):
            start = (order - 2) * cfg.heads_per_ngram
            end = start + cfg.heads_per_ngram
            mixed = shifted[0] * multipliers[0]
            for pos in range(1, order):
                mixed = torch.bitwise_xor(mixed, shifted[pos] * multipliers[pos])
            ids = torch.remainder(mixed.unsqueeze(-1), vocab[start:end].view(1, 1, -1))
            blocks.append(ids + offsets[start:end].view(1, 1, -1))
        return torch.cat(blocks, dim=-1)[:, -seq:]

    def _ple(
        self, hidden: torch.Tensor, token_ids: torch.Tensor, layer: int, cache: HybridCache | None
    ) -> torch.Tensor:
        cfg = self.config
        batch, seq, _ = hidden.shape

        ngram_ids = self._ngram_ids(token_ids, cache, layer)  # (B, S, ngram_heads)
        flat = ngram_ids.reshape(-1)
        rows = self.store.get_rows("per_layer_token_embd.weight", flat)
        embeddings = rows.reshape(batch, seq, cfg.ngram_heads * cfg.ple_head_dim)

        key = rms_norm(
            F.linear(embeddings, self.bw(layer, "ple_key.weight")),
            self.bw(layer, "ple_norm_key.weight"),
            cfg.rms_norm_eps,
            group_size=cfg.hidden_size,
        ).unflatten(-1, (cfg.hc_count, cfg.hidden_size))
        value = F.linear(embeddings, self.bw(layer, "ple_value.weight"))
        query = rms_norm(
            hidden, self.bw(layer, "ple_norm_query.weight"), cfg.rms_norm_eps, group_size=cfg.hidden_size
        ).unflatten(-1, (cfg.hc_count, cfg.hidden_size))

        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(cfg.hidden_size)
        # signed square root keeps the gate's sign but compresses its magnitude
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_normed = rms_norm(
            gated.flatten(-2), self.bw(layer, "ple_norm_conv.weight"), cfg.rms_norm_eps, group_size=cfg.hidden_size
        )
        gated = gated.flatten(-2)

        conv_w = self.bw(layer, "ple_conv1d.weight")
        state_len = (cfg.ple_conv_kernel - 1) * cfg.ngram_size
        x = gated_normed.transpose(1, 2)
        if cache is not None and cache[layer].ple_conv_state is not None:
            x = torch.cat([cache[layer].ple_conv_state, x], dim=-1)
        x = F.pad(x, (state_len, 0))[:, :, -(state_len + seq) :]
        if cache is not None:
            cache[layer].ple_conv_state = x[:, :, -state_len:].clone() if state_len else x[:, :, :0]
        conv = F.silu(F.conv1d(x, conv_w.unsqueeze(1), groups=conv_w.shape[0], dilation=cfg.ngram_size))
        return gated + conv.transpose(1, 2)

    # -- linear attention (Gated DeltaNet) --------------------------------

    def _linear_attention(
        self, hidden: torch.Tensor, layer: int, cache: HybridCache | None
    ) -> torch.Tensor:
        cfg = self.config
        batch, seq, _ = hidden.shape
        n_v, n_k, hd = cfg.linear_num_v_heads, cfg.linear_num_k_heads, cfg.linear_head_dim

        qkv = F.linear(hidden, self.bw(layer, "attn_qkv.weight")).transpose(1, 2)
        z = F.linear(hidden, self.bw(layer, "attn_gate.weight")).reshape(batch, seq, -1, hd)
        b = F.linear(hidden, self.bw(layer, "ssm_beta.weight"))
        a = F.linear(hidden, self.bw(layer, "ssm_alpha.weight"))

        conv_w = self.bw(layer, "ssm_conv1d.weight")
        state_len = cfg.conv_kernel - 1
        if cache is not None:
            state = cache[layer].conv_state
            if state is None:
                state = qkv.new_zeros(batch, cfg.conv_dim, state_len)
            if seq == 1:
                qkv, new_state = causal_conv1d_step(qkv, state, conv_w)
                cache[layer].conv_state = new_state
            else:
                full = torch.cat([state, qkv], dim=-1)
                cache[layer].conv_state = full[:, :, -state_len:].clone()
                qkv = causal_conv1d(full, conv_w)[:, :, -seq:]
        else:
            qkv = causal_conv1d(qkv, conv_w)

        qkv = qkv.transpose(1, 2)
        key_dim = cfg.linear_key_dim
        query, key, value = torch.split(qkv, [key_dim, key_dim, cfg.linear_value_dim], dim=-1)
        query = query.reshape(batch, seq, n_k, hd)
        key = key.reshape(batch, seq, n_k, hd)
        value = value.reshape(batch, seq, n_v, hd)

        beta = b.sigmoid()
        # The converter stores A = -exp(A_log), not A_log, so there is no
        # exp() here: `if name.endswith(".A_log"): data_torch = -torch.exp(...)`.
        a_decay = self.bw(layer, "ssm_a")
        dt_bias = self.bw(layer, "ssm_dt.bias")
        g = a_decay.float() * F.softplus(a.float() + dt_bias.float())

        # The checkpoint stores V heads tiled over K heads, so the K-side heads
        # are tiled to match -- repeat, not repeat_interleave.
        if n_v > n_k:
            reps = n_v // n_k
            query = query.repeat(1, 1, reps, 1)
            key = key.repeat(1, 1, reps, 1)

        prev_state = cache[layer].recurrent_state if cache is not None else None
        rule = recurrent_gated_delta_rule if seq == 1 else chunk_gated_delta_rule
        out, new_state = rule(query, key, value, g, beta, initial_state=prev_state)
        if cache is not None:
            cache[layer].recurrent_state = new_state

        out = rms_norm_gated(
            out.reshape(-1, hd), z.reshape(-1, hd), self.bw(layer, "ssm_norm.weight"), cfg.rms_norm_eps, cfg.output_gate_type
        ).reshape(batch, seq, -1)
        return F.linear(out, self.bw(layer, "ssm_out.weight"))

    # -- QSA indexer -------------------------------------------------------

    def _indexer_mask(
        self,
        hidden: torch.Tensor,
        layer: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: HybridCache | None,
        kv_len: int,
        query_offset: int,
    ) -> torch.Tensor:
        """Boolean (B, 1, S, kv_len) mask of the tokens QSA lets each query see.

        Keys are averaged into blocks of `compress_ratio`, scored against the
        indexer queries, and the top `budget/compress_ratio` blocks are kept.
        Tokens in the trailing partial block are always visible.
        """
        cfg = self.config
        batch, seq, _ = hidden.shape
        dim = cfg.indexer_head_dim
        ratio = cfg.indexer_compress_ratio
        block_topk = cfg.indexer_budget // ratio

        q = F.linear(hidden, self.bw(layer, "indexer.q_proj.weight")).reshape(batch, seq, -1, dim)
        q = rms_norm(q, self.bw(layer, "indexer.q_norm.weight"), cfg.rms_norm_eps)
        q = apply_rotary(q, cos[:, -seq:], sin[:, -seq:], unsqueeze_dim=2)

        raw_k = F.linear(hidden, self.bw(layer, "indexer.k_proj.weight")).reshape(batch, seq, dim)
        if cache is not None:
            if cache[layer].indexer_keys is not None:
                raw_k = torch.cat([cache[layer].indexer_keys, raw_k], dim=1)
            cache[layer].indexer_keys = raw_k
        keys = raw_k[:, :kv_len]

        n_blocks = kv_len // ratio
        mask = torch.zeros(batch, seq, kv_len, dtype=torch.bool, device=hidden.device)
        if n_blocks == 0:
            # No complete block yet: everything causally visible is in the tail.
            pos = torch.arange(kv_len, device=hidden.device)
            qpos = torch.arange(seq, device=hidden.device)[:, None] + query_offset
            return (pos[None, None, :] <= qpos[None]).expand(batch, seq, kv_len).unsqueeze(1)

        pooled = keys[:, : n_blocks * ratio].reshape(batch, n_blocks, ratio, dim).float().mean(dim=2)
        pooled = rms_norm(pooled.to(keys.dtype), self.bw(layer, "indexer.k_norm.weight"), cfg.rms_norm_eps)
        starts = torch.arange(n_blocks, device=hidden.device) * ratio
        block_cos = cos[:, starts]
        block_sin = sin[:, starts]
        pooled = apply_rotary(pooled.unsqueeze(2), block_cos, block_sin, unsqueeze_dim=2).squeeze(2)

        # relu-then-sum over heads: a head only ever votes for a block
        scores = torch.einsum("bqhd,bkd->bqhk", q.float(), pooled.float())
        scores = torch.relu(scores).sum(dim=2) / math.sqrt(dim)

        query_pos = torch.arange(seq, device=hidden.device) + query_offset
        # a block is eligible only once all `ratio` of its tokens are visible
        block_end = starts + ratio - 1
        eligible = block_end[None, :] <= query_pos[:, None]  # (S, n_blocks)
        scores = scores.masked_fill(~eligible[None], float("-inf"))

        k = min(block_topk, n_blocks)
        chosen = scores.topk(k, dim=-1).indices  # (B, S, k)
        chosen_valid = eligible[None].expand(batch, -1, -1).gather(-1, chosen)

        token_idx = chosen.unsqueeze(-1) * ratio + torch.arange(ratio, device=hidden.device)
        token_idx = token_idx.flatten(2)
        keep = chosen_valid.unsqueeze(-1).expand(-1, -1, -1, ratio).flatten(2)
        mask.scatter_(2, token_idx, keep)

        # The trailing partial block is always visible -- the one *this* query is
        # in, not the one the last query is in. Deriving it from the global
        # `kv_len` made a whole-prompt call disagree with the same prompt fed
        # token by token, which is how QSA decodes: for a query at position p
        # only blocks ending at or before p are eligible, so everything after
        # the last such block has to come from the tail. With kv_len an exact
        # multiple of `ratio` the global form left no tail at all, and every
        # query before the last got a fully masked row.
        tail_start = ((query_pos + 1) // ratio) * ratio
        pos = torch.arange(kv_len, device=hidden.device)
        tail = (pos[None, :] >= tail_start[:, None]) & (pos[None, :] <= query_pos[:, None])
        mask |= tail[None]
        return mask.unsqueeze(1)

    # -- full attention ----------------------------------------------------

    def _full_attention(
        self,
        hidden: torch.Tensor,
        layer: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: HybridCache | None,
        query_offset: int,
    ) -> torch.Tensor:
        cfg = self.config
        batch, seq, _ = hidden.shape
        hd = cfg.head_dim

        # attn_q packs [q | gate] per head
        qg = F.linear(hidden, self.bw(layer, "attn_q.weight")).view(batch, seq, -1, hd * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(batch, seq, -1)

        q = rms_norm(q, self.bw(layer, "attn_q_norm.weight"), cfg.rms_norm_eps).transpose(1, 2)
        k = F.linear(hidden, self.bw(layer, "attn_k.weight")).view(batch, seq, -1, hd)
        k = rms_norm(k, self.bw(layer, "attn_k_norm.weight"), cfg.rms_norm_eps).transpose(1, 2)
        v = F.linear(hidden, self.bw(layer, "attn_v.weight")).view(batch, seq, -1, hd).transpose(1, 2)

        cur_cos, cur_sin = cos[:, -seq:], sin[:, -seq:]
        q = apply_rotary(q, cur_cos, cur_sin)
        k = apply_rotary(k, cur_cos, cur_sin)

        if cache is not None:
            if cache[layer].keys is not None:
                k = torch.cat([cache[layer].keys, k], dim=2)
                v = torch.cat([cache[layer].values, v], dim=2)
            cache[layer].keys, cache[layer].values = k, v
        kv_len = k.shape[2]

        qsa = self._indexer_mask(hidden, layer, cos, sin, cache, kv_len, query_offset)

        pos = torch.arange(kv_len, device=hidden.device)
        qpos = torch.arange(seq, device=hidden.device)[:, None] + query_offset
        causal = (pos[None, :] <= qpos)[None, None]
        allowed = causal & qsa

        groups = cfg.num_attention_heads // cfg.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(hd)
        scores = scores.masked_fill(~allowed, torch.finfo(torch.float32).min)
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v.float()).to(hidden.dtype)

        out = out.transpose(1, 2).reshape(batch, seq, -1)
        out = out * torch.sigmoid(gate)
        return F.linear(out, self.bw(layer, "attn_output.weight"))

    # -- MoE ----------------------------------------------------------------

    def _moe(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        cfg = self.config
        batch, seq, dim = hidden.shape
        flat = hidden.reshape(-1, dim)

        shared = F.silu(F.linear(flat, self.bw(layer, "ffn_gate_shexp.weight"))) * F.linear(
            flat, self.bw(layer, "ffn_up_shexp.weight")
        )
        shared = F.linear(shared, self.bw(layer, "ffn_down_shexp.weight"))
        shared_gate = torch.sigmoid(flat @ self.bw(layer, "ffn_gate_inp_shexp.weight").reshape(-1, 1))
        shared = shared * shared_gate

        logits = F.linear(flat, self.bw(layer, "ffn_gate_inp.weight"))
        probs = torch.softmax(logits.float(), dim=-1)
        weights, indices = torch.topk(probs, cfg.num_experts_per_tok, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # norm_topk_prob=True
        weights = weights.to(flat.dtype)

        out = torch.zeros_like(flat)
        # Only the experts actually selected are dequantised, one at a time.
        for expert in torch.unique(indices):
            e = int(expert)
            slot, token = torch.where(indices == e)
            x = flat[slot]
            gate_w = self.store.get_rows(f"blk.{layer}.ffn_gate_exps.weight", [e])[0]
            up_w = self.store.get_rows(f"blk.{layer}.ffn_up_exps.weight", [e])[0]
            down_w = self.store.get_rows(f"blk.{layer}.ffn_down_exps.weight", [e])[0]
            h = F.silu(F.linear(x, gate_w)) * F.linear(x, up_w)
            out.index_add_(0, slot, F.linear(h, down_w) * weights[slot, token, None])

        return (out + shared).reshape(batch, seq, dim)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: HybridCache | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        batch, seq = input_ids.shape

        past = cache.seq_len if cache is not None else 0
        if position_ids is None:
            position_ids = (torch.arange(seq, device=input_ids.device) + past)[None].expand(batch, -1)

        if cache is not None:
            full_positions = cache.extend_positions(position_ids)
        else:
            full_positions = position_ids[None].expand(3, -1, -1) if position_ids.ndim == 2 else position_ids
        cos, sin = self.rotary(full_positions)

        embeds = self.store.get_rows("token_embd.weight", input_ids.reshape(-1))
        hidden = embeds.reshape(batch, seq, cfg.hidden_size).repeat(1, 1, cfg.hc_count)

        for layer in range(cfg.num_layers):
            if layer in self._ple_layers:
                hidden = hidden + self._ple(hidden, input_ids, layer, cache)

            mixed, hyper, inject = self._gated_residual(hidden, layer, "attn")
            if cfg.is_full_attention(layer):
                branch = self._full_attention(mixed, layer, cos, sin, cache, past)
            else:
                branch = self._linear_attention(mixed, layer, cache)
            hidden = self._reinject(hyper, branch, inject)

            mixed, hyper, inject = self._gated_residual(hidden, layer, "ffn")
            hidden = self._reinject(hyper, self._moe(mixed, layer), inject)

            if self.probe is not None:
                self.probe(layer, hidden)

        hidden, _, _ = self._gated_residual(hidden, None, "")
        return hidden

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.w("output.weight"))
