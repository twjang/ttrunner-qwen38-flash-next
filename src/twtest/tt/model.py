"""qwen4exp on a 4 x Blackhole mesh.

Structure of the forward, and why it is shaped this way:

* The 10240-wide hyper-connection residual is **replicated** on every device, as
  is every dense weight. Only the three expert stacks per layer and the LM head
  are sharded, so there is exactly one all-reduce per layer (the MoE down
  projection) plus one all-gather for the logits -- 49 collectives per step
  rather than ~100. See plan.py.
* Token generation runs one token at a time through `step()`. The prompt is
  consumed by the same path, which keeps a single code path under test; a
  chunked prefill using `gated_delta_attn_seq` is a throughput optimisation on
  top of it, not a correctness requirement.
* **QSA below 2048 tokens is exactly dense causal attention.** The indexer keeps
  the top `budget/compress_ratio = 512` complete blocks of 4 tokens; with fewer
  than 512 complete blocks, `topk` keeps them all and the only remaining
  constraint is causality. The sparse selection path therefore only engages past
  2048 tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import ttnn

from ..reference.config import Qwen4ExpConfig
from ..reference.weights import WeightStore
from . import linear_attn, moe
from .ops import HIFI4, gated_residual_mix, grouped_rms_norm, reinject, rms_norm
from .weights import TTWeights


# Largest MoE row-group that still computes a per-token MoE.
#
# `moe.moe_block` must be exact per row -- each row picks its own experts and is
# weighted by its own probabilities -- so grouping rows can only change speed.
# It does not: `scripts/dev/moe_rows_check.py` builds the answer one row at a
# time and compares, and row-groups of 1, 8, 16 and 32 reproduce it exactly
# while 64 gets **33 of 64 rows wrong**, worst row 102.9 %. 32 is one tile.
# See `TTModel.prefill` and handoff 5.8.
_MAX_MOE_CHUNK = 32


@dataclass
class LayerState:
    conv: list | None = None                   # ring of kernel-1 single columns
    conv_step: int = 0
    recurrent: ttnn.Tensor | None = None       # [BH, 1, Dk, Dv]
    keys: ttnn.Tensor | None = None            # [1, n_kv, T, head_dim]
    values: ttnn.Tensor | None = None
    # The QSA indexer's compressed keys: one row per `indexer_compress_ratio`
    # tokens, already mean-pooled, normed and roped, so a block never changes
    # once complete. `indexer_ring` holds the last `ratio` raw keys the pool is
    # taken over.
    indexer_blocks: ttnn.Tensor | None = None  # [batch, 1, T/ratio, indexer_dim]
    indexer_ring: list | None = None
    ple_conv: list | None = None      # ring of `state_len` single-column tensors
    ple_step: int = 0


class TTState:
    """Decoding state for a batch of `batch` sequences advancing in lockstep.

    Token histories live here rather than per layer: only the PLE layer reads
    them, and they are a property of the sequence, not of a layer.
    """

    def __init__(self, num_layers: int, batch: int = 1):
        self.layers = [LayerState() for _ in range(num_layers)]
        self.batch = batch
        self.positions = [0] * batch
        self.histories: list[list[int]] = [[] for _ in range(batch)]

    def __getitem__(self, i: int) -> LayerState:
        return self.layers[i]

    @property
    def position(self) -> int:
        return self.positions[0]


class TTModel:
    def __init__(
        self,
        config: Qwen4ExpConfig,
        weights: TTWeights,
        host_store: WeightStore,
        mesh,
        max_seq_len: int = 4096,
        sdpa_k_chunk: int = 128,
        traceable_kv: bool = False,
        state_dtype=None,
        cache_update_batch: int = 64,
    ):
        self.cfg = config
        self.n_dev = weights.n_dev
        # The DeltaNet is head-sharded: each device owns 48/n_dev value heads and
        # 16/n_dev key heads, so its slice of the 10240-wide q|k|v channel axis is
        # 2*key_dim/n_dev + value_dim/n_dev wide.
        self.n_v_local = config.linear_num_v_heads // self.n_dev
        self.n_k_local = config.linear_num_k_heads // self.n_dev
        self.value_dim_local = config.linear_value_dim // self.n_dev
        # Q and K are sharded by the heads each device's V heads pair with, not
        # chunked -- so a device holds as many q/k heads as v heads, and the
        # pairing is local (v-head i reads k-head i) with no expansion. See
        # `split_qkv_channels` in tt/convert.py and docs/iterations/014.
        self.key_dim_local = self.n_v_local * config.linear_head_dim
        self.conv_dim_local = 2 * self.key_dim_local + self.value_dim_local
        self.max_seq_len = max_seq_len
        # `update_cache` takes the position as an int, which a captured trace
        # bakes in. `paged_update_cache` takes it as a tensor -- traceable -- but
        # requires the k/v input to be L1 height-sharded. Off by default so the
        # validated interleaved path stays the reference behaviour.
        self.traceable_kv = traceable_kv
        # The DeltaNet recurrent state dominates memory traffic at large batch
        # (its matmuls run memory-bound). bf16 halves that, but the state is
        # *recurrent*, so precision has to be checked, not assumed.
        self.state_dtype = state_dtype or ttnn.float32
        # Set by TracedDecoder before capture: rings then advance by device copy
        # at fixed indices instead of by host-side Python rebinding, which a
        # trace cannot record. Off by default -- the eager path pays nothing.
        self.trace_safe_rings = False
        # Fuse the experts' gate and up projections into one sparse_matmul,
        # reading the prebuilt blk.N.ffn_gateup_exps weights. Verified
        # token-for-token identical to the split path, and neutral-to-better at
        # every batch size once the constant conv taps stopped being re-sliced
        # every token (two runs per mode, 25 samples each):
        #
        #     batch    split                fused
        #        1   522.4 / 520.3 ms     504.8 / 523.0 ms     neutral
        #       32   57.57 / 57.97 tok/s  59.51 / 58.49 tok/s  +2 %
        #       64   86.66 / 86.60 tok/s  97.47 / 97.36 tok/s  +12 %
        #
        # The weights cannot be swapped at run time -- both sets resident does not
        # fit in DRAM -- so this is fixed at construction.
        self.fuse_expert_gate_up = False
        # Per-tap columns of the depthwise conv weights. The weights are constant,
        # but the taps were being re-sliced on every token: 4 slices x 36 DeltaNet
        # layers + 4 for the PLE = 148 ops per step recomputing the same thing.
        self._conv_taps: dict[tuple, list] = {}
        # Sequences per `update_cache` call; see _attention_step.
        self.cache_update_batch = cache_update_batch
        # sdpa_decode sizes its L1 circular buffers from the K chunk. The default
        # overflows Blackhole's 1.5 MB per core for 24 heads at head_dim 256
        # ("circular buffers grow to 1917696 B beyond max L1 size of 1572864 B"),
        # so the chunk is set explicitly rather than left to the default.
        # -- QSA sparse selection ------------------------------------------
        # Below the budget QSA keeps every complete block, so dense causal
        # attention is exactly right and cheaper; the selection only earns its
        # keep past it. Whether it runs is therefore a property of the *model*,
        # not of the position -- a captured trace replays one graph, so this
        # cannot be decided per step.
        self.indexer_ratio = config.indexer_compress_ratio
        self.indexer_budget = config.indexer_budget
        self.indexer_topk = config.indexer_budget // config.indexer_compress_ratio
        self.max_blocks = max_seq_len // config.indexer_compress_ratio
        # The compact attention window: the budget, plus one K chunk for the
        # block the query sits inside. That block is *extra* to the budget, not
        # one of it -- the reference allows the selected blocks and the trailing
        # partial one, so spending a selection slot on it keeps 511 blocks where
        # it should keep 512 (measured: 2047 tokens visible against 2051).
        #
        # A whole K chunk rather than a tile, because sdpa_decode asserts
        # `mask_shape[3] % k_chunk_size == 0`. Only `ratio` of the 128 slots
        # carry a token; the rest are masked off.
        self.sdpa_k_chunk = sdpa_k_chunk
        self.indexer_window = config.indexer_budget + sdpa_k_chunk
        # `ttnn.scatter` takes uint16 indices -- int32 and uint32 both assert --
        # so the selection can only address 65536 cache positions. Past that the
        # path falls back to dense attention, which is *not* the model beyond the
        # budget; TTEngine says so at construction.
        self.indexer_max_seq = 1 << 16
        self._mask_base = None
        self.use_indexer = (
            traceable_kv
            and config.indexer_budget < max_seq_len <= self.indexer_max_seq
        )
        self._block_offsets = None

        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=32,
            k_chunk_size=sdpa_k_chunk,
            exp_approx_mode=False,
        )
        self.w = weights
        self.host = host_store
        self.mesh = mesh
        self.replicate = ttnn.ReplicateTensorToMesh(mesh)
        self.compose = ttnn.ConcatMeshToTensor(mesh, dim=0)
        self._ple_layers = {idx: n for n, idx in enumerate(config.ple_layers)}
        self._rope_cache: dict[int, tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        # When tracing, per-step inputs must live at fixed device addresses: the
        # trace records the graph, and only the *contents* of these buffers may
        # change between replays. `None` means untraced, and every input is
        # created fresh (identical numerics, just more dispatch).
        self.bound: dict[str, ttnn.Tensor] | None = None
        # While a trace is being captured, the graph must contain device ops
        # only -- the host->device copies that fill the bound buffers happen
        # outside it, so they are suppressed here.
        self._skip_copy = False
        # Debug hook: called as probe(layer, hidden) after every layer in both
        # `step` and `prefill`, so the two paths can be bisected layer by layer.
        # Never set while tracing (host callbacks are invisible to capture).
        self.probe = None

    # -- host <-> device -------------------------------------------------

    def to_dev(self, t: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.mesh, mesh_mapper=self.replicate)

    def _input(self, name: str, host: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        """A per-step input, either fresh or written into its bound buffer.

        Writing into a persistent buffer is what lets the whole 48-layer step be
        replayed from a captured trace; the copy itself happens outside the trace.
        """
        if self.bound is None:
            return self.to_dev(host, dtype, layout)
        buf = self.bound.get(name)
        if buf is None:
            buf = self.to_dev(host, dtype, layout)
            self.bound[name] = buf
            return buf
        if self._skip_copy:
            return buf
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host, dtype=dtype, layout=layout, mesh_mapper=self.replicate), buf
        )
        return buf

    def from_dev(self, t: ttnn.Tensor) -> torch.Tensor:
        """First device's copy (all devices agree for replicated results)."""
        return ttnn.to_torch(t, mesh_composer=self.compose)[0:1]

    def all_reduce(self, t: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.all_reduce(t, cluster_axis=1, topology=ttnn.Topology.Linear)

    # -- rope -------------------------------------------------------------

    def rope(self, positions: int | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for one absolute position per sequence -> [1, B, 1, rope_dim].

        Text-only input uses the same position on all three mrope axes, so the
        interleaved mrope reduces to plain rope here.
        """
        cfg = self.cfg
        pos = [positions] if isinstance(positions, int) else list(positions)
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rope_dim, 2, dtype=torch.float) / cfg.rope_dim))
        freqs = torch.outer(torch.tensor(pos, dtype=torch.float), inv)   # [B, rope_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)                          # [B, rope_dim]
        shape = (1, len(pos), 1, cfg.rope_dim)
        return emb.cos().reshape(shape), emb.sin().reshape(shape)

    @staticmethod
    def _apply_rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        rot = cos.shape[-1]
        x_rot, x_pass = x[..., :rot], x[..., rot:]
        half = rot // 2
        rotated = torch.cat([-x_rot[..., half:], x_rot[..., :half]], dim=-1)
        return torch.cat([x_rot * cos + rotated * sin, x_pass], dim=-1)

    # -- embedding + PLE (host gathers) ------------------------------------

    def embed(self, token_ids: list[int]) -> torch.Tensor:
        rows = self.host.get_rows("token_embd.weight", token_ids)
        return rows.reshape(1, 1, len(token_ids), self.cfg.hidden_size)

    def ngram_embed(self, histories: list[list[int]]) -> torch.Tensor:
        """Hash each sequence's trailing n-grams and gather those rows on host.

        The table is 28.8 GB and only `ngram_heads` rows per sequence are needed,
        so it stays in the GGUF mmap and just these rows cross to the device.

        Vectorised over (sequence x n-gram order x head). The natural nested-loop
        form runs `batch * (ngram_size-1) * heads_per_ngram` Python iterations of
        *big-integer* arithmetic per token -- 1024 of them at batch 64, which is
        why one PLE layer cost 51 ms, six times a whole DeltaNet layer. The
        multipliers reach ~2.4e13 and tokens ~2.5e5, so the products stay under
        int64 and numpy can do the whole thing at once.
        """
        cfg = self.cfg
        eos = cfg.ple_eos_token_id
        context = cfg.ngram_size
        mults = np.asarray(cfg.ngram_layer_multipliers, dtype=np.int64)
        vocab = np.asarray(cfg.ngram_head_vocab_sizes, dtype=np.int64)
        offsets = np.asarray(cfg.ngram_head_offsets, dtype=np.int64)

        # [batch, ngram_size] window of trailing tokens, EOS-padded on the left
        window = np.full((len(histories), context), eos, dtype=np.int64)
        for row, tokens in enumerate(histories):
            tail = tokens[-context:]
            if tail:
                window[row, context - len(tail) :] = tail

        blocks = []
        for order in range(2, context + 1):
            mixed = window[:, -1] * mults[0]
            for pos in range(1, order):
                mixed = np.bitwise_xor(mixed, window[:, -1 - pos] * mults[pos])
            lo = (order - 2) * cfg.heads_per_ngram
            hi = lo + cfg.heads_per_ngram
            blocks.append(mixed[:, None] % vocab[lo:hi] + offsets[lo:hi])
        ids = np.concatenate(blocks, axis=1).reshape(-1)

        rows = self.host.get_rows("per_layer_token_embd.weight", ids)
        return rows.reshape(1, 1, len(histories), cfg.ngram_heads * cfg.ple_head_dim)

    # -- small helpers ------------------------------------------------------

    @staticmethod
    def _l2norm(x: ttnn.Tensor, eps: float = 1e-6) -> ttnn.Tensor:
        sq = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
        return ttnn.multiply(x, ttnn.rsqrt(ttnn.add(sq, eps)))

    @staticmethod
    def _slice_last(x: ttnn.Tensor, start: int, stop: int) -> ttnn.Tensor:
        s = list(x.shape)
        return ttnn.slice(x, (0, 0, 0, start), (s[0], s[1], s[2], stop))

    def _apply_rope_dev(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """Rotate the leading rope_dim dims; leave the rest untouched."""
        rot = self.cfg.rope_dim
        half = rot // 2
        x_rot = self._slice_last(x, 0, rot)
        x_pass = self._slice_last(x, rot, x.shape[-1])
        first = self._slice_last(x_rot, 0, half)
        second = self._slice_last(x_rot, half, rot)
        rotated = ttnn.concat([ttnn.neg(second), first], dim=-1)
        out = ttnn.add(ttnn.multiply(x_rot, cos), ttnn.multiply(rotated, sin))
        return ttnn.concat([out, x_pass], dim=-1)

    # Blackhole's Tensix grid is 11 x 10 = 110 cores, and paged_update_cache
    # needs one per sequence, so that is the hard batch ceiling for this path.
    GRID_WIDTH = 11
    MAX_BATCH = 110

    @staticmethod
    def _core_range_set(n_cores: int, width: int = 11) -> "ttnn.CoreRangeSet":
        """A CoreRangeSet covering `n_cores` cores, row-major over `width` columns."""
        rows, rem = divmod(n_cores, width)
        ranges = []
        if rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, rows - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(rem - 1, rows)))
        return ttnn.CoreRangeSet(set(ranges))

    def _l1_height_sharded(self, t: ttnn.Tensor, width: int) -> ttnn.Tensor:
        """L1 height-sharded copy with shard width == the last dimension.

        `paged_update_cache` requires exactly this (and rejects WIDTH_SHARDED);
        it is the precondition for taking the cache position as a tensor, which
        is in turn what lets the attention layers be captured in a trace.

        The shard count must come from the **padded** height, not the logical
        one. For k/v shaped [1, B, n_kv, hd] the n_kv=2 axis pads to a full
        32-row tile, so the physical height is B*32 and the tensor splits into B
        shards -- sizing from the logical height (1*B*n_kv) asks for one shard on
        one core and fails with "Number of shards along height B must not exceed
        number of cores 1" as soon as B > 1.
        """
        shape = list(t.shape)
        tile = 32
        n_shards = 1
        for d in shape[:-2]:
            n_shards *= d
        n_shards *= (shape[-2] + tile - 1) // tile
        # Exactly one shard per core is mandatory, not a tuning choice:
        # paged_update_cache "dispatches one user per core" and asserts
        # num_cores == batch. Packing several tiles per shard to use fewer cores
        # therefore fails, and the Tensix grid (11x10) is a hard ceiling on batch.
        grid = self._core_range_set(n_shards)
        spec = ttnn.ShardSpec(grid, [tile, width], ttnn.ShardOrientation.ROW_MAJOR)
        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
        return ttnn.to_memory_config(t, mem)

    def conv_taps(self, key: tuple, weight: ttnn.Tensor, channels: int, k: int) -> list:
        """The kernel's `k` per-tap columns, sliced once and reused."""
        hit = self._conv_taps.get(key)
        if hit is None:
            hit = [
                ttnn.slice(weight, (0, 0, 0, tap), (1, 1, channels, tap + 1)) for tap in range(k)
            ]
            self._conv_taps[key] = hit
        return hit

    def _causal_conv_step(
        self, x_col: ttnn.Tensor, weight: ttnn.Tensor, state: list | None,
        channels: int, batch: int = 1, step: int = 0, layer: int = 0
    ) -> tuple[ttnn.Tensor, list]:
        """One step of a 4-tap depthwise causal conv, per sequence.

        The window is a ring of single columns for the same reason as the PLE
        conv: advancing it is then a Python list assignment rather than a
        slice + concat + copy over [1, B, channels, kernel] every token, on all
        36 linear-attention layers. With a kernel of 4 the conv itself is just
        `sum_i w[c,i] * x[c,i]`, so no conv op is needed either.

        Each sequence keeps its own window (columns are [1, B, C, 1]);
        concatenating along a shared axis would make the window k-1+B wide and
        break the kernel-width broadcast for any B > 1.
        """
        k = self.cfg.conv_kernel
        depth = k - 1
        if state is None:
            state = [
                ttnn.zeros((1, batch, channels, 1), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=self.mesh)
                for _ in range(depth)
            ]
        # Rotating by index (`state[pos] = x_col`, pos = step % depth) is host-side
        # Python: free, but invisible to a trace, which records only device ops.
        # A captured step would replay one frozen phase for ever and read a
        # convolution history stuck at capture time -- measured as tokens
        # diverging from eager at the very first one. `trace_safe_rings` instead
        # keeps the read indices fixed and shifts the contents with device copies,
        # so the graph is the same every step and can be captured.
        taps = self.conv_taps(("ssm", layer), weight, channels, k)
        if self.trace_safe_rings:
            acc = None
            for tap in range(k):
                age = depth - tap                   # 3, 2, 1, 0 steps back
                piece = x_col if age == 0 else state[age - 1]
                w_tap = taps[tap]
                term = ttnn.multiply(piece, w_tap)
                acc = term if acc is None else ttnn.add(acc, term)
            for i in range(depth - 1, 0, -1):       # oldest drops off the end
                ttnn.copy(state[i - 1], state[i])
            ttnn.copy(x_col, state[0])
            return ttnn.silu(acc), state

        pos = step % depth
        acc = None
        for tap in range(k):
            age = depth - tap                       # 3, 2, 1, 0 steps back
            piece = x_col if age == 0 else state[(pos + (depth - age)) % depth]
            w_tap = taps[tap]
            term = ttnn.multiply(piece, w_tap)
            acc = term if acc is None else ttnn.add(acc, term)
        state[pos] = x_col
        return ttnn.silu(acc), state

    def _linear_attention_step(self, mixed: ttnn.Tensor, layer: int, st: LayerState) -> ttnn.Tensor:
        cfg = self.cfg
        # local (per-device) head count: q, k and v are all sharded to the same
        # twelve heads, so there is a single head count here
        n_v, hd = self.n_v_local, cfg.linear_head_dim
        # `mixed` is [1, 1, B, hidden]; B sequences decode together.
        batch = mixed.shape[-2]

        qkv = ttnn.linear(mixed, self.w.blk(layer, "attn_qkv.weight"), compute_kernel_config=HIFI4)
        z = ttnn.linear(mixed, self.w.blk(layer, "attn_gate.weight"), compute_kernel_config=HIFI4)

        # [1,1,B,conv_dim] -> [1,B,conv_dim,1] so each sequence owns a window
        qkv_col = ttnn.transpose(ttnn.permute(qkv, (0, 2, 1, 3)), -2, -1)
        conv_w = self.w.blk(layer, "ssm_conv1d.weight")
        conv_out, st.conv = self._causal_conv_step(
            qkv_col, conv_w, st.conv, self.conv_dim_local, batch, st.conv_step, layer
        )
        st.conv_step += 1
        # [1,B,conv_dim,1] -> [1,1,B,conv_dim]
        qkv = ttnn.permute(ttnn.transpose(conv_out, -2, -1), (0, 2, 1, 3))

        kd = self.key_dim_local
        q = self._slice_last(qkv, 0, kd)
        k = self._slice_last(qkv, kd, 2 * kd)
        v = self._slice_last(qkv, 2 * kd, 2 * kd + self.value_dim_local)

        # No expansion: the converter gives this device exactly the twelve q/k
        # heads its twelve v heads pair with, in matching order, so v-head i
        # reads k-head i. Chunking q/k four ways instead and tiling the local
        # heads is what cost the engine 25.5 % next-token accuracy against the
        # reference's 80.9 % (docs/iterations/014).
        q = ttnn.reshape(q, (batch * n_v, 1, 1, hd))
        k = ttnn.reshape(k, (batch * n_v, 1, 1, hd))
        v = ttnn.reshape(v, (batch * n_v, 1, 1, hd))

        q = ttnn.multiply(self._l2norm(q), hd**-0.5)
        k = self._l2norm(k)

        a = ttnn.linear(mixed, self.w.blk(layer, "ssm_alpha.weight"), compute_kernel_config=HIFI4)
        b = ttnn.linear(mixed, self.w.blk(layer, "ssm_beta.weight"), compute_kernel_config=HIFI4)
        dt = self.w.blk(layer, "ssm_dt.bias")
        a_decay = self.w.blk(layer, "ssm_a")
        # g = A * softplus(a + dt_bias); A is stored already negated (= -exp(A_log))
        g = ttnn.multiply(a_decay, ttnn.softplus(ttnn.add(a, dt)))
        g_exp = ttnn.reshape(ttnn.exp(g), (batch * n_v, 1, 1, 1))
        beta = ttnn.reshape(ttnn.sigmoid(b), (batch * n_v, 1, 1, 1))

        if st.recurrent is None:
            st.recurrent = ttnn.zeros(
                (batch * n_v, 1, hd, hd), dtype=self.state_dtype,
                layout=ttnn.TILE_LAYOUT, device=self.mesh,
            )
        # The recurrent state is float32 already, and casting q/k/v/g/beta to
        # float32 here changes nothing measurable (24.35 % vs 24.14 % at step 24
        # of `branch_sequence_check.py`): the error is in those tensors when they
        # arrive, from bf16 projections and the conv, and a wider container does
        # not recover bits that were never there.
        # writes st.recurrent in place and returns the output
        out = linear_attn.decode_step(q, k, v, g_exp, beta, st.recurrent)

        out = ttnn.reshape(out, (1, 1, batch * n_v, hd))
        z_heads = ttnn.reshape(z, (1, 1, batch * n_v, hd))
        normed = ttnn.rms_norm(
            out, epsilon=cfg.rms_norm_eps, weight=self.w.blk(layer, "ssm_norm.weight"),
            compute_kernel_config=HIFI4,
        )
        # output_gate_type is "sigmoid" for this model, not the silu that
        # hidden_act would imply, and the GGUF does not record the field.
        gated = ttnn.multiply(normed, ttnn.sigmoid(z_heads))
        gated = ttnn.reshape(gated, (1, 1, batch, self.value_dim_local))
        # ssm_out is row-sharded on its contraction dim, so each device produces a
        # partial sum -- this is the collective that head-sharding costs.
        out = ttnn.linear(gated, self.w.blk(layer, "ssm_out.weight"), compute_kernel_config=HIFI4)
        return self.all_reduce(out)

    # -- full attention (QSA), one token -------------------------------------

    def _indexer_select(
        self, mixed: ttnn.Tensor, layer: int, st: LayerState, positions: list[int],
        q_cos: ttnn.Tensor, q_sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """QSA's sparse selection, as an additive attention mask.

        Returns [B, n_q, max_seq_len] worth of 0 / -1e9 for
        `scaled_dot_product_attention_decode`, marking the `indexer_budget`
        tokens this query may attend to.

        Every step is unconditional, because a trace replays one graph:

        * The pooled block for position p is written every step at index
          `p // ratio`, over the last `ratio` raw keys. For p not at a block
          boundary that value is wrong -- and the block is not yet eligible, and
          is overwritten before it becomes so. The pool is a mean, so the ring's
          order does not matter.
        * A block is eligible once all `ratio` of its tokens are visible,
          `ratio*j + ratio - 1 <= p`. Ineligible blocks are pushed to -inf by a
          host-built bias, and the block p sits inside is forced *in* when it is
          incomplete -- that is the reference's "the trailing partial block is
          always visible" rule (`_indexer_mask` in reference/model.py).
        * Whatever `topk` then returns, the mask keeps exactly the tokens with
          `index <= p`. That covers the three cases at once: a fully ineligible
          block lies entirely beyond p, the straddling block keeps its visible
          prefix, and an eligible block keeps everything. It is also why the
          path is exact below the budget, where fewer than `topk` blocks exist
          and the surplus is filled with whatever scored highest among -inf.
        """
        cfg = self.cfg
        d, ratio = cfg.indexer_head_dim, self.indexer_ratio
        batch = mixed.shape[-2]
        nb, k = self.max_blocks, self.indexer_topk

        # -- compressed keys ------------------------------------------------
        k_raw = ttnn.linear(
            mixed, self.w.blk(layer, "indexer.k_proj.weight"), compute_kernel_config=HIFI4
        )                                                        # [1,1,B,d]
        if st.indexer_ring is None:
            st.indexer_ring = [
                ttnn.zeros((1, 1, batch, d), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=self.mesh)
                for _ in range(ratio)
            ]
        for i in range(ratio - 1, 0, -1):
            ttnn.copy(st.indexer_ring[i - 1], st.indexer_ring[i])
        ttnn.copy(k_raw, st.indexer_ring[0])
        pooled = st.indexer_ring[0]
        for i in range(1, ratio):
            pooled = ttnn.add(pooled, st.indexer_ring[i])
        pooled = ttnn.multiply(pooled, 1.0 / ratio)
        pooled = rms_norm(pooled, self.w.blk(layer, "indexer.k_norm.weight"), cfg.rms_norm_eps)
        pooled = ttnn.reshape(pooled, (1, batch, 1, d))
        # roped at the block's *start*, which is where its first token sat
        b_cos, b_sin = self.rope([ratio * (p // ratio) for p in positions])
        pooled = self._apply_rope_dev(
            pooled,
            self._input("idx_block_cos", b_cos, ttnn.float32),
            self._input("idx_block_sin", b_sin, ttnn.float32),
        )
        if st.indexer_blocks is None:
            st.indexer_blocks = ttnn.zeros(
                (batch, 1, nb, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh
            )
        ttnn.experimental.paged_update_cache(
            st.indexer_blocks,
            self._l1_height_sharded(ttnn.typecast(pooled, ttnn.bfloat16), d),
            update_idxs_tensor=self._input(
                "idx_block_pos", torch.tensor([p // ratio for p in positions], dtype=torch.int32),
                ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
        )

        # -- scores: sum over heads of relu(q . block), one matmul per head --
        # Contracting the *block* cache against a single query column keeps the
        # 65536 x 128 cache where it is; scoring the other way round would
        # transpose 16 MB a layer a step.
        q_idx = ttnn.linear(
            mixed, self.w.blk(layer, "indexer.q_proj.weight"), compute_kernel_config=HIFI4
        )
        q_idx = ttnn.reshape(q_idx, (1, batch, cfg.indexer_heads, d))
        q_idx = rms_norm(q_idx, self.w.blk(layer, "indexer.q_norm.weight"), cfg.rms_norm_eps)
        # the caller has already bound rope at p for the attention heads
        q_idx = self._apply_rope_dev(q_idx, q_cos, q_sin)
        scores = None
        for h in range(cfg.indexer_heads):
            qh = ttnn.reshape(
                ttnn.slice(q_idx, (0, 0, h, 0), (1, batch, h + 1, d)), (batch, 1, d, 1)
            )
            part = ttnn.relu(ttnn.matmul(st.indexer_blocks, qh, compute_kernel_config=HIFI4))
            scores = part if scores is None else ttnn.add(scores, part)
        scores = ttnn.multiply(ttnn.transpose(scores, -2, -1), d**-0.5)   # [B,1,1,nb]
        scores = ttnn.add(scores, self._input("idx_bias", self._block_bias(positions), ttnn.float32))

        # -- select, expand to tokens, and build the mask -------------------
        blocks = ttnn.topk(scores, k, dim=-1)[1]                  # uint16 [B,1,1,k]
        if self._block_offsets is None:
            self._block_offsets = self.to_dev(
                torch.arange(ratio, dtype=torch.float32).reshape(1, 1, 1, ratio), ttnn.float32
            )
        tokens = ttnn.add(
            ttnn.reshape(ttnn.multiply(ttnn.typecast(blocks, ttnn.float32), float(ratio)),
                         (batch, 1, k, 1)),
            self._block_offsets,
        )                                                          # [B,1,k,ratio]
        tokens = ttnn.reshape(tokens, (batch, 1, 1, k * ratio))

        visible = ttnn.le(
            tokens,
            self._input(
                "idx_cur_pos_f",
                torch.tensor(positions, dtype=torch.float32).reshape(batch, 1, 1, 1).expand(
                    batch, 1, 1, k * ratio
                ).contiguous(),
                ttnn.float32,
            ),
        )
        # This one comparison covers every case `topk` can hand back: a block
        # entirely beyond p (which is what the -inf fills are, when fewer than
        # `topk` blocks are eligible) has all its tokens beyond p, and an
        # eligible block has none.
        tail_idx, tail_vis = self._tail_block(positions)
        tokens = ttnn.concat(
            [tokens, self._input("idx_tail", tail_idx, ttnn.float32)], dim=-1
        )
        visible = ttnn.concat(
            [visible, self._input("idx_tail_vis", tail_vis, ttnn.float32)], dim=-1
        )

        # Mark the selection in a full-length row and hand back an additive mask
        # over the *existing* cache. The alternative -- gathering the selected
        # K/V into a budget-sized buffer -- reads the whole cache per call in
        # this ttnn build: `ttnn.gather` is linear in the source, 35 ms at 4096
        # tokens and 1161 ms at 131072, against 0.3-0.7 ms for this scatter.
        # The base has to be a *fresh* zero row each step, and it cannot come
        # from `ttnn.zeros(device=...)`: that is a host-to-device write, which
        # trace capture refuses ("Writes are not supported during trace
        # capture"). Scaling a persistent buffer by zero is a device op, so it
        # is recordable -- and it leaves the persistent one untouched whether or
        # not `scatter` writes its base in place.
        if self._mask_base is None:
            self._mask_base = ttnn.zeros(
                (batch, 1, 1, self.max_seq_len), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=self.mesh,
            )
        row = ttnn.scatter(
            ttnn.multiply(self._mask_base, 0.0),
            -1,
            ttnn.typecast(tokens, ttnn.uint16),
            ttnn.typecast(visible, ttnn.bfloat16),
        )
        mask = ttnn.multiply(ttnn.subtract(row, 1.0), 1e9)
        return ttnn.repeat(mask, (1, 1, cfg.num_attention_heads, 1))

    def _block_bias(self, positions: list[int]) -> torch.Tensor:
        """Additive score bias: 0 for blocks this query may select, -inf otherwise.

        Eligibility is `ratio*j + ratio - 1 <= p`. The block p sits inside is
        pushed *below* the other ineligible ones while it is incomplete, so
        `topk` can never return it -- it is appended separately as the trailing
        partial block, which is how the reference treats it, and appending is
        what keeps it from costing a selection slot.

        Once that block completes (`p % ratio == ratio - 1`) it is eligible like
        any other and competes normally, and nothing is appended.
        """
        ratio, nb = self.indexer_ratio, self.max_blocks
        ends = torch.arange(nb, dtype=torch.float32) * ratio + (ratio - 1)
        rows = []
        for p in positions:
            row = torch.where(ends <= p, 0.0, -1e9)
            if p % ratio != ratio - 1:
                row[p // ratio] = -2e9        # never selectable; appended instead
            rows.append(row)
        return torch.stack(rows).reshape(len(positions), 1, 1, nb)

    def _tail_block(self, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """The trailing partial block: cache indices, and which of them are visible.

        A whole K chunk rather than `ratio` entries, because sdpa_decode asserts
        the mask width is a multiple of `k_chunk_size`. Everything here is
        derived from the position alone, so it is data a captured trace re-reads,
        not a branch it would have to record: when the block is already complete
        the whole chunk is masked off and the block competes in `topk` instead.
        """
        ratio, tile = self.indexer_ratio, self.sdpa_k_chunk
        idx = torch.zeros(len(positions), 1, 1, tile)
        vis = torch.zeros(len(positions), 1, 1, tile)
        for b, p in enumerate(positions):
            # Padding slots point at a position that is never visible, so
            # scattering a zero there cannot un-select a token some selected
            # block legitimately contributed. Index 0 would: block 0 may be
            # selected, and token 0 is visible from the very first step.
            idx[b, 0, 0, :] = min(p + 1, self.max_seq_len - 1)
            if p % ratio == ratio - 1:
                continue                      # complete: it is in the topk pool
            start = ratio * (p // ratio)
            for t in range(ratio):
                idx[b, 0, 0, t] = start + t
                if start + t <= p:
                    vis[b, 0, 0, t] = 1.0
        return idx, vis

    def _attention_step(self, mixed: ttnn.Tensor, layer: int, st: LayerState, position: int) -> ttnn.Tensor:
        """One decode step of QSA attention.

        The K/V cache is **preallocated** to `max_seq_len` and written in place by
        `paged_update_cache`, and attention is `scaled_dot_product_attention_decode`
        with `cur_pos_tensor`. Both take the position as a *tensor*, so every shape
        in the step is static regardless of how far decoding has progressed --
        which is what makes the whole 48-layer step capturable as a single ttnn
        trace. A growing `ttnn.concat` cache would change shapes every token and
        force a re-dispatch (and a reallocation) each time.

        Below 2048 tokens QSA selects every complete 4-token block, so plain
        causal attention is exactly right; the sparse selection path engages
        beyond that.
        """
        cfg = self.cfg
        hd, n_q, n_kv = cfg.head_dim, cfg.num_attention_heads, cfg.num_kv_heads
        batch = mixed.shape[-2]

        qg = ttnn.linear(mixed, self.w.blk(layer, "attn_q.weight"), compute_kernel_config=HIFI4)
        qg = ttnn.reshape(qg, (1, batch, n_q, hd * 2))      # [q | gate] interleaved per head
        q = self._slice_last(qg, 0, hd)
        gate = self._slice_last(qg, hd, hd * 2)

        k = ttnn.reshape(
            ttnn.linear(mixed, self.w.blk(layer, "attn_k.weight"), compute_kernel_config=HIFI4),
            (1, batch, n_kv, hd),
        )
        v = ttnn.reshape(
            ttnn.linear(mixed, self.w.blk(layer, "attn_v.weight"), compute_kernel_config=HIFI4),
            (1, batch, n_kv, hd),
        )
        q = rms_norm(q, self.w.blk(layer, "attn_q_norm.weight"), cfg.rms_norm_eps)
        k = rms_norm(k, self.w.blk(layer, "attn_k_norm.weight"), cfg.rms_norm_eps)

        positions_for_rope = list(position) if isinstance(position, (list, tuple)) else [position] * batch
        cos_t, sin_t = self.rope(positions_for_rope)
        cos = self._input("rope_cos", cos_t, ttnn.float32)
        sin = self._input("rope_sin", sin_t, ttnn.float32)
        q = self._apply_rope_dev(q, cos, sin)
        k = self._apply_rope_dev(k, cos, sin)

        if st.keys is None:
            shape = (batch, n_kv, self.max_seq_len, hd)
            st.keys = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)
            st.values = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)
        positions = list(position) if isinstance(position, (list, tuple)) else [position] * batch
        positions_for_rope = positions
        # `update_cache` accepts an INTERLEAVED input and takes the position as a
        # plain int. `paged_update_cache` would take it as a tensor -- which is
        # what a captured trace needs, since an int is baked into the recorded
        # program -- but it *also* asserts the input is L1 height-sharded with
        # shard width == head_dim. Getting the position out of the graph therefore
        # means sharding k/v first; until that is in place this path uses the
        # interleaved int form, and tracing the attention layers is blocked on it.
        # `update_cache` asserts input.shape[0] == 1, input.shape[1] ==
        # cache.shape[1] (the KV-head count) and input.dtype == cache.dtype. The
        # projections give [1, batch, n_kv, hd], so heads and batch swap; and rope
        # multiplies by f32 cos/sin, which can promote the dtype, so cast back.
        if self.traceable_kv:
            # paged_update_cache wants [1, batch, n_kv, hd] -- the projections'
            # own shape -- L1 height-sharded, with the index tensor in DRAM.
            pos_tensor = self._input(
                "cur_pos", torch.tensor(positions, dtype=torch.int32), ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            k_s = self._l1_height_sharded(ttnn.typecast(k, ttnn.bfloat16), hd)
            v_s = self._l1_height_sharded(ttnn.typecast(v, ttnn.bfloat16), hd)
            ttnn.experimental.paged_update_cache(st.keys, k_s, update_idxs_tensor=pos_tensor)
            ttnn.experimental.paged_update_cache(st.values, v_s, update_idxs_tensor=pos_tensor)
            if self.use_indexer:
                out = ttnn.transformer.scaled_dot_product_attention_decode(
                    q, st.keys, st.values, is_causal=False,
                    attn_mask=self._indexer_select(mixed, layer, st, positions, cos, sin),
                    cur_pos_tensor=pos_tensor,
                    scale=hd**-0.5, program_config=self.sdpa_program_config,
                    compute_kernel_config=HIFI4,
                )
            else:
                out = ttnn.transformer.scaled_dot_product_attention_decode(
                    q, st.keys, st.values, is_causal=True, cur_pos_tensor=pos_tensor,
                    scale=hd**-0.5, program_config=self.sdpa_program_config,
                    compute_kernel_config=HIFI4,
                )
        else:
            k_upd = ttnn.typecast(ttnn.permute(k, (0, 2, 1, 3)), ttnn.bfloat16)
            v_upd = ttnn.typecast(ttnn.permute(v, (0, 2, 1, 3)), ttnn.bfloat16)
            # `update_cache` sizes its circular buffers from the whole batch and
            # overflows L1 past ~64 sequences ("grow to 2437888 B beyond max L1
            # size of 1572864 B"). Splitting it with `batch_offset` does not help:
            # the op asserts `batch_offset == 0` once the cache batch reaches 32.
            # Past 64 sequences use traceable_kv=True, whose paged_update_cache
            # shards across cores instead.
            #
            # The position is a single int, so it writes *every* sequence's K/V at
            # the same cache index. That holds only while the batch advances in
            # lockstep from a common start. Continuous batching breaks it the
            # moment a slot is refilled mid-flight: the new sequence sits at
            # position 0 while its neighbours are at 500, and one index cannot be
            # right for both -- reads below already use per-sequence `cur_pos`.
            # Caught by a test that reset one slot and saw a neighbour's logits
            # move by 5.6. Refuse rather than silently corrupt; traceable_kv=True
            # takes the position as a per-sequence tensor and is the correct path.
            if len(set(positions)) > 1:
                raise ValueError(
                    "update_cache writes one cache index for the whole batch, but "
                    f"sequences are at differing positions ({sorted(set(positions))[:4]}...). "
                    "Construct TTModel(traceable_kv=True) for continuous batching."
                )
            ttnn.update_cache(st.keys, k_upd, positions[0])
            ttnn.update_cache(st.values, v_upd, positions[0])
            # sdpa_decode wants q as [1, b, nh, dh]; it handles n_kv < n_q internally
            out = ttnn.transformer.scaled_dot_product_attention_decode(
                q, st.keys, st.values, is_causal=True, cur_pos=positions,
                scale=hd**-0.5, program_config=self.sdpa_program_config,
                compute_kernel_config=HIFI4,
            )

        out = ttnn.reshape(out, (1, 1, batch, n_q * hd))
        gate = ttnn.reshape(gate, (1, 1, batch, n_q * hd))
        out = ttnn.multiply(out, ttnn.sigmoid(gate))
        return ttnn.linear(out, self.w.blk(layer, "attn_output.weight"), compute_kernel_config=HIFI4)

    # -- PLE (n-gram) injection, one token ------------------------------------

    # -- k tokens of one sequence, in one step --------------------------------

    def _linear_attention_step_n(
        self, mixed: ttnn.Tensor, layer: int, st: LayerState, k: int
    ) -> ttnn.Tensor:
        """DeltaNet over `k` consecutive tokens of *one* sequence.

        The projections, the output norm and `ssm_out` are per-token, so they
        run over all k rows at once and cost what one row costs. The convolution
        and the recurrence are not: row i's window is rows i-3..i and row i's
        state is row i-1's, so those two are unrolled.

        That is the whole trade. A batched step is flat in batch up to 64 rows,
        so verifying k drafts costs one step plus k-1 unrollings of the two
        sequential pieces -- against k full steps for feeding them one at a time.
        """
        cfg = self.cfg
        n_v, hd = self.n_v_local, cfg.linear_head_dim
        kd, vd = self.key_dim_local, self.value_dim_local

        qkv = ttnn.linear(mixed, self.w.blk(layer, "attn_qkv.weight"), compute_kernel_config=HIFI4)
        z = ttnn.linear(mixed, self.w.blk(layer, "attn_gate.weight"), compute_kernel_config=HIFI4)
        a = ttnn.linear(mixed, self.w.blk(layer, "ssm_alpha.weight"), compute_kernel_config=HIFI4)
        b = ttnn.linear(mixed, self.w.blk(layer, "ssm_beta.weight"), compute_kernel_config=HIFI4)
        g = ttnn.multiply(
            self.w.blk(layer, "ssm_a"),
            ttnn.softplus(ttnn.add(a, self.w.blk(layer, "ssm_dt.bias"))),
        )

        # [1,1,k,conv_dim] -> [1,k,conv_dim,1]; one column per token
        qkv_col = ttnn.transpose(ttnn.permute(qkv, (0, 2, 1, 3)), -2, -1)
        conv_w = self.w.blk(layer, "ssm_conv1d.weight")
        cols = []
        for i in range(k):
            col = ttnn.slice(qkv_col, (0, i, 0, 0), (1, i + 1, self.conv_dim_local, 1))
            out_i, st.conv = self._causal_conv_step(
                col, conv_w, st.conv, self.conv_dim_local, 1, st.conv_step, layer
            )
            st.conv_step += 1
            cols.append(out_i)
        conv_out = cols[0] if k == 1 else ttnn.concat(cols, dim=1)
        qkv = ttnn.permute(ttnn.transpose(conv_out, -2, -1), (0, 2, 1, 3))

        q = ttnn.reshape(self._slice_last(qkv, 0, kd), (k * n_v, 1, 1, hd))
        kk = ttnn.reshape(self._slice_last(qkv, kd, 2 * kd), (k * n_v, 1, 1, hd))
        v = ttnn.reshape(self._slice_last(qkv, 2 * kd, 2 * kd + vd), (k * n_v, 1, 1, hd))
        q = ttnn.multiply(self._l2norm(q), hd**-0.5)
        kk = self._l2norm(kk)
        g_exp = ttnn.reshape(ttnn.exp(g), (k * n_v, 1, 1, 1))
        beta = ttnn.reshape(ttnn.sigmoid(b), (k * n_v, 1, 1, 1))

        if st.recurrent is None:
            st.recurrent = ttnn.zeros(
                (n_v, 1, hd, hd), dtype=self.state_dtype,
                layout=ttnn.TILE_LAYOUT, device=self.mesh,
            )
        outs = []
        for i in range(k):
            lo, hi = i * n_v, (i + 1) * n_v
            outs.append(
                linear_attn.decode_step(
                    ttnn.slice(q, (lo, 0, 0, 0), (hi, 1, 1, hd)),
                    ttnn.slice(kk, (lo, 0, 0, 0), (hi, 1, 1, hd)),
                    ttnn.slice(v, (lo, 0, 0, 0), (hi, 1, 1, hd)),
                    ttnn.slice(g_exp, (lo, 0, 0, 0), (hi, 1, 1, 1)),
                    ttnn.slice(beta, (lo, 0, 0, 0), (hi, 1, 1, 1)),
                    st.recurrent,
                )
            )
        out = outs[0] if k == 1 else ttnn.concat(outs, dim=0)

        out = ttnn.reshape(out, (1, 1, k * n_v, hd))
        z_heads = ttnn.reshape(z, (1, 1, k * n_v, hd))
        normed = ttnn.rms_norm(
            out, epsilon=cfg.rms_norm_eps, weight=self.w.blk(layer, "ssm_norm.weight"),
            compute_kernel_config=HIFI4,
        )
        gated = ttnn.reshape(
            ttnn.multiply(normed, ttnn.sigmoid(z_heads)), (1, 1, k, vd)
        )
        return self.all_reduce(
            ttnn.linear(gated, self.w.blk(layer, "ssm_out.weight"), compute_kernel_config=HIFI4)
        )

    def _ple_step_n(
        self, hidden: ttnn.Tensor, layer: int, st: LayerState, histories: list[list[int]]
    ) -> ttnn.Tensor:
        """PLE injection for `k` rows of one sequence.

        The dilated conv reads a ring that has to advance once per token, and
        the n-gram hash reads each row's own history, so this is one `_ple_step`
        per row rather than a batched call. It costs little: PLE runs on one
        layer of forty-eight.
        """
        cfg = self.cfg
        outs = []
        for i, hist in enumerate(histories):
            row = ttnn.slice(hidden, (0, 0, i, 0), (1, 1, i + 1, cfg.hc_hidden_size))
            outs.append(
                self._ple_step(
                    row, layer, st, None, histories=[hist],
                    ngram_name=f"ngram_n{len(histories)}_{i}",
                )
            )
        return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=-2)

    def _attention_step_n(
        self, mixed: ttnn.Tensor, layer: int, st: LayerState, start: int, k: int
    ) -> ttnn.Tensor:
        """QSA over `k` consecutive tokens of one sequence.

        The projections and the output gate are per-token and run over all k
        rows. The attention itself is k `sdpa_decode` calls: all k rows are
        written to the cache first, then row i reads with `cur_pos = start + i`,
        which is exactly its causal window and already contains the rows before
        it.

        Deliberately not the chunk path's single masked
        `scaled_dot_product_attention`: that slices the cache to `start + k`
        rounded up to a tile, so its shapes grow with position and a captured
        trace would only be valid inside the tile it was captured in.
        `sdpa_decode` takes the position as a tensor, so the graph is fixed --
        and it is flash-decode rather than a dense read. It also leaves room for
        the sparse selection, which is per-row and per-position.
        """
        cfg = self.cfg
        hd, n_q, n_kv = cfg.head_dim, cfg.num_attention_heads, cfg.num_kv_heads

        qg = ttnn.reshape(
            ttnn.linear(mixed, self.w.blk(layer, "attn_q.weight"), compute_kernel_config=HIFI4),
            (1, k, n_q, hd * 2),
        )
        q = rms_norm(self._slice_last(qg, 0, hd), self.w.blk(layer, "attn_q_norm.weight"),
                     cfg.rms_norm_eps)
        gate = self._slice_last(qg, hd, hd * 2)
        kt = rms_norm(
            ttnn.reshape(
                ttnn.linear(mixed, self.w.blk(layer, "attn_k.weight"), compute_kernel_config=HIFI4),
                (1, k, n_kv, hd),
            ),
            self.w.blk(layer, "attn_k_norm.weight"), cfg.rms_norm_eps,
        )
        vt = ttnn.reshape(
            ttnn.linear(mixed, self.w.blk(layer, "attn_v.weight"), compute_kernel_config=HIFI4),
            (1, k, n_kv, hd),
        )

        positions = list(range(start, start + k))
        cos_t, sin_t = self.rope(positions)
        # Names carry k: one capture per k, and their bound buffers differ in
        # shape, so sharing a name would hand a k=4 host tensor to a k=2 buffer.
        cos = self._input(f"stepn{k}_cos", cos_t, ttnn.float32)
        sin = self._input(f"stepn{k}_sin", sin_t, ttnn.float32)
        q = self._apply_rope_dev(q, cos, sin)
        kr = self._apply_rope_dev(kt, cos, sin)

        if st.keys is None:
            shape = (1, n_kv, self.max_seq_len, hd)
            st.keys = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)
            st.values = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)

        # every row is written before any is read, so row i's window already
        # contains rows 0..i
        idxs = []
        for i, pos in enumerate(positions):
            idx = self._input(
                f"stepn{k}_pos{i}", torch.tensor([pos], dtype=torch.int32), ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            idxs.append(idx)
            k_row = ttnn.reshape(ttnn.slice(kr, (0, i, 0, 0), (1, i + 1, n_kv, hd)), (1, 1, n_kv, hd))
            v_row = ttnn.reshape(ttnn.slice(vt, (0, i, 0, 0), (1, i + 1, n_kv, hd)), (1, 1, n_kv, hd))
            ttnn.experimental.paged_update_cache(
                st.keys, self._l1_height_sharded(ttnn.typecast(k_row, ttnn.bfloat16), hd),
                update_idxs_tensor=idx,
            )
            ttnn.experimental.paged_update_cache(
                st.values, self._l1_height_sharded(ttnn.typecast(v_row, ttnn.bfloat16), hd),
                update_idxs_tensor=idx,
            )

        outs = []
        for i in range(k):
            q_row = ttnn.reshape(ttnn.slice(q, (0, i, 0, 0), (1, i + 1, n_q, hd)), (1, 1, n_q, hd))
            outs.append(
                ttnn.transformer.scaled_dot_product_attention_decode(
                    q_row, st.keys, st.values, is_causal=True, cur_pos_tensor=idxs[i],
                    scale=hd**-0.5, program_config=self.sdpa_program_config,
                    compute_kernel_config=HIFI4,
                )
            )
        out = outs[0] if k == 1 else ttnn.concat(outs, dim=1)

        out = ttnn.reshape(out, (1, 1, k, n_q * hd))
        gate = ttnn.reshape(gate, (1, 1, k, n_q * hd))
        return ttnn.linear(
            ttnn.multiply(out, ttnn.sigmoid(gate)),
            self.w.blk(layer, "attn_output.weight"), compute_kernel_config=HIFI4,
        )


    def _ple_step(
        self, hidden: ttnn.Tensor, layer: int, st: LayerState, state: TTState | None,
        histories: list[list[int]] | None = None, ngram_name: str = "ngram",
    ) -> ttnn.Tensor:
        cfg = self.cfg
        batch = hidden.shape[-2]
        if histories is None:
            histories = state.histories
        emb = self._input(ngram_name, self.ngram_embed(histories), ttnn.bfloat16)

        key = grouped_rms_norm(
            ttnn.linear(emb, self.w.blk(layer, "ple_key.weight"), compute_kernel_config=HIFI4),
            self.w.blk(layer, "ple_norm_key.weight"), cfg.rms_norm_eps, cfg.hidden_size, cfg.hc_count,
        )
        value = ttnn.linear(emb, self.w.blk(layer, "ple_value.weight"), compute_kernel_config=HIFI4)
        query = grouped_rms_norm(
            hidden, self.w.blk(layer, "ple_norm_query.weight"), cfg.rms_norm_eps,
            cfg.hidden_size, cfg.hc_count,
        )

        # Per-stream dot product of key and query, in channel-major layout
        # [1, hc, B, hidden]. Doing this grouped instead -- reshaping to
        # (1, 1, B*hc, hidden) and expanding `value` with ttnn.repeat -- silently
        # mismatches for B > 1: `repeat` tiles the batch (b0,b1,b0,b1,...) while
        # the gate is grouped (b0c0,b0c1,...). The two orders coincide only at
        # B == 1, so it passes every single-sequence test.
        def to_streams(t):
            t = ttnn.reshape(t, (1, batch, cfg.hc_count, cfg.hidden_size))
            return ttnn.permute(t, (0, 2, 1, 3))                      # [1, hc, B, hidden]

        prod = ttnn.multiply(to_streams(key), to_streams(query))
        gate = ttnn.multiply(ttnn.sum(prod, dim=-1, keepdim=True), 1.0 / math.sqrt(cfg.hidden_size))
        # signed square root: keeps the sign, compresses the magnitude.
        # clamp_min matches the reference exactly (it floors |gate| at 1e-6).
        gate = ttnn.multiply(
            ttnn.sqrt(ttnn.clamp(ttnn.abs(gate), min=1e-6)), ttnn.sign(gate)
        )
        gate = ttnn.sigmoid(gate)                                     # [1, hc, B, 1]

        val = ttnn.reshape(value, (1, 1, batch, cfg.hidden_size))     # broadcasts over hc
        gated = ttnn.multiply(gate, val)                              # [1, hc, B, hidden]
        gated = ttnn.reshape(
            ttnn.permute(gated, (0, 2, 1, 3)), (1, 1, batch, cfg.hc_hidden_size)
        )

        gated_normed = grouped_rms_norm(
            gated, self.w.blk(layer, "ple_norm_conv.weight"), cfg.rms_norm_eps,
            cfg.hidden_size, cfg.hc_count,
        )
        # Dilated depthwise conv: dilation == ngram_size, so the window spans
        # (kernel-1)*ngram_size past steps and only every ngram_size-th tap is used.
        conv_w = self.w.blk(layer, "ple_conv1d.weight")
        # per-sequence window: [1,1,B,C] -> [1,B,C,1]
        col = ttnn.transpose(ttnn.permute(gated_normed, (0, 2, 1, 3)), -2, -1)
        state_len = (cfg.ple_conv_kernel - 1) * cfg.ngram_size
        c_dim = cfg.hc_hidden_size
        if st.ple_conv is None:
            st.ple_conv = [
                ttnn.zeros((1, batch, c_dim, 1), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=self.mesh)
                for _ in range(state_len)
            ]
            st.ple_step = 0

        # The window is a ring of single columns, so advancing it costs a Python
        # list assignment instead of device data movement. Holding it as one
        # [1, B, 10240, state_len] tensor means a slice + concat + copy every
        # token -- ~12 M element-ops at batch 64 -- to shuffle data that does not
        # change. The dilated taps only ever read 3, 6 and 9 steps back, and the
        # oldest of those is exactly the slot about to be overwritten.
        ring = st.ple_conv
        ple_taps = self.conv_taps(("ple", layer), conv_w, c_dim, cfg.ple_conv_kernel)
        acc = None
        if self.trace_safe_rings:
            # fixed read indices + a device-copy shift; see _causal_conv_step
            for tap in range(cfg.ple_conv_kernel):
                age = state_len - tap * cfg.ngram_size      # 9, 6, 3, 0 steps back
                piece = col if age == 0 else ring[age - 1]
                w_tap = ple_taps[tap]
                term = ttnn.multiply(piece, w_tap)
                acc = term if acc is None else ttnn.add(acc, term)
            for i in range(state_len - 1, 0, -1):
                ttnn.copy(ring[i - 1], ring[i])
            ttnn.copy(col, ring[0])
        else:
            pos = st.ple_step % state_len
            for tap in range(cfg.ple_conv_kernel):
                age = state_len - tap * cfg.ngram_size      # 9, 6, 3, 0 steps back
                piece = col if age == 0 else ring[(pos + (state_len - age)) % state_len]
                w_tap = ple_taps[tap]
                term = ttnn.multiply(piece, w_tap)
                acc = term if acc is None else ttnn.add(acc, term)
            ring[pos] = col
        conv = acc
        st.ple_step += 1

        # [1,B,C,1] -> [1,1,B,C]
        conv = ttnn.silu(ttnn.permute(ttnn.transpose(conv, -2, -1), (0, 2, 1, 3)))
        return ttnn.add(gated, conv)

    # -- one decoder layer ----------------------------------------------------

    def _layer(self, hidden: ttnn.Tensor, layer: int, state: TTState, position: int) -> ttnn.Tensor:
        cfg = self.cfg
        st = state[layer]

        if layer in self._ple_layers:
            hidden = ttnn.add(hidden, self._ple_step(hidden, layer, st, state))

        mixed, inject = gated_residual_mix(
            hidden,
            self.w.blk(layer, "hc_attn_norm.weight"),
            self.w.blk(layer, "hc_attn_down.weight"),
            self.w.blk(layer, "hc_attn_up.weight"),
            self.w.blk(layer, "hc_attn_inject.weight"),
            cfg.rms_norm_eps, cfg.hc_count, cfg.hidden_size,
        )
        branch = (
            self._attention_step(mixed, layer, st, position)
            if cfg.is_full_attention(layer)
            else self._linear_attention_step(mixed, layer, st)
        )
        hidden = reinject(hidden, branch, inject, cfg.hc_count)

        mixed, inject = gated_residual_mix(
            hidden,
            self.w.blk(layer, "hc_ffn_norm.weight"),
            self.w.blk(layer, "hc_ffn_down.weight"),
            self.w.blk(layer, "hc_ffn_up.weight"),
            self.w.blk(layer, "hc_ffn_inject.weight"),
            cfg.rms_norm_eps, cfg.hc_count, cfg.hidden_size,
        )
        return reinject(hidden, self._moe_block(mixed, layer), inject, cfg.hc_count)

    def _moe_block(self, mixed: ttnn.Tensor, layer: int) -> ttnn.Tensor:
        """Routed experts plus the shared one. Per-token, so it is unchanged by
        how many rows the caller brings."""
        cfg = self.cfg
        if self.fuse_expert_gate_up:
            gate_w, up_w = self.w.fused_gate_up(layer), None
        else:
            gate_w = self.w.blk(layer, "ffn_gate_exps.weight")
            up_w = self.w.blk(layer, "ffn_up_exps.weight")
        routed = moe.moe_block(
            mixed,
            self.w.blk(layer, "ffn_gate_inp.weight"),
            gate_w,
            up_w,
            self.w.blk(layer, "ffn_down_exps.weight"),
            cfg.num_experts_per_tok, cfg.num_experts, cfg.hidden_size, cfg.expert_intermediate,
        )
        # down_proj is sharded on its contraction dim, so each device holds a
        # partial sum -- this is the single collective per layer.
        routed = self.all_reduce(routed)
        shared = moe.shared_expert(
            mixed,
            self.w.blk(layer, "ffn_gate_shexp.weight"),
            self.w.blk(layer, "ffn_up_shexp.weight"),
            self.w.blk(layer, "ffn_down_shexp.weight"),
            self.w.blk(layer, "ffn_gate_inp_shexp.weight"),
        )
        return ttnn.add(routed, shared)

    # -- public API ------------------------------------------------------------

    def step(self, tokens: int | list[int], state: TTState) -> ttnn.Tensor:
        """Advance every sequence by one token.

        `tokens` is one id per sequence (an int is accepted for batch 1).
        Returns the final hidden state [1, 1, B, hidden].
        """
        cfg = self.cfg
        ids = [tokens] if isinstance(tokens, int) else list(tokens)
        for seq, tok in enumerate(ids):
            state.histories[seq].append(tok)

        emb = self._input("embed", self.embed(ids), ttnn.bfloat16)
        hidden = ttnn.repeat(emb, (1, 1, 1, cfg.hc_count))

        positions = list(state.positions)
        for layer in range(cfg.num_layers):
            hidden = self._layer(hidden, layer, state, positions)
            if self.probe is not None:
                self.probe(layer, hidden)

        mixed, _ = gated_residual_mix(
            hidden,
            self.w.get("output_hc_norm.weight"),
            self.w.get("output_hc_down.weight"),
            self.w.get("output_hc_up.weight"),
            None,
            cfg.rms_norm_eps, cfg.hc_count, cfg.hidden_size,
        )
        state.positions = [p + 1 for p in state.positions]
        return mixed

    def step_n(self, tokens: list[int], state: TTState) -> ttnn.Tensor:
        """Advance one sequence by `k` tokens in a single step.

        Returns the mixed hidden for all k positions, [1, 1, k, hidden], so a
        caller can read the logits the sequence would have produced at each --
        which is what a speculative verifier needs.

        The point is that a step is flat in batch up to 64 rows, so the k tokens
        ride the batch axis for everything that is per-token, and only the
        convolution and the DeltaNet recurrence are unrolled. Measured against
        the alternative of feeding them one at a time
        (`scripts/dev/short_chunk_bench.py` and `step_n_check.py`).

        Single-sequence: `state` must have `batch == 1`, and its caches and rings
        stay single-sequence while the activations carry k rows. On return the
        state has consumed all k tokens -- committing only a prefix means
        snapshotting first and replaying, which is a caller's problem and the
        remaining piece of speculative decoding.

        The chunked path is *not* the right verifier, which is why this exists:
        the DeltaNet op fixes its chunk at 128 and pads up to it, so a k-token
        pass does most of a 128-token pass's work -- 850 ms for k=1 against a
        236 ms traced step (`short_chunk_bench.py`). It is no longer *host*
        bound -- `prepare_device` moved that on device, which took the fixed
        cost from ~1.3 s to 850 ms -- but 850 ms is still the wrong shape for
        verifying two or three drafted tokens.
        """
        cfg = self.cfg
        k = len(tokens)
        if state.batch != 1:
            raise NotImplementedError("step_n advances one sequence at a time")
        if not 0 < k <= 64:
            raise ValueError(f"k must be in 1..64 (the batch cliff), got {k}")
        if self.use_indexer:
            # `_attention_step_n` reads with `sdpa_decode` per row, which leaves
            # room for a per-row selection mask, but the selection is not wired
            # in yet -- and running dense here while `step` runs sparse would make
            # the two disagree beyond the budget, silently.
            raise NotImplementedError(
                "step_n does not carry the QSA selection yet; construct the model "
                f"with max_seq_len <= {self.cfg.indexer_budget} to use it"
            )
        start = state.positions[0]

        # Each row's PLE n-gram hash reads that row's own history, so the
        # histories have to grow as the rows do.
        base = list(state.histories[0])
        histories = []
        for i, tok in enumerate(tokens):
            base.append(tok)
            histories.append(list(base))
        state.histories[0] = base

        emb = self._input(f"embed_n{k}", self.embed(tokens), ttnn.bfloat16)
        hidden = ttnn.repeat(emb, (1, 1, 1, cfg.hc_count))

        for layer in range(cfg.num_layers):
            st = state[layer]
            if layer in self._ple_layers:
                hidden = ttnn.add(hidden, self._ple_step_n(hidden, layer, st, histories))
            mixed, inject = gated_residual_mix(
                hidden, self.w.blk(layer, "hc_attn_norm.weight"),
                self.w.blk(layer, "hc_attn_down.weight"), self.w.blk(layer, "hc_attn_up.weight"),
                self.w.blk(layer, "hc_attn_inject.weight"),
                cfg.rms_norm_eps, cfg.hc_count, cfg.hidden_size,
            )
            branch = (
                self._attention_step_n(mixed, layer, st, start, k)
                if cfg.is_full_attention(layer)
                else self._linear_attention_step_n(mixed, layer, st, k)
            )
            hidden = reinject(hidden, branch, inject, cfg.hc_count)

            mixed, inject = gated_residual_mix(
                hidden, self.w.blk(layer, "hc_ffn_norm.weight"),
                self.w.blk(layer, "hc_ffn_down.weight"), self.w.blk(layer, "hc_ffn_up.weight"),
                self.w.blk(layer, "hc_ffn_inject.weight"),
                cfg.rms_norm_eps, cfg.hc_count, cfg.hidden_size,
            )
            hidden = reinject(hidden, self._moe_block(mixed, layer), inject, cfg.hc_count)
            if self.probe is not None:
                self.probe(layer, hidden)

        mixed, _ = gated_residual_mix(
            hidden, self.w.get("output_hc_norm.weight"), self.w.get("output_hc_down.weight"),
            self.w.get("output_hc_up.weight"), None,
            cfg.rms_norm_eps, cfg.hc_count, cfg.hidden_size,
        )
        state.positions = [start + k]
        return mixed

    def logits(self, hidden: ttnn.Tensor) -> torch.Tensor:
        """Vocabulary-sharded LM head; all-gathers the logits to host.

        Returns [B, vocab_size].
        """
        batch = hidden.shape[-2]
        part = ttnn.linear(hidden, self.w.get("output.weight"), compute_kernel_config=HIFI4)
        full = ttnn.to_torch(part, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))
        return full.reshape(batch, -1)[:, : self.cfg.vocab_size]

    def greedy_tokens(self, hidden: ttnn.Tensor) -> list[int] | None:
        """argmax over the vocabulary without gathering the logits to host.

        `logits()` moves [B, vocab] float32 off four devices every step -- 119 ms
        at B=32, 17 % of the whole step -- when greedy decoding needs exactly one
        integer per sequence.

        The LM head is column-sharded, so device d owns vocabulary columns
        [d*W, (d+1)*W). A per-device (max, argmax) therefore composes into the
        global argmax: pick the device holding the largest value and offset its
        local index. That is only sound when the split is exact -- otherwise the
        last shard is zero-padded and a padded column can win against genuinely
        negative logits -- so an uneven vocabulary returns None and the caller
        falls back to `logits()`. For this model 248320/4 = 62080, itself a whole
        number of 32-wide tiles, so nothing is padded.

        Returns one token id per sequence, or None if the fast path does not apply.
        """
        vocab, n_dev = self.cfg.vocab_size, self.n_dev
        if vocab % n_dev != 0:
            return None
        width = vocab // n_dev

        part = ttnn.linear(hidden, self.w.get("output.weight"), compute_kernel_config=HIFI4)
        best = ttnn.max(part, dim=-1, keepdim=True)
        where = ttnn.argmax(part, dim=-1, keepdim=True)
        # [n_dev, 1, B, 1] once the per-device results are stacked
        vals = ttnn.to_torch(best, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        idxs = ttnn.to_torch(where, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        vals = vals.reshape(n_dev, -1).float()
        idxs = idxs.reshape(n_dev, -1).long()
        winner = vals.argmax(dim=0)                       # which device won, per sequence
        cols = torch.arange(vals.shape[1])
        return (idxs[winner, cols] + winner * width).tolist()

    def new_state(self, batch: int = 1) -> TTState:
        return TTState(self.cfg.num_layers, batch)

    def snapshot(self, state: TTState, into: dict | None = None) -> dict:
        """Copy everything a single-sequence state carries, for a later rewind.

        Speculation needs this and a transformer does not: verifying k drafts
        advances the DeltaNet recurrent state and the convolution rings by all k
        tokens, and unlike a K/V cache -- which attention simply reads less of --
        a recurrence cannot be truncated back to the accepted prefix.

        The K/V and the indexer's block cache are deliberately *not* copied.
        Attention reads only up to `cur_pos`, so whatever a rejected draft left
        beyond the accepted position is never looked at, and copying 6.4 GB a
        slot to protect data nobody reads would cost more than the speculation
        saves. What must be copied is everything that accumulates
        unconditionally.
        """
        if state.batch != 1:
            raise NotImplementedError("snapshot is single-sequence")

        # Pass a previous snapshot back as `into` to reuse its buffers. A
        # speculative round snapshots every time, and allocating ~200 tensors
        # per round is both slow and an allocation while a trace is live, which
        # is the thing this file keeps warning about.
        reuse = into is not None

        def take(t, dst):
            if dst is None:
                dst = ttnn.zeros(list(t.shape), dtype=t.dtype, layout=t.layout, device=self.mesh)
            ttnn.copy(t, dst)
            return dst

        layers = []
        for i, st in enumerate(state.layers):
            prev = into["layers"][i] if reuse else {}
            entry = {"conv_step": st.conv_step, "ple_step": st.ple_step}
            entry["recurrent"] = (
                None if st.recurrent is None else take(st.recurrent, prev.get("recurrent"))
            )
            for name in ("conv", "ple_conv"):
                src = getattr(st, name)
                if src is None:
                    entry[name] = None
                    continue
                held = prev.get(name) or [None] * len(src)
                entry[name] = [take(c, held[j]) for j, c in enumerate(src)]
            layers.append(entry)
        return {
            "layers": layers,
            "positions": list(state.positions),
            "histories": [list(h) for h in state.histories],
        }

    def restore(self, state: TTState, snap: dict) -> None:
        """Rewind to a `snapshot`, writing in place.

        In place because a captured trace replays against the addresses it
        recorded: rebinding the rings to the snapshot's tensors would leave the
        trace reading whatever used to be there.
        """
        for st, saved in zip(state.layers, snap["layers"]):
            if saved["recurrent"] is not None:
                ttnn.copy(saved["recurrent"], st.recurrent)
            for name in ("conv", "ple_conv"):
                if saved[name] is None:
                    continue
                for src, dst in zip(saved[name], getattr(st, name)):
                    ttnn.copy(src, dst)
            st.conv_step = saved["conv_step"]
            st.ple_step = saved["ple_step"]
        state.positions = list(snap["positions"])
        state.histories = [list(h) for h in snap["histories"]]

    def reset_slot(self, state: TTState, slot: int) -> None:
        """Clear one sequence's state so a fresh sequence can take the slot.

        This is what makes continuous batching possible: slots free and refill
        independently while the other sequences keep decoding, so the batch does
        not have to drain between requests.

        The K/V cache deliberately is *not* cleared. Attention reads only up to
        `cur_pos`, so whatever the previous occupant left beyond the new position
        is never looked at, and zeroing 48 layers of cache per admission would
        cost far more than it saves. What must be cleared is everything that
        accumulates unconditionally: the DeltaNet recurrent state and the two
        convolution rings.

        Every write is in place (`output_tensor=`). Rebinding these to fresh
        tensors would change their addresses, which silently invalidates any
        captured trace replaying against the old ones.
        """
        b = state.batch
        if not 0 <= slot < b:
            raise IndexError(f"slot {slot} out of range for batch {b}")

        def mask(shape, zero_rows, dtype):
            keep = torch.ones(*shape)
            keep[zero_rows] = 0.0
            return ttnn.from_torch(
                keep, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        nv = self.n_v_local
        # conv rings are [1, B, C, 1]; the recurrent state is [B*n_v, 1, Dk, Dv]
        # with head h of sequence i at row i*n_v + h. Both masks are built on
        # first use: a state whose tensors are not allocated yet (a slot admitted
        # before its first step) has nothing to clear and needs no device work.
        cache: dict[str, object] = {}

        def ring_mask():
            if "ring" not in cache:
                cache["ring"] = mask((1, b, 1, 1), (slice(None), slot), ttnn.bfloat16)
            return cache["ring"]

        def rec_mask():
            if "rec" not in cache:
                cache["rec"] = mask(
                    (b * nv, 1, 1, 1), slice(slot * nv, (slot + 1) * nv), self.state_dtype
                )
            return cache["rec"]

        for st in state.layers:
            if st.recurrent is not None:
                ttnn.multiply(st.recurrent, rec_mask(), output_tensor=st.recurrent)
            for ring in (st.conv, st.ple_conv):
                if ring is not None:
                    for t in ring:
                        ttnn.multiply(t, ring_mask(), output_tensor=t)

        state.positions[slot] = 0
        state.histories[slot] = []

    # -- chunked prefill --------------------------------------------------------

    def _ring_oldest_first(self, ring: list, step: int) -> list:
        """The ring's columns in age order, oldest first.

        The two decode modes lay a ring out differently and neither is simply
        "oldest at index 0": `trace_safe_rings` keeps the newest at index 0 and
        shifts on every step, while the rotating path leaves the oldest at
        `step % len(ring)` and moves that index instead. A chunk consuming a ring
        a previous chunk (or a decode step) left has to honour whichever is in
        force -- reading it in the wrong order is silent, and just permutes the
        convolution's taps.
        """
        n = len(ring)
        if self.trace_safe_rings:
            return list(reversed(ring))
        return [ring[(step + i) % n] for i in range(n)]

    def _ring_from_oldest_first(self, cols: list, step: int, into: list | None = None) -> list:
        """Lay `cols` (oldest first) out the way a decode step at `step` reads it.

        The inverse of `_ring_oldest_first`, so that a chunk leaves behind
        exactly the ring the decode path would have left after the same tokens
        -- which is what makes prefill's state comparable to decode's at all.

        With `into`, the columns are *copied* into those buffers instead of
        replacing them. Rebinding a ring to fresh tensors moves it, and a
        captured trace replays against the addresses it recorded -- so a prefill
        that rebound the rings would leave any traced decoder reading whatever
        used to be there. That is what kept chunked prefill out of the engine
        even once it was correct.
        """
        n = len(cols)
        order = list(reversed(cols)) if self.trace_safe_rings else [None] * n
        if not self.trace_safe_rings:
            for i, col in enumerate(cols):
                order[(step + i) % n] = col
        if into is None:
            return order
        for dst, src in zip(into, order):
            ttnn.copy(src, dst)
        return into

    def _causal_conv_chunk(
        self, x: ttnn.Tensor, weight: ttnn.Tensor, state: list | None, channels: int, seq: int,
        layer: int = 0, step: int = 0,
    ) -> tuple[ttnn.Tensor, list, int]:
        """4-tap depthwise causal conv over `seq` positions at once.

        x: [1, 1, channels, seq] (channel-major). Expressed as a sum of four
        shifted slices of the (state ++ x) window rather than a conv op, for the
        same reason as the single-step version: with a kernel of 4 it is just a
        weighted sum, and the rolling window stays a slice.
        """
        k = self.cfg.conv_kernel
        depth = k - 1
        # The decode path keeps this window as a *ring* of single [1,1,C,1]
        # columns, so prefill has to speak the same representation: it consumes
        # the ring left by any earlier chunk and leaves one the decode step can
        # pick up. (Before this it took a single [1,1,C,k-1] tensor and died in
        # concat the moment the two paths met.)
        if state is None:
            state = [
                ttnn.zeros((1, 1, channels, 1), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=self.mesh)
                for _ in range(depth)
            ]
        # oldest first, then this chunk: [1,1,C,depth+seq]
        window = ttnn.concat([*self._ring_oldest_first(state, step), x], dim=-1)
        acc = None
        # Keyed by layer, like the decode path. Keyed by channel count alone (as
        # this was), every DeltaNet layer after the first silently reused layer
        # 0's taps -- layer 0 matched decode to 0.005 and layer 1 was off by 30 %.
        for tap, w_tap in enumerate(self.conv_taps(("ssm_chunk", layer), weight, channels, k)):
            piece = ttnn.slice(window, (0, 0, 0, tap), (1, 1, channels, tap + seq))
            acc = ttnn.multiply(piece, w_tap) if acc is None else ttnn.add(
                acc, ttnn.multiply(piece, w_tap)
            )
        # The ring the next decode step reads: the final `depth` columns, laid
        # out for the counter this chunk leaves behind, so that consuming a
        # prompt by chunks and consuming it token by token leave the *same*
        # state -- ring and counter both.
        total = depth + seq
        cols = [
            ttnn.slice(window, (0, 0, 0, total - depth + i), (1, 1, channels, total - depth + i + 1))
            for i in range(depth)
        ]                                   # oldest .. newest
        new_step = step + seq
        # `state` is the ring this chunk consumed; writing back into it keeps the
        # buffers where a captured trace expects them.
        return ttnn.silu(acc), self._ring_from_oldest_first(cols, new_step, into=state), new_step

    def _linear_attention_chunk(
        self, mixed: ttnn.Tensor, layer: int, st: LayerState, seq: int
    ) -> ttnn.Tensor:
        """Gated DeltaNet over a whole chunk via `gated_delta_attn_seq`.

        The op's eight inputs are prepared on device by `prepare_device`, so a
        chunk costs no host round trip at all. The equivalent host routine is
        kept as `deltanet.prepare` and is what the port is checked against
        (`scripts/dev/deltanet_prepare_device_check.py`): they agree to the
        hardware's fp32 matmul floor, which is 0.16 % for a 128x128 matmul even
        at HiFi4 with fp32 accumulation.
        """
        from .deltanet import CHUNK, prepare_device

        cfg = self.cfg
        # Local head counts: the DeltaNet weights are head-sharded, so each device
        # owns a different slice of the heads and its own conv channels. Using the
        # global counts here (as this path did before) both mis-shapes the reshape
        # and, worse, silently prepares only device 0's heads for all four.
        n_v, hd = self.n_v_local, cfg.linear_head_dim

        qkv = ttnn.linear(mixed, self.w.blk(layer, "attn_qkv.weight"), compute_kernel_config=HIFI4)
        z = ttnn.linear(mixed, self.w.blk(layer, "attn_gate.weight"), compute_kernel_config=HIFI4)
        conv_out, st.conv, st.conv_step = self._causal_conv_chunk(
            ttnn.transpose(qkv, -2, -1), self.w.blk(layer, "ssm_conv1d.weight"),
            st.conv, self.conv_dim_local, seq, layer, st.conv_step,
        )
        qkv = ttnn.transpose(conv_out, -2, -1)              # [1,1,seq,conv_dim_local]

        a = ttnn.linear(mixed, self.w.blk(layer, "ssm_alpha.weight"), compute_kernel_config=HIFI4)
        b = ttnn.linear(mixed, self.w.blk(layer, "ssm_beta.weight"), compute_kernel_config=HIFI4)
        g = ttnn.multiply(
            self.w.blk(layer, "ssm_a"),
            ttnn.softplus(ttnn.add(a, self.w.blk(layer, "ssm_dt.bias"))),
        )

        # Device preparation, per device, in place. Each device already holds
        # exactly its own heads, so there is nothing to gather: the eight op
        # inputs are built from the sharded q/k/v with `prepare_device`, which
        # is `prepare` rewritten in ttnn ops. That removes a ~30 MB round trip
        # per layer per chunk, and -- the reason it was worth doing -- leaves a
        # graph a trace capture can see, where host arithmetic is invisible.
        kd = self.key_dim_local
        total = -(-seq // CHUNK) * CHUNK
        n_chunks = total // CHUNK

        def heads(t, lo, hi):
            """[1, 1, seq, n_v*hd] -> [n_v, NC, CHUNK, hd], zero-padded."""
            x = self._slice_last(t, lo, hi)
            width = (hi - lo) // n_v
            x = ttnn.permute(ttnn.reshape(x, (1, seq, n_v, width)), (2, 0, 1, 3))
            if total != seq:
                # `prepare` pads with F.pad; zeros here mean decay 0 and beta 0,
                # so the padded positions contribute nothing to the scan.
                pad = ttnn.zeros(
                    (n_v, 1, total - seq, width), dtype=x.dtype,
                    layout=ttnn.TILE_LAYOUT, device=self.mesh,
                )
                x = ttnn.concat([x, pad], dim=-2)
            return ttnn.reshape(ttnn.typecast(x, ttnn.float32), (n_v, n_chunks, CHUNK, width))

        hq, hk, hv = (
            heads(qkv, 0, kd), heads(qkv, kd, 2 * kd),
            heads(qkv, 2 * kd, 2 * kd + self.value_dim_local),
        )
        hg, hb = heads(g, 0, n_v), heads(ttnn.sigmoid(b), 0, n_v)
        dev = prepare_device(hq, hk, hv, hg, hb, mesh=self.mesh)
        initial = None
        if st.recurrent is not None:
            initial = ttnn.reshape(st.recurrent, (n_v, hd, hd))
        # HiFi4 with fp32 accumulation, for consistency with every other matmul
        # in the model. Measured: it changes nothing here, bit for bit -- the
        # op's inputs are already float32 and it does not appear to drop
        # precision in the places this config controls. The intra-chunk error
        # (0.36 % at position 0 of a 128-token chunk, 60 % by position 127,
        # against the reference's own chunked delta rule) is not this.
        out, final_state = ttnn.transformer.gated_delta_attn_seq(
            dev["L_unit"], dev["v_beta_sc"], dev["k_bd_sc"], dev["intra_attn"],
            dev["q_decay"], dev["k_decay_t"], dev["dl_exp"], dev["L_inv"],
            initial_state=initial, compute_kernel_config=HIFI4,
        )
        if st.recurrent is None:
            st.recurrent = ttnn.zeros(
                (n_v, 1, hd, hd), dtype=self.state_dtype, layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
            )
        ttnn.copy(ttnn.reshape(final_state, (n_v, 1, hd, hd)), st.recurrent)

        # [BH, NC, C, Dv] -> [1, 1, seq*n_v, hd], on device and per device.
        out = ttnn.slice(
            ttnn.reshape(out, (n_v, 1, n_chunks * CHUNK, hd)), (0, 0, 0, 0), (n_v, 1, seq, hd)
        )
        out_d = ttnn.typecast(
            ttnn.reshape(ttnn.permute(out, (1, 2, 0, 3)), (1, 1, seq * n_v, hd)), ttnn.bfloat16
        )

        z_heads = ttnn.reshape(z, (1, 1, seq * n_v, hd))
        normed = ttnn.rms_norm(
            out_d, epsilon=cfg.rms_norm_eps, weight=self.w.blk(layer, "ssm_norm.weight"),
            compute_kernel_config=HIFI4,
        )
        gated = ttnn.multiply(normed, ttnn.sigmoid(z_heads))
        gated = ttnn.reshape(gated, (1, 1, seq, self.value_dim_local))
        # ssm_out is row-sharded on its contraction dim, exactly as in the decode
        # step: every device holds a partial sum until the all-reduce. Without it
        # each device saw only its own quarter of the DeltaNet output while the
        # recurrent state (which does not pass through ssm_out) stayed correct.
        out = ttnn.linear(gated, self.w.blk(layer, "ssm_out.weight"), compute_kernel_config=HIFI4)
        return self.all_reduce(out)

    def _attention_chunk(
        self, mixed: ttnn.Tensor, layer: int, st: LayerState, start: int, seq: int
    ) -> ttnn.Tensor:
        """QSA attention over `seq` queries starting at absolute position `start`.

        `chunked_scaled_dot_product_attention` would fit but requires a paged
        cache and a page table; with an explicit additive mask the plain
        `scaled_dot_product_attention` handles history plus chunk directly, and
        the mask is small (seq x (start+seq) floats).

        Below 2048 tokens QSA retains every complete 4-token block, so the mask is
        exactly the causal one.
        """
        cfg = self.cfg
        hd, n_q, n_kv = cfg.head_dim, cfg.num_attention_heads, cfg.num_kv_heads

        qg = ttnn.linear(mixed, self.w.blk(layer, "attn_q.weight"), compute_kernel_config=HIFI4)
        qg = ttnn.reshape(qg, (1, seq, n_q, hd * 2))
        q = self._slice_last(qg, 0, hd)
        gate = self._slice_last(qg, hd, hd * 2)
        q = rms_norm(q, self.w.blk(layer, "attn_q_norm.weight"), cfg.rms_norm_eps)

        k = ttnn.reshape(
            ttnn.linear(mixed, self.w.blk(layer, "attn_k.weight"), compute_kernel_config=HIFI4),
            (1, seq, n_kv, hd),
        )
        v = ttnn.reshape(
            ttnn.linear(mixed, self.w.blk(layer, "attn_v.weight"), compute_kernel_config=HIFI4),
            (1, seq, n_kv, hd),
        )
        k = rms_norm(k, self.w.blk(layer, "attn_k_norm.weight"), cfg.rms_norm_eps)

        # rope: cos/sin come out [1, seq, 1, rope_dim]; transposed to
        # [1, 1, seq, rope_dim] they broadcast over the head axis of
        # q/k laid out as [1, n_heads, seq, hd].
        cos_t, sin_t = self.rope(list(range(start, start + seq)))
        cos = ttnn.permute(self.to_dev(cos_t, ttnn.float32), (0, 2, 1, 3))
        sin = ttnn.permute(self.to_dev(sin_t, ttnn.float32), (0, 2, 1, 3))
        q = self._apply_rope_dev(ttnn.permute(q, (0, 2, 1, 3)), cos, sin)
        k = self._apply_rope_dev(ttnn.permute(k, (0, 2, 1, 3)), cos, sin)

        if st.keys is None:
            shape = (1, n_kv, self.max_seq_len, hd)
            st.keys = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)
            st.values = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)
        ttnn.fill_cache(st.keys, k, 0, update_idx=start)
        ttnn.fill_cache(st.values, ttnn.permute(v, (0, 2, 1, 3)), 0, update_idx=start)

        total = start + seq
        # Slice to a whole number of tiles. Both the K/V slice and the mask are
        # TILE_LAYOUT, so a key length that is not a multiple of 32 gets padded
        # -- and an additive mask pads with *zeros*, which means "attend to me".
        # The softmax then spread over up to 31 all-zero key positions and
        # diluted the real ones: at one token this block returned roughly v/32
        # instead of v (99.7 % wrong against a float32 host computation), and
        # the error decayed as real positions crowded the padding out. Padding
        # explicitly and letting the causal condition run over the padded width
        # masks it, because every query position is < total by construction.
        kv_len = min(-(-total // ttnn.TILE_SIZE) * ttnn.TILE_SIZE, self.max_seq_len)
        keys = ttnn.slice(st.keys, (0, 0, 0, 0), (1, n_kv, kv_len, hd))
        values = ttnn.slice(st.values, (0, 0, 0, 0), (1, n_kv, kv_len, hd))
        groups = n_q // n_kv
        keys = ttnn.repeat_interleave(keys, groups, dim=1)
        values = ttnn.repeat_interleave(values, groups, dim=1)

        qpos = torch.arange(seq).unsqueeze(-1) + start
        kpos = torch.arange(kv_len).unsqueeze(0)
        mask = torch.where(kpos <= qpos, 0.0, float("-inf")).reshape(1, 1, seq, kv_len)
        out = ttnn.transformer.scaled_dot_product_attention(
            q, keys, values, attn_mask=self.to_dev(mask, ttnn.bfloat16), is_causal=False,
            scale=hd**-0.5, compute_kernel_config=HIFI4,
        )
        out = ttnn.reshape(ttnn.permute(out, (0, 2, 1, 3)), (1, 1, seq, n_q * hd))
        gate = ttnn.reshape(gate, (1, 1, seq, n_q * hd))
        out = ttnn.multiply(out, ttnn.sigmoid(gate))
        return ttnn.linear(out, self.w.blk(layer, "attn_output.weight"), compute_kernel_config=HIFI4)

    def prefill(self, token_ids: list[int], state: TTState, chunk: int = 128, moe_chunk: int = 32):
        """Consume a prompt in chunks; returns the final hidden state for the last token.

        **Not verified yet -- do not wire this into the engine.** It runs, and it
        is worth having: 128 prompt tokens in 5.77 s against 65.15 s through the
        decode path, 11.3x, which is the difference between a 100k-token prompt
        being minutes or hours.

        Status (2026-09-02, per-layer bisection against the decode path on a
        4-token prompt; `self.probe` is the hook, `docs/HANDOFF.md` the recipe):

        * two structural bugs found and fixed -- the DeltaNet chunk returned the
          `ssm_out` partial sums without the all-reduce, and its conv-tap cache
          was keyed without the layer, so every DeltaNet layer after the first
          used layer 0's conv weights. Layers 0-2 now agree with decode to ~1 %.
        * what remains is small and grows with depth: 5 % relative at layer 3
          (the first sparse-attention layer), ~25 % by layer 47, and the greedy
          token still differs at every prompt length tried. Whether this is the
          bf16 noise floor of two different-but-correct computations amplified by
          48 MoE layers, or a third bug in `_attention_chunk`, is undecided; the
          discriminating experiment is the per-layer diff of *decode* against
          the CPU reference on the same prompt, which sets the noise floor.

        Earlier fixes worth keeping: `_causal_conv_chunk` and `_ple_chunk` speak
        the ring representation the decode path uses, `_linear_attention_chunk`
        uses the local (head-sharded) head counts and prepares each device's own
        heads instead of broadcasting device 0's, and the MoE selects the same
        expert weights as decode -- naming the split halves while the model is
        fused loaded another ~11 GB per device and ran out of DRAM at the first
        MoE layer.

        `moe_chunk` matters: the MoE's broadcast formulation computes
        |union of selected experts| x M rows, and the union approaches all 512 as
        M grows. Splitting the chunk's MoE into `moe_chunk`-sized pieces trades
        dispatches against that waste (M=128 wastes ~51x, M=16 about 8x).

        Which way that trade falls is a measurement, and the FLOP count is the
        wrong end of it: a 128-token chunk issues 19733 device calls
        (`op_count.py --prefill 128`) and is dispatch-bound, so wasting compute
        to issue fewer calls wins -- up to a point. `moe_chunk_sweep.py`:

            moe_chunk    8   3332.8 ms    38.4 tok/s
            moe_chunk   16   2032.6 ms    63.0 tok/s
            moe_chunk   32   1101.0 ms   116.3 tok/s     <- default
            moe_chunk   64    936.1 ms   136.7 tok/s
            moe_chunk  128    998.7 ms   128.2 tok/s

        The default had been 16, chosen from the waste figures alone; 32 is
        1.85x faster.

        Past 32 the answer changes, and `_MAX_MOE_CHUNK` refuses rather than
        documents it, because the fastest setting is on the wrong side and it
        fails silently. Two independent measurements say so:

        * prefill is bit-deterministic in `moe_chunk` -- three repeats at one
          setting give identical logits -- and across settings the final logits
          move 55.9 % at 64 and 144.9 % at 128, with a different argmax
          (`moe_chunk_noise.py`).
        * `moe_block` stops being per-token. Each row picks its own experts, so
          grouping may only change speed; `moe_rows_check.py` computes the
          answer a row at a time and finds groups of 1, 8, 16 and 32 exact, and
          64 wrong on 33 of 64 rows, worst row 102.9 %.

        Where it is *not*: routing is exact at 64 (`keep`, `weights` and the
        expert union all match the host), and `topk`, the threshold mask,
        `max` over the row axis, `permute`, the expert-axis reduction and
        `sparse_matmul` itself are each exact at every row count in isolation,
        as are four `sparse_program_config` variants. The defect only appears
        in `expert_ffn` at production scale, so it does not reduce to a small
        case -- see handoff 5.8 before spending a day on it.
        """
        from .deltanet import CHUNK

        cfg = self.cfg
        # `prepare_device` pads a short sequence up to the op's 128-wide chunk
        # with zero decay and zero beta, which leave the carried state untouched,
        # so a chunk *smaller* than CHUNK is well defined -- just wasteful. It is
        # more accurate: the error grows with position inside a chunk (against
        # the reference's own chunked delta rule, `branch_chunk_check.py --seq
        # 128`: 0.35 % at position 0 rising to ~32 % by position 127, ~32-33 %
        # overall at layers 0, 4 and 8), so keeping the real tokens near the top
        # of the chunk is what buys the accuracy back. Half of that growth was
        # the unstable chunk inverse and is gone -- it read 60 % at position 127
        # before `block_diag_inverse` -- and what remains is the op's own
        # accumulation. Note the *token* is unaffected: prefilling 128 tokens
        # scores 53.1 % next-token top-1 against 51.6 % for stepping the same
        # ones, which is iteration 013's lesson that a per-layer distance cannot
        # tell noise from a bug. A chunk *larger* than CHUNK would
        # need the op's inter-chunk scan, which this path does not build here.
        # It must also be a whole number of tiles: `fill_cache` asserts
        # `update_idx % TILE_HEIGHT == 0`, and the attention chunk writes the K/V
        # cache at the chunk's absolute start.
        if not 0 < chunk <= CHUNK or chunk % ttnn.TILE_SIZE:
            raise ValueError(
                f"chunk must be a multiple of {ttnn.TILE_SIZE} in "
                f"{ttnn.TILE_SIZE}..{CHUNK} (the op's chunk width), got {chunk}"
            )
        if not 0 < moe_chunk <= _MAX_MOE_CHUNK:
            raise ValueError(
                f"moe_chunk must be in 1..{_MAX_MOE_CHUNK}, got {moe_chunk}. Above "
                f"{_MAX_MOE_CHUNK} the MoE block silently returns something worse: "
                "on a 128-token chunk next-token top-1 falls from 53.1 % to 43.8 % "
                "at 64 and 21.9 % at 128, while 8, 16 and 32 agree on every token. "
                "Raise the cap with a measurement, not a reason."
            )
        if state.batch != 1:
            raise NotImplementedError("chunked prefill is single-sequence for now")
        # Absolute position the state is already at, so a prompt can be
        # prefilled onto a slot that a previous turn left populated. `fill_cache`
        # wants a tile-aligned index, so a caller resuming mid-tile has to step
        # up to the boundary first.
        base = state.positions[0]
        if base % ttnn.TILE_SIZE:
            raise ValueError(
                f"prefill resumes only on a {ttnn.TILE_SIZE}-token boundary, "
                f"and this slot is at {base}"
            )
        final = None
        for begin in range(0, len(token_ids), chunk):
            ids = token_ids[begin : begin + chunk]
            seq = len(ids)
            state.histories[0].extend(ids)
            hidden = ttnn.repeat(
                self.to_dev(self.embed(ids), ttnn.bfloat16), (1, 1, 1, cfg.hc_count)
            )
            for layer in range(cfg.num_layers):
                st = state[layer]
                if layer in self._ple_layers:
                    hidden = ttnn.add(hidden, self._ple_chunk(hidden, layer, st, state, begin, seq))
                mixed, inject = gated_residual_mix(
                    hidden, self.w.blk(layer, "hc_attn_norm.weight"),
                    self.w.blk(layer, "hc_attn_down.weight"), self.w.blk(layer, "hc_attn_up.weight"),
                    self.w.blk(layer, "hc_attn_inject.weight"),
                    cfg.rms_norm_eps, cfg.hc_count, cfg.hidden_size,
                )
                branch = (
                    self._attention_chunk(mixed, layer, st, base + begin, seq)
                    if cfg.is_full_attention(layer)
                    else self._linear_attention_chunk(mixed, layer, st, seq)
                )
                hidden = reinject(hidden, branch, inject, cfg.hc_count)

                mixed, inject = gated_residual_mix(
                    hidden, self.w.blk(layer, "hc_ffn_norm.weight"),
                    self.w.blk(layer, "hc_ffn_down.weight"), self.w.blk(layer, "hc_ffn_up.weight"),
                    self.w.blk(layer, "hc_ffn_inject.weight"),
                    cfg.rms_norm_eps, cfg.hc_count, cfg.hidden_size,
                )
                pieces = []
                for sub in range(0, seq, moe_chunk):
                    width = min(moe_chunk, seq - sub)
                    part = ttnn.slice(mixed, (0, 0, sub, 0), (1, 1, sub + width, cfg.hidden_size))
                    # Same weight selection as decode. Naming the split halves
                    # here while the model is fused loads them lazily *on top of*
                    # the fused tensor -- another ~11 GB per device, which is an
                    # out-of-memory at the first MoE layer, not a slow path.
                    if self.fuse_expert_gate_up:
                        gate_w, up_w = self.w.fused_gate_up(layer), None
                    else:
                        gate_w = self.w.blk(layer, "ffn_gate_exps.weight")
                        up_w = self.w.blk(layer, "ffn_up_exps.weight")
                    routed = self.all_reduce(
                        moe.moe_block(
                            part, self.w.blk(layer, "ffn_gate_inp.weight"),
                            gate_w, up_w,
                            self.w.blk(layer, "ffn_down_exps.weight"),
                            cfg.num_experts_per_tok, cfg.num_experts,
                            cfg.hidden_size, cfg.expert_intermediate,
                        )
                    )
                    shared = moe.shared_expert(
                        part, self.w.blk(layer, "ffn_gate_shexp.weight"),
                        self.w.blk(layer, "ffn_up_shexp.weight"),
                        self.w.blk(layer, "ffn_down_shexp.weight"),
                        self.w.blk(layer, "ffn_gate_inp_shexp.weight"),
                    )
                    pieces.append(ttnn.add(routed, shared))
                ffn = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=-2)
                hidden = reinject(hidden, ffn, inject, cfg.hc_count)
                if self.probe is not None:
                    self.probe(layer, hidden)

            final, _ = gated_residual_mix(
                hidden, self.w.get("output_hc_norm.weight"), self.w.get("output_hc_down.weight"),
                self.w.get("output_hc_up.weight"), None,
                cfg.rms_norm_eps, cfg.hc_count, cfg.hidden_size,
            )
            state.positions = [base + begin + seq]
        # last position only
        last = ttnn.slice(final, (0, 0, final.shape[-2] - 1, 0), (1, 1, final.shape[-2], cfg.hidden_size))
        return last

    def _ple_chunk(
        self, hidden: ttnn.Tensor, layer: int, st: LayerState, state: TTState, begin: int, seq: int
    ) -> ttnn.Tensor:
        """PLE over a chunk: per-token n-gram hashes, then the dilated conv."""
        cfg = self.cfg
        hist = state.histories[0]
        base = len(hist) - seq
        per_token = [hist[: base + i + 1] for i in range(seq)]
        emb = self.to_dev(self.ngram_embed(per_token), ttnn.bfloat16)  # [1,1,seq,ple_dim]

        key = grouped_rms_norm(
            ttnn.linear(emb, self.w.blk(layer, "ple_key.weight"), compute_kernel_config=HIFI4),
            self.w.blk(layer, "ple_norm_key.weight"), cfg.rms_norm_eps, cfg.hidden_size, cfg.hc_count,
        )
        value = ttnn.linear(emb, self.w.blk(layer, "ple_value.weight"), compute_kernel_config=HIFI4)
        query = grouped_rms_norm(
            hidden, self.w.blk(layer, "ple_norm_query.weight"), cfg.rms_norm_eps,
            cfg.hidden_size, cfg.hc_count,
        )
        prod = ttnn.reshape(ttnn.multiply(key, query), (1, 1, seq * cfg.hc_count, cfg.hidden_size))
        gate = ttnn.multiply(ttnn.sum(prod, dim=-1, keepdim=True), 1.0 / math.sqrt(cfg.hidden_size))
        gate = ttnn.multiply(ttnn.sqrt(ttnn.clamp(ttnn.abs(gate), min=1e-6)), ttnn.sign(gate))
        gate = ttnn.sigmoid(gate)                                    # [1,1,seq*hc,1]

        val = ttnn.reshape(value, (1, 1, seq, cfg.hidden_size))
        val = ttnn.reshape(ttnn.repeat(val, (1, 1, 1, cfg.hc_count)), (1, 1, seq * cfg.hc_count, cfg.hidden_size))
        gated = ttnn.reshape(ttnn.multiply(val, gate), (1, 1, seq, cfg.hc_hidden_size))

        gated_normed = grouped_rms_norm(
            gated, self.w.blk(layer, "ple_norm_conv.weight"), cfg.rms_norm_eps,
            cfg.hidden_size, cfg.hc_count,
        )
        state_len = (cfg.ple_conv_kernel - 1) * cfg.ngram_size
        col = ttnn.transpose(gated_normed, -2, -1)                   # [1,1,C,seq]
        c_dim = cfg.hc_hidden_size
        # Same ring contract as the DeltaNet conv: decode keeps this window as a
        # list of single columns, so prefill consumes and returns one.
        if st.ple_conv is None:
            st.ple_conv = [
                ttnn.zeros((1, 1, c_dim, 1), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=self.mesh)
                for _ in range(state_len)
            ]
        window = ttnn.concat(
            [*self._ring_oldest_first(st.ple_conv, st.ple_step), col], dim=-1
        )                                                       # [1,1,C,state_len+seq]
        conv_w = self.w.blk(layer, "ple_conv1d.weight")
        acc = None
        for tap, w_tap in enumerate(
            self.conv_taps(("ple_chunk", layer), conv_w, c_dim, cfg.ple_conv_kernel)
        ):
            off = tap * cfg.ngram_size
            piece = ttnn.slice(window, (0, 0, 0, off), (1, 1, c_dim, off + seq))
            term = ttnn.multiply(piece, w_tap)
            acc = term if acc is None else ttnn.add(acc, term)
        total = state_len + seq
        cols = [
            ttnn.slice(window, (0, 0, 0, total - state_len + i), (1, 1, c_dim, total - state_len + i + 1))
            for i in range(state_len)
        ]                                                       # oldest .. newest
        st.ple_step += seq
        st.ple_conv = self._ring_from_oldest_first(cols, st.ple_step, into=st.ple_conv)
        # prefill is single-sequence: [1,1,C,seq] -> [1,1,seq,C]
        conv = ttnn.silu(ttnn.transpose(acc, -2, -1))
        return ttnn.add(gated, conv)

