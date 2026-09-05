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
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import ttnn

from ..reference.config import Qwen4ExpConfig
from ..reference.weights import WeightStore
from . import linear_attn, moe
from .ops import (linear_rows, HIFI4, fast_linear, ksplit_linear, gated_residual_mix,
                  grouped_rms_norm, reinject, rms_norm)
from .weights import TTWeights


# K/V page size. One tile, so every cache write stays tile-aligned and a chunk
# start is a legal `chunk_start_idx` for the chunked SDPA program config below.
_MISSING = object()

KV_BLOCK = 32

# The DeltaNet op's fixed chunk width, which is also `prefill`'s default chunk
# and therefore the width the traced prefill path will capture.
DELTANET_CHUNK = 128

# The alpha|beta k-split is **off**. Its original measurement was taken while the
# split was silently computing zeros (see `ops.output_tiles`), so it had never
# actually been timed. Re-measured once it worked: 59.81/59.05 ms against
# 59.67/58.92 for `linear_rows` -- no gain either way, and `ssm_alpha` and
# `ssm_beta` are float32 on purpose ("a wrong expert is not a small
# perturbation", plan.py) while the split accumulates its partials in the
# activation's bfloat16. No speed to pay for that, so it stays on the op.
_NO_AB_KSPLIT = os.environ.get("TT_AB_KSPLIT", "0") != "1"


# Largest MoE row-group that still computes a per-token MoE.
#
# `moe.moe_block` must be exact per row -- each row picks its own experts and is
# weighted by its own probabilities -- so grouping rows can only change speed.
# It does not: `scripts/dev/moe_rows_check.py` builds the answer one row at a
# time and compares, and row-groups of 1, 8, 16 and 32 reproduce it exactly
# while 64 gets **33 of 64 rows wrong**, worst row 102.9 %. 32 is one tile.
# See `TTModel.prefill` and handoff 5.8.
_MAX_MOE_CHUNK = 32

# Prefill chunk width. Wider is faster -- 512 tokens go 3476 -> 3135 ms, 10.9 % --
# because the dense matmuls get wider, and it is not paid for in accuracy. Of the
# ten linear shapes this path uses, five are bit-identical at 512 rows and 128,
# and the other five all land *closer* to the exact float64 product at 512 than at
# 128, five out of five, never further (`row_count_stability_check.py`). Wider
# blocking means fewer partial-sum roundings. The DeltaNet op still sees its own
# 128-wide chunks -- `_linear_attention_chunk` builds n_chunks from `seq` -- so
# this also batches its scan for free.
PREFILL_CHUNK = 512


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
    # Whether the QSA selection actually runs. A Python bool, so it is baked
    # into whatever trace gets captured -- which is the point: the two regimes
    # are two graphs, and the engine recaptures when a sequence crosses
    # `indexer_budget`. Default on, so nothing changes until that wiring lands.
    selection_active = True

    def __init__(
        self,
        config: Qwen4ExpConfig,
        weights: TTWeights,
        host_store: WeightStore,
        mesh,
        max_seq_len: int = 4096,
        sdpa_k_chunk: int = 128,
        pin_sdpa_config: bool = False,
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
        # so a *dense* mask row can only address 65536 cache positions.
        self.indexer_max_seq = 1 << 16

        # -- the compact attention window ------------------------------------
        #
        # The dense mask is one column per cache position, so at the model's
        # 262144 it is both unaddressable by uint16 and beside the point: SDPA
        # reads every one of those positions, 537 MB a layer and 6.4 GB a token
        # over twelve QSA layers, to attend to 2048 of them.
        #
        # The selection already names `indexer_topk` blocks of `ratio` tokens,
        # and those live in at most that many pages of the 32-token paged cache.
        # Handing SDPA a page table listing only *those* pages makes the window
        # a constant `compact_len` however long the context is -- and small
        # enough that uint16 addresses it, which is what lets the selection run
        # past 65536 at all.
        #
        # A page may appear twice (two selected blocks can share one 32-token
        # page); each slot then enables only its own block's tokens, so nothing
        # is counted twice. The extra 32 slots are the trailing partial block
        # plus padding, kept a whole tile wide so the concat and the mask stay
        # tile-aligned and the width stays a multiple of `sdpa_k_chunk`.
        self.compact_slots = self.indexer_topk + 32
        self.compact_len = self.compact_slots * KV_BLOCK
        self._mask_base = None
        # What each bound input's contents were last derived from; see `_input`.
        #
        # Compact mode is worth it only where the dense row is the larger of the
        # two, which is every context past `compact_len` and no shorter one.
        # `topk` returns uint16 block ids, so the block count itself caps the
        # model at ratio << 16 = 262144 positions -- exactly the model's maximum.
        self.compact_attention = (
            traceable_kv
            and max_seq_len > self.compact_len
            and self.max_blocks <= (1 << 16)
        )
        self.use_indexer = (
            traceable_kv
            and config.indexer_budget < max_seq_len
            and (self.compact_attention or max_seq_len <= self.indexer_max_seq)
        )
        self._block_offsets = None
        self._compact_slot_base = None
        self._head_sum = None

        # **No program config for the decode attention, deliberately.**
        #
        # This used to pin q_chunk_size=32 and k_chunk_size=`sdpa_k_chunk`, and
        # that was the cause of handoff 4b: decode's next-token accuracy fell from
        # ~80 % to 0 % past ~224 tokens of context. The op's online softmax over
        # the K/V cache is supposed to give the same answer however it is chunked,
        # and with a pinned config it does not -- for one fixed set of inputs,
        # changing only k_chunk_size moved the result by up to 1e7 relative once
        # the cache spanned more than one chunk, growing with the chunk count
        # (`sdpa_decode_accuracy_check.py`). The collapse tracked the setting: at
        # k=64 accuracy broke from ~64 tokens, at k=128 from ~128, at k=256 it
        # decayed gently from 128 and sharply at 256.
        #
        # Letting ttnn choose fixes it. Against the float32 CPU reference on the
        # same passage (`reference_context_decay_check.py`), which itself declines
        # on this text, the device now tracks it the whole way:
        #
        #     positions      0-127  128-159  160-191  192-223  224-255  256-287
        #     reference      75-81%   62.5%    62.5%    50.0%    46.9%    50.0%
        #     device now     78-84%   59.4%    59.4%    50.0%    40.6%    46.9%
        #     device before  78-84%   53.1%    12.5%     6.2%     0.0%     0.0%
        #
        # `sdpa_k_chunk` is kept because the QSA indexer sizes its window from it;
        # it no longer reaches the attention op. Set `pin_sdpa_config=True` to get
        # the old behaviour back for an A/B, and expect it to be wrong.
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=32,
            k_chunk_size=sdpa_k_chunk,
            exp_approx_mode=False,
        ) if pin_sdpa_config else None
        # `chunked_scaled_dot_product_attention` requires the chunk's start to be
        # a multiple of *both* chunk sizes, and violating that is silent: at
        # q_chunk_size=128 a start of 32 returns 257 % nonsense rather than an
        # error (`scripts/dev/paged_attention_probe.py`). `prefill` guarantees
        # only 32-alignment -- its chunk is a multiple of TILE_SIZE and it can
        # resume from any tile-aligned base -- so 32 is the size that covers
        # everything it can produce, and it is exact at every start measured.
        self.chunked_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=KV_BLOCK,
            k_chunk_size=KV_BLOCK,
            exp_approx_mode=False,
        )
        # ... and a wide one for the common case. A 32-wide chunk config is
        # correct at every start prefill can produce but does four times the
        # inner iterations of a 128-wide one, which cost 1060 -> 1442 ms a chunk
        # when it was used unconditionally. `_attention_chunk` picks the wide
        # config only when the start is 128-aligned *and* the chunk is a full
        # 128 -- the default, and the only shape the traced path will capture --
        # and falls back to the narrow one otherwise.
        # Wide in q, narrow in k. The two axes buy different things: q_chunk_size
        # is the outer iteration count and drives the speed, k_chunk_size sets
        # the accumulation order and therefore the answer. At k=128 prefill's
        # NLL moved 5.648 -> 5.980; at k=32 it reproduces the pre-paged path
        # exactly, and q=128 keeps the speed.
        self.chunked_sdpa_wide_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=DELTANET_CHUNK,
            k_chunk_size=KV_BLOCK,
            exp_approx_mode=False,
        )
        self._page_tables: dict[int, ttnn.Tensor] = {}
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

    @property
    def _host_memo(self) -> dict:
        """Host tensors derived from the position alone, cached across the
        per-layer loop for the same reason `_input` takes a key.

        Lazily created rather than set in `__init__`: the indexer tests build a
        TTModel without running the full constructor.
        """
        m = self.__dict__.get("_host_memo_store")
        if m is None:
            m = self.__dict__["_host_memo_store"] = {}
        return m

    @property
    def _input_key(self) -> dict:
        """What each bound input's contents were last derived from; see `_input`."""
        m = self.__dict__.get("_input_key_store")
        if m is None:
            m = self.__dict__["_input_key_store"] = {}
        return m

    def _input(self, name: str, host: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
               key=None):
        """A per-step input, either fresh or written into its bound buffer.

        Writing into a persistent buffer is what lets the whole 48-layer step be
        replayed from a captured trace; the copy itself happens outside the trace.

        `key` names what the contents were derived from. Most of these inputs
        depend on the position and nothing else, but they are built and uploaded
        inside the per-layer loop, so all twelve QSA layers were writing
        byte-identical data into the same buffer -- eleven redundant
        host-to-device copies per tensor per token, and the host-side torch work
        to build each one. With a key the copy happens once per distinct value.
        """
        if self.bound is None:
            return self.to_dev(host, dtype, layout)
        buf = self.bound.get(name)
        if buf is None:
            buf = self.to_dev(host, dtype, layout)
            self.bound[name] = buf
            self._input_key[name] = key
            return buf
        if self._skip_copy:
            return buf
        if key is not None and self._input_key.get(name, _MISSING) == key:
            return buf
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host, dtype=dtype, layout=layout, mesh_mapper=self.replicate), buf
        )
        self._input_key[name] = key
        return buf

    def from_dev(self, t: ttnn.Tensor) -> torch.Tensor:
        """First device's copy (all devices agree for replicated results)."""
        return ttnn.to_torch(t, mesh_composer=self.compose)[0:1]

    def all_reduce(self, t: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.all_reduce(t, cluster_axis=1, topology=ttnn.Topology.Linear)

    # -- rope -------------------------------------------------------------

    def rope(self, positions: int | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        # Two callers per layer with different arguments -- the attention heads
        # at the position, the indexer at its block's start -- so a single slot
        # would alternate and never hit. A handful of entries covers both, and
        # is bounded because the keys repeat across the twelve layers.
        k = positions if isinstance(positions, int) else tuple(positions)
        memo = self._host_memo.setdefault("rope", {})
        if k in memo:
            return memo[k]
        if len(memo) >= 8:
            memo.clear()
        out = self._rope_uncached(positions)
        memo[k] = out
        return out

    def _rope_uncached(self, positions: int | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
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
    def _l2norm(x: ttnn.Tensor, eps: float = 1e-6, scale: float = 1.0) -> ttnn.Tensor:
        """`scale * x / sqrt(sum(x^2) + eps)`, as two ops rather than five.

        The obvious form is multiply, sum, add, rsqrt, multiply. But an RMS norm
        is the same reduction with a mean instead of a sum, and `ttnn.rms_norm`
        is a real fused op:

            rms_norm(x, eps/D) = x * rsqrt(sum(x^2)/D + eps/D)
                               = sqrt(D) * x * rsqrt(sum(x^2) + eps)
                               = sqrt(D) * l2norm(x)

        so `l2norm(x) = rms_norm(x, eps/D) / sqrt(D)`, and any following scalar
        folds into that one multiply -- which is why the caller passes `scale`
        instead of doing its own. Verified against the closed form in float64:
        max difference 5.6e-17.

        DeltaNet runs this twice a layer, so it is 8 fewer ttnn calls a layer and
        288 a token, at ~5.5 us apiece (invariant 42).
        """
        d = x.shape[-1]
        return ttnn.multiply(ttnn.rms_norm(x, epsilon=eps / d), scale / math.sqrt(d))

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

    def _kv_page_table(self, batch: int) -> ttnn.Tensor:
        """Logical block -> physical block, one row per sequence.

        Each slot owns a contiguous run of blocks, so the mapping is the identity
        shifted by the slot: row b is `[b*n, b*n+1, ...]`. Nothing here is
        dynamic -- the point of the page table is not paging, it is that the ops
        take the *position* from device memory rather than from a Python int,
        which is what lets the attention layers be captured (5.2).

        Built once per batch and cached, because a trace capture must not
        allocate.
        """
        hit = self._page_tables.get(batch)
        if hit is not None:
            return hit
        n = self.max_seq_len // KV_BLOCK
        rows = torch.arange(batch * n, dtype=torch.int32).reshape(batch, n)
        table = ttnn.from_torch(
            rows, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh, mesh_mapper=self.replicate,
        )
        self._page_tables[batch] = table
        return table

    def _ensure_kv(self, st: "LayerState", batch: int, n_kv: int, hd: int) -> None:
        """Allocate this layer's K/V cache the first time it is written.

        Paged when `traceable_kv`: `[batch * T/32, n_kv, 32, head_dim]`, the same
        total bytes as the flat `[batch, n_kv, T, head_dim]` it replaces, but the
        layout `paged_fill_cache`, `paged_update_cache` and both paged SDPA ops
        want. The flat form stays for the legacy path, which uses `update_cache`
        and cannot take a position tensor at all.
        """
        if st.keys is not None:
            return
        if self.traceable_kv:
            shape = (batch * (self.max_seq_len // KV_BLOCK), n_kv, KV_BLOCK, hd)
        else:
            shape = (batch, n_kv, self.max_seq_len, hd)
        st.keys = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)
        st.values = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)

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
            # Four multiplies and three adds, deliberately, and *not* the
            # obvious "concat the pieces and the taps, one multiply and one
            # reduce". That form has half the calls and measured **4.5 ms a
            # token slower** (82.73 -> 87.26): each piece is [1, 1, C, 1], which
            # TILE layout pads to [C, 32], so concatenating four of them repacks
            # four tensors' worth of tiles, and `sum` over the padded width is a
            # full-tile reduction. Invariant 42's ~5.5 us floor is for
            # *elementwise* ops; `concat` and `sum` are data movement and cost
            # with the padding, so fewer calls is not automatically less work.
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

    _FUSED_PAIRS: dict = {}

    def _fused_pair(self, layer: int, left: str, right: str) -> ttnn.Tensor:
        """Two same-input projections concatenated on their output axis.

        Built once per layer at first use. Cost here follows output *width* and
        only once the grid is full, so two narrow matmuls against one input are
        strictly worse than one wider one -- see invariant 55, and
        `scripts/dev/narrow_linear_cost.py` for the measurements.
        """
        key = (layer, left, right)
        got = self._FUSED_PAIRS.get(key)
        if got is None:
            got = ttnn.concat([self.w.blk(layer, left), self.w.blk(layer, right)], dim=-1)
            self._FUSED_PAIRS[key] = got
        return got

    def _linear_attention_step(self, mixed: ttnn.Tensor, layer: int, st: LayerState) -> ttnn.Tensor:
        cfg = self.cfg
        # local (per-device) head count: q, k and v are all sharded to the same
        # twelve heads, so there is a single head count here
        n_v, hd = self.n_v_local, cfg.linear_head_dim
        # `mixed` is [1, 1, B, hidden]; B sequences decode together.
        batch = mixed.shape[-2]

        # Two matmuls, not one. attn_qkv and attn_gate read the same `mixed` and
        # are both column shards, so `_fused_pair` concatenates them into a
        # [2560, 4096] and one call -- and it is worth **nothing**: 62.40 ms a
        # token fused against 62.37 unfused, with the ranges overlapping. It
        # also keeps a third copy of both weights resident, ~376 MB a device.
        # Invariant 62 again: fewer calls is not automatically less work, and
        # 2560 or 1536 output columns already spread over enough of the grid
        # that widening to 4096 buys no bandwidth.
        qkv = linear_rows(mixed, self.w.blk(layer, "attn_qkv.weight"),
                          compute_kernel_config=HIFI4)
        z = linear_rows(mixed, self.w.blk(layer, "attn_gate.weight"),
                        compute_kernel_config=HIFI4)

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

        q = self._l2norm(q, scale=hd**-0.5)
        k = self._l2norm(k)

        # `ssm_alpha` and `ssm_beta` are [2560, 48] each and take the same input,
        # so they are one matmul with a wider output. Width costs nothing until
        # it fills the grid -- [2560, 48] and [2560, 96] both measure 31.6 us --
        # so this halves 2.28 ms a token to 1.14 (invariant 55).
        ab = self._fused_pair(layer, "ssm_alpha.weight", "ssm_beta.weight")
        # Three output tiles, the narrowest in the model after the fusion above,
        # so the grid affords more reduction groups here than anywhere else.
        both_ab = None if _NO_AB_KSPLIT else ksplit_linear(mixed, ab)
        if both_ab is None:
            both_ab = linear_rows(mixed, ab, compute_kernel_config=HIFI4)
        half = both_ab.shape[-1] // 2
        a = self._slice_last(both_ab, 0, half)
        b = self._slice_last(both_ab, half, 2 * half)
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
        out = linear_rows(gated, self.w.blk(layer, "ssm_out.weight"), compute_kernel_config=HIFI4)
        return self.all_reduce(out)

    # -- full attention (QSA), one token -------------------------------------

    def _indexer_select(
        self, mixed: ttnn.Tensor, layer: int, st: LayerState, positions: list[int],
        q_cos: ttnn.Tensor, q_sin: ttnn.Tensor,
    ) -> tuple:
        """Maintain the block cache, then select from it. See the two halves.

        Returns `(mask, page_table, cur_pos)`. The last two are None unless
        `compact_attention` is on, in which case they replace the identity page
        table and the real position for the SDPA call -- the selection is then
        addressed against a `compact_len` window rather than the whole cache.
        """
        self._indexer_update(mixed, layer, st, positions)
        return self._indexer_mask(mixed, layer, st, positions, q_cos, q_sin)

    def _indexer_update(
        self, mixed: ttnn.Tensor, layer: int, st: LayerState, positions: list[int],
    ) -> None:
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
        k_raw = fast_linear(
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
            self._input("idx_block_cos", b_cos, ttnn.float32, key=tuple(positions)),
            self._input("idx_block_sin", b_sin, ttnn.float32, key=tuple(positions)),
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
                ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, key=tuple(positions),
            ),
        )

        return None

    def _indexer_mask(
        self, mixed: ttnn.Tensor, layer: int, st: LayerState, positions: list[int],
        q_cos: ttnn.Tensor, q_sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Score the cached blocks, take the top `k`, and paint them into a mask.

        This half is what costs: `ttnn.topk` alone measured **22.3 ms** of a
        146 ms decode step -- k=512 out of nb=1024 blocks is half a sort, run in
        all twelve QSA layers. The scatter behind it is only 1.8 ms.

        It is also skippable below the budget. Eligible blocks at position p
        number p//ratio, so while p < indexer_budget there are fewer than k of
        them and `topk` returns every visible block plus -inf padding -- the mask
        it builds is exactly plain causal attention. The block cache still has to
        be filled on those steps, which is why `_indexer_update` is separate:
        skip this half, keep that one, and read causally.
        """
        cfg = self.cfg
        d, ratio = cfg.indexer_head_dim, self.indexer_ratio
        batch = mixed.shape[-2]
        nb, k = self.max_blocks, self.indexer_topk

        # -- scores: sum over heads of relu(q . block), one matmul per head --
        # Contracting the *block* cache against a single query column keeps the
        # 65536 x 128 cache where it is; scoring the other way round would
        # transpose 16 MB a layer a step.
        q_idx = fast_linear(
            mixed, self.w.blk(layer, "indexer.q_proj.weight"), compute_kernel_config=HIFI4
        )
        q_idx = ttnn.reshape(q_idx, (1, batch, cfg.indexer_heads, d))
        q_idx = rms_norm(q_idx, self.w.blk(layer, "indexer.q_norm.weight"), cfg.rms_norm_eps)
        # the caller has already bound rope at p for the attention heads
        q_idx = self._apply_rope_dev(q_idx, q_cos, q_sin)
        # All `indexer_heads` in one matmul. relu is elementwise and the sum is
        # over heads, so relu-then-sum is the same function either way -- but the
        # per-head loop read the whole block cache once per head. At the model's
        # maximum context that cache is 16.8 MB a layer and four reads of it are
        # 67; it also cost nineteen ops a layer against five, 168 a token.
        qt = ttnn.reshape(ttnn.transpose(q_idx, -2, -1), (batch, 1, d, cfg.indexer_heads))
        parts = ttnn.relu(ttnn.matmul(st.indexer_blocks, qt, compute_kernel_config=HIFI4))
        # Summed with a constant rather than `ttnn.sum`: the head axis is four
        # wide in a 32-wide tile, and a matmul against a [heads, 1] of ones
        # contracts exactly the logical four whatever the padding holds.
        if self._head_sum is None:
            self._head_sum = self.to_dev(
                torch.ones(1, 1, cfg.indexer_heads, 1, dtype=torch.float32), ttnn.bfloat16)
        scores = ttnn.matmul(parts, self._head_sum, compute_kernel_config=HIFI4)
        scores = ttnn.multiply(ttnn.transpose(scores, -2, -1), d**-0.5)   # [B,1,1,nb]
        scores = ttnn.add(scores, self._input("idx_bias", self._block_bias(positions), ttnn.float32, key=tuple(positions)))

        # -- select, expand to tokens, and build the mask -------------------
        blocks = ttnn.topk(scores, k, dim=-1)[1]                  # uint16 [B,1,1,k]
        if self._block_offsets is None:
            self._block_offsets = self.to_dev(
                torch.arange(ratio, dtype=torch.float32).reshape(1, 1, 1, ratio), ttnn.float32
            )
        blocks_f = ttnn.typecast(blocks, ttnn.float32)             # [B,1,1,k]
        tokens = ttnn.add(
            ttnn.reshape(ttnn.multiply(blocks_f, float(ratio)), (batch, 1, k, 1)),
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
                ttnn.float32, key=tuple(positions),
            ),
        )
        # This one comparison covers every case `topk` can hand back: a block
        # entirely beyond p (which is what the -inf fills are, when fewer than
        # `topk` blocks are eligible) has all its tokens beyond p, and an
        # eligible block has none.
        if self.compact_attention:
            return self._compact_mask(blocks_f, visible, positions, batch, k, cfg)

        tail_idx, tail_vis = self._tail_block(positions)
        tokens = ttnn.concat(
            [tokens, self._input("idx_tail", tail_idx, ttnn.float32, key=tuple(positions))], dim=-1
        )
        visible = ttnn.concat(
            [visible, self._input("idx_tail_vis", tail_vis, ttnn.float32, key=tuple(positions))], dim=-1
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
        return ttnn.repeat(mask, (1, 1, cfg.num_attention_heads, 1)), None, None

    def _compact_mask(self, blocks_f, visible, positions, batch: int, k: int, cfg):
        """The same selection, addressed against a compact page table.

        Returns `(mask, page_table, cur_pos)`, where the page table lists the
        pages the selection actually touches and the mask is `compact_len` wide
        instead of `max_seq_len`. Slot j of the table is block j's page, so a
        block that shares a page with another still gets its own slot and its own
        four mask columns -- repeated physical pages are read twice and enable
        disjoint tokens, which is why no de-duplication is needed.

        `visible` comes in already computed against the *absolute* positions,
        because visibility is a fact about the sequence and not about the layout.
        """
        ratio = self.indexer_ratio
        per_page = KV_BLOCK // ratio                  # blocks inside one page
        slots, width = self.compact_slots, self.compact_len

        # page = block // per_page, and the block's offset inside that page.
        # `floor_div` on floats: these are small exact integers in float32.
        pages = ttnn.floor(ttnn.multiply(blocks_f, 1.0 / per_page))
        intra = ttnn.subtract(blocks_f, ttnn.multiply(pages, float(per_page)))

        if self._compact_slot_base is None:
            self._compact_slot_base = self.to_dev(
                (torch.arange(k, dtype=torch.float32) * KV_BLOCK).reshape(1, 1, 1, k),
                ttnn.float32,
            )
        ctok = ttnn.add(
            ttnn.reshape(
                ttnn.add(ttnn.multiply(intra, float(ratio)), self._compact_slot_base),
                (batch, 1, k, 1)),
            self._block_offsets,
        )                                                          # [B,1,k,ratio]
        ctok = ttnn.reshape(ctok, (batch, 1, 1, k * ratio))

        # The trailing partial block, in the extra tile of slots. One whole tile
        # so both concats stay tile-aligned; only the first slot of it carries a
        # page, the rest are padding that the mask never marks.
        tail_ctok, tail_vis, tail_pages = self._compact_tail(positions)
        key = tuple(positions)
        ctok = ttnn.concat(
            [ctok, self._input("cidx_tail", tail_ctok, ttnn.float32, key=key)], dim=-1)
        visible = ttnn.concat(
            [visible, self._input("cidx_tail_vis", tail_vis, ttnn.float32, key=key)], dim=-1)

        if self._mask_base is None:
            self._mask_base = ttnn.zeros(
                (batch, 1, 1, width), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=self.mesh,
            )
        row = ttnn.scatter(
            ttnn.multiply(self._mask_base, 0.0), -1,
            ttnn.typecast(ctok, ttnn.uint16),
            ttnn.typecast(visible, ttnn.bfloat16),
        )
        mask = ttnn.multiply(ttnn.subtract(row, 1.0), 1e9)
        mask = ttnn.repeat(mask, (1, 1, cfg.num_attention_heads, 1))

        # `_kv_page_table` numbers pages b * n_pages + p, so a compact table for
        # batch b carries the same offset.
        table = ttnn.concat(
            [pages, self._input("cidx_tail_pages", tail_pages, ttnn.float32, key=key)],
            dim=-1)                                                # [B,1,1,slots]
        if batch > 1:
            table = ttnn.add(table, self._compact_batch_offset(batch, slots))
        table = ttnn.reshape(
            ttnn.to_layout(ttnn.typecast(table, ttnn.int32), ttnn.ROW_MAJOR_LAYOUT),
            (batch, slots))
        cur = self._input(
            "cidx_cur", torch.full((batch,), width - 1, dtype=torch.int32),
            ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, key=("compact", width))
        return mask, table, cur

    def _compact_batch_offset(self, batch: int, slots: int):
        hit = self._host_memo.get("_compact_batch_offset")
        if hit is None or hit[0] != (batch, slots):
            n_pages = self.max_seq_len // KV_BLOCK
            host = (torch.arange(batch, dtype=torch.float32) * n_pages).reshape(
                batch, 1, 1, 1).expand(batch, 1, 1, slots).contiguous()
            hit = ((batch, slots), self.to_dev(host, ttnn.float32))
            self._host_memo["_compact_batch_offset"] = hit
        return hit[1]

    def _compact_tail(self, positions: list[int]):
        k = tuple(positions)
        hit = self._host_memo.get("_compact_tail")
        if hit is not None and hit[0] == k:
            return hit[1]
        out = self._compact_tail_uncached(positions)
        self._host_memo["_compact_tail"] = (k, out)
        return out

    def _compact_tail_uncached(self, positions: list[int]):
        """The trailing partial block, addressed inside its own compact slot.

        The block containing `p` is pushed below every other by `_block_bias`
        while it is incomplete, so `topk` can never return it and it is appended
        here instead -- the same rule the dense path uses, only the indices are
        compact. Its page goes in slot `indexer_topk`; the remaining 31 slots of
        the tile are padding and point at page 0 with nothing marked visible.

        Only the incomplete block's own tokens are marked, never the whole page,
        so a selected block sharing that page cannot have its tokens counted a
        second time.
        """
        ratio, b = self.indexer_ratio, len(positions)
        k, slots = self.indexer_topk, self.compact_slots
        idx = torch.zeros(b, 1, 1, KV_BLOCK)
        vis = torch.zeros(b, 1, 1, KV_BLOCK)
        pages = torch.zeros(b, 1, 1, slots - k)
        for i, p in enumerate(positions):
            page = p // KV_BLOCK
            pages[i, 0, 0, 0] = float(page)
            # Every column of the tail slot addresses its own position, so a
            # column the loop below does not mark visible scatters a zero over a
            # zero. Columns of the *padding* slots are never addressed at all.
            base = k * KV_BLOCK                        # first column of the tail slot
            idx[i, 0, 0, :] = torch.arange(KV_BLOCK, dtype=torch.float32) + base
            if p % ratio == ratio - 1:
                continue                              # complete: it competes in topk
            start = ratio * (p // ratio)
            for t in range(ratio):
                tok = start + t
                if tok <= p and tok // KV_BLOCK == page:
                    vis[i, 0, 0, tok - page * KV_BLOCK] = 1.0
        return idx, vis, pages

    def _block_bias(self, positions: list[int]) -> torch.Tensor:
        k = tuple(positions)
        hit = self._host_memo.get("_block_bias")
        if hit is not None and hit[0] == k:
            return hit[1]
        out = self._block_bias_uncached(positions)
        self._host_memo["_block_bias"] = (k, out)
        return out

    def _block_bias_uncached(self, positions: list[int]) -> torch.Tensor:
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
        k = tuple(positions)
        hit = self._host_memo.get("_tail_block")
        if hit is not None and hit[0] == k:
            return hit[1]
        out = self._tail_block_uncached(positions)
        self._host_memo["_tail_block"] = (k, out)
        return out

    def _tail_block_uncached(self, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
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

        qg = fast_linear(mixed, self.w.blk(layer, "attn_q.weight"), compute_kernel_config=HIFI4)
        qg = ttnn.reshape(qg, (1, batch, n_q, hd * 2))      # [q | gate] interleaved per head
        q = self._slice_last(qg, 0, hd)
        gate = self._slice_last(qg, hd, hd * 2)

        k = ttnn.reshape(
            fast_linear(mixed, self.w.blk(layer, "attn_k.weight"), compute_kernel_config=HIFI4),
            (1, batch, n_kv, hd),
        )
        v = ttnn.reshape(
            fast_linear(mixed, self.w.blk(layer, "attn_v.weight"), compute_kernel_config=HIFI4),
            (1, batch, n_kv, hd),
        )
        q = rms_norm(q, self.w.blk(layer, "attn_q_norm.weight"), cfg.rms_norm_eps)
        k = rms_norm(k, self.w.blk(layer, "attn_k_norm.weight"), cfg.rms_norm_eps)

        positions_for_rope = list(position) if isinstance(position, (list, tuple)) else [position] * batch
        cos_t, sin_t = self.rope(positions_for_rope)
        cos = self._input("rope_cos", cos_t, ttnn.float32, key=tuple(positions_for_rope))
        sin = self._input("rope_sin", sin_t, ttnn.float32, key=tuple(positions_for_rope))
        q = self._apply_rope_dev(q, cos, sin)
        k = self._apply_rope_dev(k, cos, sin)

        self._ensure_kv(st, batch, n_kv, hd)
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
                layout=ttnn.ROW_MAJOR_LAYOUT, key=tuple(positions),
            )
            k_s = self._l1_height_sharded(ttnn.typecast(k, ttnn.bfloat16), hd)
            v_s = self._l1_height_sharded(ttnn.typecast(v, ttnn.bfloat16), hd)
            page_table = self._kv_page_table(batch)
            ttnn.experimental.paged_update_cache(
                st.keys, k_s, update_idxs_tensor=pos_tensor, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                st.values, v_s, update_idxs_tensor=pos_tensor, page_table=page_table
            )
            # The paged decode op takes the same `attn_mask` shape the flat one
            # does, so the QSA selection passes through unchanged: the indexer
            # masks *logical* positions and never addresses the cache.
            mask, sel_table, sel_pos = None, None, None
            if self.use_indexer:
                if self.selection_active:
                    mask, sel_table, sel_pos = self._indexer_select(
                        mixed, layer, st, positions, cos, sin)
                else:
                    # Below the budget the selection reduces to plain causal
                    # attention (see `_indexer_mask`), so skip it and read
                    # causally -- but keep filling the block cache, because it
                    # has to be right on the step the selection starts mattering.
                    self._indexer_update(mixed, layer, st, positions)
            # In compact mode the read is re-addressed: a page table listing only
            # the pages the selection touches, and a position that bounds the
            # compact window rather than the sequence. The cache itself is
            # untouched -- `paged_update_cache` above still writes at the real
            # position through the identity table.
            out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q, st.keys, st.values,
                page_table_tensor=page_table if sel_table is None else sel_table,
                is_causal=mask is None, attn_mask=mask,
                cur_pos_tensor=pos_tensor if sel_pos is None else sel_pos,
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
        return fast_linear(out, self.w.blk(layer, "attn_output.weight"), compute_kernel_config=HIFI4)

    # -- PLE (n-gram) injection, one token ------------------------------------

    # -- k tokens of one sequence, in one step --------------------------------

    def _linear_attention_step_n(
        self, mixed: ttnn.Tensor, layer: int, st: LayerState, k: int,
        accept: ttnn.Tensor | None = None, conv_sel: ttnn.Tensor | None = None,
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

        qkv = linear_rows(mixed, self.w.blk(layer, "attn_qkv.weight"), compute_kernel_config=HIFI4)
        z = linear_rows(mixed, self.w.blk(layer, "attn_gate.weight"), compute_kernel_config=HIFI4)
        a = linear_rows(mixed, self.w.blk(layer, "ssm_alpha.weight"), compute_kernel_config=HIFI4)
        b = linear_rows(mixed, self.w.blk(layer, "ssm_beta.weight"), compute_kernel_config=HIFI4)
        g = ttnn.multiply(
            self.w.blk(layer, "ssm_a"),
            ttnn.softplus(ttnn.add(a, self.w.blk(layer, "ssm_dt.bias"))),
        )

        # [1,1,k,conv_dim] -> [1,k,conv_dim,1]; one column per token
        qkv_col = ttnn.transpose(ttnn.permute(qkv, (0, 2, 1, 3)), -2, -1)
        conv_w = self.w.blk(layer, "ssm_conv1d.weight")

        # The convolution ring is the one piece the accept mask cannot make
        # inert: a rejected step still shifts a column in, and masking the shift
        # itself would be three blends a step a layer -- 864 ops at k=8, ~14 ms,
        # more than the verify it is protecting.
        #
        # Instead, keep every column this call could possibly need. The ring
        # after accepting j is columns j-2, j-1, j, and when j < depth the older
        # ones come from *before* this call -- so the history is the entry ring
        # followed by the k new columns, captured here while the ring is still
        # intact. One small matmul against a selection matrix picks the right
        # three out at the end (see below); the matrix is a bound tensor, so the
        # choice stays data and one capture serves any j.
        history = None
        if accept is not None:
            depth = self.cfg.conv_kernel - 1
            if st.conv is None:
                # `_causal_conv_step` would make this on its first call, but the
                # history has to be captured *before* the loop touches it.
                st.conv = [
                    ttnn.zeros((1, 1, self.conv_dim_local, 1), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.mesh)
                    for _ in range(depth)
                ]
            prior = [
                ttnn.reshape(ttnn.transpose(st.conv[d], -2, -1),
                             (1, 1, 1, self.conv_dim_local))
                for d in range(depth - 1, -1, -1)          # oldest first
            ]
            fresh = ttnn.reshape(ttnn.transpose(qkv_col, -2, -1),
                                 (1, 1, k, self.conv_dim_local))
            history = ttnn.concat(prior + [fresh], dim=-2)  # [1,1,depth+k,C]

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

        # Rewind the ring to where accepting j tokens would have left it. `sel`
        # is [depth, depth+k] with one 1 a row, so this is a gather written as a
        # matmul -- and being a bound tensor it is data, which is the whole point.
        if history is not None:
            depth = self.cfg.conv_kernel - 1
            picked = ttnn.matmul(conv_sel, history)        # [1,1,depth,C]
            for slot in range(depth):
                col = ttnn.reshape(
                    ttnn.slice(picked, (0, 0, slot, 0),
                               (1, 1, slot + 1, self.conv_dim_local)),
                    (1, 1, self.conv_dim_local, 1),
                )
                ttnn.copy(col, st.conv[slot])

        q = ttnn.reshape(self._slice_last(qkv, 0, kd), (k * n_v, 1, 1, hd))
        kk = ttnn.reshape(self._slice_last(qkv, kd, 2 * kd), (k * n_v, 1, 1, hd))
        v = ttnn.reshape(self._slice_last(qkv, 2 * kd, 2 * kd + vd), (k * n_v, 1, 1, hd))
        q = self._l2norm(q, scale=hd**-0.5)
        kk = self._l2norm(kk)
        g_exp = ttnn.reshape(ttnn.exp(g), (k * n_v, 1, 1, 1))
        beta = ttnn.reshape(ttnn.sigmoid(b), (k * n_v, 1, 1, 1))

        # Speculative acceptance, as data rather than as a shape.
        #
        # A captured trace advances the state by exactly its k, and two capture
        # widths are the one thing that provokes the alternation hang (see
        # `ttnn_bug_report/`: "B superset of A is safe; B and A disagreeing about
        # a shape is not"). So the *advance* has to be maskable instead.
        #
        # The recurrence is `state = state * g + k^T delta` with
        # `delta = (v - predicted) * beta`, so a step with `g = 1` and
        # `beta = 0` leaves the state exactly as it was. Both come from tensors,
        # and `_input` binds them, so one capture serves every acceptance count:
        #
        #     g_exp <- g_exp * acc + (1 - acc)      -> 1 where rejected
        #     beta  <- beta * acc                   -> 0 where rejected
        #
        # Rejected steps still pollute the convolution ring, but only for later
        # steps that are themselves rejected (acceptance is a prefix), and the
        # ring is rebuilt from the columns this call already computed.
        if accept is not None:
            g_exp = ttnn.add(ttnn.multiply(g_exp, accept),
                             ttnn.subtract(ttnn.full_like(accept, 1.0), accept))
            beta = ttnn.multiply(beta, accept)

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
            fast_linear(gated, self.w.blk(layer, "ssm_out.weight"), compute_kernel_config=HIFI4)
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
            fast_linear(mixed, self.w.blk(layer, "attn_q.weight"), compute_kernel_config=HIFI4),
            (1, k, n_q, hd * 2),
        )
        q = rms_norm(self._slice_last(qg, 0, hd), self.w.blk(layer, "attn_q_norm.weight"),
                     cfg.rms_norm_eps)
        gate = self._slice_last(qg, hd, hd * 2)
        kt = rms_norm(
            ttnn.reshape(
                fast_linear(mixed, self.w.blk(layer, "attn_k.weight"), compute_kernel_config=HIFI4),
                (1, k, n_kv, hd),
            ),
            self.w.blk(layer, "attn_k_norm.weight"), cfg.rms_norm_eps,
        )
        vt = ttnn.reshape(
            fast_linear(mixed, self.w.blk(layer, "attn_v.weight"), compute_kernel_config=HIFI4),
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

        self._ensure_kv(st, 1, n_kv, hd)
        page_table = self._kv_page_table(1)

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
                update_idxs_tensor=idx, page_table=page_table,
            )
            ttnn.experimental.paged_update_cache(
                st.values, self._l1_height_sharded(ttnn.typecast(v_row, ttnn.bfloat16), hd),
                update_idxs_tensor=idx, page_table=page_table,
            )

        outs = []
        for i in range(k):
            q_row = ttnn.reshape(ttnn.slice(q, (0, i, 0, 0), (1, i + 1, n_q, hd)), (1, 1, n_q, hd))
            outs.append(
                ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q_row, st.keys, st.values, page_table_tensor=page_table,
                    is_causal=True, cur_pos_tensor=idxs[i],
                    scale=hd**-0.5, program_config=self.sdpa_program_config,
                    compute_kernel_config=HIFI4,
                )
            )
        out = outs[0] if k == 1 else ttnn.concat(outs, dim=1)

        out = ttnn.reshape(out, (1, 1, k, n_q * hd))
        gate = ttnn.reshape(gate, (1, 1, k, n_q * hd))
        return fast_linear(
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
            fast_linear(emb, self.w.blk(layer, "ple_key.weight"), compute_kernel_config=HIFI4),
            self.w.blk(layer, "ple_norm_key.weight"), cfg.rms_norm_eps, cfg.hidden_size, cfg.hc_count,
        )
        value = fast_linear(emb, self.w.blk(layer, "ple_value.weight"), compute_kernel_config=HIFI4)
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

    def _bind_accept(self, k: int, accept: list[float] | None):
        """Bind the two tensors that make speculative acceptance *data*.

        Split out of `step_n` so a captured replay can rebind them between
        replays: `_input` writes into the bound buffer when one exists, which is
        exactly the mechanism the rope tables and cache positions already use.
        One capture therefore serves every acceptance count, and the ladder of
        widths that provokes the alternation hang is never needed.

        `accept` must be a prefix of ones -- anything else would give a `j` that
        does not mean what the caller thinks, so it is rejected rather than
        interpreted.
        """
        if accept is None:
            return None, None
        if len(accept) != k:
            raise ValueError(f"accept has {len(accept)} entries, expected k={k}")
        j = int(sum(accept))
        if j < 1 or accept[:j] != [1.0] * j or any(accept[j:]):
            raise ValueError(f"accept must be ones then zeros, got {accept}")

        n_v = self.n_v_local
        acc = self._input(
            "accept_mask",
            torch.tensor(accept, dtype=torch.float32)
            .repeat_interleave(n_v).reshape(k * n_v, 1, 1, 1),
            ttnn.bfloat16, key=tuple(accept),
        )
        # `history` inside the layer is the entry ring, oldest first, then this
        # call's k columns, so history index h holds the column from time
        # `h - depth`. After accepting j, ring slot s (0 is newest) must hold
        # time `j - 1 - s`, i.e. index `depth + j - 1 - s`.
        depth = self.cfg.conv_kernel - 1
        sel = torch.zeros(1, 1, depth, depth + k, dtype=torch.float32)
        for slot in range(depth):
            sel[0, 0, slot, depth + j - 1 - slot] = 1.0
        csel = self._input("conv_sel", sel, ttnn.bfloat16, key=(k, j))
        return acc, csel

    def step_n(self, tokens: list[int], state: TTState,
               accept: list[float] | None = None) -> ttnn.Tensor:
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
        # 32, not the batch cliff of 64: `step_n` is exact to k=16 (0.00 % on the
        # hidden, tokens matching, `step_n_check.py`) and wrong from k=33 --
        # 35.68 % on the hidden and a different token stream, *identically* at
        # 33, 48 and 64, which is a structural break at one tile of rows rather
        # than anything that accumulates. The range said 1..64 and nothing
        # exercised past 8, so the broken half was reachable and unnoticed.
        #
        # The mechanism is one row tile, and it is not the MoE. A plain
        # `ttnn.linear` returns a given row identically for any row count that
        # fits in one 32-row tile and differently past it
        # (`row_tile_boundary_check.py`), so by the time the experts are reached
        # every op in the layer has already diverged. Not corruption --
        # `step_n_layer_bisect.py` shows 0.385 % at layer 0 rising smoothly --
        # and not fixable here: splitting `expert_ffn` into 32-row groups to
        # keep `per_core_M` at 1 changes nothing. The same fact caps
        # `moe_chunk` and makes batch 64 decode differently from batch 1.
        #
        # `step_n` exists to reproduce k sequential steps exactly, so a path
        # that cannot is no use to it whatever the cause. Nothing needs k > 32
        # today (speculation caps its widths at 17), so it refuses.
        # `TTRUNNER_ALLOW_WIDE_STEP_N=1` lifts the cap so the break can be
        # investigated (`step_n_layer_bisect.py`); it does not make it correct.
        if not 0 < k <= (64 if os.environ.get("TTRUNNER_ALLOW_WIDE_STEP_N") else 32):
            raise ValueError(
                f"k must be in 1..32, got {k}. step_n is wrong past one tile of "
                "rows (35.68 % on the hidden at k=33); see the comment above."
            )
        if self.use_indexer:
            # `_attention_step_n` reads with `sdpa_decode` per row, which leaves
            # room for a per-row selection mask, but the selection is not wired
            # in yet -- and running dense here while `step` runs sparse would make
            # the two disagree beyond the budget, silently.
            raise NotImplementedError(
                "step_n does not carry the QSA selection yet; construct the model "
                f"with max_seq_len <= {self.cfg.indexer_budget} to use it"
            )
        acc, csel = self._bind_accept(k, accept)

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
                else self._linear_attention_step_n(mixed, layer, st, k, acc, csel)
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
        part = fast_linear(hidden, self.w.get("output.weight"), compute_kernel_config=HIFI4)
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

        part = fast_linear(hidden, self.w.get("output.weight"), compute_kernel_config=HIFI4)
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

    def _deltanet_front(self, mixed: ttnn.Tensor, layer: int, st: LayerState, seq: int):
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

        qkv = linear_rows(mixed, self.w.blk(layer, "attn_qkv.weight"), compute_kernel_config=HIFI4)
        z = linear_rows(mixed, self.w.blk(layer, "attn_gate.weight"), compute_kernel_config=HIFI4)
        conv_out, st.conv, st.conv_step = self._causal_conv_chunk(
            ttnn.transpose(qkv, -2, -1), self.w.blk(layer, "ssm_conv1d.weight"),
            st.conv, self.conv_dim_local, seq, layer, st.conv_step,
        )
        qkv = ttnn.transpose(conv_out, -2, -1)              # [1,1,seq,conv_dim_local]

        a = linear_rows(mixed, self.w.blk(layer, "ssm_alpha.weight"), compute_kernel_config=HIFI4)
        b = linear_rows(mixed, self.w.blk(layer, "ssm_beta.weight"), compute_kernel_config=HIFI4)
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
        return (hq, hk, hv, hg, hb), z

    def _deltanet_back(self, out, z, layer: int, seq: int) -> ttnn.Tensor:
        """The per-chunk tail: reshape, norm, gate, project, all-reduce.

        Runs on exactly the rows the chunk would have seen on its own, which is
        what keeps `_linear_attention_multi` bit-identical to calling
        `_linear_attention_chunk` per chunk.
        """
        from .deltanet import CHUNK

        cfg = self.cfg
        n_v, hd = self.n_v_local, cfg.linear_head_dim
        # One call may cover several of the op's 128-wide chunks: `prefill`'s
        # chunk is 512 by default, while `_linear_attention_multi` hands this one
        # chunk at a time. Derive it from `seq` rather than assuming either.
        n_chunks = -(-seq // CHUNK)
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
        out = linear_rows(gated, self.w.blk(layer, "ssm_out.weight"), compute_kernel_config=HIFI4)
        return self.all_reduce(out)
    def _deltanet_scan(self, fronts, st: "LayerState"):
        """`prepare_device` + the fused op over one or more prepared chunks.

        The chunk axis is NC, which both take vectorised, so N chunks cost the
        same ~59 dispatches as one. That is the whole point of
        `_linear_attention_multi`, and it is safe: prepared chunk 0 comes out
        bit-identical whether NC is 1 or 4 -- all eight tensors, max diff
        0.000e+00 (`prepare_nc_invariance_check.py`).
        """
        from .deltanet import prepare_device

        cfg = self.cfg
        n_v, hd = self.n_v_local, cfg.linear_head_dim
        if len(fronts) == 1:
            hq, hk, hv, hg, hb = fronts[0]
        else:
            hq, hk, hv, hg, hb = (
                ttnn.concat([f[i] for f in fronts], dim=1) for i in range(5)
            )
        dev = prepare_device(hq, hk, hv, hg, hb, mesh=self.mesh)
        initial = None
        if st.recurrent is not None:
            initial = ttnn.reshape(st.recurrent, (n_v, hd, hd))
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
        return out

    def _linear_attention_chunk(
        self, mixed: ttnn.Tensor, layer: int, st: "LayerState", seq: int
    ) -> ttnn.Tensor:
        """Gated DeltaNet over one chunk: front, scan, back."""
        front, z = self._deltanet_front(mixed, layer, st, seq)
        return self._deltanet_back(self._deltanet_scan([front], st), z, layer, seq)

    def _linear_attention_multi(self, mixed_list, layer: int, st: "LayerState", seqs):
        """The same over several chunks, with one scan instead of several.

        Everything that depends on the row count -- the q/k/v/gate projections,
        the causal convolution, the output norm and projection -- still runs per
        chunk on exactly the rows it would have seen alone, so this is
        bit-identical to calling `_linear_attention_chunk` once per chunk. Only
        `prepare_device` and the fused op are batched, and they take the chunk
        axis as a batch dimension.

        Worth doing because `prepare_device` is 158.9 ms of a 922 ms prefill
        chunk -- the largest single item -- and issues ~59 dispatches whatever
        NC is.
        """
        from .deltanet import CHUNK

        n_v, hd = self.n_v_local, self.cfg.linear_head_dim
        prepared, zs = [], []
        for mixed, seq in zip(mixed_list, seqs):
            front, z = self._deltanet_front(mixed, layer, st, seq)
            prepared.append(front)
            zs.append(z)
        out = self._deltanet_scan(prepared, st)
        # Each entry may itself span several of the op's 128-wide chunks -- the
        # prefill chunk is 512 by default -- so walk the chunk axis by each
        # entry's own count rather than assuming one apiece.
        outs, off = [], 0
        for i, seq in enumerate(seqs):
            nc = -(-seq // CHUNK)
            piece = ttnn.slice(out, (0, off, 0, 0), (n_v, off + nc, CHUNK, hd))
            outs.append(self._deltanet_back(piece, zs[i], layer, seq))
            off += nc
        return outs


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

        qg = linear_rows(mixed, self.w.blk(layer, "attn_q.weight"), compute_kernel_config=HIFI4)
        qg = ttnn.reshape(qg, (1, seq, n_q, hd * 2))
        q = self._slice_last(qg, 0, hd)
        gate = self._slice_last(qg, hd, hd * 2)
        q = rms_norm(q, self.w.blk(layer, "attn_q_norm.weight"), cfg.rms_norm_eps)

        k = ttnn.reshape(
            linear_rows(mixed, self.w.blk(layer, "attn_k.weight"), compute_kernel_config=HIFI4),
            (1, seq, n_kv, hd),
        )
        v = ttnn.reshape(
            linear_rows(mixed, self.w.blk(layer, "attn_v.weight"), compute_kernel_config=HIFI4),
            (1, seq, n_kv, hd),
        )
        k = rms_norm(k, self.w.blk(layer, "attn_k_norm.weight"), cfg.rms_norm_eps)

        # rope: cos/sin come out [1, seq, 1, rope_dim]; transposed to
        # [1, 1, seq, rope_dim] they broadcast over the head axis of
        # q/k laid out as [1, n_heads, seq, hd].
        # Bound buffers, not `to_dev`: a capture cannot see a host->device copy,
        # so the rope table has to live at a fixed address that the filler
        # rewrites before each replay -- the same treatment the decode step's
        # inputs already get. The name carries `seq` because the buffer's shape
        # depends on it, exactly as `stepn{k}_cos` carries k.
        cos_t, sin_t = self.rope(list(range(start, start + seq)))
        cos = ttnn.permute(self._input(f"chunk{seq}_cos", cos_t, ttnn.float32), (0, 2, 1, 3))
        sin = ttnn.permute(self._input(f"chunk{seq}_sin", sin_t, ttnn.float32), (0, 2, 1, 3))
        q = self._apply_rope_dev(ttnn.permute(q, (0, 2, 1, 3)), cos, sin)
        k = self._apply_rope_dev(ttnn.permute(k, (0, 2, 1, 3)), cos, sin)

        self._ensure_kv(st, 1, n_kv, hd)
        page_table = self._kv_page_table(1)
        # `paged_fill_cache` writes its input at the *start* of whatever page
        # table it is handed, so a chunk at absolute `start` passes a table
        # sliced from that block onward rather than the whole one.
        blocks = self.max_seq_len // KV_BLOCK
        fill_table = ttnn.slice(page_table, (0, start // KV_BLOCK), (1, blocks))
        ttnn.experimental.paged_fill_cache(st.keys, k, fill_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(
            st.values, ttnn.permute(v, (0, 2, 1, 3)), fill_table, batch_idx=0
        )

        # `chunked_scaled_dot_product_attention` is causal internally and reads
        # `n_kv` directly, which removes three things at once: the explicit
        # additive mask, the tile-padded `kv_len` slice that mask needed, and the
        # `repeat_interleave` that expanded 2 KV heads to 24.
        #
        # The mask deserves an obituary, because its bug cost an iteration
        # (`docs/iterations/013`). Both the K/V slice and the mask were
        # TILE_LAYOUT, so a key length that was not a multiple of 32 got padded
        # -- and an additive mask pads with *zeros*, which means "attend to me".
        # The softmax spread over up to 31 all-zero key positions: at one token
        # this block returned roughly v/32 instead of v, 99.7 % wrong against a
        # float32 host computation. Slicing to a whole tile and letting the
        # causal condition run over the padded width fixed it. None of that can
        # recur here, because there is no mask to pad.
        #
        # The start is a *device tensor*, so a capture records the read rather
        # than the value and one trace serves every chunk -- the whole point of
        # this path (5.2). It must be a multiple of both program-config chunk
        # sizes, and violating that is silent rather than an error, which is why
        # `chunked_sdpa_program_config` uses 32: `prefill` guarantees only
        # tile-alignment.
        start_t = self._input(
            "chunk_start", torch.tensor([start], dtype=torch.int32), ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        wide = seq == DELTANET_CHUNK and start % DELTANET_CHUNK == 0
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q, st.keys, st.values, page_table, chunk_start_idx_tensor=start_t,
            scale=hd**-0.5,
            program_config=(
                self.chunked_sdpa_wide_config if wide else self.chunked_sdpa_program_config
            ),
            compute_kernel_config=HIFI4,
        )
        out = ttnn.reshape(ttnn.permute(out, (0, 2, 1, 3)), (1, 1, seq, n_q * hd))
        gate = ttnn.reshape(gate, (1, 1, seq, n_q * hd))
        out = ttnn.multiply(out, ttnn.sigmoid(gate))
        return linear_rows(out, self.w.blk(layer, "attn_output.weight"), compute_kernel_config=HIFI4)

    def _prefill_ffn_half(self, hidden, layer: int, seq: int, moe_chunk: int):
        """The FFN half of one prefill chunk: mix, route, run experts, reinject.

        Split out of `prefill`'s loop so the single-chunk and the grouped path
        share one body instead of two that can drift apart. It sees exactly the
        rows of one chunk either way, which is what keeps the grouped path
        bit-identical.
        """
        cfg = self.cfg
        mixed, inject = gated_residual_mix(
            hidden, self.w.blk(layer, "hc_ffn_norm.weight"),
            self.w.blk(layer, "hc_ffn_down.weight"), self.w.blk(layer, "hc_ffn_up.weight"),
            self.w.blk(layer, "hc_ffn_inject.weight"),
            cfg.rms_norm_eps, cfg.hc_count, cfg.hidden_size,
        )
        pieces = []
        # Same weight selection as decode. Naming the split halves here
        # while the model is fused loads them lazily *on top of* the
        # fused tensor -- another ~11 GB per device, which is an
        # out-of-memory at the first MoE layer, not a slow path.
        if self.fuse_expert_gate_up:
            gate_w, up_w = self.w.fused_gate_up(layer), None
        else:
            gate_w = self.w.blk(layer, "ffn_gate_exps.weight")
            up_w = self.w.blk(layer, "ffn_up_exps.weight")
        # Route in `moe_chunk`-sized groups, compute the experts over the
        # whole chunk. The two halves want different group sizes and only
        # one of them is fussy: routing at a wider group changes the
        # answer (bf16 moves the router's probabilities ~0.5 %, which
        # reorders experts across the k-th boundary -- 5.8), while the
        # expert FFN is exactly per-token at any width now that
        # `sparse_program_config` gives it a single K block. Splitting
        # them keeps this bit-identical to routing *and* computing at
        # `moe_chunk`, for one `expert_ffn` dispatch set instead of
        # `seq / moe_chunk` of them.
        routes, keeps = [], []
        for sub in range(0, seq, moe_chunk):
            width = min(moe_chunk, seq - sub)
            part = ttnn.slice(mixed, (0, 0, sub, 0), (1, 1, sub + width, cfg.hidden_size))
            w_sub, k_sub = moe.route(
                part, self.w.blk(layer, "ffn_gate_inp.weight"),
                cfg.num_experts_per_tok,
            )
            routes.append(w_sub)
            keeps.append(k_sub)
        weights = routes[0] if len(routes) == 1 else ttnn.concat(routes, dim=-2)
        keep = keeps[0] if len(keeps) == 1 else ttnn.concat(keeps, dim=-2)
        routed = self.all_reduce(
            moe.apply_experts(
                mixed, weights, keep, gate_w, up_w,
                self.w.blk(layer, "ffn_down_exps.weight"),
                cfg.num_experts, cfg.hidden_size, cfg.expert_intermediate,
            )
        )
        # The shared expert stays sub-chunked, and it is worth saying why
        # because the opposite looks obviously right. It has no routing
        # to broadcast -- four dense linears and a sigmoid gate -- so
        # `moe_chunk`, which exists to bound the routed MoE's
        # |union| x M waste, buys it nothing, and running it once on the
        # whole chunk saves 1296 dispatches (9.5 % of the prefill's
        # total, 1089 -> 1017 ms). It is also genuinely per-token:
        # `shared_expert_rows_check.py` finds no row over 1 % at any
        # group size up to 128, unlike `moe_block` (5.8).
        #
        # It still costs accuracy, because "per-token within 1 %" is not
        # "identical": against the per-row answer a 32-row group is
        # 0.000 % and a 128-row group 0.500 %, bf16 rounding that the
        # sub-chunked form happens to avoid. Measured over 107 scored
        # positions, whole-chunk is 20.6 % top-1 / NLL 6.389 against
        # 24.3 % / 5.648 sub-chunked. 7 % of a prefill is not worth that
        # here, so the dispatches stay. See handoff 5.7.
        sub_pieces = []
        for sub in range(0, seq, moe_chunk):
            width = min(moe_chunk, seq - sub)
            part = ttnn.slice(mixed, (0, 0, sub, 0), (1, 1, sub + width, cfg.hidden_size))
            sub_pieces.append(
                moe.shared_expert(
                    part, self.w.blk(layer, "ffn_gate_shexp.weight"),
                    self.w.blk(layer, "ffn_up_shexp.weight"),
                    self.w.blk(layer, "ffn_down_shexp.weight"),
                    self.w.blk(layer, "ffn_gate_inp_shexp.weight"),
                )
            )
        shared = (
            sub_pieces[0] if len(sub_pieces) == 1 else ttnn.concat(sub_pieces, dim=-2)
        )
        ffn = ttnn.add(routed, shared)
        hidden = reinject(hidden, ffn, inject, cfg.hc_count)
        if self.probe is not None:
            self.probe(layer, hidden)
        return hidden


    def prefill(self, token_ids: list[int], state: TTState, chunk: int = PREFILL_CHUNK,
                moe_chunk: int = 32, deltanet_batch: int = 1):
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

        Those are **one unwarmed draw each** -- the harness took a single
        sample until it was fixed to warm twice and take the median of nine --
        and they are wrong in both magnitude and order. Re-measured: 16 is
        1214 ms, 32 is **918.6**, 64 is 881.6, 128 is 861.7. So 16 -> 32 is
        1.32x rather than 1.85x, and larger chunks are monotonically quicker
        where the draws had 128 slower than 64. Lifting the cap from 32 to 128
        would buy 6.6 %.

        The default had been 16, chosen from the waste figures alone.

        Past 32 the answer changes, and `_MAX_MOE_CHUNK` refuses rather than
        documents it. Most of that gap was a real defect and is gone:
        `ttnn.sparse_matmul` dropped rows past the first 32-row tile whenever
        `per_core_M > 1` and K spanned more than one block, which
        `sparse_program_config` now avoids -- `moe_block` at 64 rows went from
        33 of 64 rows wrong to 1.

        What is left is irreducible. The router's probabilities differ between
        groupings by up to 0.53 % (no row over 1 %), bf16 accumulating in a
        different order under a different tiling, and `keep = ge(probs,
        threshold)` admits ties, so one row in 64 picks a different expert set --
        on that row the threshold is bit-identical and a twelfth expert simply
        crossed it. Measured after the fix, 107 scored positions: NLL 5.648 at
        32, 6.443 at 64, 6.136 at 128. The cap stays because 1.18x is not worth
        that, not because the cause is unknown. Handoff 5.8 has the whole
        investigation, including the six places the defect turned out not to be.
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
        # The chunk used to be capped at CHUNK (128) because "a chunk larger than
        # CHUNK would need the op's inter-chunk scan, which this path does not
        # build here". That was stale: `_linear_attention_chunk` computes
        # `n_chunks` from `seq`, lays the heads out as [n_v, NC, CHUNK, width],
        # and `gated_delta_attn_seq` carries the scan through
        # initial_state/final_state. What a wider chunk really changes is the row
        # count the dense linears see, and that was measured before it was
        # allowed: see `PREFILL_CHUNK`.
        if chunk <= 0 or chunk % ttnn.TILE_SIZE:
            raise ValueError(
                f"chunk must be a positive multiple of {ttnn.TILE_SIZE}, got {chunk}"
            )
        if not 0 < moe_chunk <= _MAX_MOE_CHUNK:
            raise ValueError(
                f"moe_chunk must be in 1..{_MAX_MOE_CHUNK}, got {moe_chunk}. Above "
                f"{_MAX_MOE_CHUNK} the MoE block silently returns something worse: "
                "on a 128-token chunk next-token top-1 falls from 53.1 % to 43.8 % "
                "at 64 and 21.9 % at 128, while 8, 16 and 32 agree on every token. "
                "Raise the cap with a measurement, not a reason."
            )
        if not self.traceable_kv:
            raise NotImplementedError(
                "chunked prefill reads a paged K/V cache, which only "
                "traceable_kv=True allocates; the legacy path's flat cache has "
                "no page table for `chunked_scaled_dot_product_attention`"
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
        # Chunks run in groups of `deltanet_batch`. Inside a group each chunk still
        # sees exactly its own rows for everything that depends on the row count --
        # embedding, PLE, the hyper-connection mixes, QSA attention, the DeltaNet
        # projections and convolution, the MoE -- so this is bit-identical to
        # running them one at a time. The one batched thing is the DeltaNet scan,
        # whose chunk axis is a batch dimension: `prepare_device` issues ~59
        # dispatches whatever NC is, and it is 158.9 ms of a 922 ms chunk, the
        # largest single item on a dispatch-bound path. Prepared chunk 0 is
        # bit-identical at NC=1 and NC=4, all eight tensors
        # (`prepare_nc_invariance_check.py`).
        #
        # deltanet_batch=1 reduces to the original chunk-at-a-time loop.
        #
        # It is worth 0.7 %, not the 10 % first estimated, and the gap is the
        # useful part: `prepare_device` turns out to be compute-bound rather than
        # dispatch-bound, so batching keeps its op count but quadruples the data
        # each op moves and most of the saving never appears. The 10 % that a
        # *wider* chunk buys comes from the dense matmuls getting wider, which
        # changes their blocking and so their rounding -- a different trade, and
        # not one this takes.
        group = chunk * max(deltanet_batch, 1)
        for gstart in range(0, len(token_ids), group):
            gids = token_ids[gstart : gstart + group]
            offs = list(range(0, len(gids), chunk))
            subs = [gids[o : o + chunk] for o in offs]
            seqs = [len(x) for x in subs]
            hist_ends = []
            for x in subs:
                state.histories[0].extend(x)
                hist_ends.append(len(state.histories[0]))
            hiddens = [
                ttnn.repeat(self.to_dev(self.embed(x), ttnn.bfloat16), (1, 1, 1, cfg.hc_count))
                for x in subs
            ]
            for layer in range(cfg.num_layers):
                st = state[layer]
                mixes = []
                for i in range(len(subs)):
                    if layer in self._ple_layers:
                        hiddens[i] = ttnn.add(
                            hiddens[i],
                            self._ple_chunk(hiddens[i], layer, st, state,
                                            gstart + offs[i], seqs[i], hist_ends[i]),
                        )
                    mixes.append(gated_residual_mix(
                        hiddens[i], self.w.blk(layer, "hc_attn_norm.weight"),
                        self.w.blk(layer, "hc_attn_down.weight"),
                        self.w.blk(layer, "hc_attn_up.weight"),
                        self.w.blk(layer, "hc_attn_inject.weight"),
                        cfg.rms_norm_eps, cfg.hc_count, cfg.hidden_size,
                    ))
                if cfg.is_full_attention(layer):
                    branches = [
                        self._attention_chunk(mixes[i][0], layer, st,
                                              base + gstart + offs[i], seqs[i])
                        for i in range(len(subs))
                    ]
                else:
                    branches = self._linear_attention_multi(
                        [m for m, _ in mixes], layer, st, seqs
                    )
                for i in range(len(subs)):
                    hiddens[i] = reinject(hiddens[i], branches[i], mixes[i][1], cfg.hc_count)
                    hiddens[i] = self._prefill_ffn_half(hiddens[i], layer, seqs[i], moe_chunk)
                    if self.probe is not None:
                        self.probe(layer, hiddens[i])

            final, _ = gated_residual_mix(
                hiddens[-1], self.w.get("output_hc_norm.weight"),
                self.w.get("output_hc_down.weight"),
                self.w.get("output_hc_up.weight"), None,
                cfg.rms_norm_eps, cfg.hc_count, cfg.hidden_size,
            )
            state.positions = [base + gstart + len(gids)]
        # last position only
        last = ttnn.slice(final, (0, 0, final.shape[-2] - 1, 0), (1, 1, final.shape[-2], cfg.hidden_size))
        return last

    def _ple_chunk(
        self, hidden: ttnn.Tensor, layer: int, st: LayerState, state: TTState, begin: int,
        seq: int, hist_end: int | None = None,
    ) -> ttnn.Tensor:
        """PLE over a chunk: per-token n-gram hashes, then the dilated conv.

        `hist_end` is where this chunk ends in the token history. It defaults to
        the end of the history, which is right when chunks are appended and
        consumed one at a time. The grouped prefill path appends a whole group
        before running any layer, so it passes the chunk's own end and the
        n-gram context stays causal.
        """
        cfg = self.cfg
        hist = state.histories[0]
        base = (len(hist) if hist_end is None else hist_end) - seq
        per_token = [hist[: base + i + 1] for i in range(seq)]
        # Bound for the same reason as the chunk's rope table: host work is
        # invisible to a trace capture.
        emb = self._input(f"chunk{seq}_ngram", self.ngram_embed(per_token), ttnn.bfloat16)

        key = grouped_rms_norm(
            linear_rows(emb, self.w.blk(layer, "ple_key.weight"), compute_kernel_config=HIFI4),
            self.w.blk(layer, "ple_norm_key.weight"), cfg.rms_norm_eps, cfg.hidden_size, cfg.hc_count,
        )
        value = linear_rows(emb, self.w.blk(layer, "ple_value.weight"), compute_kernel_config=HIFI4)
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

