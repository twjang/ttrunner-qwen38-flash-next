"""Capture a decode step as a single ttnn trace.

Why this matters here: a step issues roughly 5 000 device ops (48 layers of
hyper-connections, DeltaNet or QSA, and MoE), and the measurements say the path
is dispatch-bound, not FLOP-bound -- a MoE layer costs 1.79 ms for one token and
only 3.21 ms for eight. A trace records the op sequence once and replays it with
a single dispatch, so it removes the per-op host cost outright.

Two preconditions, both already satisfied by the model:

* **Static shapes.** The K/V cache is preallocated and written by
  `paged_update_cache(..., update_idxs_tensor=)`, and attention is
  `sdpa_decode(..., cur_pos_tensor=)`, so the position is data, not a shape.
* **Fixed addresses.** Per-step inputs live in bound buffers filled by
  `TTModel._input`, and every piece of recurrent state (DeltaNet state, the two
  conv windows, the K/V cache) is updated in place via `ttnn.copy` rather than
  rebound to a new tensor.
* **Phase-invariant rings.** The conv windows rotate by index in the eager path
  (`ring[step % depth] = col`), which is host-side Python and therefore invisible
  to capture: a replayed step would read a convolution history frozen at capture
  time. `TTModel.trace_safe_rings`, set below before the warmup step, switches
  them to fixed read indices with a device-copy shift. Without it the traced path
  diverges from eager at the first generated token -- it did, and the trace
  benchmarks that measured only wall-clock did not notice.

Replay therefore only needs the host to refresh the input buffers.
"""

from __future__ import annotations

import torch
import ttnn

from .model import TTModel, TTState


class TracedStepN:
    """A captured `step_n`: k tokens of one sequence, replayable at any position.

    The single-token trace is what makes decoding fast; this is what makes a
    speculative *verifier* fast. Without it there is nothing to gain: `step_n`
    at k=4 costs 864 ms eager against 2071 for four eager steps, but four
    **traced** steps are 944 ms, so an untraced verifier is already beaten by
    the thing it is supposed to replace.

    One capture per k -- the graph has k unrolled convolution and recurrence
    steps in it, so k is a shape, not data. The position is not: rope, the cache
    indices and the n-gram rows are all bound buffers refreshed before replay,
    exactly as for the single-token step.
    """

    def __init__(self, model: TTModel, state: TTState, k: int, warmup_token: int = 1000,
                 cq_id: int = 0):
        if state.batch != 1:
            raise ValueError("step_n is single-sequence, so its trace is too")
        self.model = model
        self.state = state
        self.k = k
        self.cq_id = cq_id

        if model.bound is None:
            model.bound = {}
        model.trace_safe_rings = True
        warm = [warmup_token] * k
        # Two real calls first: the buffers and every kernel have to exist
        # before capture, and the LM head has to be warmed for the same reason
        # `TracedDecoder` warms it -- a caller reads the logits of the k rows.
        model.step_n(warm, state)
        ttnn.synchronize_device(model.mesh)
        warm_hidden = model.step_n(warm, state)
        model.logits(warm_hidden)
        model.greedy_tokens(warm_hidden)
        ttnn.synchronize_device(model.mesh)

        model._skip_copy = True
        try:
            self.trace_id = ttnn.begin_trace_capture(model.mesh, cq_id=cq_id)
            self.output = model.step_n(warm, state)
            ttnn.end_trace_capture(model.mesh, self.trace_id, cq_id=cq_id)
        finally:
            model._skip_copy = False
        ttnn.synchronize_device(model.mesh)

    def _fill_inputs(self, tokens: list[int]) -> None:
        model, state = self.model, self.state
        bound = model.bound
        start = state.positions[0]

        def write(name, host, dtype, layout=ttnn.TILE_LAYOUT):
            buf = bound.get(name)
            if buf is None:
                return
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(host, dtype=dtype, layout=layout, mesh_mapper=model.replicate), buf
            )

        k = self.k
        write(f"embed_n{k}", model.embed(tokens), ttnn.bfloat16)
        positions = list(range(start, start + k))
        cos, sin = model.rope(positions)
        write(f"stepn{k}_cos", cos, ttnn.float32)
        write(f"stepn{k}_sin", sin, ttnn.float32)
        for i, pos in enumerate(positions):
            write(f"stepn{k}_pos{i}", torch.tensor([pos], dtype=torch.int32),
                  ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        # each row's n-gram hash reads the history up to and including its token
        base = list(state.histories[0])
        for i, tok in enumerate(tokens):
            base.append(tok)
            write(f"ngram_n{k}_{i}", model.ngram_embed([list(base)]), ttnn.bfloat16)

    def step_n(self, tokens: list[int]) -> ttnn.Tensor:
        """Replay the capture for these k tokens. Returns [1, 1, k, hidden]."""
        if len(tokens) != self.k:
            raise ValueError(f"this trace was captured for k={self.k}, got {len(tokens)}")
        state = self.state
        self._fill_inputs(tokens)
        state.histories[0].extend(tokens)
        ttnn.execute_trace(self.model.mesh, self.trace_id, cq_id=self.cq_id, blocking=True)
        state.positions = [state.positions[0] + self.k]
        return self.output

    def release(self) -> None:
        ttnn.release_trace(self.model.mesh, self.trace_id)


class TracedDecoder:
    """A captured decode step, replayable at any position."""

    def __init__(self, model: TTModel, state: TTState, warmup_token: int = 1000, cq_id: int = 0):
        self.model = model
        self.state = state
        self.cq_id = cq_id
        self.batch = state.batch

        model.bound = {}
        # Must be set before the warmup step, so the rings are allocated and
        # advanced by the same rule that capture will record. A trace cannot see
        # the host-side `ring[pos] = col` rotation the eager path uses, and would
        # replay one frozen phase for ever.
        model.trace_safe_rings = True
        # A real step first: it allocates the bound buffers and compiles every
        # kernel, neither of which may happen inside the capture region.
        model.step([warmup_token] * self.batch, state)
        ttnn.synchronize_device(model.mesh)

        # Everything the steady-state loop allocates must exist before capture.
        # The warmup step above covers the decode graph, but not the LM head:
        # `output.weight` is loaded lazily and `greedy_tokens`/`logits` allocate
        # their own intermediates, and in the engine those first happen *after*
        # the capture region. Force them now.
        warm_hidden = model.step([warmup_token] * self.batch, state)
        model.logits(warm_hidden)
        model.greedy_tokens(warm_hidden)
        ttnn.synchronize_device(model.mesh)

        model._skip_copy = True
        try:
            self.trace_id = ttnn.begin_trace_capture(model.mesh, cq_id=cq_id)
            self.output = model.step([warmup_token] * self.batch, state)
            ttnn.end_trace_capture(model.mesh, self.trace_id, cq_id=cq_id)
        finally:
            model._skip_copy = False
        ttnn.synchronize_device(model.mesh)


    # -- host-side input refresh -------------------------------------------

    def _fill_inputs(self, tokens: list[int], state: TTState) -> None:
        """Write this step's inputs into the bound buffers.

        Mirrors exactly the inputs `TTModel` reads through `_input`: the token
        embedding rows, the n-gram rows for the PLE layer, the rope tables, and
        the per-sequence cache position.
        """
        model = self.model
        bound = model.bound
        assert bound is not None

        def write(name: str, host: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT) -> None:
            buf = bound.get(name)
            if buf is None:
                return
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(host, dtype=dtype, layout=layout, mesh_mapper=model.replicate), buf
            )

        write("embed", model.embed(tokens), ttnn.bfloat16)
        if "ngram" in bound:
            write("ngram", model.ngram_embed(state.histories), ttnn.bfloat16)
        cos, sin = model.rope(list(state.positions))
        write("rope_cos", cos, ttnn.float32)
        write("rope_sin", sin, ttnn.float32)
        write(
            "cur_pos", torch.tensor(list(state.positions), dtype=torch.int32),
            ttnn.int32, ttnn.ROW_MAJOR_LAYOUT,
        )
        if "idx_bias" in bound:
            # QSA's sparse selection reads five more per-step inputs. They are
            # all derived from the position, and they have to be written here
            # like every other one: a host-to-device copy is refused *during*
            # capture, which is what `TTModel._skip_copy` is for.
            ratio = model.indexer_ratio
            starts = [ratio * (p // ratio) for p in state.positions]
            b_cos, b_sin = model.rope(starts)
            write("idx_block_cos", b_cos, ttnn.float32)
            write("idx_block_sin", b_sin, ttnn.float32)
            write(
                "idx_block_pos",
                torch.tensor([p // ratio for p in state.positions], dtype=torch.int32),
                ttnn.int32, ttnn.ROW_MAJOR_LAYOUT,
            )
            write("idx_bias", model._block_bias(list(state.positions)), ttnn.float32)
            width = model.indexer_topk * ratio
            write(
                "idx_cur_pos_f",
                torch.tensor(list(state.positions), dtype=torch.float32)
                .reshape(-1, 1, 1, 1).expand(-1, 1, 1, width).contiguous(),
                ttnn.float32,
            )
            tail_idx, tail_vis = model._tail_block(list(state.positions))
            write("idx_tail", tail_idx, ttnn.float32)
            write("idx_tail_vis", tail_vis, ttnn.float32)

    # -- public API ---------------------------------------------------------

    def step(self, tokens: int | list[int]) -> ttnn.Tensor:
        """Advance one token per sequence by replaying the trace."""
        ids = [tokens] * self.batch if isinstance(tokens, int) else list(tokens)
        state = self.state
        for seq, tok in enumerate(ids):
            state.histories[seq].append(tok)
        self._fill_inputs(ids, state)
        # blocking=True: the caller reads `self.output` as soon as this returns,
        # and with blocking=False the engine read it before the replay had
        # written it -- ' Paris.' came back as '!!!!'. It reproduced only in the
        # engine because every standalone harness happened to put a synchronising
        # to_torch or a benchmark barrier between the replay and the read, which
        # is also why interleaving an eager shadow step made the corruption
        # disappear. Correctness is not something to leave to timing.
        ttnn.execute_trace(self.model.mesh, self.trace_id, cq_id=self.cq_id, blocking=True)
        state.positions = [p + 1 for p in state.positions]
        return self.output

    def reset(self) -> None:
        """Rewind to position 0, keeping every buffer at its captured address.

        Capture consumes two warmup tokens, which leaves the recurrent state,
        conv windows and K/V cache populated. The trace is bound to *those*
        buffers, so they cannot be replaced -- they have to be zeroed in place
        and the host-side bookkeeping rewound.
        """
        state = self.state

        def clear(buf) -> None:
            zeros = ttnn.zeros(
                list(buf.shape), dtype=buf.dtype, layout=buf.layout, device=self.model.mesh
            )
            ttnn.copy(zeros, buf)

        for layer in state.layers:
            for name in ("recurrent", "conv", "ple_conv", "keys", "values"):
                buf = getattr(layer, name)
                if buf is None:
                    continue
                # conv/ple_conv became rings (lists of single columns) when the
                # window stopped being one tensor; the rest are still tensors.
                if isinstance(buf, list):
                    for entry in buf:
                        clear(entry)
                else:
                    clear(buf)
            layer.conv_step = 0
            layer.ple_step = 0
        state.positions = [0] * self.batch
        state.histories = [[] for _ in range(self.batch)]

    def release(self) -> None:
        ttnn.release_trace(self.model.mesh, self.trace_id)
        self.model.bound = None
