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
