"""Async inference engine over the ttnn device model.

The mesh is a single resource and tt-metal is not re-entrant, so all device work
happens on one dedicated worker thread. Concurrency comes from the scheduler
above it: many requests may be in flight, each with its own `TTState`, and the
worker interleaves them token by token. That is the same continuous-batching
shape vLLM uses -- admit new sequences as slots free up, evict finished ones --
decoding every active sequence in a single batched device step.

All sequences share one `TTState` of `max_concurrency` slots and advance in
lockstep; a finished sequence frees its slot and `TTModel.reset_slot` clears it
for the next admission, so the batch never has to drain. This is what makes the
device numbers reachable through the API: stepping sequences one at a time cost
a full step *per sequence*, so 32 concurrent requests ran ~32x slower than the
batched path they now share.

Slot count is a real trade-off, and the measured step times are flat in batch up
to a point -- 518 ms at 1, 561 at 16, 572 at 32, 742 at 64 -- so 32 slots buy
32x the concurrency for 10 % more latency than a single sequence, while 64 slots
cost 43 % for the next doubling.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ..engine import Engine, EngineStats, GenerationRequest, TokenEvent


@dataclass
class _Sequence:
    request: GenerationRequest
    out_queue: "queue.SimpleQueue"
    slot: int | None = None
    next_token: int | None = None
    emitted: int = 0
    prompt_pos: int = 0
    done: bool = False
    finish_reason: str | None = None
    created: float = field(default_factory=time.time)


def reusable_prefix(held: list[int] | None, prompt: list[int]) -> bool:
    """Can a slot holding `held` be handed to a request for `prompt` as is?

    Only when `held` is an exact prefix of `prompt` and leaves at least one
    token to feed -- the step that consumes it is what produces the first
    output logits, and a recurrent state cannot be rewound to get it back.
    `None` means the slot's state is not accounted for and must be reset.
    """
    return held is not None and len(held) < len(prompt) and prompt[: len(held)] == held


class TTEngine(Engine):
    def __init__(
        self,
        cache_dir: str,
        gguf_dir: str,
        tokenizer_path: str,
        mesh_shape: tuple[int, int] = (1, 4),
        max_concurrency: int = 1,
        max_seq_len: int | None = None,
        chunked_prefill: bool = False,
        trace_region_bytes: int = 128 << 20,
        use_trace: bool = True,
    ):
        import ttnn

        from ..gguf.reader import GGUFModel
        from ..reference.config import Qwen4ExpConfig
        from ..reference.tokenizer import Qwen4ExpTokenizer
        from ..reference.weights import WeightStore
        from .model import TTModel
        from .weights import TTWeights

        # CCL is dead without this: every collective fails on
        # `fabric_context_ != nullptr` inside the control plane.
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        # The trace region has to be reserved at open time; a decode step records
        # on the order of 5 000 ops, so it needs real space.
        self.mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(*mesh_shape), trace_region_size=trace_region_bytes
        )

        gguf = GGUFModel.from_dir(gguf_dir)
        self.config = Qwen4ExpConfig.from_gguf(gguf.metadata)
        self.config.validate()
        # Host store serves only the two gather tensors (token_embd and the
        # 51 B-param n-gram table); everything else lives on device.
        self.host_store = WeightStore(gguf, cache_bytes=2 << 30, row_cache_bytes=8 << 30)
        self.weights = TTWeights(cache_dir, self.mesh)
        # traceable_kv=True: paged_update_cache takes the cache index as a
        # per-sequence tensor. Continuous batching needs that -- slots are
        # refilled independently, so sequences sit at different positions, and
        # the plain update_cache path writes one index for the whole batch.
        # Default to the model's full context. The K/V cache is the only thing
        # that grows with it -- 12 sparse-attention layers of
        # (batch, 2 kv-heads, seq, 256) in bf16, replicated per device, which is
        # 6.4 GB per sequence at 262144 -- and decode latency is flat in position
        # (measured 496 ms at 4 and 501 ms at 65536), because the step is
        # dominated by the MoE and DeltaNet and the attention budget is fixed at
        # 2048. So long context costs memory, not time.
        seq = max_seq_len if max_seq_len is not None else self.config.context_length
        kv_bytes = 2 * max_concurrency * 2 * seq * self.config.head_dim * 2 * (
            self.config.num_layers // 4
        )
        budget = 7 << 30  # what is left per device after 24.94 GB of weights
        if kv_bytes > budget:
            raise ValueError(
                f"K/V cache for {max_concurrency} slots at {seq} tokens is "
                f"{kv_bytes / 1e9:.1f} GB per device, over the ~{budget / 1e9:.0f} GB "
                "left after weights. Lower max_concurrency or max_seq_len: the two "
                "trade directly, and one slot at the full 262144 context fits."
            )
        self.model = TTModel(
            self.config, self.weights, self.host_store, self.mesh,
            max_seq_len=seq, traceable_kv=True,
        )
        # QSA attends to `indexer_budget` selected tokens, not to everything.
        # Below the budget every complete block is retained, so dense causal
        # attention is exactly right and cheaper; above it, dense is a different
        # model. The selection addresses the cache with uint16 indices (the only
        # dtype `ttnn.scatter` takes), so past 65536 tokens it cannot run and
        # this is a fidelity caveat the caller should know about rather than
        # discover.
        if not self.model.use_indexer and seq > self.config.indexer_budget:
            print(
                f"[tt] context {seq} exceeds the QSA budget "
                f"({self.config.indexer_budget}) and the sparse selection cannot "
                f"address it (limit {self.model.indexer_max_seq}); attention is "
                "dense beyond the budget, which is not what the model does."
            )
        # Fused gate|up experts: one sparse_matmul instead of two, verified
        # token-for-token identical. Two runs per mode, 25 samples each:
        #
        #            split                 fused
        #     B=1    522.4 / 520.3 ms      504.8 / 523.0 ms      (neutral)
        #     B=32   57.57 / 57.97 tok/s   59.51 / 58.49 tok/s   (+2 %)
        #     B=64   86.66 / 86.60 tok/s   97.47 / 97.36 tok/s   (+12 %)
        #
        # An earlier single run showed 0.94x at B=1 and B=32 and this was gated on
        # slot count because of it; that run had just written 40 GB of new weight
        # files and was reading them cold (6.1 s load against 5.3 s here). Once
        # the constant conv taps stopped being re-sliced every token the fused
        # path is neutral-to-better everywhere, so it is simply on when the
        # weights exist. Falls back silently when they do not -- they are an
        # optional artefact of scripts/fuse_expert_gate_up.py.
        if "blk.0.ffn_gateup_exps.weight" in self.weights:
            self.model.fuse_expert_gate_up = True
        self.tokenizer = Qwen4ExpTokenizer(
            tokenizer_path, self.config.chat_template,
            [t for t in (self.config.eos_token_id, 248044) if t is not None],
        )
        self.stop_token_ids = tuple(self.tokenizer.eos_token_ids)
        self.model_name = "Qwen3.8-Flash-Next"

        self._stats = EngineStats()
        self._max_concurrency = max_concurrency
        # Chunked prefill consumes 128 prompt tokens at a time through
        # `gated_delta_attn_seq` instead of replaying the decode step per token:
        # 5.77 s against 65.15 s for 128 tokens. It is single-sequence -- it
        # takes a whole prompt before returning, which a shared lockstep batch
        # cannot express -- so it is available only where there is one slot,
        # which is the configuration this engine is tuned for anyway.
        #
        # It was off entirely until `docs/iterations/013`, when it was still
        # wrong. It now predicts the next token of real text as well as the
        # decode path does (87.5% against a same-positions control of 86.7%),
        # and the rings it leaves are written in place, so a captured trace
        # replays against the buffers it recorded.
        if chunked_prefill and max_concurrency != 1:
            raise NotImplementedError(
                "chunked_prefill is incompatible with batched decoding: "
                "TTModel.prefill runs one sequence at a time, so it needs "
                f"max_concurrency=1 (got {max_concurrency})"
            )
        self._chunked_prefill = bool(chunked_prefill)
        # Chunked prefill and a captured trace do not coexist. Prefill runs
        # eagerly and allocates gigabytes of temporaries per call, and a trace
        # replays a recorded graph against the addresses it captured -- after a
        # few prefills the replay came back as token 0 repeated, the same class
        # of failure the LM head caused before `TracedDecoder` warmed it. Eager
        # prefill is *correct* (verified: warm and cold turns agree token for
        # token); it is the combination that is not.
        #
        # Which to prefer depends on the shape of the work, so it is worth
        # stating. A 2000-token prompt with 200 output tokens costs roughly
        # 2000*0.50 + 200*0.236 = 1047 s traced with the prompt stepped, against
        # 2000*0.045 + 200*0.52 = 194 s eager with the prompt prefilled. Short
        # prompts invert it, and prefix reuse makes later turns of a chat cheap
        # either way.
        if self._chunked_prefill and use_trace:
            print(
                "[tt] chunked_prefill needs eager execution; trace capture is off. "
                "Prompt ingestion is ~11x faster, each generated token ~2.2x slower."
            )
        # Trace replays a captured step with a single dispatch, and for a single
        # user that is the whole game: 2.02x at batch 1 (516 -> 255 ms), 1.52x at
        # 16, 1.29x at 32, ~1.01x by 48 where each op carries enough device work
        # to hide the dispatch cost. Verified token-for-token against eager.
        #
        # This was off by default while a captured decoder was correct through
        # TTModel and corrupt through this engine. The cause was the LM head:
        # `output.weight` loads lazily and greedy_tokens/logits allocate their own
        # intermediates, and in this engine all of that first happened *after* the
        # capture region, landing on memory the recorded graph depends on.
        # TracedDecoder now runs them before capturing. Above batch 48 the gain is
        # ~1 %, so capture is not worth its memory there.
        self._use_trace = use_trace and max_concurrency < 48 and not self._chunked_prefill
        self._decoder = None
        self._admit: queue.SimpleQueue = queue.SimpleQueue()
        self._shutdown = threading.Event()
        self._worker = threading.Thread(target=self._device_loop, name="tt-device", daemon=True)
        self._worker.start()

    # -- Engine interface ---------------------------------------------------

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)

    @property
    def stats(self) -> EngineStats:
        return self._stats

    # -- device worker -------------------------------------------------------

    def _device_loop(self) -> None:
        import torch

        from ..reference.generate import SamplingParams, sample

        TILE = 32                      # fill_cache asserts a tile-aligned index
        B = self._max_concurrency
        state = self.model.new_state(batch=B)
        slots: list[_Sequence | None] = [None] * B
        free: list[int] = list(range(B))
        # The exact token sequence each slot's state has consumed, or None when
        # the state is not reusable. This is what makes prefix reuse possible:
        # a slot that a finished sequence left behind still holds that whole
        # conversation as recurrent state, conv rings and K/V, so the next turn
        # -- which contains the previous turn as an exact prefix -- only has to
        # feed the tokens it added.
        prefix: list[list[int] | None] = [None] * B

        # Captured here rather than in __init__: it runs a warmup step, which
        # compiles every kernel and takes tens of seconds, and it must happen on
        # the device thread. A failure (most often no room for the trace buffer
        # alongside 24.94 GB of weights) is not fatal -- eager is correct, just
        # slower -- so it degrades instead of refusing to serve.
        if self._use_trace:
            try:
                from .traced import TracedDecoder

                decoder = TracedDecoder(self.model, state)
                decoder.reset()
                self._decoder = decoder
            except Exception as exc:
                self._decoder = None
                print(f"[tt] trace capture failed, falling back to eager: {exc}")

        def advance(tokens: list[int]):
            if self._decoder is not None:
                return self._decoder.step(tokens)
            return self.model.step(tokens, state)
        # A slot with no sequence still occupies a lane in the batched step; it is
        # fed a harmless token and its logits are dropped. Shrinking the batch to
        # the live count instead would mean a differently-shaped state (and a new
        # trace) every time a request arrives or finishes.
        FILLER = 0

        def params_of(seq: _Sequence) -> SamplingParams:
            r = seq.request
            return SamplingParams(
                max_tokens=r.max_tokens, temperature=r.temperature, top_p=r.top_p,
                top_k=r.top_k, seed=r.seed, stop_token_ids=r.stop_token_ids,
            )

        def retire(slot: int, seq: _Sequence) -> None:
            seq.out_queue.put(None)
            slots[slot] = None
            free.append(slot)
            self._stats.running = sum(x is not None for x in slots)

        def admit(seq: _Sequence, slot: int) -> None:
            """Give `seq` a slot, reusing the state already there when it can.

            A slot is reusable only while its `prefix` is exactly what its state
            consumed. That holds for as long as nothing steps it: an idle engine
            does not step at all, so with one slot a follow-up turn always hits.
            With several slots, a step taken for someone else feeds this one a
            filler token and invalidates it -- handled below, where the fed
            tokens are recorded.

            At least one token must be left to feed, because the step that
            consumes it is what produces the first output logits. A prompt that
            is *exactly* the cached prefix therefore starts over.
            """
            held = prefix[slot]
            prompt = seq.request.prompt_token_ids
            if reusable_prefix(held, prompt):
                seq.prompt_pos = len(held)
                self._stats.cached_prompt_tokens += len(held)
            else:
                self.model.reset_slot(state, slot)
                prefix[slot] = []
            seq.slot = slot
            slots[slot] = seq
            self._stats.running = sum(x is not None for x in slots)

            # Consume the bulk of the prompt in one go. The last token is left
            # for the decode loop, because the step that consumes it is what
            # produces the first output logits. `TTModel.prefill` resumes from
            # wherever the slot already is, so this composes with prefix reuse,
            # but `fill_cache` wants a tile-aligned start -- so prefill only a
            # whole number of tiles and let the loop step the remainder.
            if self._chunked_prefill:
                available = len(prompt) - 1 - seq.prompt_pos
                take = available - available % TILE
                if take and seq.prompt_pos % TILE == 0:
                    chunk = prompt[seq.prompt_pos : seq.prompt_pos + take]
                    self.model.prefill(chunk, state)
                    seq.prompt_pos += take
                    if prefix[slot] is not None:
                        prefix[slot].extend(chunk)

        while not self._shutdown.is_set():
            while free:
                try:
                    seq = self._admit.get_nowait()
                except queue.Empty:
                    break
                admit(seq, free.pop())
            if all(x is None for x in slots):
                try:
                    seq = self._admit.get(timeout=0.05)
                except queue.Empty:
                    continue
                admit(seq, free.pop())

            # One token per slot: the next prompt token while the prompt is still
            # being consumed, otherwise the token this slot sampled last round.
            tokens: list[int] = []
            sampling: list[int] = []          # slots whose logits we will read
            for i, seq in enumerate(slots):
                if seq is None:
                    tokens.append(FILLER)
                    continue
                prompt = seq.request.prompt_token_ids
                if seq.prompt_pos < len(prompt):
                    tokens.append(prompt[seq.prompt_pos])
                    seq.prompt_pos += 1
                    if seq.prompt_pos >= len(prompt):
                        sampling.append(i)     # last prompt token -> first output
                else:
                    tokens.append(seq.next_token)
                    sampling.append(i)

            # Greedy decoding needs one integer per sequence, not the whole
            # vocabulary: gathering [B, 248320] off four devices costs 194 ms at
            # B=32 against 38 ms for a device-side argmax. Taken only when every
            # slot sampling this step is greedy -- any slot needing a real
            # distribution (temperature > 0) forces the full gather, which then
            # serves the greedy slots too.
            greedy_only = all(
                slots[i] is not None and slots[i].request.temperature <= 0 for i in sampling
            )
            try:
                hidden = advance(tokens)
                logits = None
                argmax = self.model.greedy_tokens(hidden) if greedy_only else None
                if argmax is None:             # uneven vocab shard, or sampling needed
                    logits = self.model.logits(hidden)
            except Exception as exc:           # a device fault kills every slot
                prefix[:] = [None] * B
                for i, seq in enumerate(slots):
                    if seq is not None:
                        seq.out_queue.put(exc)
                        retire(i, seq)
                continue

            # The step happened, so every slot's state moved on by exactly one
            # token. An occupied slot extends its prefix; an empty one just ate
            # a filler token, which puts its state out of step with any prefix
            # we could claim for it.
            for i, seq in enumerate(slots):
                if seq is None:
                    prefix[i] = None
                elif prefix[i] is not None:
                    prefix[i].append(tokens[i])

            for i in sampling:
                seq = slots[i]
                if seq is None:
                    continue
                try:
                    params = params_of(seq)
                    if argmax is not None:
                        token = argmax[i]
                    else:
                        generator = None
                        if params.seed is not None:
                            generator = torch.Generator().manual_seed(params.seed + seq.emitted)
                        token = sample(logits[i], params, generator)
                    seq.next_token = token
                    seq.emitted += 1
                    self._stats.completion_tokens += 1

                    if token in params.stop_token_ids:
                        seq.out_queue.put(TokenEvent(token, "", seq.emitted, finish_reason="stop"))
                        retire(i, seq)
                    elif seq.emitted >= params.max_tokens:
                        seq.out_queue.put(TokenEvent(token, self.decode([token]), seq.emitted - 1))
                        seq.out_queue.put(TokenEvent(-1, "", seq.emitted, finish_reason="length"))
                        retire(i, seq)
                    else:
                        seq.out_queue.put(TokenEvent(token, self.decode([token]), seq.emitted - 1))
                except Exception as exc:
                    seq.out_queue.put(exc)
                    retire(i, seq)

    # -- request path ---------------------------------------------------------

    async def generate(self, request: GenerationRequest) -> AsyncIterator[TokenEvent]:
        out: queue.SimpleQueue = queue.SimpleQueue()
        seq = _Sequence(request=request, out_queue=out)
        self._stats.prompt_tokens += len(request.prompt_token_ids)
        self._stats.queued += 1
        self._admit.put(seq)

        loop = asyncio.get_running_loop()
        try:
            while True:
                item = await loop.run_in_executor(None, out.get)
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self._stats.queued = max(0, self._stats.queued - 1)

    async def close(self) -> None:
        import ttnn

        self._shutdown.set()
        self._worker.join(timeout=5)
        ttnn.close_mesh_device(self.mesh)
