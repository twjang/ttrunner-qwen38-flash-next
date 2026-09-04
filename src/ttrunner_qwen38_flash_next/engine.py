"""Engine abstraction shared by the reference (CPU) and ttnn backends.

The server talks only to this interface, so the slow PyTorch reference and the
ttnn engine are interchangeable. Generation is expressed as an async iterator of
token events, which is what both a thread-offloaded synchronous engine and a
natively async device engine can provide.
"""

from __future__ import annotations

import abc
import asyncio
import itertools
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass(slots=True)
class GenerationRequest:
    prompt_token_ids: list[int]
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    stop_strings: tuple[str, ...] = ()
    request_id: str = field(default_factory=lambda: f"req-{next(_COUNTER)}")


_COUNTER = itertools.count(1)


@dataclass(slots=True)
class TokenEvent:
    token_id: int
    text: str
    index: int
    finish_reason: str | None = None


@dataclass(slots=True)
class EngineStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Prompt tokens a backend did not have to feed because a slot already held
    # them as state -- see TTEngine's prefix reuse.
    cached_prompt_tokens: int = 0
    queued: int = 0
    running: int = 0
    started_at: float = field(default_factory=time.time)


class Engine(abc.ABC):
    """Backend interface."""

    model_name: str

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abc.abstractmethod
    def decode(self, token_ids: list[int]) -> str: ...

    @abc.abstractmethod
    def apply_chat_template(
        self, messages: list[dict], add_generation_prompt: bool = True, **kwargs
    ) -> str: ...

    @abc.abstractmethod
    def generate(self, request: GenerationRequest) -> AsyncIterator[TokenEvent]: ...

    @property
    @abc.abstractmethod
    def stats(self) -> EngineStats: ...

    async def close(self) -> None:
        return None


class ReferenceEngine(Engine):
    """Wraps the synchronous CPU reference model.

    The model holds no per-request device state but is not re-entrant (it walks
    a single HybridCache), so requests are serialised through one worker thread.
    Concurrency is still real from the caller's point of view: requests queue and
    stream independently, they just do not overlap on the CPU. The ttnn engine
    replaces this with genuine batched execution.
    """

    def __init__(self, model, tokenizer, config, max_concurrency: int = 1):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_name = "Qwen3.8-Flash-Next"
        self._sem = asyncio.Semaphore(max_concurrency)
        self._stats = EngineStats()
        self._lock = asyncio.Lock()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def apply_chat_template(
        self, messages: list[dict], add_generation_prompt: bool = True, **kwargs
    ) -> str:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, **kwargs
        )

    @property
    def stats(self) -> EngineStats:
        return self._stats

    async def generate(self, request: GenerationRequest) -> AsyncIterator[TokenEvent]:
        from .reference.generate import SamplingParams, generate as sync_generate

        params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            seed=request.seed,
            stop_token_ids=request.stop_token_ids,
        )

        self._stats.queued += 1
        async with self._sem:
            self._stats.queued -= 1
            self._stats.running += 1
            self._stats.prompt_tokens += len(request.prompt_token_ids)
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[TokenEvent | None | BaseException] = asyncio.Queue()

            def worker() -> None:
                # Runs in a thread; hands tokens back through the event loop so
                # the HTTP response streams as they are produced.
                try:
                    for i, token in enumerate(sync_generate(self.model, request.prompt_token_ids, params)):
                        event = TokenEvent(token, self.tokenizer.decode([token]), i)
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                except BaseException as exc:  # surfaced to the request
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                else:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            future = loop.run_in_executor(None, worker)
            emitted = 0
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    emitted += 1
                    self._stats.completion_tokens += 1
                    if item.token_id in request.stop_token_ids:
                        yield TokenEvent(item.token_id, "", item.index, finish_reason="stop")
                        break
                    yield item
                else:  # pragma: no cover
                    pass
                if emitted >= request.max_tokens:
                    yield TokenEvent(-1, "", emitted, finish_reason="length")
            finally:
                self._stats.running -= 1
                await future
