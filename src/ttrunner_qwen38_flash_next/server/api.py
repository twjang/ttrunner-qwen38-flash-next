"""OpenAI-compatible HTTP API.

Implements the subset of the OpenAI surface that vLLM exposes and that clients
actually depend on: /v1/models, /v1/completions and /v1/chat/completions, each
with server-sent-event streaming. The engine is injected, so this serves either
the CPU reference or the ttnn backend unchanged.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..engine import Engine, GenerationRequest

router = APIRouter()


# --------------------------------------------------------------------------
# request/response schemas
# --------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None


class _Sampling(BaseModel):
    max_tokens: int | None = Field(default=128, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=20, ge=-1)
    seed: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    n: int = Field(default=1, ge=1, le=1)  # only n=1 is supported

    def stop_strings(self) -> tuple[str, ...]:
        if self.stop is None:
            return ()
        return tuple([self.stop] if isinstance(self.stop, str) else self.stop)


class ChatCompletionRequest(_Sampling):
    model: str
    messages: list[ChatMessage]
    max_completion_tokens: int | None = None
    # Passed straight to the Jinja chat template. This checkpoint's template
    # reads `enable_thinking` (default true) and `reasoning_effort`, so
    # {"enable_thinking": false} is how a caller asks for a direct answer. The
    # name matches vLLM and SGLang so existing clients need no special case.
    chat_template_kwargs: dict[str, Any] | None = None


class CompletionRequest(_Sampling):
    model: str
    prompt: str | list[str]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _engine(request: Request) -> Engine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    return engine


def _build(req: _Sampling, engine: Engine, prompt_ids: list[int], max_tokens: int) -> GenerationRequest:
    stop_ids = tuple(getattr(engine, "stop_token_ids", ()) or ())
    return GenerationRequest(
        prompt_token_ids=prompt_ids,
        max_tokens=max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        seed=req.seed,
        stop_token_ids=stop_ids,
        stop_strings=req.stop_strings(),
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


THINK_CLOSE = "</think>"


def _thinking_is_open(prompt: str) -> bool:
    """Does the rendered prompt leave the model inside a reasoning block?

    Read off the prompt rather than assumed from the request, because the
    template decides. With thinking on it ends `<|im_start|>assistant\n<think>\n`
    and the model generates reasoning, then `</think>`, then the answer. With
    `enable_thinking: false` it ends `<think>\n\n</think>\n\n` -- already
    closed -- and everything generated is answer.
    """
    return prompt.rstrip().endswith("<think>")


class _ThinkSplitter:
    """Split a reasoning model's output into reasoning and answer.

    The closing tag can straddle two token deltas, so while the block is open
    everything is buffered and released when the tag arrives. Reasoning is
    therefore delivered in one piece rather than incrementally, which costs
    nothing that matters: the answer still streams from the moment it starts,
    and that is the part a caller renders.

    When the block never closes -- the token budget ran out mid-thought -- the
    buffer is reasoning, and the answer is empty. That is the honest shape, and
    an empty `content` with `finish_reason: length` says exactly what happened.
    """

    def __init__(self, open_block: bool) -> None:
        self.open = open_block
        self._buf = ""
        # the template puts a blank line after the tag, and it is not part of the
        # answer. It cannot simply be stripped from the delta that carried the
        # tag: if the tag straddles two deltas the newlines arrive in a later
        # one, so the trim is a state that outlives a single delta.
        self._trim = False

    def _answer(self, text: str) -> str:
        if self._trim:
            text = text.lstrip("\n")
            if text:
                self._trim = False
        return text

    def feed(self, delta: str) -> tuple[str, str]:
        """One delta in, (reasoning, content) out. Either may be empty."""
        if not self.open:
            return "", self._answer(delta)
        self._buf += delta
        cut = self._buf.find(THINK_CLOSE)
        if cut < 0:
            return "", ""
        reasoning = self._buf[:cut]
        rest = self._buf[cut + len(THINK_CLOSE):]
        self._buf = ""
        self.open = False
        self._trim = True
        return reasoning, self._answer(rest)

    def flush(self) -> tuple[str, str]:
        buf, self._buf = self._buf, ""
        return (buf, "") if self.open else ("", self._answer(buf))


def split_thinking(text: str, open_block: bool) -> tuple[str, str]:
    """`_ThinkSplitter` over a whole string. -> (reasoning, content)."""
    sp = _ThinkSplitter(open_block)
    r1, c1 = sp.feed(text)
    r2, c2 = sp.flush()
    return r1 + r2, c1 + c2


async def _collect(
    engine: Engine, gen_req: GenerationRequest, stop_strings: tuple[str, ...]
) -> tuple[str, str, int]:
    """Run to completion, honouring stop strings. Returns (text, reason, n_tokens)."""
    pieces: list[str] = []
    reason = "length"
    count = 0
    async for event in engine.generate(gen_req):
        if event.finish_reason:
            reason = event.finish_reason
            break
        pieces.append(event.text)
        count += 1
        text = "".join(pieces)
        hit = next((s for s in stop_strings if s and s in text), None)
        if hit:
            pieces = [text[: text.index(hit)]]
            reason = "stop"
            break
    return "".join(pieces), reason, count


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    engine = _engine(request)
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": engine.model_name,
                    "object": "model",
                    "created": int(engine.stats.started_at),
                    "owned_by": "local",
                }
            ],
        }
    )


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    s = engine.stats
    return JSONResponse(
        {
            "status": "ok",
            "model": engine.model_name,
            "uptime_s": round(time.time() - s.started_at, 1),
            "queued": s.queued,
            "running": s.running,
            "prompt_tokens": s.prompt_tokens,
            "completion_tokens": s.completion_tokens,
        }
    )


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    engine = _engine(request)
    messages = [m.model_dump() for m in body.messages]
    try:
        prompt = engine.apply_chat_template(
            messages, add_generation_prompt=True, **(body.chat_template_kwargs or {})
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"chat template failed: {exc}") from exc

    # Whether the model is starting inside a reasoning block is a property of
    # the prompt the template just produced, so read it there.
    open_block = _thinking_is_open(prompt)
    prompt_ids = engine.encode(prompt)
    max_tokens = body.max_completion_tokens or body.max_tokens or 128
    gen_req = _build(body, engine, prompt_ids, max_tokens)
    created = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    stops = body.stop_strings()

    if not body.stream:
        text, reason, n = await _collect(engine, gen_req, stops)
        reasoning, content = split_thinking(text, open_block)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            # the key DeepSeek-R1 and vLLM use, so clients that know about it
            # find the reasoning and clients that do not still get a clean answer
            message["reasoning_content"] = reasoning
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": n,
                    "total_tokens": len(prompt_ids) + n,
                },
            }
        )

    async def stream() -> AsyncIterator[str]:
        yield _sse(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
        acc: list[str] = []
        reason = "length"
        splitter = _ThinkSplitter(open_block)

        def chunk(delta: dict) -> str:
            return _sse(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
            )

        try:
            async for event in engine.generate(gen_req):
                if event.finish_reason:
                    reason = event.finish_reason
                    break
                acc.append(event.text)
                joined = "".join(acc)
                hit = next((s for s in stops if s and s in joined), None)
                if hit:
                    reason = "stop"
                    break
                reasoning, content = splitter.feed(event.text)
                if reasoning:
                    yield chunk({"reasoning_content": reasoning})
                if content:
                    yield chunk({"content": content})
            # whatever is still buffered: reasoning if the block never closed
            reasoning, content = splitter.flush()
            if reasoning:
                yield chunk({"reasoning_content": reasoning})
            if content:
                yield chunk({"content": content})
        except Exception as exc:  # surface mid-stream failures to the client
            yield _sse({"error": {"message": str(exc), "type": type(exc).__name__}})
            return
        yield _sse(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/v1/completions")
async def completions(body: CompletionRequest, request: Request):
    engine = _engine(request)
    if isinstance(body.prompt, list):
        if len(body.prompt) != 1:
            raise HTTPException(status_code=400, detail="only a single prompt is supported")
        prompt = body.prompt[0]
    else:
        prompt = body.prompt

    prompt_ids = engine.encode(prompt)
    gen_req = _build(body, engine, prompt_ids, body.max_tokens or 128)
    created = int(time.time())
    cid = f"cmpl-{uuid.uuid4().hex[:24]}"
    stops = body.stop_strings()

    if not body.stream:
        text, reason, n = await _collect(engine, gen_req, stops)
        return JSONResponse(
            {
                "id": cid,
                "object": "text_completion",
                "created": created,
                "model": body.model,
                "choices": [{"index": 0, "text": text, "finish_reason": reason, "logprobs": None}],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": n,
                    "total_tokens": len(prompt_ids) + n,
                },
            }
        )

    async def stream() -> AsyncIterator[str]:
        acc: list[str] = []
        reason = "length"
        async for event in engine.generate(gen_req):
            if event.finish_reason:
                reason = event.finish_reason
                break
            acc.append(event.text)
            if any(s and s in "".join(acc) for s in stops):
                reason = "stop"
                break
            yield _sse(
                {
                    "id": cid,
                    "object": "text_completion",
                    "created": created,
                    "model": body.model,
                    "choices": [{"index": 0, "text": event.text, "finish_reason": None}],
                }
            )
        yield _sse(
            {
                "id": cid,
                "object": "text_completion",
                "created": created,
                "model": body.model,
                "choices": [{"index": 0, "text": "", "finish_reason": reason}],
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def create_app(engine: Engine | None = None) -> FastAPI:
    app = FastAPI(title="ttrunner_qwen38_flash_next Qwen3.8-Flash-Next server", version="0.1.0")
    app.state.engine = engine
    app.include_router(router)
    return app
