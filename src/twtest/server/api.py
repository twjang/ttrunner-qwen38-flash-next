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
        prompt = engine.apply_chat_template(messages, add_generation_prompt=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"chat template failed: {exc}") from exc

    prompt_ids = engine.encode(prompt)
    max_tokens = body.max_completion_tokens or body.max_tokens or 128
    gen_req = _build(body, engine, prompt_ids, max_tokens)
    created = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    stops = body.stop_strings()

    if not body.stream:
        text, reason, n = await _collect(engine, gen_req, stops)
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
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
                yield _sse(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.model,
                        "choices": [{"index": 0, "delta": {"content": event.text}, "finish_reason": None}],
                    }
                )
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
    app = FastAPI(title="twtest Qwen3.8-Flash-Next server", version="0.1.0")
    app.state.engine = engine
    app.include_router(router)
    return app
