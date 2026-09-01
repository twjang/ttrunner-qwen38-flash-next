# 006 — OpenAI-compatible server

Date: 2026-09-01
Status: **working**

## Goal
Deliverable (4): a vLLM-style OpenAI-compatible server, with async support.

---

## Design decision — the server must not depend on which engine runs

Deliverable (4) says "a server around (3)", but (3) is the long pole and (2)
already works. Binding the HTTP layer to the ttnn engine would mean nothing is
demonstrable until the very end, and would bake device assumptions into request
handling.

**Remedy** — `src/twtest/engine.py` defines an `Engine` interface: encode,
decode, chat template, `stats`, and `generate(request) -> AsyncIterator[TokenEvent]`.
The server talks only to that. `--backend reference` serves the CPU model today;
`--backend tt` is the same server once the ttnn engine lands.

Expressing generation as an async iterator is what makes both shapes work: a
synchronous CPU engine offloaded to a thread, and a natively async device
engine, present the same interface.

---

## Observation — the reference engine is not re-entrant

It walks a single `HybridCache`, so two concurrent requests would corrupt each
other's conv/recurrent/KV state. Rather than pretend otherwise, `ReferenceEngine`
serialises through a semaphore and says so in its docstring: requests queue and
stream independently, but do not overlap on the CPU. The token loop runs in a
worker thread and hands tokens back with `loop.call_soon_threadsafe`, so the HTTP
response streams as tokens are produced rather than after the whole completion.

Real batched concurrency is the ttnn engine's job.

---

## Endpoints

`GET /v1/models`, `GET /health`, `POST /v1/completions`,
`POST /v1/chat/completions` — each with SSE streaming, `stop` strings,
`temperature`/`top_p`/`top_k`/`seed`, and OpenAI-shaped `usage`.

## Verified against the running server

```
$ curl /v1/chat/completions -d '{"messages":[{"role":"user",
    "content":"What is the capital of France? Answer in one word."}],"max_tokens":6}'
{"object":"chat.completion", ...
 "choices":[{"message":{"role":"assistant","content":"We need to answer user's"},
             "finish_reason":"length"}],
 "usage":{"prompt_tokens":64,"completion_tokens":6,"total_tokens":70}}
```

The chat template applied cleanly (64 prompt tokens from a short question — this
model's template opens a thinking block, which is why the reply reads as
reasoning rather than "Paris").

Streaming:
```
data: {"choices":[{"index":0,"text":" Paris","finish_reason":null}]}
data: {"choices":[{"index":0,"text":".","finish_reason":null}]}
data: {"choices":[{"index":0,"text":" The","finish_reason":null}]}
data: {"choices":[{"index":0,"text":"","finish_reason":"length"}]}
data: [DONE]
```

Throughput is the reference engine's: ~12 s/token, ~7 min for a 64-token
prefill. Correct, and slow exactly where expected.

## Run it

```
python -m twtest.server \
  --model    /home/twjang/models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS \
  --tokenizer /home/twjang/models/Qwen3.8-Flash-Next-tokenizer/tokenizer.json \
  --port 8000
```
