# mellea-openai

A FastAPI library for building OpenAI-compatible API servers backed by [Mellea](https://github.com/fractalego/mellea). Subclass `OpenAIRoutes`, override the endpoints you need, and get a drop-in replacement for the OpenAI API — pointing at whatever backend you like.

## How it works

`mellea_openai_server` provides two things:

- **`OpenAIRoutes`** — a base class that registers all standard OpenAI v1 endpoints on a FastAPI app. Every handler returns `501 Not Implemented` by default. Subclass it and override whichever methods you need.
- **`stream_chunks(mot, model)`** — an async generator helper that converts a Mellea model output token (`mot`) into a well-formed SSE stream of `ChatCompletionChunk` objects, compatible with any OpenAI client.

The `examples/` directory contains a complete working server backed by [LMStudio](https://lmstudio.ai/) running locally.

## Project structure

```
mellea_openai/
├── examples/
│   └── lmstudio_server.py       # Full example: LMStudio-backed server
├── main.py                       # Uvicorn entry point
├── mellea_openai_server/
│   ├── __init__.py               # Public API: OpenAIRoutes, stream_chunks
│   ├── helpers.py                # stream_chunks() SSE generator
│   ├── routes.py                 # OpenAIRoutes base class + all stubs
│   └── schemas.py                # Pydantic schemas for the full OpenAI API surface
└── pyproject.toml
```

## Quickstart

**Requirements:** Python ≥ 3.12, [uv](https://docs.astral.sh/uv/)

```bash
uv sync
python main.py
```

The server starts at `http://localhost:8000`. The included example expects LMStudio to be running at `http://localhost:1234`.

## Writing your own server

Subclass `OpenAIRoutes` and override the methods you want:

```python
from fastapi import FastAPI
from mellea_openai_server import OpenAIRoutes, stream_chunks
from mellea_openai_server.schemas import ChatCompletionRequest, ChatCompletionResponse

class MyRoutes(OpenAIRoutes):
    async def chat_completions(self, request: ChatCompletionRequest):
        # your backend logic here
        ...

app = FastAPI()
routes = MyRoutes()
routes.register(app)
```

All other endpoints continue to return `501` until you override them.

### Streaming

Use `stream_chunks` to stream tokens from any Mellea model output token (`mot`):

```python
from fastapi.responses import StreamingResponse
from mellea_openai_server import stream_chunks

# inside an overridden chat_completions:
if request.stream:
    return StreamingResponse(
        stream_chunks(mot, request.model),
        media_type="text/event-stream",
    )
```

`stream_chunks` handles chunk IDs, timestamps, role injection on the first delta, finish reasons, and the final `[DONE]` sentinel.

## Covered endpoints

All standard OpenAI v1 endpoints are registered as stubs:

| Group | Endpoints |
|---|---|
| Chat | `POST /v1/chat/completions` |
| Completions (legacy) | `POST /v1/completions` |
| Models | `GET/DELETE /v1/models`, `GET /v1/models/{model}` |
| Embeddings | `POST /v1/embeddings` |
| Images | `POST /v1/images/generations`, `/edits`, `/variations` |
| Audio | `POST /v1/audio/speech`, `/transcriptions`, `/translations` |
| Files | `GET/POST /v1/files`, `GET/DELETE /v1/files/{id}`, content retrieval |
| Fine-tuning | Full CRUD + events + checkpoints under `/v1/fine_tuning/jobs` |
| Moderations | `POST /v1/moderations` |
| Batches | Full CRUD under `/v1/batches` |

## Dependencies

- [`mellea`](https://github.com/fractalego/mellea) ≥ 0.3.2 — LLM abstraction layer
- `fastapi` — web framework (pulled in via mellea)
- `uvicorn[standard]` — ASGI server
- `python-multipart` — required for file/audio upload endpoints