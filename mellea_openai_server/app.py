import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from mellea.backends.model_options import ModelOption
from mellea.stdlib.components.chat import Message
from mellea.stdlib.context import ChatContext

from mellea_extra import LMStudioBackend

from .models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChunkChoice,
    DeltaContent,
)

app = FastAPI(title="Mellea OpenAI-Compatible Server")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Build model options
    model_options: dict = {ModelOption.STREAM: True}
    if request.temperature is not None:
        model_options[ModelOption.TEMPERATURE] = request.temperature
    if request.max_tokens is not None:
        model_options[ModelOption.MAX_NEW_TOKENS] = request.max_tokens

    # Convert messages to mellea Message objects
    mellea_messages = [
        Message(role=msg.role, content=msg.content) for msg in request.messages
    ]

    # Build ChatContext from all messages except the last; last is the action
    ctx = ChatContext()
    for msg in mellea_messages[:-1]:
        ctx = ctx.add(msg)

    action = mellea_messages[-1]

    # Create backend and generate
    backend = LMStudioBackend(
        model_name=request.model, model_options=model_options
    )
    mot, _ = await backend.generate_from_context(
        action=action, ctx=ctx, model_options=model_options
    )

    if request.stream:
        return StreamingResponse(
            _stream_chunks(mot, request.model),
            media_type="text/event-stream",
        )

    # Non-streaming: await the full result
    full_text = await mot.avalue()
    return ChatCompletionResponse(
        model=request.model,
        choices=[
            Choice(message=ChatMessage(role="assistant", content=full_text))
        ],
    )


async def _stream_chunks(mot, model: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    prev_len = 0
    first = True

    while not mot.is_computed():
        text = await mot.astream()
        new_text = text[prev_len:]
        prev_len = len(text)

        if not new_text:
            continue

        delta = DeltaContent(content=new_text)
        if first:
            delta.role = "assistant"
            first = False

        chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model,
            choices=[ChunkChoice(delta=delta)],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Yield any remaining text after computed
    text = mot.value
    if text and len(text) > prev_len:
        new_text = text[prev_len:]
        delta = DeltaContent(content=new_text)
        if first:
            delta.role = "assistant"
            first = False
        chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model,
            choices=[ChunkChoice(delta=delta)],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Send finish chunk
    chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[ChunkChoice(delta=DeltaContent(), finish_reason="stop")],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Helper for stub routes
# ---------------------------------------------------------------------------

def _not_implemented():
    return JSONResponse(
        status_code=501,
        content={"error": {"message": "Not implemented", "type": "not_implemented"}},
    )


# ---------------------------------------------------------------------------
# Completions (legacy)
# ---------------------------------------------------------------------------

@app.post("/v1/completions")
async def completions():
    return _not_implemented()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    return _not_implemented()


@app.get("/v1/models/{model}")
async def retrieve_model(model: str):
    return _not_implemented()


@app.delete("/v1/models/{model}")
async def delete_model(model: str):
    return _not_implemented()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@app.post("/v1/embeddings")
async def embeddings():
    return _not_implemented()


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

@app.post("/v1/images/generations")
async def image_generations():
    return _not_implemented()


@app.post("/v1/images/edits")
async def image_edits():
    return _not_implemented()


@app.post("/v1/images/variations")
async def image_variations():
    return _not_implemented()


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

@app.post("/v1/audio/speech")
async def audio_speech():
    return _not_implemented()


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions():
    return _not_implemented()


@app.post("/v1/audio/translations")
async def audio_translations():
    return _not_implemented()


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

@app.get("/v1/files")
async def list_files():
    return _not_implemented()


@app.post("/v1/files")
async def upload_file():
    return _not_implemented()


@app.get("/v1/files/{file_id}")
async def retrieve_file(file_id: str):
    return _not_implemented()


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str):
    return _not_implemented()


@app.get("/v1/files/{file_id}/content")
async def retrieve_file_content(file_id: str):
    return _not_implemented()


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

@app.post("/v1/fine_tuning/jobs")
async def create_fine_tuning_job():
    return _not_implemented()


@app.get("/v1/fine_tuning/jobs")
async def list_fine_tuning_jobs():
    return _not_implemented()


@app.get("/v1/fine_tuning/jobs/{job_id}")
async def retrieve_fine_tuning_job(job_id: str):
    return _not_implemented()


@app.post("/v1/fine_tuning/jobs/{job_id}/cancel")
async def cancel_fine_tuning_job(job_id: str):
    return _not_implemented()


@app.get("/v1/fine_tuning/jobs/{job_id}/events")
async def list_fine_tuning_events(job_id: str):
    return _not_implemented()


@app.get("/v1/fine_tuning/jobs/{job_id}/checkpoints")
async def list_fine_tuning_checkpoints(job_id: str):
    return _not_implemented()


# ---------------------------------------------------------------------------
# Moderations
# ---------------------------------------------------------------------------

@app.post("/v1/moderations")
async def moderations():
    return _not_implemented()


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

@app.post("/v1/batches")
async def create_batch():
    return _not_implemented()


@app.get("/v1/batches")
async def list_batches():
    return _not_implemented()


@app.get("/v1/batches/{batch_id}")
async def retrieve_batch(batch_id: str):
    return _not_implemented()


@app.post("/v1/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    return _not_implemented()
