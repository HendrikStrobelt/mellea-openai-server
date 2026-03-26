import time
import uuid

from .schemas import ChatCompletionChunk, ChunkChoice, DeltaContent


async def stream_chunks(mot, model: str):
    """Async generator that yields SSE-formatted chat completion chunks."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    prev_len = 0
    first = True

    while not mot.is_computed():
        text = await mot.astream()
        print(text, prev_len)
        new_text = text
        # prev_len = len(text)

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

    # Send finish chunk
    chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[ChunkChoice(delta=DeltaContent(), finish_reason="stop")],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
