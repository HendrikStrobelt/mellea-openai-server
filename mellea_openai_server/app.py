from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from mellea.backends.model_options import ModelOption
from mellea.stdlib.components.chat import Message
from mellea.stdlib.context import ChatContext

from mellea_extra import LMStudioBackend

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
)
from .routes import OpenAIRoutes


class MelleaRoutes(OpenAIRoutes):
    """OpenAI-compatible routes backed by Mellea / LMStudio."""

    async def chat_completions(self, request: ChatCompletionRequest):
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
                self._stream_chunks(mot, request.model),
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


# ---------------------------------------------------------------------------
# Module-level app — keeps main.py working unchanged
# ---------------------------------------------------------------------------

app = FastAPI(title="Mellea OpenAI-Compatible Server")
routes = MelleaRoutes()
routes.register(app)
