"""Example: OpenAI-compatible server backed by a local LMStudio instance.

This file shows how to:
1. Define a backend (LMStudioBackend) that wraps Mellea's OpenAI backend
   to connect to LMStudio running at localhost:1234.
2. Subclass OpenAIRoutes and override chat_completions with a real implementation.
3. Wire everything onto a FastAPI app and serve it.

Run with:
    python main.py
or directly:
    uvicorn examples.lmstudio_server:app --reload
"""

import openai
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from mellea.backends.model_options import ModelOption
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.components.chat import Message
from mellea.stdlib.context import ChatContext

from mellea_openai_server.helpers import stream_chunks
from mellea_openai_server.routes import OpenAIRoutes
from mellea_openai_server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ModelListResponse,
    ModelObject,
)


class LMStudioBackend(OpenAIBackend):
    """Mellea backend pre-configured for a local LMStudio server."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:1234/v1",
        model_options: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            api_key="lm-studio",
            base_url=base_url,
            model_id=model_name,
            model_options=model_options,
            **kwargs,
        )

    @staticmethod
    def list_models(base_url: str = "http://localhost:1234/v1") -> list[str]:
        client = openai.OpenAI(api_key="lm-studio", base_url=base_url)
        response = client.models.list()
        return [model.id for model in response.data]


class LMStudioMelleaRoutes(OpenAIRoutes):
    """OpenAI-compatible routes backed by Mellea / LMStudio."""

    async def list_models(self) -> ModelListResponse:
        model_ids = LMStudioBackend.list_models()
        return ModelListResponse(data=[ModelObject(id=m) for m in model_ids])

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
                stream_chunks(mot, request.model),
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


app = FastAPI(title="Mellea OpenAI-Compatible Server")
routes = LMStudioMelleaRoutes()
routes.register(app)
