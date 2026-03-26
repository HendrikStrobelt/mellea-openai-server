from fastapi import FastAPI, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .schemas import (
    AudioSpeechRequest,
    AudioTranscriptionResponse,
    AudioTranslationResponse,
    BatchListResponse,
    BatchObject,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    CreateBatchRequest,
    CreateFineTuningJobRequest,
    DeleteFileResponse,
    DeleteModelResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    FileListResponse,
    FileObject,
    FineTuningCheckpointListResponse,
    FineTuningEventListResponse,
    FineTuningJob,
    FineTuningJobListResponse,
    ImageGenerationRequest,
    ImageResponse,
    ModelListResponse,
    ModelObject,
    ModerationRequest,
    ModerationResponse,
)


# ---------------------------------------------------------------------------
# Helper for stub routes
# ---------------------------------------------------------------------------

def _not_implemented():
    return JSONResponse(
        status_code=501,
        content={"error": {"message": "Not implemented", "type": "not_implemented"}},
    )


# ---------------------------------------------------------------------------
# OpenAIRoutes — inheritable base class with all route handlers as stubs
# ---------------------------------------------------------------------------

class OpenAIRoutes:
    """Route handlers for an OpenAI-compatible API server.

    Subclass and override any method to customize behavior, then call
    ``register(app)`` to wire routes onto a FastAPI application.
    """

    def register(self, app: FastAPI):
        """Register all routes on the given FastAPI app."""
        # Chat / Completions
        app.add_api_route("/v1/chat/completions", self.chat_completions, methods=["POST"])
        app.add_api_route("/v1/completions", self.completions, methods=["POST"])

        # Models
        app.add_api_route("/v1/models", self.list_models, methods=["GET"])
        app.add_api_route("/v1/models/{model}", self.retrieve_model, methods=["GET"])
        app.add_api_route("/v1/models/{model}", self.delete_model, methods=["DELETE"])

        # Embeddings
        app.add_api_route("/v1/embeddings", self.embeddings, methods=["POST"])

        # Images
        app.add_api_route("/v1/images/generations", self.image_generations, methods=["POST"])
        app.add_api_route("/v1/images/edits", self.image_edits, methods=["POST"])
        app.add_api_route("/v1/images/variations", self.image_variations, methods=["POST"])

        # Audio
        app.add_api_route("/v1/audio/speech", self.audio_speech, methods=["POST"])
        app.add_api_route("/v1/audio/transcriptions", self.audio_transcriptions, methods=["POST"])
        app.add_api_route("/v1/audio/translations", self.audio_translations, methods=["POST"])

        # Files
        app.add_api_route("/v1/files", self.list_files, methods=["GET"])
        app.add_api_route("/v1/files", self.upload_file, methods=["POST"])
        app.add_api_route("/v1/files/{file_id}", self.retrieve_file, methods=["GET"])
        app.add_api_route("/v1/files/{file_id}", self.delete_file, methods=["DELETE"])
        app.add_api_route("/v1/files/{file_id}/content", self.retrieve_file_content, methods=["GET"])

        # Fine-tuning
        app.add_api_route("/v1/fine_tuning/jobs", self.create_fine_tuning_job, methods=["POST"])
        app.add_api_route("/v1/fine_tuning/jobs", self.list_fine_tuning_jobs, methods=["GET"])
        app.add_api_route("/v1/fine_tuning/jobs/{job_id}", self.retrieve_fine_tuning_job, methods=["GET"])
        app.add_api_route("/v1/fine_tuning/jobs/{job_id}/cancel", self.cancel_fine_tuning_job, methods=["POST"])
        app.add_api_route("/v1/fine_tuning/jobs/{job_id}/events", self.list_fine_tuning_events, methods=["GET"])
        app.add_api_route("/v1/fine_tuning/jobs/{job_id}/checkpoints", self.list_fine_tuning_checkpoints, methods=["GET"])

        # Moderations
        app.add_api_route("/v1/moderations", self.moderations, methods=["POST"])

        # Batches
        app.add_api_route("/v1/batches", self.create_batch, methods=["POST"])
        app.add_api_route("/v1/batches", self.list_batches, methods=["GET"])
        app.add_api_route("/v1/batches/{batch_id}", self.retrieve_batch, methods=["GET"])
        app.add_api_route("/v1/batches/{batch_id}/cancel", self.cancel_batch, methods=["POST"])

    # -------------------------------------------------------------------
    # Chat completions
    # -------------------------------------------------------------------

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse | StreamingResponse:
        return _not_implemented()

    # -------------------------------------------------------------------
    # Completions (legacy)
    # -------------------------------------------------------------------

    async def completions(self, request: CompletionRequest) -> CompletionResponse:
        return _not_implemented()

    # -------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------

    async def list_models(self) -> ModelListResponse:
        return _not_implemented()

    async def retrieve_model(self, model: str) -> ModelObject:
        return _not_implemented()

    async def delete_model(self, model: str) -> DeleteModelResponse:
        return _not_implemented()

    # -------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------

    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return _not_implemented()

    # -------------------------------------------------------------------
    # Images
    # -------------------------------------------------------------------

    async def image_generations(self, request: ImageGenerationRequest) -> ImageResponse:
        return _not_implemented()

    async def image_edits(
        self,
        image: UploadFile,
        prompt: str = Form(...),
        model: str | None = Form(None),
        n: int | None = Form(1),
        size: str | None = Form("1024x1024"),
        response_format: str | None = Form("url"),
        mask: UploadFile | None = None,
    ) -> ImageResponse:
        return _not_implemented()

    async def image_variations(
        self,
        image: UploadFile,
        model: str | None = Form(None),
        n: int | None = Form(1),
        size: str | None = Form("1024x1024"),
        response_format: str | None = Form("url"),
    ) -> ImageResponse:
        return _not_implemented()

    # -------------------------------------------------------------------
    # Audio
    # -------------------------------------------------------------------

    async def audio_speech(self, request: AudioSpeechRequest) -> Response:
        return _not_implemented()

    async def audio_transcriptions(
        self,
        file: UploadFile,
        model: str = Form(...),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str | None = Form("json"),
        temperature: float | None = Form(0),
    ) -> AudioTranscriptionResponse:
        return _not_implemented()

    async def audio_translations(
        self,
        file: UploadFile,
        model: str = Form(...),
        prompt: str | None = Form(None),
        response_format: str | None = Form("json"),
        temperature: float | None = Form(0),
    ) -> AudioTranslationResponse:
        return _not_implemented()

    # -------------------------------------------------------------------
    # Files
    # -------------------------------------------------------------------

    async def list_files(self, purpose: str | None = Query(None)) -> FileListResponse:
        return _not_implemented()

    async def upload_file(
        self,
        file: UploadFile,
        purpose: str = Form(...),
    ) -> FileObject:
        return _not_implemented()

    async def retrieve_file(self, file_id: str) -> FileObject:
        return _not_implemented()

    async def delete_file(self, file_id: str) -> DeleteFileResponse:
        return _not_implemented()

    async def retrieve_file_content(self, file_id: str) -> Response:
        return _not_implemented()

    # -------------------------------------------------------------------
    # Fine-tuning
    # -------------------------------------------------------------------

    async def create_fine_tuning_job(self, request: CreateFineTuningJobRequest) -> FineTuningJob:
        return _not_implemented()

    async def list_fine_tuning_jobs(
        self,
        after: str | None = Query(None),
        limit: int = Query(20),
    ) -> FineTuningJobListResponse:
        return _not_implemented()

    async def retrieve_fine_tuning_job(self, job_id: str) -> FineTuningJob:
        return _not_implemented()

    async def cancel_fine_tuning_job(self, job_id: str) -> FineTuningJob:
        return _not_implemented()

    async def list_fine_tuning_events(
        self,
        job_id: str,
        after: str | None = Query(None),
        limit: int = Query(20),
    ) -> FineTuningEventListResponse:
        return _not_implemented()

    async def list_fine_tuning_checkpoints(
        self,
        job_id: str,
        after: str | None = Query(None),
        limit: int = Query(10),
    ) -> FineTuningCheckpointListResponse:
        return _not_implemented()

    # -------------------------------------------------------------------
    # Moderations
    # -------------------------------------------------------------------

    async def moderations(self, request: ModerationRequest) -> ModerationResponse:
        return _not_implemented()

    # -------------------------------------------------------------------
    # Batches
    # -------------------------------------------------------------------

    async def create_batch(self, request: CreateBatchRequest) -> BatchObject:
        return _not_implemented()

    async def list_batches(
        self,
        after: str | None = Query(None),
        limit: int = Query(20),
    ) -> BatchListResponse:
        return _not_implemented()

    async def retrieve_batch(self, batch_id: str) -> BatchObject:
        return _not_implemented()

    async def cancel_batch(self, batch_id: str) -> BatchObject:
        return _not_implemented()
