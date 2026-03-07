import time
import uuid

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Usage()


class DeltaContent(BaseModel):
    role: str | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]


# ---------------------------------------------------------------------------
# Completions (legacy)
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str] = ""
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = 16
    top_p: float | None = None
    n: int | None = 1
    logprobs: int | None = None
    echo: bool | None = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = 0
    frequency_penalty: float | None = 0
    best_of: int | None = 1
    suffix: str | None = None
    user: str | None = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    logprobs: dict | None = None
    finish_reason: str | None = "stop"


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage = Usage()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "organization"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


class DeleteModelResponse(BaseModel):
    id: str
    object: str = "model"
    deleted: bool = True


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str | None = None
    user: str | None = None


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int = 0
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: Usage = Usage()


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

class ImageGenerationRequest(BaseModel):
    model: str | None = None
    prompt: str
    n: int | None = 1
    size: str | None = "1024x1024"
    response_format: str | None = "url"
    user: str | None = None


class ImageObject(BaseModel):
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None


class ImageResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ImageObject]


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

class AudioSpeechRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: str | None = "mp3"
    speed: float | None = 1.0


class AudioTranscriptionResponse(BaseModel):
    text: str


class AudioTranslationResponse(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

class FileObject(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int = Field(default_factory=lambda: int(time.time()))
    filename: str
    purpose: str


class FileListResponse(BaseModel):
    object: str = "list"
    data: list[FileObject]


class DeleteFileResponse(BaseModel):
    id: str
    object: str = "file"
    deleted: bool = True


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

class FineTuningHyperparameters(BaseModel):
    n_epochs: int | str | None = "auto"
    batch_size: int | str | None = "auto"
    learning_rate_multiplier: float | str | None = "auto"


class CreateFineTuningJobRequest(BaseModel):
    model: str
    training_file: str
    validation_file: str | None = None
    hyperparameters: FineTuningHyperparameters | None = None
    suffix: str | None = None


class FineTuningJob(BaseModel):
    id: str
    object: str = "fine_tuning.job"
    model: str
    created_at: int = Field(default_factory=lambda: int(time.time()))
    finished_at: int | None = None
    fine_tuned_model: str | None = None
    organization_id: str | None = None
    status: str = "queued"
    training_file: str
    validation_file: str | None = None
    hyperparameters: FineTuningHyperparameters = FineTuningHyperparameters()
    trained_tokens: int | None = None
    error: dict | None = None


class FineTuningJobListResponse(BaseModel):
    object: str = "list"
    data: list[FineTuningJob]
    has_more: bool = False


class FineTuningEvent(BaseModel):
    id: str
    object: str = "fine_tuning.job.event"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    level: str = "info"
    message: str
    data: dict | None = None
    type: str = "message"


class FineTuningEventListResponse(BaseModel):
    object: str = "list"
    data: list[FineTuningEvent]
    has_more: bool = False


class FineTuningCheckpoint(BaseModel):
    id: str
    object: str = "fine_tuning.job.checkpoint"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    fine_tuned_model_checkpoint: str
    fine_tuning_job_id: str
    metrics: dict = {}
    step_number: int


class FineTuningCheckpointListResponse(BaseModel):
    object: str = "list"
    data: list[FineTuningCheckpoint]
    has_more: bool = False


# ---------------------------------------------------------------------------
# Moderations
# ---------------------------------------------------------------------------

class ModerationRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


class ModerationCategories(BaseModel):
    sexual: bool = False
    hate: bool = False
    harassment: bool = False
    self_harm: bool = Field(False, alias="self-harm")
    sexual_minors: bool = Field(False, alias="sexual/minors")
    hate_threatening: bool = Field(False, alias="hate/threatening")
    violence_graphic: bool = Field(False, alias="violence/graphic")
    self_harm_intent: bool = Field(False, alias="self-harm/intent")
    self_harm_instructions: bool = Field(False, alias="self-harm/instructions")
    harassment_threatening: bool = Field(False, alias="harassment/threatening")
    violence: bool = False

    model_config = {"populate_by_name": True}


class ModerationCategoryScores(BaseModel):
    sexual: float = 0.0
    hate: float = 0.0
    harassment: float = 0.0
    self_harm: float = Field(0.0, alias="self-harm")
    sexual_minors: float = Field(0.0, alias="sexual/minors")
    hate_threatening: float = Field(0.0, alias="hate/threatening")
    violence_graphic: float = Field(0.0, alias="violence/graphic")
    self_harm_intent: float = Field(0.0, alias="self-harm/intent")
    self_harm_instructions: float = Field(0.0, alias="self-harm/instructions")
    harassment_threatening: float = Field(0.0, alias="harassment/threatening")
    violence: float = 0.0

    model_config = {"populate_by_name": True}


class ModerationResult(BaseModel):
    flagged: bool = False
    categories: ModerationCategories = ModerationCategories()
    category_scores: ModerationCategoryScores = ModerationCategoryScores()


class ModerationResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"modr-{uuid.uuid4().hex[:12]}")
    model: str
    results: list[ModerationResult]


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

class CreateBatchRequest(BaseModel):
    input_file_id: str
    endpoint: str
    completion_window: str
    metadata: dict[str, str] | None = None


class BatchObject(BaseModel):
    id: str
    object: str = "batch"
    endpoint: str
    input_file_id: str
    completion_window: str
    status: str = "validating"
    output_file_id: str | None = None
    error_file_id: str | None = None
    created_at: int = Field(default_factory=lambda: int(time.time()))
    in_progress_at: int | None = None
    expires_at: int | None = None
    finalizing_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    expired_at: int | None = None
    cancelling_at: int | None = None
    cancelled_at: int | None = None
    request_counts: dict | None = None
    metadata: dict[str, str] | None = None


class BatchListResponse(BaseModel):
    object: str = "list"
    data: list[BatchObject]
    has_more: bool = False
