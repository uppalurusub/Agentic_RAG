from typing import Any
from pydantic import BaseModel, Field, ConfigDict


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=3, max_length=8000)
    request_id: str | None = Field(default=None, max_length=128)


class SourceItem(BaseModel):
    source: str
    chunk_id: str | None = None
    page: int | str | None = None
    excerpt: str


class AskResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    tool_calls: int = 0
    execution_time_seconds: float


class IngestRequest(BaseModel):
    file_paths: list[str] = Field(
        default_factory=list,
        description="Paths relative to DOCUMENTS_DIR",
    )
    rebuild: bool = False


class IngestResponse(BaseModel):
    status: str
    files_processed: int
    pages_loaded: int
    chunks_created: int
    collection: str
    details: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationRequest(BaseModel):
    dataset_path: str = "./data/eval/rag_eval.jsonl"
    top_k: int = Field(default=4, ge=1, le=20)
    max_examples: int | None = Field(default=None, ge=1)


class EvaluationResponse(BaseModel):
    status: str
    examples_evaluated: int
    metrics: dict[str, float | int | None]
    run_id: str | None = None
    details: list[dict[str, Any]] = Field(default_factory=list)