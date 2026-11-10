import logging
import time
import uuid
from functools import lru_cache

from app.implementations.rag_implementation import AgenticRAGImplementation
from app.schemas.rag_schema import AskResponse, IngestResponse
from app.config.settings import settings
logger = settings.get_logger()


@lru_cache(maxsize=1)
def get_rag_implementation() -> AgenticRAGImplementation:
    """Create one implementation instance per application process."""
    return AgenticRAGImplementation()


class RAGService:
    def __init__(self, implementation: AgenticRAGImplementation | None = None) -> None:
        self.implementation = implementation or get_rag_implementation()

    def ask(self, question: str, request_id: str | None = None) -> AskResponse:
        resolved_request_id = request_id or str(uuid.uuid4())
        started = time.perf_counter()
        try:
            answer, sources, tool_calls = self.implementation.ask(question)
            elapsed = round(time.perf_counter() - started, 4)
            logger.info(
                "RAG completed request_id=%s tool_calls=%d sources=%d duration=%.4fs",
                resolved_request_id, tool_calls, len(sources), elapsed,
            )
            return AskResponse(
                request_id=resolved_request_id,
                answer=answer,
                sources=sources,
                tool_calls=tool_calls,
                execution_time_seconds=elapsed,
            )
        except Exception:
            logger.exception("RAG failed request_id=%s", resolved_request_id)
            raise

    def ingest(self, text: str, source: str, metadata: dict) -> IngestResponse:
        count = self.implementation.ingest(text=text, source=source, metadata=metadata)
        return IngestResponse(
            source=source,
            chunks_created=count,
            message="Document indexed successfully" if count else "No chunks were created",
        )

    def ingest_files(self, filenames: list[str], rebuild: bool = False):
        """Ingest uploaded filenames saved under DOCUMENTS_DIR."""
        if not filenames:
            raise ValueError("No uploaded filename was provided.")
        return self.impl.ingest(filenames, rebuild)

