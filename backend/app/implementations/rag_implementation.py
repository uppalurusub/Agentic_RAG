"""Implementation facade: configures dependencies, builds and invokes the graph."""
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

""" from app.core.llm_config import (
    CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, EMBEDDING_MODEL,
    MAX_AGENT_STEPS, TOP_K, VECTOR_DB_PATH, get_llm,
) """
from app.graph.rag_graph import build_rag_graph
from app.schemas.rag_schema import SourceItem
from app.tools.knowledge_search import build_knowledge_search_tool
from app.config.settings import settings
logger = settings.get_logger()

SYSTEM_PROMPT = """You are an evidence-grounded assistant using an Agentic RAG workflow.
Use knowledge-base search for questions that require information from indexed documents.
Do not invent facts or sources. The final answer must be grounded in retrieved passages.
Cite claims using [source: filename, page: N] when page metadata is available,
otherwise [source: filename]. If evidence is insufficient, say so."""


class AgenticRAGImplementation:
    """Stable facade used by the service layer."""

    def __init__(self) -> None:
        self.llm = settings.get_model_client()
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        Path(settings.vector_db_path).mkdir(parents=True, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=settings.chroma_collection,
            persist_directory=settings.vector_db_path,
            embedding_function=self.embeddings,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
        )
        self.search_tool = build_knowledge_search_tool(self.vector_store,settings.top_k)
        self.graph = build_rag_graph(
            llm=self.llm,
            tools=[self.search_tool],
            max_agent_steps=settings.max_agent_steps,
        )

    def ask(self, question: str) -> tuple[str, list[SourceItem], int]:

        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        logger.info("Agentic RAG started | question={}", question)

        result = self.graph.invoke(
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=question.strip()),
                ]
            },
            config={
                "recursion_limit": settings.max_agent_steps * 2 + 4
            },
        )

        messages = result.get("messages", [])

        logger.info(
            "Agentic RAG graph completed | message_count={}",
            len(messages),
        )

        # Find the latest non-empty assistant response.
        final_message = next(
            (
                message
                for message in reversed(messages)
                if getattr(message, "type", None) == "ai"
                and not getattr(message, "tool_calls", None)
                and isinstance(getattr(message, "content", None), str)
                and message.content.strip()
            ),
            None,
        )

        if final_message is None:
            for index, message in enumerate(messages):
                logger.warning(
                    "Graph message | index={} | type={} | tool_calls={} | content_type={}",
                    index,
                    getattr(message, "type", None),
                    len(getattr(message, "tool_calls", []) or []),
                    type(getattr(message, "content", None)).__name__,
                )

            raise RuntimeError(
                "RAG graph completed without a non-empty final assistant response."
            )

        answer = final_message.content.strip()

        tool_calls = sum(
            len(getattr(message, "tool_calls", []) or [])
            for message in messages
            if getattr(message, "type", None) == "ai"
        )

        sources = self._extract_sources(messages)

        logger.info(
            "Agentic RAG completed | answer_length={} | sources={} | tool_calls={}",
            len(answer),
            len(sources),
            tool_calls,
        )

        return answer, sources, tool_calls

    @staticmethod
    def _extract_sources(messages: list[Any]) -> list[SourceItem]:
        sources: dict[tuple[str, str | None, str | None], SourceItem] = {}
        for message in messages:
            if getattr(message, "type", None) != "tool":
                continue
            try:
                payload = json.loads(message.content)
            except (TypeError, json.JSONDecodeError):
                continue
            for item in payload.get("results", []):
                source = str(item.get("source", "unknown"))
                chunk_id = item.get("chunk_id")
                page = item.get("page")
                key = (source, str(page) if page is not None else None, chunk_id)
                sources[key] = SourceItem(
                    source=source,
                    chunk_id=chunk_id,
                    page=page,
                    excerpt=str(item.get("content", ""))[:500],
                )
        return list(sources.values())

    def ingest(self, text: str, source: str, metadata: dict[str, Any] | None = None) -> int:
        """Split and index text into the same configured collection used for search."""
        if not text or not text.strip():
            raise ValueError("Document text must not be empty.")
        base_metadata = dict(metadata or {})
        safe_metadata = {
            str(k): v for k, v in base_metadata.items()
            if isinstance(v, (str, int, float, bool))
        }
        docs = self.splitter.create_documents(
            [text], metadatas=[{**safe_metadata, "source": source}]
        )
        if not docs:
            return 0

        ids = []
        for index, doc in enumerate(docs):
            doc.metadata["chunk_id"] = str(uuid.uuid4())
            doc.metadata["chunk_index"] = index
            ids.append(doc.metadata["chunk_id"])

        self.vector_store.add_documents(documents=docs, ids=ids)
        logger.info(
            "Indexed source=%s chunks=%d collection=%s",
            source, len(docs), settings.chroma_collection,
        )
        return len(docs)

    def diagnose_vector_store(self) -> dict:
        sample = self.vector_store.get(limit=5, include=["metadatas", "documents"])
        return {
            "collection_name": settings.chroma_collection,
            "persist_directory": str(Path(settings.vector_db_path).resolve()),
            "embedding_model": settings.embedding_model,
            "document_count": self.vector_store._collection.count(),
            "sample_documents": [
                {"metadata": metadata, "preview": document[:200]}
                for metadata, document in zip(
                    sample.get("metadatas") or [],
                    sample.get("documents") or [],
                )
            ],
        }
