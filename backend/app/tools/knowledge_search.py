"""Knowledge-base search tool backed by the configured Chroma collection."""
import json
import logging
from typing import Callable

from langchain_chroma import Chroma
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    query: str = Field(description="Focused search query for the indexed knowledge base")


def build_knowledge_search_tool(vector_store: Chroma, top_k: int) -> StructuredTool:
    def search_knowledge(query: str) -> str:
        logger.info("RAG search started | query=%r", query)
        docs = vector_store.similarity_search(query, k=top_k)
        logger.info("RAG search completed | query=%r | result_count=%d", query, len(docs))

        results = []
        for doc in docs:
            metadata = doc.metadata or {}
            results.append({
                "source": str(metadata.get("source", "unknown")),
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id"),
                "content": doc.page_content,
            })

        if not results:
            return json.dumps(
                {"results": [], "message": "No documents were returned by vector search."},
                ensure_ascii=False,
            )
        return json.dumps({"results": results}, ensure_ascii=False)

    return StructuredTool.from_function(
        func=search_knowledge,
        name="search_knowledge",
        description=(
            "Search the indexed knowledge base for relevant passages. Call this when "
            "the question needs evidence from indexed documents. You may reformulate the query."
        ),
        args_schema=SearchInput,
    )
