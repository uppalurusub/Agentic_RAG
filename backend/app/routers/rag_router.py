from fastapi import APIRouter, HTTPException, status
from pathlib import Path
from app.schemas.rag_schema import AskRequest, AskResponse, IngestRequest, IngestResponse
from app.services.rag_service import RAGService
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from app.services.documents_service import DocumentsService
from app.core.config import settings
from app.core.dependencies import documents_service
import logging
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/rag", tags=["Agentic RAG"])

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

def get_service() -> RAGService:
    return RAGService()


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    try:
        return get_service().ask(request.question, request.request_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agentic RAG execution failed. Check server logs using the request_id.",
        ) from exc

@router.post("/ingest", response_model=IngestResponse, summary="Upload and ingest one document")
async def ingest(
    file: UploadFile = File(..., description="Browse and select one PDF or TXT file"),
    rebuild: bool = Form(False, description="Set to true to rebuild the vector store after ingestion"),
    service: DocumentsService = Depends(documents_service),
)-> IngestResponse:
    """Save one uploaded PDF/TXT file, then pass its filename to the ingestion service."""
    try:
        original_name = Path(file.filename or "").name
        suffix = Path(original_name).suffix.lower()

        if not original_name or suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Only PDF and TXT files are supported.",
            )

        content = await file.read(MAX_FILE_SIZE_BYTES + 1)
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="The uploaded file exceeds the 25 MB limit.",
            )
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        documents_dir = settings.resolved_documents_dir
        documents_dir.mkdir(parents=True, exist_ok=True)
        destination = documents_dir / original_name
        destination.write_bytes(content)

        # Pass a list of saved filenames, never the UploadFile object.
        return service.ingest_files([original_name], rebuild=rebuild)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc}",
        ) from exc
    finally:
        await file.close()
