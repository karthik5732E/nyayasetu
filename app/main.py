# =============================================================================
# main.py — FastAPI Entry Point
# Endpoints:
#   POST /api/query         — Legal Q&A (async, streaming)
#   POST /api/upload        — Upload & ingest documents
#   GET  /api/history/{id}   — Conversation history
#   GET  /api/health        — Health check
#   GET  /docs              — Swagger UI (auto-generated)
# =============================================================================

import os
import json
import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.models.schemas import (
    QueryRequest, QueryResponse, HealthResponse,
    UploadResponse, ChatMessage, Citation,
    ClauseDraftRequest, ClauseDraftResponse
)
from app.agent.graph import run_agent, run_agent_stream
from app.agent.state import AgentState
from app.retrieval.pgvector import (
    init_database, check_connection as check_pg_connection,
    get_vector_count, get_conversation, get_document_chunks,
    hybrid_search, get_document_summary_info, get_source_docs
)
from app.retrieval.ingest import ingest_file
from app.llm import get_ollama_client, summarize_text, draft_clause

# Logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup/shutdown events
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""
    logger.info("=" * 60)
    logger.info("  NYAYA SETU v2.0 — Starting up")
    logger.info("=" * 60)

    # Initialize database
    try:
        await init_database()
        pg_ok = await check_pg_connection()
        logger.info(f"PostgreSQL: {'connected' if pg_ok else 'FAILED'}")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    # Check Ollama
    try:
        ollama = get_ollama_client()
        ollama_ok = await ollama.health_check()
        logger.info(f"Ollama: {'connected' if ollama_ok else 'FAILED'}")
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        ollama_ok = False

    app.state.pg_connected = pg_ok
    app.state.ollama_connected = ollama_ok

    yield

    # Shutdown
    logger.info("Shutting down Nyaya Setu...")
    try:
        ollama = get_ollama_client()
        await ollama.close()
    except:
        pass


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Nyaya Setu — AI Legal Assistant",
    description="""
    Production-grade AI legal assistant for Indian law.
    Features: Hybrid search (pgvector + FTS + RRF), LangGraph agent,
    multi-turn conversations, MCP server, LangSmith tracing.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: Request timing
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_timing_header(request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    response.headers["X-Response-Time-Ms"] = str(round(elapsed, 2))
    return response


# ---------------------------------------------------------------------------
# POST /api/query — Main Legal Q&A (async streaming)
# ---------------------------------------------------------------------------

@app.post("/api/query", response_model=QueryResponse)
async def api_query(request: QueryRequest):
    """
    Main legal Q&A endpoint.
    Processes the query through the LangGraph agent and returns an answer with citations.
    """
    start = time.time()

    try:
        final_state = await run_agent(
            query=request.query,
            portal=request.portal,
            language=request.language,
            conversation_id=request.conversation_id
        )

        elapsed = (time.time() - start) * 1000

        return QueryResponse(
            answer=final_state.get("answer", ""),
            citations=final_state.get("citations", []),
            confidence_score=final_state.get("confidence_score", 0.0),
            intent=final_state.get("intent", "legal_question"),
            language=final_state.get("language", "english"),
            processing_time_ms=round(elapsed, 2)
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/api/query/stream")
async def api_query_stream(request: QueryRequest):
    """
    Streaming legal Q&A endpoint.
    Returns a stream of status updates, citations, and answer chunks.
    """
    async def event_generator():
        async for chunk in run_agent_stream(
            query=request.query,
            portal=request.portal,
            language=request.language,
            conversation_id=request.conversation_id
        ):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# ---------------------------------------------------------------------------
# POST /api/upload — Upload & ingest documents
# ---------------------------------------------------------------------------

@app.post("/api/upload", response_model=UploadResponse)
async def api_upload(file: UploadFile = File(...)):
    """
    Upload a legal document (PDF, DOCX, TXT) for ingestion into pgvector.
    """
    allowed_types = ["application/pdf", "text/plain"]
    allowed_extensions = [".pdf", ".docx", ".txt"]

    # Check file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed_extensions}"
        )

    # Save uploaded file temporarily
    temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Ingest into pgvector
        result = await ingest_file(temp_path, collection="case_files")

        return UploadResponse(
            success=result["chunks_indexed"] > 0,
            message=f"Indexed {result['chunks_indexed']} chunks from {result['pages']} pages",
            chunks_indexed=result["chunks_indexed"],
            source_doc=result["source_doc"]
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


# ---------------------------------------------------------------------------
# POST /api/upload/folder — Batch ingest a folder (for initial setup)
# ---------------------------------------------------------------------------

class FolderIngestRequest(BaseModel):
    folder_path: str
    collection: str = "indian_laws"


@app.post("/api/upload/folder")
async def api_upload_folder(request: FolderIngestRequest):
    """
    Batch ingest all documents from a folder.
    Use this for initial Indian law ingestion.
    """
    from app.retrieval.ingest import ingest_folder

    if not os.path.exists(request.folder_path):
        raise HTTPException(status_code=400, detail=f"Folder not found: {request.folder_path}")

    result = await ingest_folder(request.folder_path, collection=request.collection)
    return result


# ---------------------------------------------------------------------------
# GET /api/history/{conversation_id} — Conversation history
# ---------------------------------------------------------------------------

@app.get("/api/history/{conversation_id}")
async def api_history(conversation_id: str):
    """
    Retrieve a conversation's message history by ID.
    """
    conv = await get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = conv.get("messages", [])
    return {
        "conversation_id": conversation_id,
        "portal": conv.get("portal", "nagarik"),
        "messages": messages,
        "created_at": conv.get("created_at"),
        "updated_at": conv.get("updated_at")
    }


# ---------------------------------------------------------------------------
# GET /api/health — Health check
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def api_health():
    """
    Health check endpoint. Returns status of all services.
    """
    pg_ok = await check_pg_connection()
    vector_count = await get_vector_count() if pg_ok else 0

    ollama_ok = False
    try:
        ollama = get_ollama_client()
        ollama_ok = await ollama.health_check()
    except:
        pass

    return HealthResponse(
        status="healthy" if pg_ok and ollama_ok else "degraded",
        ollama_connected=ollama_ok,
        postgres_connected=pg_ok,
        vector_count=vector_count
    )


# ---------------------------------------------------------------------------
# GET /api/sources — List all source documents
# ---------------------------------------------------------------------------

@app.get("/api/sources")
async def api_sources():
    """List all source documents currently in the vector database."""
    sources = await get_source_docs()
    return {"sources": sources, "count": len(sources)}


# ---------------------------------------------------------------------------
# GET /api/document/{doc_name} — Get chunks for a document
# ---------------------------------------------------------------------------

@app.get("/api/document/{doc_name}")
async def api_document(doc_name: str, page: int = None):
    """
    Get chunks for a specific document.
    Optionally filter by page number.
    """
    if page:
        chunks = await get_document_chunks(doc_name, limit=1000)
        chunks = [c for c in chunks if c.page_number == page]
    else:
        chunks = await get_document_chunks(doc_name, limit=1000)

    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "doc_name": doc_name,
        "total_chunks": len(chunks),
        "chunks": [{"content": c.content, "page": c.page_number} for c in chunks[:50]]
    }


# ---------------------------------------------------------------------------
# POST /api/summarize — Document summarization
# ---------------------------------------------------------------------------

class SummarizeRequest(BaseModel):
    text: str
    portal: str = "vakeel"


@app.post("/api/summarize")
async def api_summarize(request: SummarizeRequest):
    """Summarize a legal document."""
    try:
        summary = await summarize_text(request.text, portal=request.portal)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


# ---------------------------------------------------------------------------
# POST /api/draft-clause — Clause drafting
# ---------------------------------------------------------------------------

@app.post("/api/draft-clause", response_model=ClauseDraftResponse)
async def api_draft_clause(request: ClauseDraftRequest):
    """Draft a legal clause."""
    try:
        details = request.model_dump()
        clause_type = details.pop("clause_type")
        text = await draft_clause(clause_type, details)

        warnings = []
        if clause_type == "non_compete":
            warnings.append(
                "Section 27 of Indian Contract Act 1872 makes post-employment "
                "non-compete clauses largely unenforceable in India."
            )

        return ClauseDraftResponse(
            clause_text=text,
            clause_type=clause_type,
            warnings=warnings or None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clause drafting failed: {str(e)}")


# ---------------------------------------------------------------------------
# GET /api/search — Direct hybrid search
# ---------------------------------------------------------------------------

@app.get("/api/search")
async def api_search(q: str, k: int = 5, source: str = None):
    """
    Direct hybrid search endpoint.
    Returns raw search results with page citations.
    """
    results = await hybrid_search(q, k=k, source_filter=source)
    return {
        "query": q,
        "results_count": len(results),
        "results": [
            {
                "content": r.content[:300] + "..." if len(r.content) > 300 else r.content,
                "source_doc": r.source_doc,
                "page_number": r.page_number,
                "score": r.score
            }
            for r in results
        ]
    }


# ---------------------------------------------------------------------------
# Root redirect to docs
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "name": "Nyaya Setu v2.0",
        "description": "AI Legal Assistant for Indian Law",
        "docs": "/docs",
        "health": "/api/health"
    }


# ---------------------------------------------------------------------------
# Run (for development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
