# =============================================================================
# schemas.py — Pydantic models for request/response validation
# =============================================================================

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------
class Citation(BaseModel):
    source_doc: str = Field(..., description="Source document name")
    page_number: int = Field(..., description="Page number in the document")
    score: float = Field(default=0.0, description="Relevance score")


# ---------------------------------------------------------------------------
# Retrieved Document Chunk
# ---------------------------------------------------------------------------
class DocChunk(BaseModel):
    id: Optional[int] = None
    content: str = Field(..., description="Chunk text content")
    source_doc: str = Field(..., description="Source document name")
    page_number: int = Field(default=1, description="Page number")
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = Field(default=None)


# ---------------------------------------------------------------------------
# Intent Classification
# ---------------------------------------------------------------------------
class IntentType:
    LEGAL_QUESTION = "legal_question"
    DOCUMENT_LOOKUP = "document_lookup"
    GENERAL_HELP = "general_help"
    CLAUSE_DRAFT = "clause_draft"
    DOC_SUMMARIZE = "doc_summarize"


# ---------------------------------------------------------------------------
# Query Request
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's legal question")
    language: Literal["english", "telugu", "hindi"] = Field(default="english")
    conversation_id: Optional[str] = Field(default=None, description="For multi-turn chat")
    portal: Literal["vakeel", "nagarik"] = Field(default="nagarik")


# ---------------------------------------------------------------------------
# Query Response (non-streaming)
# ---------------------------------------------------------------------------
class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    intent: str = Field(default="legal_question")
    language: str = Field(default="english")
    processing_time_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Streaming Response Chunk
# ---------------------------------------------------------------------------
class StreamChunk(BaseModel):
    type: Literal["status", "answer", "citation", "done", "error"] = "status"
    content: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Document Upload
# ---------------------------------------------------------------------------
class UploadResponse(BaseModel):
    success: bool
    message: str
    chunks_indexed: int = 0
    source_doc: Optional[str] = None


# ---------------------------------------------------------------------------
# Conversation History
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    citations: Optional[List[Citation]] = None
    timestamp: Optional[datetime] = None


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[ChatMessage]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"
    ollama_connected: bool = False
    postgres_connected: bool = False
    vector_count: int = 0


# ---------------------------------------------------------------------------
# MCP Tool Response
# ---------------------------------------------------------------------------
class DocumentSummaryResponse(BaseModel):
    doc_name: str
    summary: str
    total_pages: int = 0
    total_chunks: int = 0


class CitationResponse(BaseModel):
    doc_name: str
    page: int
    content: Optional[str] = None
    found: bool = False


# ---------------------------------------------------------------------------
# Clause Drafting Request
# ---------------------------------------------------------------------------
class ClauseDraftRequest(BaseModel):
    clause_type: Literal["nda", "employment", "non_compete", "indemnity"]
    party_a: str
    party_b: str
    jurisdiction: str = "Mumbai"
    duration: Optional[str] = None
    purpose: Optional[str] = None
    designation: Optional[str] = None
    salary: Optional[str] = None
    notice_period: Optional[str] = None
    location: Optional[str] = None
    geography: Optional[str] = None
    industry: Optional[str] = None
    contract_type: Optional[str] = None
    liability_cap: Optional[str] = None


class ClauseDraftResponse(BaseModel):
    clause_text: str
    clause_type: str
    warnings: Optional[List[str]] = None
