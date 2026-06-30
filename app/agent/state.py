# =============================================================================
# state.py — LangGraph Agent State Schema
# =============================================================================

from typing import List, Optional, Dict, Any
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from app.models.schemas import DocChunk, Citation


class AgentState(MessagesState):
    """
    Shared state for the LangGraph legal AI agent.
    Extends MessagesState to include conversation history.

    Fields:
        query: User's original question
        language: Detected language (english, telugu, hindi)
        intent: Classified intent type
        retrieved_docs: Documents retrieved from pgvector
        citations: Formatted citations with page numbers
        answer: Generated answer from LLM
        confidence_score: How grounded the answer is in retrieved docs
        portal: Which portal (vakeel/nagarik)
        portal: Which portal (vakeel/nagarik)
        conversation_id: Optional conversation ID for memory
        streaming: Whether to stream the answer
    """
    query: str = ""
    language: str = "english"
    intent: str = "legal_question"
    retrieved_docs: List[DocChunk] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    answer: str = ""
    confidence_score: float = 0.0
    portal: str = "nagarik"
    conversation_id: Optional[str] = None
    streaming: bool = False


def state_to_dict(state: AgentState) -> Dict[str, Any]:
    """Convert state to a serializable dict for LangSmith tracing."""
    return {
        "query": state.get("query", ""),
        "language": state.get("language", "english"),
        "intent": state.get("intent", "legal_question"),
        "retrieved_count": len(state.get("retrieved_docs", [])),
        "doc_sources": list(set(d.source_doc for d in state.get("retrieved_docs", []))),
        "citation_pages": [c.page_number for c in state.get("citations", [])],
        "answer_preview": state.get("answer", "")[:200],
        "confidence_score": state.get("confidence_score", 0.0),
        "portal": state.get("portal", "nagarik"),
    }
