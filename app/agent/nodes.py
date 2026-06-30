# =============================================================================
# nodes.py — All LangGraph Node Functions
# Each node reads from state, does work, and returns updates to state.
# =============================================================================

import re
import logging
from typing import Dict, Any, List

from langchain_core.messages import AIMessage, HumanMessage
from langsmith import traceable

from app.agent.state import AgentState
from app.models.schemas import DocChunk, Citation, IntentType
from app.retrieval.pgvector import hybrid_search, get_document_chunks
from app.llm import query_ollama

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node 1: Intent Classifier
# ---------------------------------------------------------------------------

INTENT_PROMPT = """You are an intent classifier for an Indian legal AI assistant.
Classify the user's query into EXACTLY one of these categories:
- legal_question: User is asking about Indian law, their rights, legal procedures
- document_lookup: User is asking about a specific document or wants to find something in uploaded docs
- general_help: User wants help numbers, general info, or non-legal queries
- clause_draft: User wants to draft a legal clause or contract
- doc_summarize: User wants to summarize a document

Respond with ONLY the category name, nothing else.

Query: {query}
Category:"""


@traceable(name="intent_classifier", run_type="llm")
async def intent_classifier(state: AgentState) -> Dict[str, Any]:
    """Classify the user's intent from their query."""
    query = state["query"]
    portal = state.get("portal", "nagarik")

    # Quick heuristic classification for common patterns
    q_lower = query.lower()

    # Draft-related keywords
    draft_keywords = ["draft", "write", "create", "prepare", "generate"]
    if any(k in q_lower for k in draft_keywords):
        # Check if it's about a specific clause type
        if any(x in q_lower for x in ["nda", "non-disclosure", "confidentiality"]):
            return {"intent": "clause_draft", "language": _detect_language(query)}
        if any(x in q_lower for x in ["employment", "job", "hire", "appointment"]):
            return {"intent": "clause_draft", "language": _detect_language(query)}
        if any(x in q_lower for x in ["non-compete", "noncompete", "competition"]):
            return {"intent": "clause_draft", "language": _detect_language(query)}
        if any(x in q_lower for x in ["indemnity", "liability", "compensation"]):
            return {"intent": "clause_draft", "language": _detect_language(query)}

    # Summarize-related keywords
    summarize_keywords = ["summarize", "summary", "brief", "overview", "tldr"]
    if any(k in q_lower for k in summarize_keywords):
        return {"intent": "doc_summarize", "language": _detect_language(query)}

    # Use LLM for more nuanced classification
    prompt = INTENT_PROMPT.format(query=query)
    try:
        result = await query_ollama(prompt, max_tokens=20, temperature=0.0)
        detected_intent = result.strip().lower().split("\n")[0]

        # Validate against allowed intents
        valid_intents = [
            IntentType.LEGAL_QUESTION,
            IntentType.DOCUMENT_LOOKUP,
            IntentType.GENERAL_HELP,
            IntentType.CLAUSE_DRAFT,
            IntentType.DOC_SUMMARIZE,
        ]
        intent = detected_intent if detected_intent in valid_intents else IntentType.LEGAL_QUESTION
    except Exception as e:
        logger.error(f"LLM generation failed: {type(e).__name__}: {e}")
        answer = "I apologize, but I'm unable to generate an answer at this moment. Please try again."

    language = _detect_language(query)

    logger.info(f"Intent: {intent}, Language: {language}, Portal: {portal}")
    return {"intent": intent, "language": language}


def _detect_language(text: str) -> str:
    """Detect if the query is in Telugu, Hindi, or English."""
    # Telugu Unicode range: U+0C00-U+0C7F
    telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
    # Hindi Unicode range: U+0900-U+097F
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))

    if telugu_chars > hindi_chars and telugu_chars > 3:
        return "telugu"
    elif hindi_chars > telugu_chars and hindi_chars > 3:
        return "hindi"
    return "english"


# ---------------------------------------------------------------------------
# Node 2: Document Retriever (Hybrid Search)
# ---------------------------------------------------------------------------

@traceable(name="document_retriever", run_type="retriever")
async def document_retriever(state: AgentState) -> Dict[str, Any]:
    """Retrieve relevant documents using hybrid search (semantic + keyword + RRF)."""
    query = state["query"]
    intent = state.get("intent", "legal_question")
    portal = state.get("portal", "nagarik")

    # Determine collection/source filter based on intent and portal
    source_filter = None
    if intent in [IntentType.DOCUMENT_LOOKUP, IntentType.DOC_SUMMARIZE] and portal == "vakeel":
        source_filter = None  # Search across all uploaded case files

    # Perform hybrid search
    try:
        results = await hybrid_search(query, k=5, source_filter=source_filter)
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        results = []

    # Build citations from results
    citations = []
    for r in results:
        citations.append(Citation(
            source_doc=r.source_doc,
            page_number=r.page_number,
            score=r.score or 0.0
        ))

    logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
    return {
        "retrieved_docs": results,
        "citations": citations
    }


# ---------------------------------------------------------------------------
# Node 3: Answer Generator (LLM with Retrieved Context)
# ---------------------------------------------------------------------------

VAKEEL_PROMPT = """You are a legal AI assistant for Indian lawyers. Answer based ONLY on the provided context documents.
Be concise, accurate, and cite specific clauses where possible.
If the answer is not in the context, say "Not found in uploaded documents."

CONTEXT DOCUMENTS:
{context}

QUESTION: {question}

ANSWER (cite sources with page numbers):"""

NAGARIK_PROMPT = """You are Nyaya Setu, an Indian legal awareness assistant. Help citizens understand their rights.
Be concise and clear. State:
1. The exact law/section that applies
2. What the citizen should do next
3. Which office or court to visit
4. Any deadlines or time limits

Answer in {language} language. Keep it simple.

RELEVANT LAWS:
{context}

PROBLEM: {question}

GUIDANCE:"""

CLAUSE_PROMPT = """You are an expert Indian legal drafter. Draft a professional, enforceable legal clause.
Use Indian law references where applicable. Be precise and thorough.

REQUEST: Draft a {clause_type} clause

DETAILS:
{details}

DRAFTED CLAUSE:"""

SUMMARIZE_PROMPT = """Summarize the following legal document concisely:
- Document type and parties
- Key obligations and terms
- Important dates and deadlines
- Any risky or unusual clauses

DOCUMENT:
{text}

SUMMARY:"""


@traceable(name="answer_generator", run_type="llm")
async def answer_generator(state: AgentState) -> Dict[str, Any]:
    """Generate an answer using the LLM with retrieved context."""
    query = state["query"]
    language = state.get("language", "english")
    intent = state.get("intent", IntentType.LEGAL_QUESTION)
    portal = state.get("portal", "nagarik")
    docs = state.get("retrieved_docs", [])

    # Build context from retrieved documents
    if docs:
        context_parts = []
        for i, doc in enumerate(docs[:3], 1):
            truncated = doc.content[:600]
            context_parts.append(
                f"[{i}] Source: {doc.source_doc}, Page: {doc.page_number}\n{truncated}"
            )
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "No relevant documents found."

    # Select prompt based on intent and portal
    if intent == IntentType.CLAUSE_DRAFT:
        prompt = CLAUSE_PROMPT.format(clause_type=query, details=context)
    elif intent == IntentType.DOC_SUMMARIZE:
        prompt = SUMMARIZE_PROMPT.format(text=context)
    elif portal == "vakeel":
        prompt = VAKEEL_PROMPT.format(context=context, question=query)
    else:
        lang_display = "English" if language == "english" else language.title()
        prompt = NAGARIK_PROMPT.format(context=context, question=query, language=lang_display)

    # Call Ollama
    try:
        answer = await query_ollama(prompt, max_tokens=400, temperature=0.1)
        answer = answer.strip()
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        answer = "I apologize, but I'm unable to generate an answer at this moment. Please try again."

    return {"answer": answer}


# ---------------------------------------------------------------------------
# Node 4: Citation Formatter
# ---------------------------------------------------------------------------

@traceable(name="citation_formatter", run_type="chain")
async def citation_formatter(state: AgentState) -> Dict[str, Any]:
    """Format citations with page numbers and append to answer."""
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    intent = state.get("intent", "")

    # For clause drafting and summarization, citations may not be needed
    if intent in [IntentType.CLAUSE_DRAFT, IntentType.DOC_SUMMARIZE]:
        return {"answer": answer}

    if not citations:
        return {"answer": answer}
    
    # Build citation string
    unique_cites = []
    seen = set()
    for c in citations:
        key = f"{c.source_doc}:{c.page_number}"
        if key not in seen:
            seen.add(key)
            unique_cites.append(f"📖 {c.source_doc} (Page {c.page_number})")

    if unique_cites:
        cite_text = "\n\n---\n**Sources:** " + "  ·  ".join(unique_cites)
        answer = answer + cite_text

    return {"answer": answer}


# ---------------------------------------------------------------------------
# Node 5: Response Validator (Grounding Check)
# ---------------------------------------------------------------------------

@traceable(name="response_validator", run_type="chain")
async def response_validator(state: AgentState) -> Dict[str, Any]:
    """
    Validate that the answer is grounded in the retrieved documents.
    Check for hallucinations and compute confidence score.
    """
    answer = state.get("answer", "")
    docs = state.get("retrieved_docs", [])
    citations = state.get("citations", [])

    if not docs:
        # No docs retrieved — confidence depends on whether answer admits this
        if "not found" in answer.lower() or "unable" in answer.lower():
            confidence = 0.5  # Honest about not knowing
        else:
            confidence = 0.1  # Likely hallucinating
        return {"confidence_score": round(confidence, 2)}

    # Check grounding: extract key phrases from docs and see if answer references them
    doc_text = " ".join(d.content.lower() for d in docs)
    doc_words = set(w for w in re.findall(r'\b\w+\b', doc_text) if len(w) > 4)

    answer_words = set(w for w in re.findall(r'\b\w+\b', answer.lower()) if len(w) > 4)

    # Calculate overlap
    if doc_words:
        overlap = len(doc_words & answer_words) / len(doc_words)
        # Also check for section numbers and citations
        section_matches = len(re.findall(r'\bsection\s+\d+', answer.lower()))
        has_citations = len(citations) > 0

        confidence = min(1.0, overlap * 2 + 0.1 * section_matches + (0.1 if has_citations else 0))
        confidence = max(0.0, min(1.0, confidence))
    else:
        confidence = 0.0

    # If confidence is too low, add a warning
    if confidence < 0.3 and len(docs) > 0:
        answer += "\n\n⚠️ **Note:** This answer has low confidence. Please verify with a registered advocate."
        return {"answer": answer, "confidence_score": round(confidence, 2)}

    return {"confidence_score": round(confidence, 2)}


# ---------------------------------------------------------------------------
# Node 6: Save to Memory (Conversation persistence)
# ---------------------------------------------------------------------------

async def save_to_memory(state: AgentState) -> Dict[str, Any]:
    """Save the conversation to the database for multi-turn chat."""
    conv_id = state.get("conversation_id")
    if not conv_id:
        return {"conversation_id": conv_id}

    from app.retrieval.pgvector import save_conversation

    # Build message list from state
    messages = []
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    # Add the current turn
    if state.get("query"):
        messages.append({"role": "user", "content": state["query"]})
    if state.get("answer"):
        messages.append({"role": "assistant", "content": state["answer"]})

    try:
        await save_conversation(conv_id, messages, portal=state.get("portal", "nagarik"))
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")

    return {"conversation_id": conv_id}
