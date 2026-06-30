# =============================================================================
# graph.py — LangGraph StateGraph wiring
# Builds the state-machine: intent → retrieve → generate → format → validate
# With memory checkpointing for multi-turn conversations.
# =============================================================================

import logging
from typing import AsyncGenerator, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from app.agent.state import AgentState
from app.agent.nodes import (
    intent_classifier,
    document_retriever,
    answer_generator,
    citation_formatter,
    response_validator,
    save_to_memory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional Routing
# ---------------------------------------------------------------------------

def should_retrieve(state: AgentState) -> str:
    """Route: if intent is general_help, skip retrieval and go straight to answer."""
    intent = state.get("intent", "legal_question")
    if intent == "general_help":
        return "generate"
    return "retrieve"


def should_format_citations(state: AgentState) -> str:
    """Always format citations after generating answer."""
    return "format"


# ---------------------------------------------------------------------------
# Build the Graph
# ---------------------------------------------------------------------------

def build_agent() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Flow:
        START → intent_classifier → [conditional]
            ├─ general_help ─→ answer_generator → citation_formatter → response_validator → save_to_memory → END
            └─ other intents ─→ document_retriever → answer_generator → citation_formatter → response_validator → save_to_memory → END
    """
    # Initialize the graph with our state schema
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("intent_classifier", intent_classifier)
    builder.add_node("document_retriever", document_retriever)
    builder.add_node("answer_generator", answer_generator)
    builder.add_node("citation_formatter", citation_formatter)
    builder.add_node("response_validator", response_validator)
    builder.add_node("save_to_memory", save_to_memory)

    # Define edges
    builder.add_edge(START, "intent_classifier")

    # Conditional routing after intent classification
    builder.add_conditional_edges(
        "intent_classifier",
        should_retrieve,
        {
            "generate": "answer_generator",      # Skip retrieval for general help
            "retrieve": "document_retriever",    # Normal flow with retrieval
        }
    )

    # Main flow
    builder.add_edge("document_retriever", "answer_generator")
    builder.add_edge("answer_generator", "citation_formatter")
    builder.add_edge("citation_formatter", "response_validator")
    builder.add_edge("response_validator", "save_to_memory")
    builder.add_edge("save_to_memory", END)

    # For general_help path (skips retrieval)
    # answer_generator already handles general_help via the portal/intent logic

    # Compile with memory saver for checkpointing
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    logger.info("LangGraph agent compiled with memory checkpointing")
    return graph


# ---------------------------------------------------------------------------
# Run Agent (Entry Point)
# ---------------------------------------------------------------------------

async def run_agent(
    query: str,
    portal: str = "nagarik",
    language: str = "english",
    conversation_id: str = None,
    streaming: bool = False
) -> AgentState:
    """
    Run the agent synchronously and return the final state.

    Args:
        query: User's question
        portal: "vakeel" or "nagarik"
        language: "english", "telugu", or "hindi"
        conversation_id: For multi-turn memory
        streaming: Whether to stream output

    Returns:
        Final AgentState with answer, citations, confidence_score
    """
    graph = build_agent()

    # Build initial state
    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        query=query,
        portal=portal,
        language=language,
        conversation_id=conversation_id,
        streaming=streaming,
    )

    # Thread ID for memory checkpointing
    thread_id = conversation_id or "default"
    config = {"configurable": {"thread_id": thread_id}}

    # Run the graph
    try:
        final_state = await graph.ainvoke(initial_state, config=config)
        return final_state
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        # Return a graceful error state
        return AgentState(
            messages=[HumanMessage(content=query)],
            query=query,
            answer=f"I apologize, but I encountered an error processing your request: {str(e)}. Please try again.",
            portal=portal,
            language=language,
            confidence_score=0.0,
        )


async def run_agent_stream(
    query: str,
    portal: str = "nagarik",
    language: str = "english",
    conversation_id: str = None
) -> AsyncGenerator[str, None]:
    """
    Run the agent with streaming output.
    Yields chunks of the answer as they are generated.

    Args:
        query: User's question
        portal: "vakeel" or "nagarik"
        language: Detected language
        conversation_id: For multi-turn memory

    Yields:
        String chunks of the answer
    """
    graph = build_agent()

    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        query=query,
        portal=portal,
        language=language,
        conversation_id=conversation_id,
        streaming=True,
    )

    thread_id = conversation_id or "default"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        async for event in graph.astream(initial_state, config=config):
            # Yield status updates
            if "intent_classifier" in event:
                intent = event["intent_classifier"].get("intent", "")
                yield f"STATUS: Detected intent: {intent}\n"

            if "document_retriever" in event:
                docs = event["document_retriever"].get("retrieved_docs", [])
                yield f"STATUS: Found {len(docs)} relevant documents\n"
                for doc in docs:
                    yield f"CITATION: {doc.source_doc} (Page {doc.page_number})\n"

            if "answer_generator" in event:
                answer = event["answer_generator"].get("answer", "")
                if answer:
                    yield f"ANSWER: {answer}\n"

            if "response_validator" in event:
                score = event["response_validator"].get("confidence_score", 0)
                yield f"CONFIDENCE: {score}\n"

            if "__end__" in event:
                yield "DONE\n"

    except Exception as e:
        logger.error(f"Agent streaming failed: {e}")
        yield f"ERROR: {str(e)}\n"
