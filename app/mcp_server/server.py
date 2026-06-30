#!/usr/bin/env python3
# =============================================================================
# server.py — MCP Server for Nyaya Setu Legal Tools
# Exposes legal document search, summarization, and citation tools.
# Run: python -m app.mcp_server.server
# =============================================================================

import asyncio
import json
import logging
from typing import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool, TextContent, ImageContent, EmbeddedResource,
    LoggingLevel
)

from app.config import settings
from app.retrieval.pgvector import (
    hybrid_search, get_document_chunks, get_document_summary_info,
    init_database, get_source_docs
)
from app.models.schemas import DocChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nyaya-mcp")


# ---------------------------------------------------------------------------
# App Context (shared across the server lifespan)
# ---------------------------------------------------------------------------

@dataclass
class AppContext:
    db_initialized: bool = False


@asynccontextmanager
async def app_lifespan(server: Server) -> AsyncIterator[AppContext]:
    """Initialize the server context."""
    logger.info("Nyaya Setu MCP Server starting...")
    try:
        await init_database()
        ctx = AppContext(db_initialized=True)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        ctx = AppContext(db_initialized=False)
    yield ctx


# ---------------------------------------------------------------------------
# Create MCP Server
# ---------------------------------------------------------------------------

server = Server(
    "nyaya-setu",
    instructions="""
    Nyaya Setu MCP Server — Legal document search and citation tools.
    Use these tools to search Indian law documents, get document summaries,
    and cite specific sections.
    """,
    lifespan=app_lifespan
)


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available legal tools."""
    return [
        Tool(
            name="search_legal_docs",
            description="Search Indian legal documents using hybrid semantic + keyword search with RRF ranking. Returns relevant document chunks with exact page citations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — describe the legal topic or question"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "source_doc": {
                        "type": "string",
                        "description": "Optional: filter by specific document name",
                        "default": None
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_document_summary",
            description="Get a summary of a specific legal document — total pages, total chunks, and a content overview.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_name": {
                        "type": "string",
                        "description": "Name of the document (e.g., 'IPC_1860.pdf')"
                    }
                },
                "required": ["doc_name"]
            }
        ),
        Tool(
            name="cite_section",
            description="Get the exact text content of a specific page/section from a legal document.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_name": {
                        "type": "string",
                        "description": "Name of the document"
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number to retrieve"
                    }
                },
                "required": ["doc_name", "page"]
            }
        ),
        Tool(
            name="list_available_documents",
            description="List all legal documents currently available in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    """Handle tool invocations."""
    logger.info(f"Tool called: {name} with args: {arguments}")

    # --- search_legal_docs ---
    if name == "search_legal_docs":
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        source_doc = arguments.get("source_doc")

        if not query:
            return [TextContent(type="text", text="Error: query is required")]

        try:
            results = await hybrid_search(query, k=top_k, source_filter=source_doc)

            if not results:
                return [TextContent(
                    type="text",
                    text=f"No relevant documents found for: '{query}'"
                )]

            lines = [f"Search results for: '{query}'\n{'='*50}"]
            for i, r in enumerate(results, 1):
                lines.append(
                    f"\n[{i}] {r.source_doc} (Page {r.page_number})\n"
                    f"Relevance: {r.score:.4f}\n"
                    f"{r.content[:500]}{'...' if len(r.content) > 500 else ''}"
                )

            return [TextContent(type="text", text="\n".join(lines))]

        except Exception as e:
            logger.error(f"search_legal_docs failed: {e}")
            return [TextContent(type="text", text=f"Error searching: {str(e)}")]

    # --- get_document_summary ---
    elif name == "get_document_summary":
        doc_name = arguments.get("doc_name", "")

        if not doc_name:
            return [TextContent(type="text", text="Error: doc_name is required")]

        try:
            pages, chunks = await get_document_summary_info(doc_name)

            if chunks == 0:
                return [TextContent(
                    type="text",
                    text=f"Document '{doc_name}' not found in the database."
                )]

            # Get a sample of content
            doc_chunks = await get_document_chunks(doc_name, limit=5)
            preview = "\n\n".join(f"[Page {c.page_number}] {c.content[:200]}..." for c in doc_chunks)

            text = (
                f"Document: {doc_name}\n"
                f"{'='*50}\n"
                f"Total Pages: {pages}\n"
                f"Total Chunks: {chunks}\n\n"
                f"Content Preview:\n{preview}"
            )
            return [TextContent(type="text", text=text)]

        except Exception as e:
            logger.error(f"get_document_summary failed: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    # --- cite_section ---
    elif name == "cite_section":
        doc_name = arguments.get("doc_name", "")
        page = arguments.get("page", 0)

        if not doc_name or page <= 0:
            return [TextContent(type="text", text="Error: doc_name and positive page number required")]

        try:
            chunks = await get_document_chunks(doc_name, limit=1000)
            page_chunks = [c for c in chunks if c.page_number == page]

            if not page_chunks:
                return [TextContent(
                    type="text",
                    text=f"No content found for {doc_name} page {page}"
                )]

            content = "\n\n".join(c.content for c in page_chunks)
            text = (
                f"Citation: {doc_name} — Page {page}\n"
                f"{'='*50}\n"
                f"{content}"
            )
            return [TextContent(type="text", text=text)]

        except Exception as e:
            logger.error(f"cite_section failed: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    # --- list_available_documents ---
    elif name == "list_available_documents":
        try:
            sources = await get_source_docs()
            if not sources:
                return [TextContent(type="text", text="No documents in the database.")]

            lines = [f"Available Legal Documents ({len(sources)}):\n{'='*50}"]
            for i, s in enumerate(sources, 1):
                lines.append(f"{i}. {s}")

            return [TextContent(type="text", text="\n".join(lines))]

        except Exception as e:
            logger.error(f"list_available_documents failed: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    # --- Unknown tool ---
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

async def main():
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
