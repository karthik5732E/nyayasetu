# =============================================================================
# pgvector.py — Hybrid Search with pgvector + Full-Text Search + RRF
# =============================================================================

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager

import asyncpg
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.models.schemas import DocChunk, Citation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global embedding model (lazy-loaded)
# ---------------------------------------------------------------------------
_embed_model: Optional[SentenceTransformer] = None


def get_embed_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformers model (FREE, runs locally)."""
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {settings.embed_model}")
        _embed_model = SentenceTransformer(settings.embed_model)
        logger.info("Embedding model loaded")
    return _embed_model


def embed_text(text: str) -> List[float]:
    """Embed a single text into a 384-dim vector."""
    model = get_embed_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed multiple texts in a batch."""
    model = get_embed_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# Database Connection Pool
# ---------------------------------------------------------------------------
_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create the asyncpg connection pool."""
    global _pool
    if _pool is None or _pool._closed:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url.replace("postgresql+asyncpg://", "postgresql://"),
            min_size=2,
            max_size=10,
            command_timeout=60,
        )
    return _pool


async def close_pool():
    """Close the connection pool."""
    global _pool
    if _pool and not _pool._closed:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_conn():
    """Context manager for a database connection."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


# ---------------------------------------------------------------------------
# Schema Setup
# ---------------------------------------------------------------------------
SETUP_SQL = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable full-text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Main embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    id              SERIAL PRIMARY KEY,
    content         TEXT NOT NULL,
    embedding       vector(384) NOT NULL,
    metadata        JSONB DEFAULT '{}',
    page_number     INT DEFAULT 1,
    source_doc      TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_embeddings_fts
    ON embeddings USING GIN (to_tsvector('english', content));

-- Source document index for fast filtering
CREATE INDEX IF NOT EXISTS idx_embeddings_source
    ON embeddings(source_doc);

-- Page number index
CREATE INDEX IF NOT EXISTS idx_embeddings_page
    ON embeddings(page_number);

-- Vector similarity index (IVFFlat for approximate nearest neighbors)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
    ON embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Conversations table for multi-turn chat memory
CREATE TABLE IF NOT EXISTS conversations (
    id              SERIAL PRIMARY KEY,
    conversation_id TEXT UNIQUE NOT NULL,
    portal          TEXT DEFAULT 'nagarik',
    messages        JSONB DEFAULT '[]',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_conv_id
    ON conversations(conversation_id);
"""


async def init_database():
    """Initialize the database schema. Run this once at startup."""
    logger.info("Initializing database schema...")
    async with get_conn() as conn:
        await conn.execute(SETUP_SQL)
    logger.info("Database schema initialized")


async def check_connection() -> bool:
    """Check if PostgreSQL is reachable."""
    try:
        async with get_conn() as conn:
            result = await conn.fetchval("SELECT 1")
            return result == 1
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        return False


# ---------------------------------------------------------------------------
# CRUD Operations
# ---------------------------------------------------------------------------

async def insert_chunks(chunks: List[Dict[str, Any]], collection: str = "default") -> int:
    """
    Insert document chunks into pgvector.
    Returns the number of chunks inserted.
    """
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    inserted = 0
    async with get_conn() as conn:
        for chunk, vector in zip(chunks, vectors):
            await conn.execute(
                """
                INSERT INTO embeddings (content, embedding, metadata, page_number, source_doc)
                VALUES ($1, $2, $3, $4, $5)
                """,
                chunk["text"],
                str(vector),  # asyncpg handles vector conversion
                json.dumps(chunk.get("metadata", {})),
                chunk.get("page", 1),
                chunk.get("source", "unknown")
            )
            inserted += 1

    logger.info(f"Inserted {inserted} chunks into pgvector")
    return inserted


async def insert_chunks_batch(chunks: List[Dict[str, Any]], collection: str = "default") -> int:
    """
    Batch insert chunks. Embeds and inserts in small batches with progress
    logging, instead of embedding the entire document in one giant call.
    """
    if not chunks:
        return 0

    BATCH_SIZE = 32
    total_inserted = 0
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end = start + BATCH_SIZE
        batch = chunks[start:end]

        texts = [c["text"] for c in batch]
        vectors = embed_texts(texts)

        records = []
        for chunk, vector in zip(batch, vectors):
            records.append((
                chunk["text"],
                str(vector),
                json.dumps(chunk.get("metadata", {})),
                chunk.get("page", 1),
                chunk.get("source", "unknown")
            ))

        async with get_conn() as conn:
            await conn.executemany(
                """
                INSERT INTO embeddings (content, embedding, metadata, page_number, source_doc)
                VALUES ($1, $2, $3, $4, $5)
                """,
                records
            )

        total_inserted += len(records)
        print(f"      batch {batch_num + 1}/{total_batches} -> {total_inserted}/{len(chunks)} chunks embedded+inserted")

    logger.info(f"Batch inserted {total_inserted} chunks into pgvector")
    return total_inserted


async def delete_by_source(source_doc: str) -> int:
    """Delete all chunks belonging to a source document."""
    async with get_conn() as conn:
        result = await conn.execute(
            "DELETE FROM embeddings WHERE source_doc = $1",
            source_doc
        )
        # Parse "DELETE N" from result string
        count = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
        logger.info(f"Deleted {count} chunks for source: {source_doc}")
        return count


async def clear_collection() -> int:
    """Clear all embeddings. Returns count of deleted rows."""
    async with get_conn() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM embeddings")
        await conn.execute("TRUNCATE TABLE embeddings RESTART IDENTITY")
        logger.info(f"Cleared {count} embeddings")
        return count


async def get_vector_count() -> int:
    """Get the total number of stored vectors."""
    async with get_conn() as conn:
        return await conn.fetchval("SELECT COUNT(*) FROM embeddings") or 0


async def get_source_docs() -> List[str]:
    """Get list of all unique source documents."""
    async with get_conn() as conn:
        rows = await conn.fetch("SELECT DISTINCT source_doc FROM embeddings")
        return [r["source_doc"] for r in rows]


# ---------------------------------------------------------------------------
# Hybrid Search: Cosine Similarity + Full-Text Search + RRF
# ---------------------------------------------------------------------------

async def search_semantic(query_embedding: List[float], k: int = 5, source_filter: Optional[str] = None) -> List[Dict]:
    """Semantic search using cosine similarity."""
    vector_str = f"[{','.join(str(v) for v in query_embedding)}]"

    async with get_conn() as conn:
        if source_filter:
            rows = await conn.fetch(
                """
                SELECT id, content, source_doc, page_number, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM embeddings
                WHERE source_doc = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                vector_str, source_filter, k
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, content, source_doc, page_number, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM embeddings
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vector_str, k
            )

    return [dict(r) for r in rows]


async def search_keyword(query: str, k: int = 5, source_filter: Optional[str] = None) -> List[Dict]:
    """Keyword search using PostgreSQL full-text search."""
    async with get_conn() as conn:
        if source_filter:
            rows = await conn.fetch(
                """
                SELECT id, content, source_doc, page_number, metadata,
                       ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) AS score
                FROM embeddings
                WHERE source_doc = $2
                  AND to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                ORDER BY score DESC
                LIMIT $3
                """,
                query, source_filter, k
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, content, source_doc, page_number, metadata,
                       ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) AS score
                FROM embeddings
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                ORDER BY score DESC
                LIMIT $2
                """,
                query, k
            )

    return [dict(r) for r in rows]


async def hybrid_search(
    query: str,
    k: int = 5,
    source_filter: Optional[str] = None,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> List[DocChunk]:
    """
    Hybrid search combining semantic (cosine similarity) + keyword (FTS) with RRF.

    Args:
        query: User's search query
        k: Number of results to return
        source_filter: Optional filter by source document
        semantic_weight: Weight for semantic search scores
        keyword_weight: Weight for keyword search scores

    Returns:
        List of DocChunk with exact page citations
    """
    if not query or not query.strip():
        return []

    # 1. Semantic search
    query_embedding = embed_text(query)
    semantic_results = await search_semantic(query_embedding, k=k * 2, source_filter=source_filter)

    # 2. Keyword search
    keyword_results = await search_keyword(query, k=k * 2, source_filter=source_filter)

    # 3. Reciprocal Rank Fusion (RRF)
    # RRF score = sum(1 / (k + rank)) for each list the item appears in
    rrf_constant = 60  # RRF parameter k

    # Track items and their ranks
    item_scores: Dict[int, Dict] = {}
    item_ranks: Dict[int, List[int]] = {}

    # Process semantic results (assign ranks 1, 2, 3, ...)
    for rank, row in enumerate(semantic_results, start=1):
        doc_id = row["id"]
        item_scores[doc_id] = row
        item_ranks.setdefault(doc_id, []).append(rank)

    # Process keyword results
    for rank, row in enumerate(keyword_results, start=1):
        doc_id = row["id"]
        item_scores[doc_id] = row
        item_ranks.setdefault(doc_id, []).append(rank)

    # Calculate RRF scores
    rrf_scores: List[Tuple[int, float]] = []
    for doc_id, ranks in item_ranks.items():
        rrf_score = sum(1.0 / (rrf_constant + rank) for rank in ranks)
        rrf_scores.append((doc_id, rrf_score))

    # Sort by RRF score descending
    rrf_scores.sort(key=lambda x: x[1], reverse=True)

    # Build final results
    results: List[DocChunk] = []
    for doc_id, rrf_score in rrf_scores[:k]:
        row = item_scores[doc_id]
        results.append(DocChunk(
            id=doc_id,
            content=row["content"],
            source_doc=row["source_doc"],
            page_number=row["page_number"],
            score=round(rrf_score, 4),
            metadata=json.loads(row["metadata"]) if isinstance(row.get("metadata"), str) else (row.get("metadata") or {})
        ))

    logger.info(f"Hybrid search for '{query[:50]}...' returned {len(results)} results")
    return results


# ---------------------------------------------------------------------------
# Get chunk by document + page (for citation lookup)
# ---------------------------------------------------------------------------

async def get_chunks_by_page(doc_name: str, page: int) -> List[DocChunk]:
    """Get all chunks for a specific document page."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, source_doc, page_number, metadata
            FROM embeddings
            WHERE source_doc = $1 AND page_number = $2
            ORDER BY id
            """,
            doc_name, page
        )
    return [DocChunk(
        id=r["id"],
        content=r["content"],
        source_doc=r["source_doc"],
        page_number=r["page_number"],
        metadata=json.loads(r["metadata"]) if isinstance(r.get("metadata"), str) else (r.get("metadata") or {})
    ) for r in rows]


async def get_document_chunks(doc_name: str, limit: int = 100) -> List[DocChunk]:
    """Get all chunks for a document (for summarization)."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, source_doc, page_number, metadata
            FROM embeddings
            WHERE source_doc = $1
            ORDER BY page_number, id
            LIMIT $2
            """,
            doc_name, limit
        )
    return [DocChunk(
        id=r["id"],
        content=r["content"],
        source_doc=r["source_doc"],
        page_number=r["page_number"],
        metadata=json.loads(r["metadata"]) if isinstance(r.get("metadata"), str) else (r.get("metadata") or {})
    ) for r in rows]


async def get_document_summary_info(doc_name: str) -> Tuple[int, int]:
    """Get total pages and chunks for a document. Returns (pages, chunks)."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT COUNT(DISTINCT page_number) as pages, COUNT(*) as chunks
            FROM embeddings
            WHERE source_doc = $1
            """,
            doc_name
        )
    return row["pages"] or 0, row["chunks"] or 0


# ---------------------------------------------------------------------------
# Conversation Memory (for LangGraph MemorySaver)
# ---------------------------------------------------------------------------

async def get_conversation(conv_id: str) -> Optional[Dict[str, Any]]:
    """Get a conversation by ID."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM conversations WHERE conversation_id = $1",
            conv_id
        )
    if row:
        return dict(row)
    return None


async def save_conversation(conv_id: str, messages: List[Dict[str, Any]], portal: str = "nagarik"):
    """Save or update a conversation."""
    messages_json = json.dumps(messages)
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO conversations (conversation_id, portal, messages, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (conversation_id)
            DO UPDATE SET messages = $3, updated_at = NOW()
            """,
            conv_id, portal, messages_json
        )


async def list_conversations(limit: int = 50) -> List[Dict[str, Any]]:
    """List recent conversations."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT $1",
            limit
        )
    return [dict(r) for r in rows]
