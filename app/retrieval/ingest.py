# =============================================================================
# ingest.py — Document loading, chunking, and pgvector ingestion
# Replaces the old ChromaDB-based ingest with pgvector-based ingest.
# Run: python -m app.retrieval.ingest <folder_path> <collection_name>
# Or use the API: POST /api/upload
# =============================================================================

import os
import sys
import time
import fitz  # PyMuPDF
from docx import Document as DocxDoc
from typing import List, Dict, Any

from app.retrieval.pgvector import (
    insert_chunks_batch, delete_by_source, get_vector_count
)


# ---------------------------------------------------------------------------
# Document Loaders
# ---------------------------------------------------------------------------

def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Load a PDF and extract text page by page."""
    docs = []
    pdf = fitz.open(file_path)
    filename = os.path.basename(file_path)
    for i in range(len(pdf)):
        text = pdf[i].get_text().strip()
        if len(text) >= 20:
            docs.append({
                "text": text,
                "page": i + 1,
                "source": filename,
                "metadata": {"type": "pdf", "total_pages": len(pdf)}
            })
    pdf.close()
    return docs


def load_docx(file_path: str) -> List[Dict[str, Any]]:
    """Load a DOCX file and split into page-like chunks."""
    docs = []
    filename = os.path.basename(file_path)
    doc = DocxDoc(file_path)
    current, wc, idx = [], 0, 1
    for para in doc.paragraphs:
        t = para.text.strip()
        if not t:
            continue
        current.append(t)
        wc += len(t.split())
        if wc >= 400:
            docs.append({
                "text": "\n".join(current),
                "page": idx,
                "source": filename,
                "metadata": {"type": "docx"}
            })
            current, wc, idx = [], 0, idx + 1
    if current:
        docs.append({
            "text": "\n".join(current),
            "page": idx,
            "source": filename,
            "metadata": {"type": "docx"}
        })
    return docs


def load_txt(file_path: str) -> List[Dict[str, Any]]:
    """Load a TXT file and split into word chunks."""
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        words = f.read().split()
    return [
        {
            "text": " ".join(words[i:i + 500]),
            "page": (i // 500) + 1,
            "source": filename,
            "metadata": {"type": "txt"}
        }
        for i in range(0, len(words), 500)
    ]


def load_document(file_path: str) -> List[Dict[str, Any]]:
    """Load any supported document type."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 300
CHUNK_OVERLAP = 30


def chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split documents into overlapping chunks.
    Uses simple sentence-based chunking for speed.
    """
    chunks = []
    for doc in docs:
        text = doc["text"]
        # Split into sentences (rough approximation)
        sentences = []
        for para in text.split("\n"):
            for sent in para.split("."):
                sent = sent.strip()
                if sent:
                    sentences.append(sent + ".")

        # Build chunks
        current_chunk = []
        current_len = 0
        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > CHUNK_SIZE and current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text.strip()) >= 30:
                    chunks.append({
                        "text": chunk_text.strip(),
                        "page": doc["page"],
                        "source": doc["source"],
                        "metadata": doc.get("metadata", {})
                    })
                # Keep overlap
                overlap_text = " ".join(current_chunk[-2:]) if len(current_chunk) >= 2 else ""
                current_chunk = [overlap_text, sent] if overlap_text else [sent]
                current_len = len(overlap_text) + sent_len if overlap_text else sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.strip()) >= 30:
                chunks.append({
                    "text": chunk_text.strip(),
                    "page": doc["page"],
                    "source": doc["source"],
                    "metadata": doc.get("metadata", {})
                })

    return chunks


# ---------------------------------------------------------------------------
# Ingestion Pipeline
# ---------------------------------------------------------------------------

async def ingest_file(file_path: str, collection: str = "default") -> Dict[str, Any]:
    """
    Ingest a single file: load → chunk → embed → store in pgvector.
    Returns: {chunks_indexed, source_doc, pages}
    """
    source_doc = os.path.basename(file_path)

    # Delete existing chunks for this source (to avoid duplicates)
    await delete_by_source(source_doc)

    # Load document
    pages = load_document(file_path)
    if not pages:
        return {"chunks_indexed": 0, "source_doc": source_doc, "pages": 0, "error": "No content found"}

    # Chunk
    chunks = chunk_documents(pages)
    if not chunks:
        return {"chunks_indexed": 0, "source_doc": source_doc, "pages": len(pages), "error": "No chunks created"}

    # Insert into pgvector
    count = await insert_chunks_batch(chunks, collection=collection)

    return {
        "chunks_indexed": count,
        "source_doc": source_doc,
        "pages": len(pages)
    }


async def ingest_folder(folder_path: str, collection: str = "indian_laws") -> Dict[str, Any]:
    """
    Ingest all supported files from a folder.
    Returns: {total_files, total_chunks, errors}
    """
    if not os.path.exists(folder_path):
        return {"total_files": 0, "total_chunks": 0, "errors": [f"Folder not found: {folder_path}"]}

    extensions = (".pdf", ".docx", ".txt")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

    if not files:
        return {"total_files": 0, "total_chunks": 0, "errors": ["No supported files found"]}

    total_chunks = 0
    errors = []
    start = time.time()

    for i, filename in enumerate(files):
        file_path = os.path.join(folder_path, filename)
        print(f"[{i + 1}/{len(files)}] Ingesting: {filename}")
        try:
            result = await ingest_file(file_path, collection=collection)
            total_chunks += result["chunks_indexed"]
            print(f"   -> {result['chunks_indexed']} chunks")
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
            print(f"   -> ERROR: {e}")

    elapsed = time.time() - start
    total = await get_vector_count()

    return {
        "total_files": len(files),
        "total_chunks": total_chunks,
        "total_vectors_in_db": total,
        "errors": errors,
        "time_seconds": round(elapsed, 1)
    }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

async def main():
    """CLI: python -m app.retrieval.ingest <folder> [collection]"""
    if len(sys.argv) < 2:
        print("Usage: python -m app.retrieval.ingest <folder_path> [collection_name]")
        print("Example: python -m app.retrieval.ingest ./indian_laws indian_laws")
        sys.exit(1)

    folder = sys.argv[1]
    collection = sys.argv[2] if len(sys.argv) > 2 else "indian_laws"

    # Initialize DB
    from app.retrieval.pgvector import init_database
    await init_database()

    print(f"\n{'='*60}")
    print(f"  NYAYA SETU — Document Ingestion (pgvector)")
    print(f"{'='*60}")
    print(f"Folder: {folder}")
    print(f"Collection: {collection}")
    print()

    result = await ingest_folder(folder, collection)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Files processed: {result['total_files']}")
    print(f"  Chunks indexed: {result['total_chunks']}")
    print(f"  Total vectors in DB: {result['total_vectors_in_db']}")
    print(f"  Time: {result['time_seconds']}s")
    if result['errors']:
        print(f"  Errors: {len(result['errors'])}")
        for e in result['errors']:
            print(f"    - {e}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
