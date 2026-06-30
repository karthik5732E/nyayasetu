# Nyaya Setu v2.0 — Private Offline AI Legal Assistant (Indian Law)

> **Fully offline · Privacy-first · Production RAG architecture**
> Built with FastAPI, LangGraph, PostgreSQL + pgvector, and a local LLM via Ollama — no API keys, no cloud costs, no data leaving your machine.

---

## What This Is

Nyaya Setu is an AI-powered legal assistant trained on real Indian law — the Indian Penal Code, Code of Criminal Procedure, RTI Act, POCSO Act, Domestic Violence Act, Consumer Protection Act, Payment of Wages Act, and the Land Acquisition Act (LARR). Ask it a question in plain English about your legal rights, and it retrieves the relevant law, generates a grounded answer, and cites the exact source document and page number — entirely on your own machine, with no internet dependency at inference time.

It runs as a 5-node **LangGraph agent**: intent classification → hybrid document retrieval (semantic + keyword search, fused with Reciprocal Rank Fusion) → LLM answer generation → citation formatting → hallucination/confidence validation.

## Why This Matters (Architecture Highlights)

- **Hybrid search, not just vector search.** Combines pgvector cosine similarity with PostgreSQL full-text search, merged via RRF — catches both semantic intent and exact legal terminology/section numbers.
- **Grounded, not hallucinated.** Every answer is generated only from retrieved chunks, with a built-in confidence scorer that checks word-overlap between the answer and source documents, flagging low-confidence responses automatically.
- **Fully local LLM.** Runs on Ollama with no per-query API cost — currently configured with `qwen2.5:1.5b` for a responsive experience on consumer hardware (8GB RAM friendly).
- **Multi-portal design.** Two personas baked into the prompt layer — `nagarik` (citizen-facing, plain-language guidance) and `vakeel` (lawyer-facing, clause-citing precision).
- **MCP server included.** Exposes 4 legal tools (search, summarize, cite, list documents) so any MCP-compatible client (e.g. Claude Desktop) can use Nyaya Setu as a tool.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama + Qwen2.5 1.5B (swappable — Phi-3 also supported on higher-RAM machines) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`), 384-dim, fully local |
| Database | PostgreSQL 15 + pgvector + full-text search |
| Agent Orchestration | LangGraph (5-node StateGraph with checkpointing) |
| API | FastAPI (async) |
| Observability | LangSmith (optional, free tier) |
| MCP Server | Python MCP SDK |
| Deployment | Docker + docker-compose, CPU-only friendly |

---

## Verified Working Endpoints

Every endpoint below was tested end-to-end against the full 8-document Indian law corpus (25,000+ indexed chunks):

| Method | Endpoint | What it does | Status |
|---|---|---|---|
| `GET` | `/api/health` | Service + DB + LLM connectivity check | ✅ |
| `GET` | `/api/sources` | List all indexed legal documents | ✅ |
| `POST` | `/api/query` | Full RAG legal Q&A with citations | ✅ |
| `GET` | `/api/search` | Direct hybrid search (no LLM) | ✅ |
| `POST` | `/api/draft-clause` | AI-drafted legal clauses (NDA, employment, non-compete, indemnity) | ✅ |
| `POST` | `/api/summarize` | Legal document summarization | ✅ |
| `GET` | `/api/document/{name}` | Retrieve all chunks for a specific document | ✅ |
| `POST` | `/api/upload` | Ingest a new document on the fly | ✅ |
| `GET` | `/docs` | Interactive Swagger UI | ✅ |

---

## Setup (Docker, Windows/Mac/Linux)

### Prerequisites
- Docker Desktop (with WSL2 backend on Windows)
- 8GB+ RAM recommended (4.5GB allocated to Docker minimum)
- Git

### Quick Start

```bash
git clone <your-repo-url>
cd nyaya-setu
copy .env.example .env      # Windows
# cp .env.example .env      # Mac/Linux
```

Edit `.env` and confirm these values (defaults work out of the box):
```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/nyayasetu
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen2.5:1.5b
```

> **Note on model choice:** `qwen2.5:1.5b` is recommended for machines with 8GB RAM or less. If you have 16GB+ RAM, you can switch to `phi3` for stronger reasoning at the cost of slower inference.

Build and start:
```bash
docker-compose build
docker-compose up -d
```

Ingest the legal documents (place PDFs in `uploads/indian_laws/` first):
```bash
docker-compose exec -T api python -m app.retrieval.ingest /app/uploads/indian_laws indian_laws
```

Verify everything is healthy:
```bash
curl http://localhost:8001/api/health
```

### Example Query

```bash
curl -X POST http://localhost:8001/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are my rights if my employer does not pay salary on time?", "language": "english", "portal": "nagarik"}'
```

---

## Known Constraints

- **CPU-only inference.** No GPU required, but expect 10-30 seconds per query on typical consumer hardware after the model is warmed up (first query after startup takes longer while the model loads into memory).
- **8GB RAM machines:** stick with `qwen2.5:1.5b`. Larger models (Phi-3, Llama 3) risk being killed by the OS under memory pressure when combined with PostgreSQL + the embedding model running simultaneously.
- **Not a substitute for legal counsel.** This is an educational/awareness tool. Always consult a registered advocate for actual legal proceedings.

---

## Roadmap

- [ ] React frontend (Nagarik citizen portal + Vakeel lawyer portal)
- [ ] Multi-language answer generation (Hindi, Telugu — language detection already implemented)
- [ ] Streaming responses in the UI
- [ ] Expanded legal corpus (more Acts, case law)

---

## Disclaimer

For educational purposes only. Not a substitute for a registered advocate. Always consult a qualified lawyer before taking legal action.

## License

MIT License — free for personal and commercial use.