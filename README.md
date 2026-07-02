# ⚖️ Nyaya Setu – Indian Legal AI

> Private Offline AI Legal Assistant for Citizens & Legal Professionals

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Next.js](https://img.shields.io/badge/Next.js-Frontend-black)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue)
![Ollama](https://img.shields.io/badge/Ollama-Qwen%201.5B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

Nyaya Setu is an offline, privacy-first AI legal assistant trained on Indian law documents using Retrieval-Augmented Generation (RAG).

It helps:
- 👨‍⚖️ Citizens (Nagrik Portal) understand legal rights
- 🧑‍💼 Legal professionals (Vakeel Portal) perform legal research & drafting

The system provides:
- grounded legal answers
- exact document citations
- page references
- confidence scores
- offline local inference

---

# ✨ Features

- 🔍 Hybrid RAG Retrieval
- 📚 Source Grounded Legal Answers
- 📄 Exact PDF + Page Citations
- ⚖️ Dual Portals:
  - Nagrik Portal (Citizen Guidance)
  - Vakeel Portal (Legal Research)
- 🧠 Local LLM using Ollama (Qwen 1.5B)
- 🗂️ pgvector Semantic Search
- 🔐 Fully Offline & Private
- 📑 Legal Clause Drafting
- 📊 Confidence Scoring
- ⚡ FastAPI Backend + Next.js Frontend

---

# 🏗️ Architecture Overview

```text
User
 ↓
Next.js Frontend
 ↓
FastAPI Backend
 ↓
LangGraph RAG Pipeline
 ↓
PostgreSQL + pgvector
 ↓
Ollama (Qwen 1.5B)
 ↓
Indian Legal Documents (PDFs)
```

---

# 🛠️ Tech Stack

## Frontend
- Next.js
- Tailwind CSS
- TypeScript

## Backend
- FastAPI
- LangGraph
- Python

## Database
- PostgreSQL
- pgvector

## AI Stack
- Ollama
- Qwen 1.5B
- sentence-transformers

## Infrastructure
- Docker
- Docker Compose

---

# 📂 Project Structure

```text
nyaya-setu/
│
├── backend/
├── frontend/
├── docs/
│   └── images/
├── data/
├── scripts/
├── docker/
├── README.md
└── docker-compose.yml
```

---

# 📸 Screenshots

## 🧑 Nagrik Portal

### Home Page

![Home](docs/images/home.png)

---

### RTI Application Guidance

![RTI](docs/images/rti-application.png)

---

### Domestic Violence Rights

![Domestic Violence](docs/images/domestic-violence.png)

---

### Salary Rights & Employer Delays

![Salary Rights](docs/images/salary-rights.png)

---

## ⚖️ Vakeel Portal

### IPC Sections & Punishments

![IPC](docs/images/ipc-sections.png)

---

### Legal Research Response

![Legal Research](docs/images/legal-research.png)

---

### Clause Drafting

![Clause Drafting](docs/images/clause-drafting.png)

---

# 📚 Indexed Indian Laws

Currently indexed legal acts include:

- IPC 1860
- CrPC 1973
- RTI Act 2005
- POCSO Act
- Domestic Violence Act
- Consumer Protection Act
- Payment of Wages Act
- LARR 2013

---

# 🚀 Quick Start

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/nyaya-setu.git
cd nyaya-setu
```

---

## 2. Start Services

```bash
docker-compose up --build
```

---

## 3. Run Backend

```bash
cd backend
uvicorn app.main:app --reload --port 8001
```

---

## 4. Run Frontend

```bash
cd frontend
npm install
npm run dev
```

---

# 🔎 API Health Check

Run:

```powershell
curl http://localhost:8001/api/health
```

Expected Output:

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "ollama_connected": true,
  "postgres_connected": true,
  "vector_count": 25412
}
```

---

# 🧪 Example Queries

## Nagrik Portal

```text
What are my rights if police arrest me without warrant?
```

```text
How do I file an RTI application?
```

```text
Can my employer delay salary legally?
```

```text
What legal protection exists for domestic violence victims?
```

---

## Vakeel Portal

```text
Provide IPC sections related to criminal intimidation.
```

```text
Summarize Section 420 IPC with punishments.
```

```text
Draft a legal notice for breach of contract.
```

```text
Provide CrPC provisions for anticipatory bail.
```

---

# 📄 Adding New Legal Documents

1. Add PDF documents into the data/documents folder

2. Run ingestion pipeline

```bash
python ingest.py
```

3. Verify indexing

```powershell
curl http://localhost:8001/api/health
```

If vector count increases, indexing succeeded.

---

# 🔐 Privacy & Security

- 100% Offline Inference
- No External API Calls
- No Data Leaves Your Machine
- Local LLM Execution
- Private Legal Research

---

# 📈 Future Improvements

- Multi-language Legal Support
- Voice-Based Legal Assistance
- Legal Judgment Summarization
- Citation Highlighting
- Advanced Clause Drafting
- Fine-tuned Indian Legal LLM

---

# 🤝 Contributing

Pull requests are welcome.

For major changes, please open an issue first to discuss improvements.

---

# 📜 License

This project is licensed under the MIT License.

---
