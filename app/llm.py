# =============================================================================
# llm.py — Ollama LLM wrapper
# All LLM calls go through here for centralized management.
# =============================================================================

import os
import logging
import httpx
from typing import Optional, AsyncGenerator

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ollama Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """Async client for Ollama API."""

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0))
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 350,
        temperature: float = 0.1,
        stream: bool = False
    ) -> str:
        """Generate text from a prompt."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 2048,
            },
            "stream": False
        }

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 350,
        temperature: float = 0.1
    ) -> AsyncGenerator[str, None]:
        """Stream text generation."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": True
        }

        try:
            async with self.client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            yield f"[Error: {str(e)}]"

    async def chat(self, messages: list, max_tokens: int = 350, temperature: float = 0.1) -> str:
        """Chat completion with message history."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False
        }

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if Ollama is reachable and model is loaded."""
        try:
            url = f"{self.base_url}/api/tags"
            response = await self.client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return self.model in models or any(self.model in m for m in models)
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get the singleton Ollama client."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


async def query_ollama(prompt: str, max_tokens: int = 350, temperature: float = 0.1) -> str:
    """Convenience: send a single prompt to Ollama and get response."""
    client = get_ollama_client()
    return await client.generate(prompt, max_tokens=max_tokens, temperature=temperature)


async def stream_ollama(prompt: str, max_tokens: int = 350, temperature: float = 0.1) -> AsyncGenerator[str, None]:
    """Convenience: stream a single prompt from Ollama."""
    client = get_ollama_client()
    async for chunk in client.generate_stream(prompt, max_tokens=max_tokens, temperature=temperature):
        yield chunk


# ---------------------------------------------------------------------------
# LangChain-compatible wrapper (for LangGraph integration)
# ---------------------------------------------------------------------------

from langchain_core.language_models.llms import LLM
from typing import Any, List, Mapping
from pydantic import Field


class OllamaLangChainLLM(LLM):
    """LangChain-compatible wrapper for Ollama."""

    base_url: str = Field(default_factory=lambda: settings.ollama_base_url)
    model_name: str = Field(default_factory=lambda: settings.ollama_model)
    temperature: float = 0.1
    max_tokens: int = 350

    @property
    def _llm_type(self) -> str:
        return "ollama"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model_name, "temperature": self.temperature}

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Synchronous call (for LangChain compatibility)."""
        import asyncio
        client = OllamaClient(base_url=self.base_url, model=self.model_name)
        try:
            return asyncio.get_event_loop().run_until_complete(
                client.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
            )
        finally:
            asyncio.get_event_loop().run_until_complete(client.close())


# ---------------------------------------------------------------------------
# Summarization helper
# ---------------------------------------------------------------------------

async def summarize_text(text: str, portal: str = "nagarik") -> str:
    """Summarize a document using Ollama."""
    words = text.split()
    if len(words) > 2000:
        text = " ".join(words[:2000]) + "\n\n[Truncated]"

    if portal == "vakeel":
        prompt = f"""Summarize this legal document concisely:
- Document type & parties
- Key obligations & terms
- Important dates/deadlines
- Any risky clauses

DOCUMENT:
{text}

SUMMARY:"""
    else:
        prompt = f"""Summarize this Indian law simply:
- What it covers  - Who it protects  - Key rights  - How to use it

LAW:
{text}

SUMMARY:"""

    return await query_ollama(prompt, max_tokens=400, temperature=0.1)


# ---------------------------------------------------------------------------
# Clause drafting helper
# ---------------------------------------------------------------------------

async def draft_clause(clause_type: str, details: dict) -> str:
    """Draft a legal clause using a template + Ollama."""
    template_path = os.path.join(
        os.path.dirname(__file__), "templates", f"{clause_type}.txt"
    )

    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            template_text = f.read()
        for key, value in details.items():
            template_text = template_text.replace(f"{{{key}}}", str(value or ""))
    else:
        template_text = f"Draft a professional {clause_type} clause for Indian law."

    prompt = f"""Draft a concise professional Indian law clause.

INSTRUCTIONS:
{template_text}

CLAUSE:"""

    return await query_ollama(prompt, max_tokens=600, temperature=0.1)
