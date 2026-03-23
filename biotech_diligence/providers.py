"""
Provider adapters — drop-in replacements for the Anthropic client.
Allows the diligence agent to run on Ollama (free, local) or Groq (free tier).

All adapters expose the same interface as the Anthropic client:
    client.messages.create(model=..., max_tokens=..., system=..., messages=[...])
    → response.content[0].text
"""

from __future__ import annotations
from dataclasses import dataclass


# ------------------------------------------------------------------
# Shared response wrapper (mimics anthropic.types.Message)
# ------------------------------------------------------------------

@dataclass
class _Content:
    text: str


@dataclass
class _Response:
    content: list[_Content]


class _MessagesNamespace:
    """Wraps an OpenAI-compatible client to match the Anthropic messages API."""

    def __init__(self, openai_client, default_model: str):
        self._client = openai_client
        self._default_model = default_model

    def create(
        self,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict],
        **kwargs,
    ) -> _Response:
        oai_messages = [{"role": "system", "content": system}] + messages
        resp = self._client.chat.completions.create(
            model=model or self._default_model,
            max_tokens=max_tokens,
            messages=oai_messages,
        )
        text = resp.choices[0].message.content or ""
        return _Response(content=[_Content(text=text)])


# ------------------------------------------------------------------
# Ollama — local, 100% free
# ------------------------------------------------------------------

def OllamaClient(model: str = "llama3.1", host: str = "http://localhost:11434"):
    """
    Client for Ollama (local models). Completely free.

    Prerequisites:
        brew install ollama          # or: curl -fsSL https://ollama.com/install.sh | sh
        ollama serve                 # start the Ollama daemon
        ollama pull llama3.1         # or: mistral, qwen2.5, gemma2, etc.

    Args:
        model: Ollama model name (default: llama3.1)
        host:  Ollama server URL (default: http://localhost:11434)
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip3 install openai")

    raw = OpenAI(base_url=f"{host}/v1", api_key="ollama")

    class _OllamaClient:
        messages = _MessagesNamespace(raw, model)

    return _OllamaClient()


# ------------------------------------------------------------------
# Groq — cloud, free tier (requires free account)
# ------------------------------------------------------------------

def GroqClient(api_key: str | None = None, model: str = "llama-3.3-70b-versatile"):
    """
    Client for Groq (cloud inference, free tier).

    Free tier limits: ~14,400 requests/day, 30 req/min.
    Sign up at https://console.groq.com — no credit card required.

    Args:
        api_key: Groq API key (or set GROQ_API_KEY env var)
        model:   Groq model (default: llama-3.1-70b-versatile)

    Good free models on Groq:
        llama-3.3-70b-versatile   ← recommended, strong reasoning
        llama-3.1-8b-instant      ← faster, lighter
        mixtral-8x7b-32768        ← good for long context
        gemma2-9b-it              ← Google's model
    """
    import os
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip3 install openai")

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "Groq API key not set. Get a free key at https://console.groq.com "
            "then set GROQ_API_KEY=... in your .env file."
        )

    raw = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)

    class _GroqClient:
        messages = _MessagesNamespace(raw, model)

    return _GroqClient()


# ------------------------------------------------------------------
# Gemini — cloud, free tier (requires free Google account)
# ------------------------------------------------------------------

def GeminiClient(api_key: str | None = None, model: str = "gemini-2.0-flash"):
    """
    Client for Google Gemini (free tier via OpenAI-compatible endpoint).

    Free tier: 15 requests/min, 1M tokens/day.
    Get a free key at https://aistudio.google.com/app/apikey

    Args:
        api_key: Gemini API key (or set GEMINI_API_KEY env var)
        model:   Gemini model (default: gemini-1.5-flash)
    """
    import os
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip3 install openai")

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "Gemini API key not set. Get a free key at https://aistudio.google.com/app/apikey "
            "then set GEMINI_API_KEY=... in your .env file."
        )

    raw = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=key,
    )

    class _GeminiClient:
        messages = _MessagesNamespace(raw, model)

    return _GeminiClient()


# ------------------------------------------------------------------
# Factory — auto-select provider from env
# ------------------------------------------------------------------

PROVIDER_HELP = """
No LLM provider configured. Set one of these in your .env file:

  ANTHROPIC_API_KEY=sk-ant-...      ← Claude (paid, best quality)
  GROQ_API_KEY=gsk_...              ← Groq free tier (llama3.1-70b, great quality)
  GEMINI_API_KEY=AIza...            ← Gemini free tier (gemini-1.5-flash)
  OLLAMA_MODEL=llama3.1             ← Local Ollama (100% free, needs install)

Get free keys:
  Groq:   https://console.groq.com        (no credit card)
  Gemini: https://aistudio.google.com     (Google account only)
  Ollama: brew install ollama && ollama pull llama3.1
"""


def auto_client() -> tuple[object, str]:
    """
    Auto-detect available provider from environment variables.
    Returns (client, model_name).
    Priority: Anthropic → Gemini → Groq → Ollama
    """
    import os

    # 1. Anthropic (paid)
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        return anthropic.Anthropic(), model

    # 2. Gemini (free tier — prioritised for recency of training data)
    if os.environ.get("GEMINI_API_KEY"):
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        return GeminiClient(model=model), model

    # 3. Groq (free tier — fallback)
    if os.environ.get("GROQ_API_KEY"):
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return GroqClient(model=model), model

    # 4. Ollama (local, fully free)
    if os.environ.get("OLLAMA_MODEL") or _ollama_running():
        model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OllamaClient(model=model, host=host), model

    raise ValueError(PROVIDER_HELP)


def _ollama_running() -> bool:
    """Check if Ollama is running locally."""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=1)
        return True
    except Exception:
        return False
