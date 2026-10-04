"""BYOK provider adapters.

No vendor SDKs — everything is plain `httpx`. Two kinds are supported:

* ``openai_compatible`` — covers OpenAI, DeepSeek, OpenRouter, Groq, Ollama
  (``/v1/chat/completions``), vLLM, LM Studio, and any OpenAI-shaped API.
* ``gemini`` — Google ``generateContent`` REST endpoint.

Keys are resolved in this order:
  1. explicit `api_key` on the provider config (stored locally in
     data/providers.json for the user's own machine),
  2. the `env_key` environment variable of the provider template.
"""
from __future__ import annotations

import abc
import time

import httpx


class ProviderError(RuntimeError):
    pass


class ChatMessage(dict):
    """{"role": ..., "content": ...} convenience."""

    @staticmethod
    def system(content: str) -> "ChatMessage":
        return {"role": "system", "content": content}

    @staticmethod
    def user(content: str) -> "ChatMessage":
        return {"role": "user", "content": content}


class BaseProvider(abc.ABC):
    kind = "base"

    def __init__(self, config: dict):
        self.config = config
        self.id = config.get("id", "unnamed")
        self.base_url = config.get("base_url", "").rstrip("/")
        self.default_model = config.get("default_model", "")
        self.env_key = config.get("env_key", "")

    def resolve_key(self) -> str:
        import os
        key = (self.config.get("api_key") or "").strip()
        if key:
            return key
        if self.env_key:
            env_val = os.environ.get(self.env_key, "").strip()
            if env_val:
                return env_val
        return ""

    # ---------------------------------------------------------------- api --

    @abc.abstractmethod
    async def chat(self, messages: list[dict], *, model: str = "",
                   temperature: float = 0.0, max_tokens: int = 2048,
                   json_mode: bool = False, timeout: float = 120.0) -> str:
        """Send a chat completion, return assistant text."""

    @abc.abstractmethod
    async def health(self, timeout: float = 15.0) -> tuple[bool, str]:
        """Cheap connectivity/auth check."""

    @abc.abstractmethod
    async def list_models(self, timeout: float = 15.0) -> list[str]:
        ...

    # ------------------------------------------------------------ helpers --

    def pick_model(self, model: str | None) -> str:
        return (model or "").strip() or self.default_model


class OpenAICompatibleProvider(BaseProvider):
    kind = "openai_compatible"

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        key = self.resolve_key()
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    async def chat(self, messages, *, model="", temperature=0.0,
                   max_tokens=2048, json_mode=False, timeout=120.0) -> str:
        if not self.base_url:
            raise ProviderError(f"[{self.id}] missing base_url")
        payload: dict = {
            "model": self.pick_model(model),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(), json=payload)
            if r.status_code >= 400:
                raise ProviderError(
                    f"[{self.id}] HTTP {r.status_code}: {r.text[:400]}")
            data = r.json()
            try:
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:
                raise ProviderError(f"[{self.id}] unexpected schema: {e}") from e

    async def health(self, timeout=15.0) -> tuple[bool, str]:
        if not self.base_url:
            return False, "missing base_url"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{self.base_url}/models",
                                     headers=self._headers())
            if r.status_code == 401:
                return False, "401 unauthorized — check API key"
            if r.status_code >= 400:
                return False, f"HTTP {r.status_code}"
            return True, "ok"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    async def list_models(self, timeout=15.0) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{self.base_url}/models",
                                     headers=self._headers())
            if r.status_code >= 400:
                return []
            items = r.json().get("data", [])
            return [m.get("id", "") for m in items if m.get("id")]
        except Exception:
            return []


class GeminiProvider(BaseProvider):
    kind = "gemini"

    async def chat(self, messages, *, model="", temperature=0.0,
                   max_tokens=2048, json_mode=False, timeout=120.0) -> str:
        key = self.resolve_key()
        if not key:
            raise ProviderError(f"[{self.id}] missing GEMINI_API_KEY")
        base = self.base_url or "https://generativelanguage.googleapis.com/v1beta"
        model_id = self.pick_model(model) or "gemini-2.0-flash"
        system_text = "\n".join(m["content"] for m in messages
                                if m.get("role") == "system")
        contents = []
        for m in messages:
            if m.get("role") == "system":
                continue
            role = "model" if m.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        body: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        url = f"{base}/models/{model_id}:generateContent?key={key}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=body)
            if r.status_code >= 400:
                raise ProviderError(f"[{self.id}] HTTP {r.status_code}: {r.text[:400]}")
            data = r.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except Exception as e:
            raise ProviderError(f"[{self.id}] unexpected schema: {e}") from e

    async def health(self, timeout=15.0) -> tuple[bool, str]:
        key = self.resolve_key()
        if not key:
            return False, "missing GEMINI_API_KEY"
        base = self.base_url or "https://generativelanguage.googleapis.com/v1beta"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{base}/models?key={key}")
            if r.status_code >= 400:
                return False, f"HTTP {r.status_code}: {r.text[:200]}"
            return True, "ok"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    async def list_models(self, timeout=15.0) -> list[str]:
        key = self.resolve_key()
        if not key:
            return []
        base = self.base_url or "https://generativelanguage.googleapis.com/v1beta"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{base}/models?key={key}")
            if r.status_code >= 400:
                return []
            return [m["name"].split("/")[-1] for m in r.json().get("models", [])]
        except Exception:
            return []


PROVIDER_CLASSES = {
    "openai_compatible": OpenAICompatibleProvider,
    "gemini": GeminiProvider,
}


def build_provider(config: dict) -> BaseProvider:
    kind = config.get("kind", "openai_compatible")
    cls = PROVIDER_CLASSES.get(kind)
    if cls is None:
        raise ProviderError(f"unknown provider kind: {kind}")
    return cls(config)
