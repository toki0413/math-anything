"""LLM client wrapper for math-anything.

Users must provide their own API keys. We do not host or proxy LLM calls.
Supports: OpenAI, Anthropic (Claude), DeepSeek, OpenRouter.
"""

import os
from typing import Dict, List, Optional

from .config import get_config


class LLMError(Exception):
    """LLM API error."""

    pass


class LLMClient:
    """Unified LLM client."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        cfg = get_config().get("llm", {})
        self.provider = (provider or cfg.get("provider", "openai")).lower()
        self.api_key = api_key or cfg.get("api_key", "")
        self.base_url = base_url or cfg.get("base_url", "")
        self.model = model or cfg.get("model", "gpt-4o-mini")
        self.temperature = cfg.get("temperature", 0.3)
        self.max_tokens = cfg.get("max_tokens", 4096)

        if not self.api_key:
            raise LLMError(
                "No API key configured. Set via:\n"
                "  math-anything config set llm.api_key YOUR_KEY\n"
                "Or environment variable: OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY"
            )

        self._client = None

    def _get_client(self):
        """Lazy initialize underlying client."""
        if self._client is not None:
            return self._client

        if self.provider in ("openai", "openrouter"):
            try:
                import openai
            except ImportError:
                raise LLMError("openai package not installed. Run: pip install openai")
            base = self.base_url or ("https://openrouter.ai/api/v1" if self.provider == "openrouter" else None)
            self._client = openai.OpenAI(api_key=self.api_key, base_url=base)

        elif self.provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise LLMError("anthropic package not installed. Run: pip install anthropic")
            self._client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url or None)

        elif self.provider == "deepseek":
            try:
                import openai
            except ImportError:
                raise LLMError("openai package not installed. Run: pip install openai")
            base = self.base_url or "https://api.deepseek.com"
            self._client = openai.OpenAI(api_key=self.api_key, base_url=base)

        else:
            raise LLMError(f"Unsupported provider: {self.provider}")

        return self._client

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send chat completion request."""
        client = self._get_client()
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "anthropic":
            system_msg = ""
            user_messages = []
            for m in messages:
                if m["role"] == "system":
                    system_msg = m["content"]
                else:
                    user_messages.append(m)
            # Anthropic requires alternating user/assistant
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tok,
                temperature=temp,
                system=system_msg,
                messages=user_messages,
            )
            return response.content[0].text  # type: ignore[no-any-return]

        else:
            # OpenAI-compatible
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                stream=stream,
            )
            if stream:
                chunks = []
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)
                return "".join(chunks)
            return response.choices[0].message.content  # type: ignore[no-any-return]

    def embed(
        self,
        texts: List[str],
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Get embeddings for texts."""
        cfg = get_config().get("embedding", {})
        prov = (provider or cfg.get("provider", "openai")).lower()
        key = api_key or cfg.get("api_key", "")
        mdl = model or cfg.get("model", "text-embedding-3-small")

        if not key:
            key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise LLMError("No embedding API key configured.")

        if prov in ("openai", "deepseek"):
            try:
                import openai
            except ImportError:
                raise LLMError("openai package not installed. Run: pip install openai")
            base = cfg.get("base_url") or None
            client = openai.OpenAI(api_key=key, base_url=base)
            response = client.embeddings.create(input=texts, model=mdl)
            return [item.embedding for item in response.data]

        raise LLMError(f"Unsupported embedding provider: {prov}")


def quick_chat(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """Quick one-shot chat."""
    client = LLMClient()
    return client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )
