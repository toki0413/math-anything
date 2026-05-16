"""Data firewall for sanitizing user data before sending to LLM APIs.

Strips numeric values from input to reduce data leakage risk while
preserving mathematical structure (operators, variable names, constraint forms).
Also provides encrypted storage for API keys via SecureKeyStore.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_NUMERIC_RE = re.compile(
    r"(?<![a-zA-Z_\.])" r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?" r"(?![a-zA-Z_\.])"
)


class DataFirewall:
    """Sanitizes numeric values from user data before LLM submission.

    Replaces concrete numbers with [COEFF] placeholders so that
    proprietary parameters (ENCUT, SIGMA, etc.) are not leaked to
    third-party APIs, while keeping the mathematical skeleton intact.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def toggle(self) -> None:
        self._enabled = not self._enabled

    def sanitize_for_llm(self, data: str | dict | list) -> str | dict | list:
        if not self._enabled:
            return data

        if isinstance(data, str):
            return _NUMERIC_RE.sub("[COEFF]", data)

        if isinstance(data, dict):
            return {
                k: self.sanitize_for_llm(v) if isinstance(v, (str, dict, list)) else v
                for k, v in data.items()
            }

        if isinstance(data, list):
            return [self.sanitize_for_llm(item) for item in data]

        return data

    def sanitize_messages(self, messages: list[dict]) -> list[dict]:
        if not self._enabled:
            return messages

        sanitized = []
        for msg in messages:
            if msg.get("role") != "user":
                sanitized.append(msg)
                continue

            new_msg = dict(msg)
            content = new_msg.get("content")

            if isinstance(content, str):
                new_msg["content"] = _NUMERIC_RE.sub("[COEFF]", content)
            elif isinstance(content, list):
                new_msg["content"] = self._sanitize_content_parts(content)

            sanitized.append(new_msg)

        return sanitized

    def _sanitize_content_parts(self, parts: list) -> list:
        result = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                new_part = dict(part)
                new_part["text"] = _NUMERIC_RE.sub("[COEFF]", part.get("text", ""))
                result.append(new_part)
            else:
                result.append(part)
        return result


class SecureKeyStore:
    """In-memory encrypted storage for LLM API keys.

    Uses Fernet symmetric encryption when the ``cryptography`` package is
    available.  Falls back to plain-text storage with a warning otherwise.
    The encryption key is generated on first use and is never persisted.
    """

    def __init__(self):
        self._keys: dict[str, Any] = {}
        self._fernet = None
        self._use_encryption = False

        try:
            from cryptography.fernet import Fernet

            self._fernet = Fernet(Fernet.generate_key())
            self._use_encryption = True
        except ImportError:
            logger.warning(
                "cryptography package not found — API keys will be stored "
                "in plain text. Install with: pip install cryptography"
            )

    def store_key(self, provider: str, api_key: str) -> None:
        if self._use_encryption and self._fernet is not None:
            self._keys[provider] = self._fernet.encrypt(api_key.encode())
        else:
            self._keys[provider] = api_key

    def get_key(self, provider: str) -> str | None:
        raw = self._keys.get(provider)
        if raw is None:
            return None

        if self._use_encryption and self._fernet is not None:
            try:
                return self._fernet.decrypt(raw).decode()
            except Exception:
                logger.error("Failed to decrypt key for provider '%s'", provider)
                return None

        return raw

    def has_key(self, provider: str) -> bool:
        return provider in self._keys

    def remove_key(self, provider: str) -> None:
        self._keys.pop(provider, None)
