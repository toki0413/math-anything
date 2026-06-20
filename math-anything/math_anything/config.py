"""Configuration management for math-anything.

Users provide their own API keys for cloud LLM services.
We do NOT provide computation or API access directly.
"""

import base64
import getpass
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CONFIG = {
    "llm": {
        "provider": "openai",  # openai, anthropic, deepseek, openrouter
        "api_key": "",
        "base_url": "",  # for custom endpoints / proxies
        "model": "gpt-4o-mini",  # default model
        "temperature": 0.3,
        "max_tokens": 4096,
    },
    "embedding": {
        "provider": "openai",
        "api_key": "",
        "base_url": "",
        "model": "text-embedding-3-small",
    },
    "rag": {
        "enabled": False,
        "db_path": "",  # auto-set to ~/.math-anything/rag_db
        "top_k": 5,
    },
    "agent": {
        "auto_confirm": False,  # if True, agent runs without asking
        "max_iterations": 10,
    },
    "watch": {
        "debounce_seconds": 1.0,
        "auto_check": True,
        "auto_explain": False,
    },
}


class Config:
    """Configuration manager."""

    def __init__(self):
        self.config_dir = Path.home() / ".math-anything"
        self.config_file = self.config_dir / "config.json"
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                self.data = self._merge_defaults(DEFAULT_CONFIG, loaded)
            except Exception:
                self.data = dict(DEFAULT_CONFIG)
        else:
            self.data = dict(DEFAULT_CONFIG)
            self.save()

    def save(self) -> None:
        """Save configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value by dot path, e.g. 'llm.api_key'."""
        keys = key_path.split(".")
        value = self.data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any) -> None:
        """Set config value by dot path."""
        keys = key_path.split(".")
        target = self.data
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
        self.save()

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration with env var fallback."""
        cfg = dict(self.data.get("llm", {}))
        # Fallback to environment variables
        if not cfg.get("api_key"):
            cfg["api_key"] = os.environ.get("OPENAI_API_KEY", "")
        if not cfg.get("api_key") and cfg.get("provider") == "anthropic":
            cfg["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
        if not cfg.get("api_key") and cfg.get("provider") == "deepseek":
            cfg["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")
        return cfg

    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration with env var fallback."""
        cfg = dict(self.data.get("embedding", {}))
        if not cfg.get("api_key"):
            cfg["api_key"] = os.environ.get("OPENAI_API_KEY", "")
        return cfg

    @staticmethod
    def _merge_defaults(defaults: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge override into defaults."""
        result = dict(defaults)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._merge_defaults(result[key], value)
            else:
                result[key] = value
        return result


class SecureConfig:
    """安全的配置存储 — API key 加密存储.

    使用机器特定的密钥对敏感信息进行 XOR 加密后存储，
    防止明文泄露。优先从环境变量读取。
    """

    def __init__(self, config_dir: Path | None = None):
        self._config_dir = config_dir or Path.home() / ".math-anything"
        self._secrets_file = self._config_dir / ".secrets"
        self._key = self._derive_key()
        self._cache: Dict[str, str] = {}
        self._load()

    def _derive_key(self) -> bytes:
        """从机器信息派生加密密钥."""
        machine_id = f"{platform.node()}-{getpass.getuser()}-{platform.machine()}"
        return hashlib.sha256(machine_id.encode()).digest()

    def _xor_encrypt(self, data: bytes) -> bytes:
        """简单 XOR 加密（非生产级，但防止明文存储）."""
        key = self._key
        return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))

    def set_secret(self, key: str, value: str) -> None:
        """存储加密的密钥."""
        self._cache[key] = value
        self._save()

    def get_secret(self, key: str) -> str | None:
        """获取解密的密钥，优先从环境变量获取."""
        # 优先检查环境变量
        env_key = key.upper().replace("-", "_")
        env_val = os.environ.get(env_key)
        if env_val:
            return env_val
        # 然后检查加密存储
        return self._cache.get(key)

    def delete_secret(self, key: str) -> None:
        """删除存储的密钥."""
        self._cache.pop(key, None)
        self._save()

    def _load(self) -> None:
        """从加密文件加载密钥."""
        if not self._secrets_file.exists():
            self._cache = {}
            return
        try:
            raw = self._secrets_file.read_bytes()
            decrypted = self._xor_encrypt(base64.b64decode(raw))
            self._cache = json.loads(decrypted.decode("utf-8"))
        except Exception:
            self._cache = {}

    def _save(self) -> None:
        """将密钥加密后写入文件."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self._cache, ensure_ascii=False).encode("utf-8")
        encrypted = self._xor_encrypt(data)
        self._secrets_file.write_bytes(base64.b64encode(encrypted))
        # 限制文件权限（仅当前用户可读写）
        try:
            os.chmod(self._secrets_file, 0o600)
        except OSError:
            pass


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
