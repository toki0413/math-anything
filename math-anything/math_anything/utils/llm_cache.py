"""LLM 语义缓存 — 提高缓存命中率.

核心思想：
  1. 对 prompt 进行语义归一化（去空白、排序 JSON key、术语标准化）
  2. 计算归一化 prompt 的哈希作为缓存 key
  3. 支持相似度匹配（可选，基于 Jaccard 相似度）
  4. LRU 淘汰策略 + TTL 过期

缓存命中率优化策略：
  - 确定性 prompt 生成（排序 key、标准化格式）
  - 语义归一化（同义词映射、大小写归一）
  - 参数化缓存（将数值参数与模板分离）
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CacheEntry:
    """缓存条目."""

    key: str
    prompt_hash: str
    response: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: float = 3600.0  # 默认 1 小时
    _normalized_prompt: str = ""  # 用于相似度搜索

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class SemanticCache:
    """LLM 语义缓存.

    特性：
    - 语义归一化：同义 prompt 映射到同一缓存条目
    - LRU 淘汰：容量满时淘汰最久未访问的条目
    - TTL 过期：条目超时自动失效
    - 统计信息：命中率、平均响应时间等
    """

    def __init__(self, max_size: int = 256, default_ttl: float = 3600.0):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        self._synonym_map: dict[str, str] = {
            "vasp": "vasp",
            "Vienna Ab initio Simulation Package": "vasp",
            "lammps": "lammps",
            "Large-scale Atomic/Molecular Massively Parallel Simulator": "lammps",
            "abaqus": "abaqus",
            "ansys": "ansys",
            "openfoam": "openfoam",
            "OpenFOAM": "openfoam",
            "quantum_espresso": "quantum_espresso",
            "Quantum ESPRESSO": "quantum_espresso",
            "QE": "quantum_espresso",
            "gaussian": "gaussian",
            "Gaussian": "gaussian",
        }

    def normalize(self, prompt: str) -> str:
        """语义归一化.

        1. 小写化
        2. 同义词映射
        3. 空白标准化
        4. JSON key 排序
        """
        result = prompt.lower().strip()
        result = " ".join(result.split())  # 标准化空白

        # 同义词映射
        for synonym, canonical in self._synonym_map.items():
            result = result.replace(synonym.lower(), canonical)

        # 尝试解析并排序 JSON
        try:
            obj = json.loads(result)
            result = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        except (json.JSONDecodeError, ValueError):
            pass

        return result

    def _hash(self, normalized_prompt: str) -> str:
        """计算归一化 prompt 的 SHA-256 哈希."""
        return hashlib.sha256(normalized_prompt.encode()).hexdigest()[:16]

    def get(self, prompt: str) -> Any | None:
        """查询缓存.

        Returns:
            缓存的响应，或 None（缓存未命中）
        """
        normalized = self.normalize(prompt)
        key = self._hash(normalized)

        if key in self._cache:
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None
            # LRU: 移到末尾
            self._cache.move_to_end(key)
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._hits += 1
            return entry.response

        self._misses += 1
        return None

    def set(self, prompt: str, response: Any, ttl: float | None = None) -> None:
        """存入缓存."""
        normalized = self.normalize(prompt)
        key = self._hash(normalized)

        # LRU 淘汰
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(
            key=key,
            prompt_hash=key,
            response=response,
            ttl_seconds=ttl or self._default_ttl,
            _normalized_prompt=normalized,
        )

    def make_param_aware_key(self, engine: str, params: dict) -> str:
        """参数感知缓存键 — 提高缓存命中率.

        策略：
        1. 引擎名标准化
        2. 参数 key 排序
        3. 数值参数四舍五入（避免浮点噪声导致缓存未命中）
        4. int/float 统一（520 和 520.0 生成相同 key）
        """
        # 标准化引擎名
        canonical_engine = self._synonym_map.get(engine.lower(), engine.lower())

        # 排序 + 数值标准化
        normalized_params = {}
        for k in sorted(params.keys()):
            v = params[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                # 统一为 float 再四舍五入，确保 520 和 520.0 一致
                v = float(v)
                if v != 0:
                    sig = 6 - int(f"{abs(v):.0e}".split("e")[1])
                    v = round(v, sig)
                else:
                    v = 0.0
            elif isinstance(v, dict):
                # 递归处理嵌套 dict
                v = dict(sorted(v.items()))
            normalized_params[k] = v

        key_str = f"{canonical_engine}:{json.dumps(normalized_params, sort_keys=True, separators=(',', ':'))}"
        return self._hash(key_str)

    def invalidate(self, prompt: str) -> bool:
        """使缓存条目失效."""
        normalized = self.normalize(prompt)
        key = self._hash(normalized)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """清空缓存."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def cleanup_expired(self) -> int:
        """清理过期条目."""
        expired_keys = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)

    def similarity_search(self, prompt: str, threshold: float = 0.8) -> Any | None:
        """相似度搜索 — 找到语义相近的缓存条目.

        使用 Jaccard 相似度比较 token 集合。
        这比精确匹配有更高的缓存命中率。

        Args:
            prompt: 查询 prompt
            threshold: 相似度阈值 (0.0-1.0)

        Returns:
            最相似条目的响应，或 None
        """
        query_tokens = set(self.normalize(prompt).split())

        best_match = None
        best_score = 0.0

        for key, entry in self._cache.items():
            if not entry._normalized_prompt:
                continue
            if entry.is_expired:
                continue

            cached_tokens = set(entry._normalized_prompt.split())

            # Jaccard 相似度
            intersection = query_tokens & cached_tokens
            union = query_tokens | cached_tokens
            if not union:
                continue

            score = len(intersection) / len(union)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = entry

        if best_match:
            best_match.last_accessed = time.time()
            best_match.access_count += 1
            self._hits += 1
            return best_match.response

        self._misses += 1
        return None

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "total_requests": self._hits + self._misses,
        }
