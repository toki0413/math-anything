"""LLM Prompt 压缩器 — 减少 token 消耗.

策略：
  1. 数学符号压缩：将冗长的数学描述替换为紧凑符号
  2. 结构化模板：用 JSON schema 替代自然语言描述
  3. 上下文裁剪：只保留与当前查询相关的上下文
  4. 缓存友好：生成确定性的 prompt 以提高缓存命中率
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# 数学符号压缩映射
_SYMBOL_COMPRESSION = {
    "governing_equations": "eqs",
    "boundary_conditions": "bc",
    "constitutive_relations": "cr",
    "numerical_methods": "nm",
    "conservation_properties": "cp",
    "computational_graph": "cg",
    "partial_differential_equation": "PDE",
    "ordinary_differential_equation": "ODE",
    "navier_stokes": "NS",
    "computational_fluid_dynamics": "CFD",
    "density_functional_theory": "DFT",
    "finite_element_method": "FEM",
    "finite_volume_method": "FVM",
    "molecular_dynamics": "MD",
}


@dataclass(slots=True)
class PromptStats:
    """Prompt 统计信息."""

    original_tokens: int = 0
    compressed_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - self.compressed_tokens / self.original_tokens

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total


class PromptCompressor:
    """LLM Prompt 压缩器."""

    def __init__(self):
        self._stats = PromptStats()
        self._symbol_map = _SYMBOL_COMPRESSION.copy()

    def compress(self, prompt: str) -> str:
        """压缩 prompt，减少 token 数量.

        策略：
        1. 替换冗长术语为缩写
        2. 移除冗余空白
        3. 结构化 JSON 输出
        """
        original_tokens = self._estimate_tokens(prompt)

        result = prompt
        # 1. 术语压缩
        for long_form, short_form in self._symbol_map.items():
            result = result.replace(long_form, short_form)

        # 2. 空白压缩
        result = re.sub(r"\n{3,}", "\n\n", result)
        result = re.sub(r" {2,}", " ", result)

        # 3. 移除注释行
        result = re.sub(r"^#.*$", "", result, flags=re.MULTILINE)
        result = re.sub(r"\n{3,}", "\n\n", result)

        compressed_tokens = self._estimate_tokens(result)
        self._stats.original_tokens += original_tokens
        self._stats.compressed_tokens += compressed_tokens

        return result.strip()

    def compress_schema(self, schema_dict: dict[str, Any]) -> str:
        """将 MathSchema 压缩为紧凑 JSON 字符串."""
        compressed = {}
        for key, value in schema_dict.items():
            short_key = self._symbol_map.get(key, key)
            compressed[short_key] = value
        return json.dumps(compressed, separators=(",", ":"), ensure_ascii=False)

    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数量 (粗略: 1 token ≈ 4 字符)."""
        return max(1, len(text) // 4)

    @property
    def stats(self) -> PromptStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = PromptStats()
