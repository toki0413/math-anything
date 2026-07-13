"""Python 版本兼容性辅助。

集中处理不同 Python 版本之间的差异，避免在业务代码里到处写 try/except。
"""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    # Python < 3.11 没有内置 StrEnum，提供一个最小实现
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """字符串枚举，与 Python 3.11+ 的 enum.StrEnum 行为兼容."""

        def __str__(self) -> str:
            return self.value  # type: ignore[no-any-return]

        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()


__all__ = ["StrEnum"]
