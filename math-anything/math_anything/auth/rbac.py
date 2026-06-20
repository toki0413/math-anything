"""基于角色的访问控制 (RBAC).

角色层级：
  admin   → 全部权限
  analyst → extract + verify + discover
  viewer  → 只读 (engines + health)

权限：
  extract    → 提取数学结构
  verify     → 验证结构
  discover   → 符号回归
  admin      → 管理操作
  view       → 只读访问
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Role(Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Permission(Enum):
    EXTRACT = "extract"
    VERIFY = "verify"
    DISCOVER = "discover"
    ADMIN = "admin"
    VIEW = "view"


# 角色 → 权限映射
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: {Permission.EXTRACT, Permission.VERIFY, Permission.DISCOVER, Permission.ADMIN, Permission.VIEW},
    Role.ANALYST: {Permission.EXTRACT, Permission.VERIFY, Permission.DISCOVER, Permission.VIEW},
    Role.VIEWER: {Permission.VIEW},
}


@dataclass
class User:
    """用户."""

    user_id: str
    username: str
    role: Role
    api_key_hash: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        return permission in ROLE_PERMISSIONS[self.role]


@dataclass
class AuditEntry:
    """审计日志条目."""

    timestamp: float = field(default_factory=time.time)
    user_id: str = ""
    action: str = ""
    resource: str = ""
    result: str = ""  # success/failure
    details: dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """审计日志.

    记录所有 API 调用，支持：
    - 按用户/时间/操作查询
    - 持久化到文件
    - 自动轮转
    """

    def __init__(self, log_dir: Path | None = None, max_entries: int = 10000):
        self._log_dir = log_dir
        self._max_entries = max_entries
        self._entries: list[AuditEntry] = []

    def log(
        self, user_id: str, action: str, resource: str, result: str = "success", details: dict[str, Any] | None = None
    ) -> None:
        """记录审计条目."""
        entry = AuditEntry(
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            details=details or {},
        )
        self._entries.append(entry)

        # 轮转
        if len(self._entries) > self._max_entries:
            self._flush()

    def _flush(self) -> None:
        """将旧条目写入文件."""
        if self._log_dir is None:
            # 无文件存储，简单截断
            self._entries = self._entries[-self._max_entries // 2 :]
            return

        self._log_dir.mkdir(parents=True, exist_ok=True)
        flush_count = len(self._entries) // 2
        to_flush = self._entries[:flush_count]
        self._entries = self._entries[flush_count:]

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = self._log_dir / f"audit_{timestamp}.jsonl"
        with open(log_file, "a") as f:
            for entry in to_flush:
                f.write(
                    json.dumps(
                        {
                            "timestamp": entry.timestamp,
                            "user_id": entry.user_id,
                            "action": entry.action,
                            "resource": entry.resource,
                            "result": entry.result,
                            "details": entry.details,
                        }
                    )
                    + "\n"
                )

    def query(
        self,
        user_id: str | None = None,
        action: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """查询审计日志."""
        results = self._entries
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if action:
            results = [e for e in results if e.action == action]
        if since:
            results = [e for e in results if e.timestamp >= since]
        return results[-limit:]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "max_entries": self._max_entries,
        }


class RBACManager:
    """RBAC 管理器.

    管理用户、角色、权限和审计日志。
    """

    def __init__(self, audit_log_dir: Path | None = None):
        self._users: dict[str, User] = {}
        self._api_key_map: dict[str, str] = {}  # api_key_hash → user_id
        self._audit = AuditLogger(log_dir=audit_log_dir)

    def create_user(self, username: str, role: Role, api_key: str | None = None) -> User:
        """创建用户."""
        user_id = f"user_{len(self._users) + 1}"
        api_key_hash = ""
        if api_key:
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:32]
            self._api_key_map[api_key_hash] = user_id

        user = User(user_id=user_id, username=username, role=role, api_key_hash=api_key_hash)
        self._users[user_id] = user
        self._audit.log(user_id, "create_user", f"role={role.value}", "success")
        return user

    def authenticate(self, api_key: str) -> User | None:
        """通过 API key 认证."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:32]
        user_id = self._api_key_map.get(key_hash)
        if user_id and user_id in self._users:
            return self._users[user_id]
        return None

    def authorize(self, user: User, permission: Permission) -> bool:
        """检查用户权限."""
        has_perm = user.has_permission(permission)
        self._audit.log(
            user.user_id,
            "authorize",
            f"permission={permission.value}",
            "success" if has_perm else "denied",
        )
        return has_perm

    def get_user(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    @property
    def audit(self) -> AuditLogger:
        return self._audit

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_users": len(self._users),
            "users_by_role": {role.value: sum(1 for u in self._users.values() if u.role == role) for role in Role},
            "audit": self._audit.stats,
        }
