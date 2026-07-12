"""Unit tests for auth/rbac.py — RBAC manager, users, roles, audit log."""

import hashlib
import json
import time
from pathlib import Path

import pytest

from math_anything.auth.rbac import (
    ROLE_PERMISSIONS,
    AuditEntry,
    AuditLogger,
    Permission,
    RBACManager,
    Role,
    User,
)

# ── Role / Permission enums ──


class TestRoleEnum:
    def test_role_values(self):
        assert Role.ADMIN.value == "admin"
        assert Role.ANALYST.value == "analyst"
        assert Role.VIEWER.value == "viewer"

    def test_role_count(self):
        assert len(list(Role)) == 3


class TestPermissionEnum:
    def test_permission_values(self):
        assert Permission.EXTRACT.value == "extract"
        assert Permission.VERIFY.value == "verify"
        assert Permission.DISCOVER.value == "discover"
        assert Permission.ADMIN.value == "admin"
        assert Permission.VIEW.value == "view"

    def test_permission_count(self):
        assert len(list(Permission)) == 5


class TestRolePermissions:
    def test_admin_has_all_permissions(self):
        perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.EXTRACT in perms
        assert Permission.VERIFY in perms
        assert Permission.DISCOVER in perms
        assert Permission.ADMIN in perms
        assert Permission.VIEW in perms

    def test_analyst_has_extract_verify_discover_view(self):
        perms = ROLE_PERMISSIONS[Role.ANALYST]
        assert Permission.EXTRACT in perms
        assert Permission.VERIFY in perms
        assert Permission.DISCOVER in perms
        assert Permission.VIEW in perms
        assert Permission.ADMIN not in perms

    def test_viewer_has_only_view(self):
        perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.VIEW in perms
        assert Permission.EXTRACT not in perms
        assert Permission.VERIFY not in perms
        assert Permission.DISCOVER not in perms
        assert Permission.ADMIN not in perms

    def test_role_hierarchy_admin_superset_of_analyst(self):
        assert ROLE_PERMISSIONS[Role.ANALYST].issubset(ROLE_PERMISSIONS[Role.ADMIN])

    def test_role_hierarchy_analyst_superset_of_viewer(self):
        assert ROLE_PERMISSIONS[Role.VIEWER].issubset(ROLE_PERMISSIONS[Role.ANALYST])


# ── User ──


class TestUser:
    def test_user_creation_minimal(self):
        u = User(user_id="u1", username="alice", role=Role.ADMIN)
        assert u.user_id == "u1"
        assert u.username == "alice"
        assert u.role == Role.ADMIN
        assert u.api_key_hash == ""
        assert u.created_at > 0
        assert u.metadata == {}

    def test_user_creation_with_api_key(self):
        u = User(
            user_id="u2",
            username="bob",
            role=Role.ANALYST,
            api_key_hash="abc123",
        )
        assert u.api_key_hash == "abc123"

    def test_user_has_permission_admin(self):
        u = User(user_id="u", username="a", role=Role.ADMIN)
        assert u.has_permission(Permission.ADMIN) is True
        assert u.has_permission(Permission.EXTRACT) is True
        assert u.has_permission(Permission.VIEW) is True

    def test_user_has_permission_viewer(self):
        u = User(user_id="u", username="v", role=Role.VIEWER)
        assert u.has_permission(Permission.VIEW) is True
        assert u.has_permission(Permission.EXTRACT) is False
        assert u.has_permission(Permission.ADMIN) is False

    def test_user_has_permission_analyst(self):
        u = User(user_id="u", username="an", role=Role.ANALYST)
        assert u.has_permission(Permission.EXTRACT) is True
        assert u.has_permission(Permission.VERIFY) is True
        assert u.has_permission(Permission.DISCOVER) is True
        assert u.has_permission(Permission.ADMIN) is False

    def test_user_metadata_default(self):
        u = User(user_id="u", username="x", role=Role.VIEWER)
        assert u.metadata == {}

    def test_user_metadata_custom(self):
        u = User(
            user_id="u",
            username="x",
            role=Role.VIEWER,
            metadata={"team": "research"},
        )
        assert u.metadata["team"] == "research"


# ── AuditEntry ──


class TestAuditEntry:
    def test_entry_defaults(self):
        e = AuditEntry()
        assert e.timestamp > 0
        assert e.user_id == ""
        assert e.action == ""
        assert e.resource == ""
        assert e.result == ""
        assert e.details == {}

    def test_entry_with_values(self):
        e = AuditEntry(
            user_id="u1",
            action="extract",
            resource="vasp",
            result="success",
            details={"k": "v"},
        )
        assert e.user_id == "u1"
        assert e.action == "extract"
        assert e.result == "success"


# ── AuditLogger ──


class TestAuditLogger:
    def test_creates_empty(self):
        log = AuditLogger()
        assert log.stats["total_entries"] == 0

    def test_log_appends_entry(self):
        log = AuditLogger()
        log.log("u1", "extract", "vasp")
        assert log.stats["total_entries"] == 1

    def test_log_with_details(self):
        log = AuditLogger()
        log.log("u1", "extract", "vasp", details={"key": "value"})
        entries = log.query()
        assert entries[0].details == {"key": "value"}

    def test_log_with_result(self):
        log = AuditLogger()
        log.log("u1", "extract", "vasp", result="denied")
        entries = log.query()
        assert entries[0].result == "denied"

    def test_query_by_user(self):
        log = AuditLogger()
        log.log("u1", "extract", "vasp")
        log.log("u2", "extract", "lammps")
        result = log.query(user_id="u1")
        assert len(result) == 1
        assert result[0].user_id == "u1"

    def test_query_by_action(self):
        log = AuditLogger()
        log.log("u1", "extract", "vasp")
        log.log("u1", "verify", "vasp")
        result = log.query(action="extract")
        assert len(result) == 1
        assert result[0].action == "extract"

    def test_query_by_since(self):
        log = AuditLogger()
        log.log("u1", "extract", "vasp")
        time.sleep(0.01)
        cutoff = time.time()
        log.log("u1", "verify", "vasp")
        result = log.query(since=cutoff)
        assert all(e.timestamp >= cutoff for e in result)

    def test_query_limit(self):
        log = AuditLogger()
        for i in range(10):
            log.log("u1", "extract", f"r{i}")
        result = log.query(limit=3)
        assert len(result) == 3

    def test_stats_max_entries(self):
        log = AuditLogger(max_entries=100)
        assert log.stats["max_entries"] == 100


class TestAuditLoggerFlush:
    def test_flush_truncates_when_no_dir(self):
        log = AuditLogger(max_entries=4)
        for i in range(6):
            log.log("u", "a", f"r{i}")
        # After flush, entries should be truncated to max_entries//2
        assert log.stats["total_entries"] <= 4

    def test_flush_to_file(self, tmp_path):
        log_dir = tmp_path / "audit"
        log = AuditLogger(log_dir=log_dir, max_entries=4)
        for i in range(6):
            log.log("u", "a", f"r{i}")
        # Should have written a jsonl file
        files = list(log_dir.glob("audit_*.jsonl"))
        assert len(files) >= 1
        # File should contain JSON lines
        content = files[0].read_text()
        lines = [line for line in content.strip().split("\n") if line]
        assert len(lines) >= 1
        parsed = json.loads(lines[0])
        assert "user_id" in parsed
        assert "action" in parsed


# ── RBACManager ──


class TestRBACManagerCreation:
    def test_creates_empty(self):
        mgr = RBACManager()
        assert mgr.stats["total_users"] == 0

    def test_has_audit_logger(self):
        mgr = RBACManager()
        assert mgr.audit is not None

    def test_creates_with_audit_dir(self, tmp_path):
        mgr = RBACManager(audit_log_dir=tmp_path / "audit")
        assert mgr is not None


class TestRBACManagerCreateUser:
    def test_create_user_returns_user(self):
        mgr = RBACManager()
        u = mgr.create_user("alice", Role.ADMIN)
        assert isinstance(u, User)
        assert u.username == "alice"
        assert u.role == Role.ADMIN

    def test_create_user_assigns_id(self):
        mgr = RBACManager()
        u1 = mgr.create_user("a", Role.VIEWER)
        u2 = mgr.create_user("b", Role.VIEWER)
        assert u1.user_id == "user_1"
        assert u2.user_id == "user_2"

    def test_create_user_with_api_key(self):
        mgr = RBACManager()
        u = mgr.create_user("alice", Role.ADMIN, api_key="secret123")
        assert u.api_key_hash != ""
        assert len(u.api_key_hash) == 32

    def test_create_user_without_api_key(self):
        mgr = RBACManager()
        u = mgr.create_user("alice", Role.ADMIN)
        assert u.api_key_hash == ""

    def test_create_user_logs_audit(self):
        mgr = RBACManager()
        mgr.create_user("alice", Role.ADMIN)
        entries = mgr.audit.query(action="create_user")
        assert len(entries) == 1

    def test_create_multiple_users(self):
        mgr = RBACManager()
        mgr.create_user("a", Role.ADMIN)
        mgr.create_user("b", Role.ANALYST)
        mgr.create_user("c", Role.VIEWER)
        assert mgr.stats["total_users"] == 3


class TestRBACManagerAuthenticate:
    def test_authenticate_valid_key(self):
        mgr = RBACManager()
        mgr.create_user("alice", Role.ADMIN, api_key="mykey")
        user = mgr.authenticate("mykey")
        assert user is not None
        assert user.username == "alice"

    def test_authenticate_invalid_key(self):
        mgr = RBACManager()
        mgr.create_user("alice", Role.ADMIN, api_key="mykey")
        user = mgr.authenticate("wrongkey")
        assert user is None

    def test_authenticate_no_users(self):
        mgr = RBACManager()
        user = mgr.authenticate("anykey")
        assert user is None

    def test_authenticate_multiple_users(self):
        mgr = RBACManager()
        mgr.create_user("alice", Role.ADMIN, api_key="key1")
        mgr.create_user("bob", Role.VIEWER, api_key="key2")
        assert mgr.authenticate("key1").username == "alice"
        assert mgr.authenticate("key2").username == "bob"

    def test_api_key_hash_is_sha256_truncated(self):
        mgr = RBACManager()
        mgr.create_user("alice", Role.ADMIN, api_key="mykey")
        expected = hashlib.sha256("mykey".encode()).hexdigest()[:32]
        user = mgr.authenticate("mykey")
        assert user.api_key_hash == expected


class TestRBACManagerAuthorize:
    def test_authorize_admin_all_permissions(self):
        mgr = RBACManager()
        u = mgr.create_user("alice", Role.ADMIN)
        assert mgr.authorize(u, Permission.EXTRACT) is True
        assert mgr.authorize(u, Permission.ADMIN) is True
        assert mgr.authorize(u, Permission.VIEW) is True

    def test_authorize_viewer_denied_extract(self):
        mgr = RBACManager()
        u = mgr.create_user("v", Role.VIEWER)
        assert mgr.authorize(u, Permission.VIEW) is True
        assert mgr.authorize(u, Permission.EXTRACT) is False

    def test_authorize_logs_audit(self):
        mgr = RBACManager()
        u = mgr.create_user("alice", Role.ADMIN)
        mgr.authorize(u, Permission.EXTRACT)
        entries = mgr.audit.query(action="authorize")
        assert len(entries) == 1
        assert entries[0].result == "success"

    def test_authorize_denied_logs_audit(self):
        mgr = RBACManager()
        u = mgr.create_user("v", Role.VIEWER)
        mgr.authorize(u, Permission.EXTRACT)
        entries = mgr.audit.query(action="authorize")
        assert entries[0].result == "denied"


class TestRBACManagerGetUser:
    def test_get_user_existing(self):
        mgr = RBACManager()
        created = mgr.create_user("alice", Role.ADMIN)
        fetched = mgr.get_user(created.user_id)
        assert fetched is not None
        assert fetched.username == "alice"

    def test_get_user_nonexistent(self):
        mgr = RBACManager()
        assert mgr.get_user("nonexistent") is None


class TestRBACManagerStats:
    def test_stats_total_users(self):
        mgr = RBACManager()
        mgr.create_user("a", Role.ADMIN)
        mgr.create_user("b", Role.VIEWER)
        assert mgr.stats["total_users"] == 2

    def test_stats_users_by_role(self):
        mgr = RBACManager()
        mgr.create_user("a", Role.ADMIN)
        mgr.create_user("b", Role.ANALYST)
        mgr.create_user("c", Role.VIEWER)
        mgr.create_user("d", Role.VIEWER)
        by_role = mgr.stats["users_by_role"]
        assert by_role["admin"] == 1
        assert by_role["analyst"] == 1
        assert by_role["viewer"] == 2

    def test_stats_includes_audit(self):
        mgr = RBACManager()
        mgr.create_user("a", Role.ADMIN)
        stats = mgr.stats
        assert "audit" in stats
        assert stats["audit"]["total_entries"] >= 1


class TestRBACAccessControl:
    """End-to-end access control scenarios."""

    def test_viewer_cannot_extract(self):
        mgr = RBACManager()
        mgr.create_user("v", Role.VIEWER, api_key="vkey")
        authenticated = mgr.authenticate("vkey")
        assert authenticated is not None
        assert mgr.authorize(authenticated, Permission.EXTRACT) is False

    def test_admin_can_extract(self):
        mgr = RBACManager()
        mgr.create_user("a", Role.ADMIN, api_key="akey")
        authenticated = mgr.authenticate("akey")
        assert mgr.authorize(authenticated, Permission.EXTRACT) is True

    def test_analyst_can_verify_but_not_admin(self):
        mgr = RBACManager()
        mgr.create_user("an", Role.ANALYST, api_key="ankey")
        authenticated = mgr.authenticate("ankey")
        assert mgr.authorize(authenticated, Permission.VERIFY) is True
        assert mgr.authorize(authenticated, Permission.ADMIN) is False

    def test_role_hierarchy_access(self):
        """Admin > Analyst > Viewer in terms of permissions."""
        mgr = RBACManager()
        admin = mgr.create_user("admin", Role.ADMIN, api_key="admin_key")
        analyst = mgr.create_user("analyst", Role.ANALYST, api_key="an_key")
        viewer = mgr.create_user("viewer", Role.VIEWER, api_key="v_key")

        # All can view
        assert mgr.authorize(admin, Permission.VIEW) is True
        assert mgr.authorize(analyst, Permission.VIEW) is True
        assert mgr.authorize(viewer, Permission.VIEW) is True

        # Only admin and analyst can extract
        assert mgr.authorize(admin, Permission.EXTRACT) is True
        assert mgr.authorize(analyst, Permission.EXTRACT) is True
        assert mgr.authorize(viewer, Permission.EXTRACT) is False

        # Only admin can admin
        assert mgr.authorize(admin, Permission.ADMIN) is True
        assert mgr.authorize(analyst, Permission.ADMIN) is False
        assert mgr.authorize(viewer, Permission.ADMIN) is False
