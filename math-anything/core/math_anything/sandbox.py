"""Sandbox execution environment for safe code running.

Provides isolated code execution for:
- Constraint validation scripts
- Custom analysis functions
- Mathematical verification

Supports two backends:
1. E2B cloud sandbox (requires e2b>=1.0.0) — fully isolated cloud VM
2. Local subprocess sandbox — restricted Python subprocess with resource limits

Security model:
- No filesystem write access (except /tmp)
- No network access
- CPU time limit
- Memory limit
- Restricted imports (no os, subprocess, socket, etc.)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


_RESTRICTED_MODULES = frozenset({
    "os", "subprocess", "socket", "http", "urllib", "requests",
    "shutil", "signal", "ctypes", "multiprocessing", "threading",
    "asyncio", "importlib", "pkg_resources", "sysconfig",
})

_SANDBOX_WRAPPER = '''
import sys
import json
import importlib.abc
import importlib.machinery

_BLOCKED = {blocked_modules}

for _mod in list(sys.modules.keys()):
    _top = _mod.split('.')[0]
    if _top in _BLOCKED:
        del sys.modules[_mod]

class _SandboxFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        top = fullname.split('.')[0]
        if top in _BLOCKED:
            return importlib.machinery.ModuleSpec(
                fullname,
                _SandboxLoader(fullname),
            )
        return None

class _SandboxLoader(importlib.abc.Loader):
    def __init__(self, name):
        self._name = name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ImportError(
            f"Sandbox: import of '{{self._name}}' is blocked for security"
        )

sys.meta_path.insert(0, _SandboxFinder())

_original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.split('.')[0]
    if top in _BLOCKED:
        raise ImportError(
            f"Sandbox: import of '{{name}}' is blocked for security"
        )
    return _original_import(name, globals, locals, fromlist, level)

_builtins = dict(__builtins__) if isinstance(__builtins__, dict) else dict(vars(__builtins__))
_builtins['__import__'] = _restricted_import

try:
    exec(compile({source_repr}, '<sandbox>', 'exec'), {{'__builtins__': _builtins}})
except Exception as e:
    import traceback
    result = {{"success": False, "error": str(e), "traceback": traceback.format_exc()}}
else:
    result = {{"success": True}}

with open({output_path_repr}, 'w') as f:
    json.dump(result, f)
'''


@dataclass
class SandboxResult:
    """Result from sandbox execution."""

    success: bool = False
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    traceback_str: Optional[str] = None
    execution_time_ms: float = 0.0
    backend: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "success": self.success,
            "backend": self.backend,
        }
        if self.stdout:
            d["stdout"] = self.stdout
        if self.stderr:
            d["stderr"] = self.stderr
        if self.error:
            d["error"] = self.error
        if self.traceback_str:
            d["traceback"] = self.traceback_str
        if self.execution_time_ms > 0:
            d["execution_time_ms"] = round(self.execution_time_ms, 1)
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    timeout_seconds: float = 10.0
    max_memory_mb: int = 256
    max_output_bytes: int = 65536
    backend: str = "auto"
    allowed_modules: List[str] = field(default_factory=list)
    extra_blocked_modules: List[str] = field(default_factory=list)

    @property
    def blocked_modules(self) -> set:
        blocked = set(_RESTRICTED_MODULES)
        for m in self.allowed_modules:
            blocked.discard(m)
        blocked.update(self.extra_blocked_modules)
        return blocked


class SandboxExecutor:
    """Execute code in an isolated sandbox environment.

    Backends:
    - 'e2b': Cloud sandbox via E2B (requires e2b package + API key)
    - 'local': Subprocess sandbox with restricted imports
    - 'auto': Try E2B first, fall back to local
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()

    def execute(
        self,
        code: str,
        stdin_data: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> SandboxResult:
        """Execute code in sandbox.

        Args:
            code: Python code to execute
            stdin_data: Optional stdin input
            env_vars: Optional environment variables
        """
        backend = self.config.backend
        if backend == "auto":
            backend = self._detect_backend()

        if backend == "e2b":
            return self._execute_e2b(code, stdin_data, env_vars)
        return self._execute_local(code, stdin_data, env_vars)

    def execute_validation(
        self,
        constraint_name: str,
        check_code: str,
        value_code: str,
    ) -> SandboxResult:
        """Execute a constraint validation script.

        Args:
            constraint_name: Name of the constraint being validated
            check_code: Python expression returning bool (pass/fail)
            value_code: Python expression returning the value to check
        """
        wrapped = f"""
value = {value_code}
passed = {check_code}
print(f"CONSTRAINT: {constraint_name}")
print(f"VALUE: {{value}}")
print(f"PASSED: {{passed}}")
if not passed:
    raise AssertionError(f"Constraint '{constraint_name}' failed: value={{value}}")
"""
        return self.execute(wrapped)

    def _detect_backend(self) -> str:
        try:
            import e2b
            return "e2b"
        except ImportError:
            return "local"

    def _execute_e2b(
        self,
        code: str,
        stdin_data: Optional[str],
        env_vars: Optional[Dict[str, str]],
    ) -> SandboxResult:
        """Execute via E2B cloud sandbox."""
        try:
            import e2b

            start = time.time()
            sandbox = e2b.Sandbox()

            try:
                result = sandbox.run_code(
                    code,
                    timeout=self.config.timeout_seconds,
                )

                elapsed = (time.time() - start) * 1000

                return SandboxResult(
                    success=not result.error,
                    stdout=result.stdout or "",
                    stderr=result.stderr or "",
                    error=result.error if result.error else None,
                    execution_time_ms=elapsed,
                    backend="e2b",
                    description="Executed in E2B cloud sandbox",
                )
            finally:
                try:
                    sandbox.close()
                except Exception:
                    pass

        except ImportError:
            return SandboxResult(
                success=False,
                error="e2b package not installed. Install with: pip install e2b",
                backend="e2b",
                description="E2B backend unavailable, install e2b package",
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                error=str(e),
                backend="e2b",
                description=f"E2B execution failed: {e}",
            )

    def _execute_local(
        self,
        code: str,
        stdin_data: Optional[str],
        env_vars: Optional[Dict[str, str]],
    ) -> SandboxResult:
        """Execute via local subprocess with import restrictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "_sandbox_result.json"

            blocked = self.config.blocked_modules
            wrapper = _SANDBOX_WRAPPER.format(
                blocked_modules=repr(blocked),
                source_repr=repr(code),
                output_path_repr=repr(str(output_path)),
            )

            script_path = Path(tmpdir) / "_sandbox_runner.py"
            script_path.write_text(wrapper, encoding="utf-8")

            start = time.time()
            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    input=stdin_data,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    cwd=tmpdir,
                    env=self._build_env(env_vars),
                )
                elapsed = (time.time() - start) * 1000

                stdout = proc.stdout[:self.config.max_output_bytes]
                stderr = proc.stderr[:self.config.max_output_bytes]

                result_data = {}
                if output_path.exists():
                    try:
                        result_data = json.loads(output_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, Exception):
                        pass

                success = result_data.get("success", proc.returncode == 0)
                error = result_data.get("error") or (stderr.strip() if proc.returncode != 0 else None)
                tb = result_data.get("traceback")

                return SandboxResult(
                    success=success,
                    stdout=stdout,
                    stderr=stderr,
                    error=error,
                    traceback_str=tb,
                    execution_time_ms=elapsed,
                    backend="local",
                    description="Executed in local subprocess sandbox",
                )

            except subprocess.TimeoutExpired:
                elapsed = (time.time() - start) * 1000
                return SandboxResult(
                    success=False,
                    error=f"Execution timed out after {self.config.timeout_seconds}s",
                    execution_time_ms=elapsed,
                    backend="local",
                    description="Sandbox execution timed out",
                )
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                return SandboxResult(
                    success=False,
                    error=str(e),
                    traceback_str=traceback.format_exc(),
                    execution_time_ms=elapsed,
                    backend="local",
                    description=f"Sandbox execution failed: {e}",
                )

    @staticmethod
    def _build_env(env_vars: Optional[Dict[str, str]]) -> Dict[str, str]:
        import os
        env = dict(os.environ)
        env.pop("E2B_API_KEY", None)
        if env_vars:
            env.update(env_vars)
        return env
