"""Lean4 + Mathlib bridge for machine-checked mathematical verification.

This module provides an optional Layer 5 verification that translates
math-anything's internal MathSchema representations into Lean4 statements
and verifies them against Mathlib theorems.

Two operational modes:
  - REPL mode: interactive proof verification via lean4-repl subprocess
  - LeanDojo mode: batch theorem querying and premise selection

Requires: lean4, lake, and optionally lean-dojo (pip install lean-dojo).
All Lean4 dependencies are optional -- if unavailable, this layer gracefully
degrades to a SKIP status.
"""

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LeanVerificationStatus(Enum):
    PROVED = "proved"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNKNOWN = "unknown"
    SKIPPED = "skipped"


@dataclass
class LeanVerificationResult:
    status: LeanVerificationStatus
    theorem_name: str
    lean_code: str
    message: str = ""
    proof_time_ms: float = 0.0
    mathlib_matches: List[Dict[str, str]] = field(default_factory=list)


_MATHLIB_MODULE_MAP = {
    "ode": "Mathlib.Analysis.ODE.Basic",
    "pde": "Mathlib.Analysis.PDE.Basic",
    "derivative": "Mathlib.Analysis.Calculus.Deriv",
    "integral": "Mathlib.MeasureTheory.Integral.Bochner",
    "topology": "Mathlib.Topology.Algebra.Module",
    "linear_algebra": "Mathlib.LinearAlgebra.Matrix.Hermitian",
    "normed_space": "Mathlib.Analysis.Normed.Space.FiniteDimension",
    "matrix": "Mathlib.Data.Matrix.Notation",
    "real": "Mathlib.Data.Real.Basic",
    "special_functions": "Mathlib.Analysis.SpecialFunctions.Pow.Real",
    "measure": "Mathlib.MeasureTheory.Measure.MeasureSpaceDef",
    "probability": "Mathlib.Probability.ProbabilityMassFunction.Basic",
    "group": "Mathlib.Algebra.Group.Basic",
    "ring": "Mathlib.Algebra.Ring.Basic",
    "field": "Mathlib.Algebra.Field.Basic",
    "category": "Mathlib.CategoryTheory.Category.Basic",
}

_SYMBOL_TRANSLATIONS = {
    "∇²": "Δ ",
    "∂²": "∂² ",
    "∂": "∂ ",
    "→": " → ",
    "∀": "∀ ",
    "∃": "∃ ",
    "∈": " ∈ ",
    "∉": " ∉ ",
    "⊆": " ⊆ ",
    "≥": " ≥ ",
    "≤": " ≤ ",
    "≠": " ≠ ",
    "∞": "⊤",
    "×": " × ",
    "⊗": " ⊗ ",
    "⊕": " ⊕ ",
    "∧": " ∧ ",
    "∨": " ∨ ",
    "¬": "¬ ",
    "⟹": " → ",
    "⟺": " ↔ ",
    "λ": "fun ",
    "Σ": "∑",
    "∫": "∫",
}


class Lean4Bridge:
    """Bridge between math-anything schemas and Lean4/Mathlib verification.

    Usage:
        bridge = Lean4Bridge()
        result = bridge.verify_statement("theorem foo : 1 + 1 = 2 := by norm_num")
        print(result.status)  # LeanVerificationStatus.PROVED
    """

    def __init__(
        self,
        lean_project_path: Optional[str] = None,
        timeout: int = 30,
        use_lean_dojo: bool = True,
    ):
        self.timeout = timeout
        self.use_lean_dojo = use_lean_dojo
        self._repl_proc: Optional[subprocess.Popen] = None
        self._lean_dojo_cache: Optional[Any] = None
        self._theorem_cache: Dict[str, Any] = {}
        self._available: Optional[bool] = None

        if lean_project_path:
            self.lean_project_path = Path(lean_project_path)
        else:
            self.lean_project_path = Path(__file__).parent / "_lean_project"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            result = subprocess.run(
                ["lean", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self._available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._available = False
        return self._available

    def _ensure_repl(self):
        if self._repl_proc is not None and self._repl_proc.poll() is None:
            return
        if not self.lean_project_path.exists():
            self.lean_project_path.mkdir(parents=True, exist_ok=True)
            lakefile = self.lean_project_path / "lakefile.lean"
            if not lakefile.exists():
                lakefile.write_text(
                    'import Lake\nopen Lake DSL\n\npackage "mathAnythingLean" where\n'
                    '  version := "0.1.0"\n\n'
                    "@[default_target]\nlean_lib MathAnythingLean where\n"
                )
            lean_dir = self.lean_project_path / "MathAnythingLean"
            lean_dir.mkdir(exist_ok=True)
            (lean_dir / "Basic.lean").write_text("import Mathlib.Data.Real.Basic\n\n")

        self._repl_proc = subprocess.Popen(
            ["lake", "env", "lean", "--repl"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.lean_project_path),
        )
        logger.info("Lean4 REPL started at %s", self.lean_project_path)

    def _init_lean_dojo(self):
        if self._lean_dojo_cache is not None:
            return
        try:
            from lean_dojo import LeanGitRepo, traced_repo

            repo = LeanGitRepo("https://github.com/leanprover-community/mathlib4")
            self._lean_dojo_cache = {"repo": repo, "traced": None}
            logger.info("LeanDojo initialized with Mathlib4")
        except ImportError:
            logger.warning("lean-dojo not installed. Install: pip install lean-dojo")
            self.use_lean_dojo = False

    def verify_statement(self, lean_code: str) -> LeanVerificationResult:
        if not self.is_available():
            return LeanVerificationResult(
                status=LeanVerificationStatus.SKIPPED,
                theorem_name=self._extract_theorem_name(lean_code),
                lean_code=lean_code,
                message="Lean4 not available on this system",
            )

        self._ensure_repl()
        start = time.time()

        command = json.dumps({"cmd": lean_code})
        try:
            self._repl_proc.stdin.write(command + "\n\n")
            self._repl_proc.stdin.flush()
        except BrokenPipeError:
            self._repl_proc = None
            return LeanVerificationResult(
                status=LeanVerificationStatus.ERROR,
                theorem_name=self._extract_theorem_name(lean_code),
                lean_code=lean_code,
                message="REPL process crashed",
                proof_time_ms=(time.time() - start) * 1000,
            )

        try:
            response_lines = []
            while True:
                line = self._repl_proc.stdout.readline()
                if not line or line.strip() == "":
                    break
                response_lines.append(line)
                if len(response_lines) > 100:
                    break

            elapsed = (time.time() - start) * 1000
            response_text = "".join(response_lines)

            if not response_text.strip():
                return LeanVerificationResult(
                    status=LeanVerificationStatus.ERROR,
                    theorem_name=self._extract_theorem_name(lean_code),
                    lean_code=lean_code,
                    message="Empty response from REPL",
                    proof_time_ms=elapsed,
                )

            try:
                response = json.loads(response_text)
            except json.JSONDecodeError:
                response = {"raw": response_text}

            has_error = (
                "error" in str(response).lower() or "unsolved" in str(response).lower()
            )
            is_sorry = "sorry" in lean_code

            status = LeanVerificationStatus.ERROR
            if not has_error and not is_sorry:
                status = LeanVerificationStatus.PROVED
            elif has_error and "unsolved" in str(response).lower():
                status = LeanVerificationStatus.UNKNOWN

            return LeanVerificationResult(
                status=status,
                theorem_name=self._extract_theorem_name(lean_code),
                lean_code=lean_code,
                message=str(response)[:500],
                proof_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            if elapsed > self.timeout * 1000:
                return LeanVerificationResult(
                    status=LeanVerificationStatus.TIMEOUT,
                    theorem_name=self._extract_theorem_name(lean_code),
                    lean_code=lean_code,
                    message=f"Timeout after {self.timeout}s",
                    proof_time_ms=elapsed,
                )
            return LeanVerificationResult(
                status=LeanVerificationStatus.ERROR,
                theorem_name=self._extract_theorem_name(lean_code),
                lean_code=lean_code,
                message=str(e),
                proof_time_ms=elapsed,
            )

    def query_mathlib(self, pattern: str) -> List[Dict[str, Any]]:
        if self.use_lean_dojo:
            return self._query_via_lean_dojo(pattern)
        return self._query_via_repl(pattern)

    def _query_via_lean_dojo(self, pattern: str) -> List[Dict[str, Any]]:
        self._init_lean_dojo()
        if not self.use_lean_dojo:
            return self._query_via_repl(pattern)

        from lean_dojo import traced_repo

        repo = self._lean_dojo_cache["repo"]
        if self._lean_dojo_cache["traced"] is None:
            logger.info("Tracing Mathlib4 repo (first run, this is slow)...")
            self._lean_dojo_cache["traced"] = traced_repo(repo)

        traced = self._lean_dojo_cache["traced"]
        regex = re.compile(pattern, re.IGNORECASE)

        results = []
        for theorem in traced.theorems:
            if regex.search(theorem.name):
                results.append(
                    {
                        "name": theorem.name,
                        "statement": str(theorem.statement),
                        "file": str(theorem.file_path),
                    }
                )
            if len(results) >= 50:
                break
        return results

    def _query_via_repl(self, pattern: str) -> List[Dict[str, Any]]:
        if not self.is_available():
            return []
        self._ensure_repl()

        code = f"import Mathlib.Tactic.Find\n#find {pattern}"
        command = json.dumps({"cmd": code})
        try:
            self._repl_proc.stdin.write(command + "\n\n")
            self._repl_proc.stdin.flush()
            response_line = self._repl_proc.stdout.readline()
            if response_line:
                return [{"raw": response_line.strip()}]
        except BrokenPipeError:
            self._repl_proc = None
        return []

    def math_schema_to_lean(self, schema: Any) -> str:
        lines = ["import Mathlib.Data.Real.Basic"]

        model = getattr(schema, "mathematical_model", None)
        if model:
            for eq in getattr(model, "governing_equations", []):
                eq_type = getattr(eq, "equation_type", "")
                for keyword, module in _MATHLIB_MODULE_MAP.items():
                    if keyword in eq_type.lower():
                        if f"import {module}" not in lines:
                            lines.append(f"import {module}")

        lines.append("")
        lines.append("-- Auto-generated from math-anything MathSchema")
        lines.append("")

        model = getattr(schema, "mathematical_model", None)
        if model:
            for i, eq in enumerate(getattr(model, "governing_equations", [])):
                math_form = getattr(eq, "mathematical_form", "")
                lean_stmt = self._translate_equation(math_form, i)
                if lean_stmt:
                    lines.append(lean_stmt)
                    lines.append("")

            for i, bc in enumerate(getattr(model, "boundary_conditions", [])):
                bc_type = getattr(bc, "type", "")
                domain = getattr(bc, "domain", {})
                bc_stmt = self._translate_boundary_condition(bc_type, domain, i)
                if bc_stmt:
                    lines.append(bc_stmt)
                    lines.append("")

        constraints = getattr(schema, "symbolic_constraints", [])
        for i, c in enumerate(constraints):
            expr = getattr(c, "expression", "")
            if expr:
                lines.append(f"-- Constraint {i}: {self._sanitize_lean(expr)}")

        return "\n".join(lines)

    def _translate_equation(self, math_form: str, index: int) -> Optional[str]:
        if not math_form:
            return None

        lean_form = self._sanitize_lean(math_form)

        if "=" in lean_form and "∂" not in lean_form and "∇" not in lean_form:
            lhs, rhs = lean_form.split("=", 1)
            return (
                f"theorem governing_eq_{index} : "
                f"({lhs.strip()}) = ({rhs.strip()}) := by sorry"
            )
        elif "∂" in lean_form or "∇" in lean_form:
            return (
                f"-- Differential equation {index}: {lean_form}\n"
                f"-- (Requires manual Lean4 formalization)"
            )
        return f"-- Equation {index}: {lean_form}"

    def _translate_boundary_condition(
        self, bc_type: str, domain: Dict, index: int
    ) -> Optional[str]:
        if not bc_type:
            return None
        return f"-- Boundary condition {index}: type={bc_type}, " f"domain={domain}"

    def _sanitize_lean(self, text: str) -> str:
        result = text
        for old, new in _SYMBOL_TRANSLATIONS.items():
            result = result.replace(old, new)
        result = re.sub(r"\\[a-zA-Z]+", "", result)
        result = re.sub(r"[{}\\]", " ", result)
        result = re.sub(r"\s+", " ", result).strip()
        return result

    @staticmethod
    def _extract_theorem_name(lean_code: str) -> str:
        match = re.search(r"theorem\s+(\w+)", lean_code)
        return match.group(1) if match else "unnamed"

    def suggest_mathlib_imports(self, schema: Any) -> List[str]:
        suggestions = set()
        model = getattr(schema, "mathematical_model", None)
        if not model:
            return []

        for eq in getattr(model, "governing_equations", []):
            math_form = getattr(eq, "mathematical_form", "").lower()
            eq_type = getattr(eq, "equation_type", "").lower()
            combined = math_form + " " + eq_type

            for keyword, module in _MATHLIB_MODULE_MAP.items():
                if keyword in combined:
                    suggestions.add(module)

        for bc in getattr(model, "boundary_conditions", []):
            bc_type = getattr(bc, "type", "").lower()
            if "periodic" in bc_type:
                suggestions.add("Mathlib.Topology.Periodic")
            elif "dirichlet" in bc_type or "neumann" in bc_type:
                suggestions.add("Mathlib.Analysis.PDE.Basic")

        return sorted(suggestions)

    def close(self):
        if self._repl_proc and self._repl_proc.poll() is None:
            self._repl_proc.terminate()
            try:
                self._repl_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._repl_proc.kill()
            self._repl_proc = None

    def __del__(self):
        self.close()


_lean4_bridge: Optional[Lean4Bridge] = None


def get_lean4_bridge(**kwargs) -> Lean4Bridge:
    global _lean4_bridge
    if _lean4_bridge is None:
        _lean4_bridge = Lean4Bridge(**kwargs)
    return _lean4_bridge


def verify_with_lean4(lean_code: str, timeout: int = 30) -> LeanVerificationResult:
    bridge = get_lean4_bridge(timeout=timeout)
    return bridge.verify_statement(lean_code)


def schema_to_lean(schema: Any) -> str:
    bridge = get_lean4_bridge()
    return bridge.math_schema_to_lean(schema)
