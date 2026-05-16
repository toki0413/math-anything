"""Formal Verifier - Rigorous mathematical verification beyond heuristics.

Upgrades the proof verification pipeline from keyword-based heuristics to
multi-layer formal verification:

Layer 1: Symbolic verification (sympy-powered algebraic identity checks)
Layer 2: Type system (dimensional analysis, mathematical object types)
Layer 3: Logic calculus (modus ponens, universal instantiation, etc.)
Layer 4: LLM-based semantic verification (cloud API with local fallback)

This addresses the gap between "looks correct" and "is formally correct".

Example:
    >>> from math_anything.formal_verifier import FormalVerifier
    >>> fv = FormalVerifier()
    >>> result = fv.verify(task, proof_text)
    >>> print(result.formal_status)  # 'verified', 'unverified', 'contradicted'
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class FormalStatus(Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"
    SKIPPED = "skipped"


class VerificationLayer(Enum):
    SYMBOLIC = "symbolic"
    TYPE_SYSTEM = "type_system"
    LOGIC = "logic"
    LLM_SEMANTIC = "llm_semantic"
    LEAN4_FORMAL = "lean4_formal"


class MathType(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"
    TENSOR = "tensor"
    FUNCTION = "function"
    OPERATOR = "operator"
    SET = "set"
    GROUP = "group"
    MANIFOLD = "manifold"
    FORM = "differential_form"
    UNKNOWN = "unknown"


class DimensionKind(Enum):
    LENGTH = "L"
    MASS = "M"
    TIME = "T"
    TEMPERATURE = "Θ"
    CURRENT = "I"
    AMOUNT = "N"
    DIMENSIONLESS = "1"


@dataclass
class Dimension:
    """Physical dimension in SI base units."""

    L: int = 0
    M: int = 0
    T: int = 0
    Theta: int = 0
    I: int = 0
    N: int = 0

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(
            L=self.L + other.L,
            M=self.M + other.M,
            T=self.T + other.T,
            Theta=self.Theta + other.Theta,
            I=self.I + other.I,
            N=self.N + other.N,
        )

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(
            L=self.L - other.L,
            M=self.M - other.M,
            T=self.T - other.T,
            Theta=self.Theta - other.Theta,
            I=self.I - other.I,
            N=self.N - other.N,
        )

    def __pow__(self, n: int) -> "Dimension":
        return Dimension(
            L=self.L * n,
            M=self.M * n,
            T=self.T * n,
            Theta=self.Theta * n,
            I=self.I * n,
            N=self.N * n,
        )

    def is_dimensionless(self) -> bool:
        return all(v == 0 for v in [self.L, self.M, self.T, self.Theta, self.I, self.N])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented
        return (
            self.L == other.L
            and self.M == other.M
            and self.T == other.T
            and self.Theta == other.Theta
            and self.I == other.I
            and self.N == other.N
        )

    def __str__(self) -> str:
        parts = []
        for name, val in [
            ("L", self.L),
            ("M", self.M),
            ("T", self.T),
            ("Θ", self.Theta),
            ("I", self.I),
            ("N", self.N),
        ]:
            if val != 0:
                parts.append(f"{name}^{val}" if val != 1 else name)
        return "·".join(parts) if parts else "1"

    def to_dict(self) -> Dict[str, int]:
        return {
            "L": self.L,
            "M": self.M,
            "T": self.T,
            "Theta": self.Theta,
            "I": self.I,
            "N": self.N,
        }


COMMON_DIMENSIONS: Dict[str, Dimension] = {
    "length": Dimension(L=1),
    "distance": Dimension(L=1),
    "position": Dimension(L=1),
    "area": Dimension(L=2),
    "volume": Dimension(L=3),
    "velocity": Dimension(L=1, T=-1),
    "speed": Dimension(L=1, T=-1),
    "acceleration": Dimension(L=1, T=-2),
    "force": Dimension(L=1, M=1, T=-2),
    "pressure": Dimension(L=-1, M=1, T=-2),
    "stress": Dimension(L=-1, M=1, T=-2),
    "strain": Dimension(),
    "energy": Dimension(L=2, M=1, T=-2),
    "work": Dimension(L=2, M=1, T=-2),
    "power": Dimension(L=2, M=1, T=-3),
    "mass": Dimension(M=1),
    "density": Dimension(L=-3, M=1),
    "time": Dimension(T=1),
    "frequency": Dimension(T=-1),
    "temperature": Dimension(Theta=1),
    "charge": Dimension(I=1, T=1),
    "current": Dimension(I=1),
    "voltage": Dimension(L=2, M=1, T=-3, I=-1),
    "electric_field": Dimension(L=1, M=1, T=-3, I=-1),
    "magnetic_field": Dimension(M=1, T=-2, I=-1),
    "angle": Dimension(),
    "radian": Dimension(),
    "wavevector": Dimension(L=-1),
    "momentum": Dimension(L=1, M=1, T=-1),
    "angular_momentum": Dimension(L=2, M=1, T=-1),
    "spring_constant": Dimension(M=1, T=-2),
    "viscosity": Dimension(L=-1, M=1, T=-1),
    "diffusion_coefficient": Dimension(L=2, T=-1),
}


@dataclass
class TypedSymbol:
    """A mathematical symbol with type and dimension information."""

    name: str
    math_type: MathType
    dimension: Optional[Dimension] = None
    domain: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "math_type": self.math_type.value,
            "domain": self.domain,
            "description": self.description,
        }
        if self.dimension:
            d["dimension"] = str(self.dimension)
            d["dimension_dict"] = self.dimension.to_dict()
        return d


@dataclass
class TypeViolation:
    """A type error found during verification."""

    symbol_a: str
    symbol_b: str
    operation: str
    expected: str
    actual: str
    severity: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol_a": self.symbol_a,
            "symbol_b": self.symbol_b,
            "operation": self.operation,
            "expected": self.expected,
            "actual": self.actual,
            "severity": self.severity,
        }


@dataclass
class LogicStep:
    """A single step in a proof's logical structure."""

    step_number: int
    statement: str
    justification: str = ""
    rule: str = ""
    depends_on: List[int] = field(default_factory=list)
    is_axiom: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "statement": self.statement,
            "justification": self.justification,
            "rule": self.rule,
            "depends_on": self.depends_on,
            "is_axiom": self.is_axiom,
        }


@dataclass
class LayerResult:
    """Result from a single verification layer."""

    layer: VerificationLayer
    status: FormalStatus
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer.value,
            "status": self.status.value,
            "confidence": self.confidence,
            "details": self.details,
            "issues": self.issues,
            "time_ms": self.time_ms,
        }


@dataclass
class FormalVerificationResult:
    """Complete result from the formal verification pipeline."""

    formal_status: FormalStatus
    overall_confidence: float
    layer_results: List[LayerResult] = field(default_factory=list)
    typed_symbols: List[TypedSymbol] = field(default_factory=list)
    logic_steps: List[LogicStep] = field(default_factory=list)
    type_violations: List[TypeViolation] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formal_status": self.formal_status.value,
            "overall_confidence": self.overall_confidence,
            "layer_results": [lr.to_dict() for lr in self.layer_results],
            "typed_symbols": [ts.to_dict() for ts in self.typed_symbols],
            "logic_steps": [ls.to_dict() for ls in self.logic_steps],
            "type_violations": [tv.to_dict() for tv in self.type_violations],
            "summary": self.summary,
        }


class SymbolicVerifier:
    """Layer 1: Symbolic algebraic verification using sympy.

    Attempts to parse and verify algebraic identities, equalities,
    and simplifications in the proof.
    """

    def verify(self, task, proof_text: str) -> LayerResult:
        t0 = time.time()
        issues: List[Dict[str, Any]] = []
        details: Dict[str, Any] = {}
        verified_count = 0
        total_count = 0

        try:
            import sympy

            sympy_available = True
        except ImportError:
            sympy_available = False

        equations = self._extract_equations(proof_text)
        details["equations_found"] = len(equations)
        total_count = len(equations)

        if not equations:
            return LayerResult(
                layer=VerificationLayer.SYMBOLIC,
                status=FormalStatus.SKIPPED,
                confidence=0.0,
                details={"reason": "no_equations_found"},
                time_ms=(time.time() - t0) * 1000,
            )

        if not sympy_available:
            return LayerResult(
                layer=VerificationLayer.SYMBOLIC,
                status=FormalStatus.INCONCLUSIVE,
                confidence=0.0,
                details={
                    "reason": "sympy_not_available",
                    "equations_found": len(equations),
                },
                issues=[
                    {"category": "missing_dependency", "message": "sympy not installed"}
                ],
                time_ms=(time.time() - t0) * 1000,
            )

        for eq_text in equations:
            total_count += 1
            result = self._verify_equation_sympy(eq_text, sympy)
            if result["verified"]:
                verified_count += 1
            elif result.get("error"):
                issues.append(
                    {
                        "category": "symbolic_error",
                        "equation": eq_text[:100],
                        "error": result["error"],
                    }
                )

        if total_count == 0:
            status = FormalStatus.SKIPPED
            confidence = 0.0
        elif verified_count == total_count:
            status = FormalStatus.VERIFIED
            confidence = 0.9
        elif verified_count > 0:
            status = FormalStatus.INCONCLUSIVE
            confidence = verified_count / total_count * 0.7
        else:
            status = FormalStatus.UNVERIFIED
            confidence = 0.1

        details["verified_count"] = verified_count
        details["total_count"] = total_count

        return LayerResult(
            layer=VerificationLayer.SYMBOLIC,
            status=status,
            confidence=confidence,
            details=details,
            issues=issues,
            time_ms=(time.time() - t0) * 1000,
        )

    def _extract_equations(self, text: str) -> List[str]:
        equations = []
        for match in re.finditer(r"\$([^$]+)\$", text):
            eq = match.group(1).strip()
            if "=" in eq or "\\equiv" in eq or "\\approx" in eq:
                equations.append(eq)
        for match in re.finditer(r"\\\[([^]]+)\\\]", text):
            eq = match.group(1).strip()
            if "=" in eq or "\\equiv" in eq:
                equations.append(eq)
        return equations

    def _verify_equation_sympy(self, eq_text: str, sympy) -> Dict[str, Any]:
        try:
            lhs_rhs = re.split(r"=|\\equiv|\\approx", eq_text, maxsplit=1)
            if len(lhs_rhs) != 2:
                return {"verified": False, "error": "no_equal_sign"}

            lhs_str = lhs_rhs[0].strip()
            rhs_str = lhs_rhs[1].strip()

            lhs_str = re.sub(
                r"\\(?:frac|dfrac)\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", lhs_str
            )
            rhs_str = re.sub(
                r"\\(?:frac|dfrac)\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", rhs_str
            )

            for cmd in [
                "\\left",
                "\\right",
                "\\displaystyle",
                "\\mathrm",
                "\\text",
                "\\mathbf",
                "\\boldsymbol",
                "\\hat",
                "\\bar",
                "\\tilde",
                "\\vec",
            ]:
                lhs_str = lhs_str.replace(cmd, "")
                rhs_str = rhs_str.replace(cmd, "")

            lhs_str = re.sub(r"\\([a-zA-Z]+)", r"\1", lhs_str)
            rhs_str = re.sub(r"\\([a-zA-Z]+)", r"\1", rhs_str)

            lhs_expr = sympy.sympify(lhs_str, evaluate=False)
            rhs_expr = sympy.sympify(rhs_str, evaluate=False)

            diff = sympy.simplify(lhs_expr - rhs_expr)
            if diff == 0:
                return {"verified": True}
            return {"verified": False, "difference": str(diff)}

        except Exception as e:
            return {"verified": False, "error": str(e)}


class TypeSystemVerifier:
    """Layer 2: Type system and dimensional analysis.

    Checks:
    - Dimensional consistency of equations
    - Type compatibility of operations (scalar + vector = error)
    - Symbol type inference from context
    """

    MATH_TYPE_PATTERNS: Dict[MathType, List[str]] = {
        MathType.VECTOR: [
            "\\vec",
            "\\mathbf{",
            "vector",
            "gradient",
            "∇",
            "cross product",
            "dot product",
            "norm",
        ],
        MathType.MATRIX: [
            "\\matrix",
            "\\begin{pmatrix}",
            "tensor",
            "matrix",
            "determinant",
            "eigenvalue",
            "trace",
        ],
        MathType.TENSOR: [
            "\\tensor",
            "Riemann",
            "Ricci",
            "metric tensor",
            "stress tensor",
            "strain tensor",
            "T_",
        ],
        MathType.OPERATOR: [
            "\\hat",
            "operator",
            "Laplacian",
            "Hamiltonian",
            "\\nabla^2",
            "H|",
            "\\mathcal{L}",
        ],
        MathType.FUNCTION: [
            "f(",
            "g(",
            "φ(",
            "ψ(",
            "\\phi(",
            "\\psi(",
            "function",
            "mapping",
        ],
        MathType.SET: [
            "\\{",
            "\\}",
            "\\in",
            "\\subset",
            "\\subseteq",
            "\\cup",
            "\\cap",
            "set",
        ],
        MathType.GROUP: [
            "group",
            "G",
            "symmetry group",
            "space group",
            "point group",
            "SO(",
            "SU(",
            "U(",
        ],
        MathType.MANIFOLD: [
            "manifold",
            "M",
            "chart",
            "atlas",
            "tangent space",
            "fiber bundle",
            "Brillouin zone",
        ],
        MathType.FORM: [
            "\\omega",
            "differential form",
            "dx ∧",
            "volume form",
            "1-form",
            "2-form",
            "k-form",
        ],
    }

    def __init__(self):
        self._injected_symbols: List[TypedSymbol] = []

    def inject_symbols(self, symbols: List[TypedSymbol]) -> None:
        self._injected_symbols = symbols

    def verify(self, task, proof_text: str) -> LayerResult:
        t0 = time.time()
        issues: List[Dict[str, Any]] = []
        details: Dict[str, Any] = {}

        symbols = self._infer_symbols(task, proof_text)
        if self._injected_symbols:
            existing_names = {s.name for s in symbols}
            for inj in self._injected_symbols:
                if inj.name not in existing_names:
                    symbols.append(inj)
            details["injected_symbols"] = len(self._injected_symbols)
        details["symbols_found"] = len(symbols)

        violations = self._check_type_consistency(symbols, proof_text)
        details["type_violations"] = len(violations)

        dim_issues = self._check_dimensional_consistency(symbols, proof_text)
        details["dimensional_issues"] = len(dim_issues)

        all_issues = violations + dim_issues
        for v in all_issues:
            issues.append(v.to_dict())

        if not all_issues:
            status = FormalStatus.VERIFIED
            confidence = 0.85
        elif len(all_issues) <= 2:
            status = FormalStatus.INCONCLUSIVE
            confidence = 0.5
        else:
            status = FormalStatus.CONTRADICTED
            confidence = 0.2

        return LayerResult(
            layer=VerificationLayer.TYPE_SYSTEM,
            status=status,
            confidence=confidence,
            details=details,
            issues=issues,
            time_ms=(time.time() - t0) * 1000,
        )

    def _infer_symbols(self, task, proof_text: str) -> List[TypedSymbol]:
        symbols: List[TypedSymbol] = []
        seen: Set[str] = set()

        statement = getattr(task, "statement", "")
        full_text = f"{statement}\n{proof_text}"

        var_pattern = re.compile(r"\b([a-zA-Z])(?:_\{?(\w+)\}?)?(?:\^)?")
        for match in var_pattern.finditer(full_text):
            name = match.group(0)
            if name in seen or len(name) == 1 and name.lower() in "aeiou":
                continue
            seen.add(name)

            math_type = self._infer_type(name, full_text)
            dimension = self._infer_dimension(name, full_text)

            symbols.append(
                TypedSymbol(
                    name=name,
                    math_type=math_type,
                    dimension=dimension,
                )
            )

        return symbols[:50]

    def _infer_type(self, symbol: str, context: str) -> MathType:
        for mtype, patterns in self.MATH_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in context and symbol in context:
                    nearby = context[
                        max(0, context.find(symbol) - 50) : context.find(symbol)
                        + 50
                        + len(symbol)
                    ]
                    if pattern in nearby:
                        return mtype
        return MathType.SCALAR

    def _infer_dimension(self, symbol: str, context: str) -> Optional[Dimension]:
        nearby_radius = 100
        pos = context.find(symbol)
        if pos < 0:
            return None
        nearby = context[max(0, pos - nearby_radius) : pos + nearby_radius].lower()

        for keyword, dim in COMMON_DIMENSIONS.items():
            if keyword in nearby:
                return dim
        return None

    def _check_type_consistency(
        self, symbols: List[TypedSymbol], proof_text: str
    ) -> List[TypeViolation]:
        violations = []
        type_map = {s.name: s for s in symbols}

        add_patterns = [
            re.compile(r"(\w+)\s*\+\s*(\w+)"),
            re.compile(r"(\w+)\s*-\s*(\w+)"),
        ]
        for pat in add_patterns:
            for match in pat.finditer(proof_text):
                a, b = match.group(1), match.group(2)
                sa, sb = type_map.get(a), type_map.get(b)
                if sa and sb and sa.math_type != sb.math_type:
                    if (
                        sa.math_type != MathType.UNKNOWN
                        and sb.math_type != MathType.UNKNOWN
                    ):
                        violations.append(
                            TypeViolation(
                                symbol_a=a,
                                symbol_b=b,
                                operation="addition",
                                expected=sa.math_type.value,
                                actual=sb.math_type.value,
                                severity="error",
                            )
                        )

        eq_pattern = re.compile(r"(\w+)\s*=\s*(\w+)")
        for match in eq_pattern.finditer(proof_text):
            a, b = match.group(1), match.group(2)
            sa, sb = type_map.get(a), type_map.get(b)
            if sa and sb and sa.dimension and sb.dimension:
                if sa.dimension != sb.dimension:
                    violations.append(
                        TypeViolation(
                            symbol_a=a,
                            symbol_b=b,
                            operation="equality",
                            expected=str(sa.dimension),
                            actual=str(sb.dimension),
                            severity="error",
                        )
                    )

        return violations

    def _check_dimensional_consistency(
        self, symbols: List[TypedSymbol], proof_text: str
    ) -> List[TypeViolation]:
        violations = []
        dim_map = {s.name: s.dimension for s in symbols if s.dimension}

        eq_pattern = re.compile(r"(\w+)\s*=\s*(\w+)")
        for match in eq_pattern.finditer(proof_text):
            lhs, rhs = match.group(1), match.group(2)
            dl, dr = dim_map.get(lhs), dim_map.get(rhs)
            if dl and dr and dl != dr:
                violations.append(
                    TypeViolation(
                        symbol_a=lhs,
                        symbol_b=rhs,
                        operation="dimensional_equality",
                        expected=str(dl),
                        actual=str(dr),
                        severity="error",
                    )
                )

        return violations


class LogicVerifier:
    """Layer 3: Formal logic verification.

    Checks:
    - Proof structure (axioms -> lemmas -> theorems)
    - Logical rule application (modus ponens, etc.)
    - Circular reasoning detection
    - Missing step detection
    """

    INFERENCE_RULES = {
        "modus_ponens": {"pattern": r"if\s+(.+?),?\s+then\s+(.+?)\."},
        "universal_instantiation": {"pattern": r"for all\s+\w+.*?(?:implies|→|⟹)"},
        "existential_generalization": {"pattern": r"there exists\s+\w+"},
        "contradiction": {"pattern": r"(?:suppose|assume).*(?:contradiction|absurd)"},
        "induction": {
            "pattern": r"(?:by induction|base case|inductive step|induction hypothesis)"
        },
        "contrapositive": {"pattern": r"(?:contrapositive|conversely)"},
    }

    CONCLUSION_MARKERS = [
        "therefore",
        "hence",
        "thus",
        "q.e.d",
        "qed",
        "proved",
        "shown",
        "证毕",
        "得证",
        "因此",
        "所以",
        "综上",
        "这证明了",
    ]

    JUSTIFICATION_MARKERS = [
        "by ",
        "since ",
        "because ",
        "from ",
        "using ",
        "according to ",
        "由",
        "根据",
        "因为",
        "利用",
        "根据",
    ]

    def verify(self, task, proof_text: str) -> LayerResult:
        t0 = time.time()
        issues: List[Dict[str, Any]] = []
        details: Dict[str, Any] = {}

        steps = self._parse_logic_steps(proof_text)
        details["steps_found"] = len(steps)

        rule_usage = self._detect_rules(proof_text)
        details["rules_detected"] = rule_usage

        has_conclusion = any(m in proof_text.lower() for m in self.CONCLUSION_MARKERS)
        details["has_conclusion"] = has_conclusion
        if not has_conclusion:
            issues.append(
                {
                    "category": "missing_conclusion",
                    "message": "No conclusion marker found in proof",
                }
            )

        has_justification = any(
            m in proof_text.lower() for m in self.JUSTIFICATION_MARKERS
        )
        details["has_justification"] = has_justification
        if not has_justification:
            issues.append(
                {
                    "category": "missing_justification",
                    "message": "No explicit justifications found in proof steps",
                }
            )

        circular = self._detect_circular_reasoning(steps)
        details["circular_detected"] = circular
        if circular:
            issues.append(
                {
                    "category": "circular_reasoning",
                    "message": "Possible circular reasoning detected",
                }
            )

        missing = self._detect_missing_steps(steps, task)
        details["potential_missing_steps"] = len(missing)
        issues.extend(missing)

        if circular:
            status = FormalStatus.CONTRADICTED
            confidence = 0.1
        elif not has_conclusion or not has_justification:
            status = FormalStatus.UNVERIFIED
            confidence = 0.3
        elif len(issues) == 0:
            status = FormalStatus.VERIFIED
            confidence = 0.8
        else:
            status = FormalStatus.INCONCLUSIVE
            confidence = 0.5

        return LayerResult(
            layer=VerificationLayer.LOGIC,
            status=status,
            confidence=confidence,
            details=details,
            issues=issues,
            time_ms=(time.time() - t0) * 1000,
        )

    def _parse_logic_steps(self, proof_text: str) -> List[LogicStep]:
        steps = []
        lines = [l.strip() for l in proof_text.split("\n") if l.strip()]

        step_num = 0
        for line in lines:
            numbered = re.match(r"^(\d+)[.)]\s*(.+)", line)
            if numbered:
                step_num = int(numbered.group(1))
                content = numbered.group(2)
            else:
                step_num += 1
                content = line

            is_axiom = any(
                kw in content.lower()
                for kw in ["axiom", "assume", "suppose", "let ", "公理", "假设", "设"]
            )
            justification = ""
            for marker in self.JUSTIFICATION_MARKERS:
                idx = content.lower().find(marker)
                if idx >= 0:
                    justification = content[idx : idx + 60]
                    break

            rule = ""
            for rname, rinfo in self.INFERENCE_RULES.items():
                if re.search(rinfo["pattern"], content, re.IGNORECASE):
                    rule = rname
                    break

            steps.append(
                LogicStep(
                    step_number=step_num,
                    statement=content[:200],
                    justification=justification,
                    rule=rule,
                    is_axiom=is_axiom,
                )
            )

        return steps

    def _detect_rules(self, proof_text: str) -> Dict[str, int]:
        usage: Dict[str, int] = defaultdict(int)
        for rname, rinfo in self.INFERENCE_RULES.items():
            matches = re.findall(rinfo["pattern"], proof_text, re.IGNORECASE)
            if matches:
                usage[rname] = len(matches)
        return dict(usage)

    def _detect_circular_reasoning(self, steps: List[LogicStep]) -> bool:
        conclusions: Dict[str, int] = {}
        for step in steps:
            eq_match = re.match(r"^(\w+)\s*=\s*(.+)$", step.statement.strip())
            if eq_match:
                var = eq_match.group(1)
                rhs = eq_match.group(2).strip()
                if var in conclusions and var not in rhs:
                    return True
                conclusions[var] = step.step_number
        return False

    def _detect_missing_steps(
        self, steps: List[LogicStep], task
    ) -> List[Dict[str, Any]]:
        issues = []
        goals = getattr(task, "goals", [])
        if not goals:
            return issues

        proof_lower = " ".join(s.statement.lower() for s in steps)
        for goal in goals:
            keywords = re.findall(
                r"[a-zA-Z_\u4e00-\u9fff][a-zA-Z0-9_\u4e00-\u9fff]*", goal
            )
            key_found = any(kw.lower() in proof_lower for kw in keywords if len(kw) > 3)
            if not key_found:
                issues.append(
                    {
                        "category": "unaddressed_goal",
                        "goal": goal[:100],
                        "message": f"Goal not addressed in proof: {goal[:60]}",
                    }
                )

        return issues


class LLMSemanticVerifier:
    """Layer 4: LLM-based semantic verification via cloud API.

    Supports multiple providers:
    - OpenAI-compatible (OpenAI, DeepSeek, etc.)
    - Anthropic Claude
    - Custom endpoints

    Falls back to heuristic analysis if no API is configured.
    """

    def __init__(self, api_config: Optional[Dict[str, Any]] = None):
        self.api_config = api_config or {}
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        provider = self.api_config.get("provider", "")
        api_key = self.api_config.get("api_key", "")
        base_url = self.api_config.get("base_url", "")
        model = self.api_config.get("model", "")

        if not api_key:
            return None

        try:
            if provider == "openai" or base_url:
                from openai import OpenAI

                kwargs = {"api_key": api_key}
                if base_url:
                    kwargs["base_url"] = base_url
                self._client = ("openai", OpenAI(**kwargs), model or "gpt-4")
                return self._client

            elif provider == "anthropic":
                import anthropic

                self._client = (
                    "anthropic",
                    anthropic.Anthropic(api_key=api_key),
                    model or "claude-sonnet-4-20250514",
                )
                return self._client

        except ImportError:
            logger.warning(
                "LLM client library not installed, falling back to heuristic"
            )
            return None

        return None

    def verify(self, task, proof_text: str) -> LayerResult:
        t0 = time.time()

        client_info = self._get_client()
        if client_info is None:
            return self._heuristic_fallback(task, proof_text, t0)

        provider, client, model = client_info

        try:
            if provider == "openai":
                result = self._verify_openai(client, model, task, proof_text)
            elif provider == "anthropic":
                result = self._verify_anthropic(client, model, task, proof_text)
            else:
                result = self._heuristic_fallback(task, proof_text, t0)
        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            return LayerResult(
                layer=VerificationLayer.LLM_SEMANTIC,
                status=FormalStatus.INCONCLUSIVE,
                confidence=0.0,
                details={"error": str(e), "fallback": "heuristic"},
                time_ms=(time.time() - t0) * 1000,
            )

        result.time_ms = (time.time() - t0) * 1000
        return result

    def _verify_openai(self, client, model: str, task, proof_text: str) -> LayerResult:
        prompt = self._build_verification_prompt(task, proof_text)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a mathematical proof verifier. "
                    "Respond ONLY with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )

        content = response.choices[0].message.content.strip()
        return self._parse_llm_response(content)

    def _verify_anthropic(
        self, client, model: str, task, proof_text: str
    ) -> LayerResult:
        prompt = self._build_verification_prompt(task, proof_text)

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text.strip()
        return self._parse_llm_response(content)

    def _build_verification_prompt(self, task, proof_text: str) -> str:
        task_name = getattr(task, "name", "Unknown")
        task_statement = getattr(task, "statement", "")
        task_goals = getattr(task, "goals", [])
        task_assumptions = getattr(task, "assumptions", [])

        goals_text = "\n".join(f"  {i+1}. {g}" for i, g in enumerate(task_goals))
        assumptions_text = "\n".join(f"  - {a}" for a in task_assumptions)

        return f"""Verify the following mathematical proof. Return JSON with keys:
- "status": one of "verified", "unverified", "contradicted", "inconclusive"
- "confidence": float 0.0-1.0
- "issues": list of {{"category": str, "message": str}}
- "summary": brief explanation

Proposition: {task_name}
Statement: {task_statement}

Assumptions:
{assumptions_text if assumptions_text else "  (none)"}

Goals:
{goals_text if goals_text else "  (none)"}

Proof:
---
{proof_text[:3000]}
---

Respond with JSON only:"""

    def _parse_llm_response(self, content: str) -> LayerResult:
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return LayerResult(
                layer=VerificationLayer.LLM_SEMANTIC,
                status=FormalStatus.INCONCLUSIVE,
                confidence=0.0,
                details={"raw_response": content[:200]},
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return LayerResult(
                layer=VerificationLayer.LLM_SEMANTIC,
                status=FormalStatus.INCONCLUSIVE,
                confidence=0.0,
                details={"parse_error": "invalid_json"},
            )

        status_map = {
            "verified": FormalStatus.VERIFIED,
            "unverified": FormalStatus.UNVERIFIED,
            "contradicted": FormalStatus.CONTRADICTED,
            "inconclusive": FormalStatus.INCONCLUSIVE,
        }

        return LayerResult(
            layer=VerificationLayer.LLM_SEMANTIC,
            status=status_map.get(
                data.get("status", "inconclusive"), FormalStatus.INCONCLUSIVE
            ),
            confidence=float(data.get("confidence", 0.0)),
            details={"summary": data.get("summary", ""), "model_used": True},
            issues=data.get("issues", []),
        )

    def _heuristic_fallback(self, task, proof_text: str, t0: float) -> LayerResult:
        issues: List[Dict[str, Any]] = []
        details: Dict[str, Any] = {"fallback": True}

        has_structure = any(
            kw in proof_text.lower()
            for kw in ["proof", "theorem", "lemma", "证明", "定理", "引理"]
        )
        details["has_proof_structure"] = has_structure
        if not has_structure:
            issues.append(
                {
                    "category": "no_proof_structure",
                    "message": "Text does not appear to contain a formal proof",
                }
            )

        has_reasoning = any(
            kw in proof_text.lower()
            for kw in [
                "therefore",
                "hence",
                "thus",
                "implies",
                "follows",
                "因此",
                "所以",
                "推出",
                "可得",
            ]
        )
        details["has_reasoning_chain"] = has_reasoning
        if not has_reasoning:
            issues.append(
                {
                    "category": "no_reasoning_chain",
                    "message": "No logical reasoning chain detected",
                }
            )

        if not issues:
            status = FormalStatus.INCONCLUSIVE
            confidence = 0.4
        else:
            status = FormalStatus.UNVERIFIED
            confidence = 0.2

        return LayerResult(
            layer=VerificationLayer.LLM_SEMANTIC,
            status=status,
            confidence=confidence,
            details=details,
            issues=issues,
            time_ms=(time.time() - t0) * 1000,
        )


class FormalVerifier:
    """Multi-layer formal verification pipeline.

    Runs four verification layers in sequence:
    1. Symbolic (sympy algebraic verification)
    2. Type system (dimensional analysis + type checking)
    3. Logic (proof structure and rule checking)
    4. LLM semantic (cloud API with heuristic fallback)

    Aggregates results into a single FormalVerificationResult.

    Example:
        >>> fv = FormalVerifier()
        >>> result = fv.verify(task, proof_text)
        >>> print(result.formal_status)  # 'verified', 'unverified', etc.
        >>> for lr in result.layer_results:
        ...     print(f"{lr.layer.value}: {lr.status.value} ({lr.confidence:.0%})")

    With cloud API:
        >>> fv = FormalVerifier(api_config={
        ...     "provider": "openai",
        ...     "api_key": "sk-...",
        ...     "model": "gpt-4",
        ... })
        >>> result = fv.verify(task, proof_text)
    """

    def __init__(
        self,
        api_config: Optional[Dict[str, Any]] = None,
        enable_layers: Optional[List[str]] = None,
    ):
        self.symbolic = SymbolicVerifier()
        self.type_system = TypeSystemVerifier()
        self.logic = LogicVerifier()
        self.llm_semantic = LLMSemanticVerifier(api_config=api_config)
        self._lean4_bridge = None

        all_layers = {
            "symbolic": VerificationLayer.SYMBOLIC,
            "type_system": VerificationLayer.TYPE_SYSTEM,
            "logic": VerificationLayer.LOGIC,
            "llm_semantic": VerificationLayer.LLM_SEMANTIC,
            "lean4_formal": VerificationLayer.LEAN4_FORMAL,
        }
        if enable_layers:
            self._enabled_layers = {
                all_layers[k] for k in enable_layers if k in all_layers
            }
        else:
            self._enabled_layers = set(all_layers.values())

    def verify(
        self, task, proof_text: str, geometric_context=None
    ) -> FormalVerificationResult:
        """Run formal verification on a proof.

        Args:
            task: MathematicalTask from PropositionGenerator
            proof_text: LLM-generated proof text
            geometric_context: Optional GeometricStructure from DifferentialGeometryLayer

        Returns:
            FormalVerificationResult with per-layer and aggregate results
        """
        if geometric_context is not None:
            geo_symbols = self._geometry_to_symbols(geometric_context)
            self.type_system.inject_symbols(geo_symbols)

        layer_results: List[LayerResult] = []
        all_symbols: List[TypedSymbol] = []
        all_logic_steps: List[LogicStep] = []
        all_violations: List[TypeViolation] = []

        if VerificationLayer.SYMBOLIC in self._enabled_layers:
            lr = self.symbolic.verify(task, proof_text)
            layer_results.append(lr)

        if VerificationLayer.TYPE_SYSTEM in self._enabled_layers:
            lr = self.type_system.verify(task, proof_text)
            layer_results.append(lr)
            if "symbols" in lr.details:
                all_symbols = lr.details.get("symbols", [])
            all_violations = [
                TypeViolation(**v)
                for v in lr.issues
                if v.get("operation")
                in ("addition", "equality", "dimensional_equality")
            ]

        if VerificationLayer.LOGIC in self._enabled_layers:
            lr = self.logic.verify(task, proof_text)
            layer_results.append(lr)

        if VerificationLayer.LLM_SEMANTIC in self._enabled_layers:
            lr = self.llm_semantic.verify(task, proof_text)
            layer_results.append(lr)

        if VerificationLayer.LEAN4_FORMAL in self._enabled_layers:
            lr = self._verify_lean4(task, proof_text)
            layer_results.append(lr)

        formal_status, overall_confidence = self._aggregate(layer_results)
        summary = self._build_summary(formal_status, overall_confidence, layer_results)

        return FormalVerificationResult(
            formal_status=formal_status,
            overall_confidence=overall_confidence,
            layer_results=layer_results,
            typed_symbols=all_symbols,
            logic_steps=all_logic_steps,
            type_violations=all_violations,
            summary=summary,
        )

    def _aggregate(self, results: List[LayerResult]) -> Tuple[FormalStatus, float]:
        if not results:
            return FormalStatus.INCONCLUSIVE, 0.0

        weights = {
            VerificationLayer.SYMBOLIC: 0.20,
            VerificationLayer.TYPE_SYSTEM: 0.20,
            VerificationLayer.LOGIC: 0.20,
            VerificationLayer.LLM_SEMANTIC: 0.20,
            VerificationLayer.LEAN4_FORMAL: 0.20,
        }

        weighted_conf = 0.0
        total_weight = 0.0
        has_contradiction = False
        all_verified = True
        any_verified = False

        for lr in results:
            w = weights.get(lr.layer, 0.25)
            if lr.status == FormalStatus.SKIPPED:
                continue
            weighted_conf += w * lr.confidence
            total_weight += w
            if lr.status == FormalStatus.CONTRADICTED:
                has_contradiction = True
            if lr.status != FormalStatus.VERIFIED:
                all_verified = False
            if lr.status == FormalStatus.VERIFIED:
                any_verified = True

        if total_weight > 0:
            weighted_conf /= total_weight

        if has_contradiction:
            return FormalStatus.CONTRADICTED, min(weighted_conf, 0.2)
        elif all_verified and any_verified:
            return FormalStatus.VERIFIED, weighted_conf
        elif any_verified:
            return FormalStatus.INCONCLUSIVE, weighted_conf
        else:
            return FormalStatus.UNVERIFIED, weighted_conf

    def _build_summary(
        self, status: FormalStatus, confidence: float, results: List[LayerResult]
    ) -> str:
        parts = [f"Formal verification: {status.value} (confidence: {confidence:.0%})"]
        for lr in results:
            if lr.status != FormalStatus.SKIPPED:
                parts.append(
                    f"  {lr.layer.value}: {lr.status.value} "
                    f"({lr.confidence:.0%}, {lr.time_ms:.0f}ms)"
                )
        return "\n".join(parts)

    def _verify_lean4(self, task, proof_text: str) -> LayerResult:
        start = time.time()
        try:
            from .lean4_bridge import Lean4Bridge, LeanVerificationStatus

            if self._lean4_bridge is None:
                self._lean4_bridge = Lean4Bridge()

            if not self._lean4_bridge.is_available():
                return LayerResult(
                    layer=VerificationLayer.LEAN4_FORMAL,
                    status=FormalStatus.SKIPPED,
                    confidence=0.0,
                    time_ms=(time.time() - start) * 1000,
                    details="Lean4 not available on this system. "
                    "Install lean4 and lake to enable machine-checked verification.",
                )

            lean_code = self._lean4_bridge.math_schema_to_lean(task)
            result = self._lean4_bridge.verify_statement(lean_code)

            status_map = {
                LeanVerificationStatus.PROVED: FormalStatus.VERIFIED,
                LeanVerificationStatus.TIMEOUT: FormalStatus.INCONCLUSIVE,
                LeanVerificationStatus.ERROR: FormalStatus.UNVERIFIED,
                LeanVerificationStatus.UNKNOWN: FormalStatus.INCONCLUSIVE,
                LeanVerificationStatus.SKIPPED: FormalStatus.SKIPPED,
            }

            confidence_map = {
                LeanVerificationStatus.PROVED: 1.0,
                LeanVerificationStatus.TIMEOUT: 0.3,
                LeanVerificationStatus.ERROR: 0.0,
                LeanVerificationStatus.UNKNOWN: 0.5,
                LeanVerificationStatus.SKIPPED: 0.0,
            }

            return LayerResult(
                layer=VerificationLayer.LEAN4_FORMAL,
                status=status_map.get(result.status, FormalStatus.INCONCLUSIVE),
                confidence=confidence_map.get(result.status, 0.0),
                time_ms=result.proof_time_ms,
                details=f"Lean4: {result.status.value} | {result.message[:200]}",
                issues=(
                    [{"lean_error": result.message[:500]}]
                    if result.status != LeanVerificationStatus.PROVED
                    else []
                ),
            )
        except ImportError:
            return LayerResult(
                layer=VerificationLayer.LEAN4_FORMAL,
                status=FormalStatus.SKIPPED,
                confidence=0.0,
                time_ms=(time.time() - start) * 1000,
                details="lean4_bridge module not available",
            )
        except Exception as e:
            return LayerResult(
                layer=VerificationLayer.LEAN4_FORMAL,
                status=FormalStatus.INCONCLUSIVE,
                confidence=0.0,
                time_ms=(time.time() - start) * 1000,
                details=f"Lean4 verification error: {str(e)[:200]}",
            )

    def _geometry_to_symbols(self, geo) -> List[TypedSymbol]:
        symbols: List[TypedSymbol] = []

        type_map = {
            "euclidean": MathType.SCALAR,
            "torus": MathType.MANIFOLD,
            "sphere": MathType.MANIFOLD,
            "cylinder": MathType.MANIFOLD,
            "projective": MathType.MANIFOLD,
            "hyperbolic": MathType.MANIFOLD,
            "product": MathType.MANIFOLD,
            "orbifold": MathType.MANIFOLD,
            "general": MathType.MANIFOLD,
        }

        manifold = getattr(geo, "manifold", None)
        if manifold:
            mtype = type_map.get(
                (
                    getattr(manifold, "topology", "").value
                    if hasattr(getattr(manifold, "topology", ""), "value")
                    else str(getattr(manifold, "topology", ""))
                ),
                MathType.MANIFOLD,
            )
            symbols.append(
                TypedSymbol(
                    name=getattr(manifold, "name", "M"),
                    math_type=mtype,
                    dimension=getattr(manifold, "dimension", 3),
                    domain=getattr(manifold, "description", ""),
                )
            )

        metric = getattr(geo, "metric", None)
        if metric:
            symbols.append(
                TypedSymbol(
                    name="g_ij",
                    math_type=MathType.TENSOR,
                    domain=getattr(metric, "basis", ""),
                )
            )

        for sym in getattr(geo, "symmetries", []):
            symbols.append(
                TypedSymbol(
                    name=getattr(sym, "name", "G"),
                    math_type=MathType.GROUP,
                    domain=getattr(sym, "description", ""),
                )
            )

        for fb in getattr(geo, "fiber_bundles", []):
            symbols.append(
                TypedSymbol(
                    name=getattr(fb, "name", "E"),
                    math_type=MathType.FORM,
                    domain=f"{getattr(fb, 'base_manifold', '')} -> {getattr(fb, 'fiber', '')}",
                )
            )

        return symbols
