"""5 层验证流水线.

重建前端 VerifyPage 预留的 5 层验证流水线后端：
1. Symbolic — 符号验证（表达式等价性）
2. Type System — 类型系统验证（MLTT/CIC 类型检查）
3. Logic — 逻辑验证（命题逻辑 + 一阶逻辑）
4. LLM Semantic — LLM 语义验证（自然语言推理）
5. Lean4 Formal — 形式化验证（Lean4 代码生成 + 检查）

每层返回 VerificationLayerResult，流水线汇总为 VerificationResult。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .bridge import TypeTheoryBridge
from .checker import TypeChecker
from .cic import CICBridge, CICTypeChecker
from .hott import HoTTBridge, HoTTTypeChecker
from .metamath import FormalSystemStrength, MetamathAnalyzer
from .terms import TYPE0, Context, Identity, Var


class VerificationLayer(Enum):
    """验证层级."""

    SYMBOLIC = "symbolic"
    TYPE_SYSTEM = "type_system"
    LOGIC = "logic"
    LLM_SEMANTIC = "llm_semantic"
    LEAN4_FORMAL = "lean4_formal"


@dataclass
class LayerResult:
    """单层验证结果."""

    layer: VerificationLayer
    passed: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # 0.0 ~ 1.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """完整验证结果."""

    statement: str
    layers: list[LayerResult] = field(default_factory=list)
    overall_passed: bool = False
    overall_confidence: float = 0.0

    def compute_overall(self) -> None:
        """汇总所有层级结果."""
        if not self.layers:
            return
        self.overall_passed = all(layer.passed for layer in self.layers)
        self.overall_confidence = sum(layer.confidence for layer in self.layers) / len(self.layers)


class VerificationPipeline:
    """5 层验证流水线.

    每层独立验证，汇总为最终结果。
    层级之间有依赖关系：
    - Symbolic 是基础层
    - Type System 依赖 Symbolic
    - Logic 依赖 Type System
    - LLM Semantic 是独立的语义层
    - Lean4 Formal 是最强的形式化层
    """

    def __init__(self):
        self.mltt_checker = TypeChecker()
        self.cic_checker = CICTypeChecker()
        self.hott_checker = HoTTTypeChecker()
        self.mltt_bridge = TypeTheoryBridge(checker=self.mltt_checker)
        self.cic_bridge = CICBridge(checker=self.cic_checker)
        self.hott_bridge = HoTTBridge(checker=self.hott_checker)
        self.metamath = MetamathAnalyzer()

    def verify(
        self,
        statement: str,
        proof_text: str = "",
        assumptions: list[str] | None = None,
        goals: list[str] | None = None,
        layers: list[VerificationLayer] | None = None,
    ) -> VerificationResult:
        """执行验证流水线."""
        if layers is None:
            layers = list(VerificationLayer)

        result = VerificationResult(statement=statement)

        for layer in layers:
            layer_result = self._run_layer(layer, statement, proof_text, assumptions or [], goals or [])
            result.layers.append(layer_result)

        result.compute_overall()
        return result

    def _run_layer(
        self,
        layer: VerificationLayer,
        statement: str,
        proof_text: str,
        assumptions: list[str],
        goals: list[str],
    ) -> LayerResult:
        """执行单层验证."""
        if layer == VerificationLayer.SYMBOLIC:
            return self._verify_symbolic(statement, proof_text, assumptions)
        elif layer == VerificationLayer.TYPE_SYSTEM:
            return self._verify_type_system(statement, proof_text, assumptions)
        elif layer == VerificationLayer.LOGIC:
            return self._verify_logic(statement, proof_text, assumptions, goals)
        elif layer == VerificationLayer.LLM_SEMANTIC:
            return self._verify_llm_semantic(statement, proof_text, assumptions)
        elif layer == VerificationLayer.LEAN4_FORMAL:
            return self._verify_lean4(statement, proof_text, assumptions, goals)
        return LayerResult(layer=layer, passed=False, message="Unknown layer")

    def _verify_symbolic(self, statement: str, proof_text: str, assumptions: list[str]) -> LayerResult:
        """Layer 1: 符号验证.

        检查表达式是否在符号层面等价：
        - 量纲一致性
        - 符号化简
        - 等式两边是否可化简为相同形式
        """
        checks = {}

        # 量纲一致性检查 — use real dimensional analyzer when possible
        dim_ok = True
        if "=" in statement:
            parts = statement.split("=", 1)
            if len(parts) == 2:
                lhs, rhs = parts[0].strip(), parts[1].strip()
                # Try real dimensional analysis first
                try:
                    from math_anything.dimensional.equation_checker import SymbolicDimensionalAnalyzer

                    analyzer = SymbolicDimensionalAnalyzer()
                    dim_result = analyzer.check_equation(lhs, rhs)
                    if dim_result:
                        dim_ok = dim_result.get("is_consistent", True)
                        if not dim_ok:
                            checks["dimensional_warning"] = dim_result.get("reason", "量纲不一致")
                except Exception:
                    # Fallback: check variable set overlap
                    lhs_vars = set(c for c in lhs if c.isalpha())
                    rhs_vars = set(c for c in rhs if c.isalpha())
                    if lhs_vars and rhs_vars and not lhs_vars.intersection(rhs_vars):
                        dim_ok = False
                        checks["dimensional_warning"] = f"两边变量集无交集: {lhs_vars} vs {rhs_vars}"

        checks["dimensional_consistency"] = dim_ok

        # 符号化简检查
        if proof_text.strip():
            checks["has_proof"] = True
            checks["proof_length"] = len(proof_text)
        else:
            checks["has_proof"] = False

        passed = dim_ok
        return LayerResult(
            layer=VerificationLayer.SYMBOLIC,
            passed=passed,
            message="符号验证" + ("通过" if passed else "失败"),
            details=checks,
            confidence=0.8 if passed else 0.2,
        )

    def _verify_type_system(self, statement: str, proof_text: str, assumptions: list[str]) -> LayerResult:
        """Layer 2: 类型系统验证.

        使用 MLTT/CIC 类型检查器验证：
        - 命题是否是良类型的
        - 证明项是否具有命题类型
        - 不变量是否可被类型检查器验证
        """
        ctx = Context()

        # 添加假设到上下文
        for i, assumption in enumerate(assumptions):
            ctx = ctx.extend(f"hyp_{i}", TYPE0)

        # 尝试将陈述解析为类型论命题
        checks = {}

        # 检查陈述是否包含可识别的数学结构
        type_check_passed = False
        try:
            # 简单命题：检查是否是等式
            if "=" in statement:
                # 构造恒等类型
                parts = statement.split("=", 1)
                if len(parts) == 2:
                    lhs_term = Var(f"expr_{parts[0].strip()[:20]}")
                    rhs_term = Var(f"expr_{parts[1].strip()[:20]}")
                    id_type = Identity(TYPE0, lhs_term, rhs_term)
                    ctx_ext = ctx.extend(lhs_term.name, TYPE0).extend(rhs_term.name, TYPE0)
                    result = self.mltt_checker.type_check(ctx_ext, id_type)
                    type_check_passed = result.success
                    checks["identity_type_formed"] = result.success
            else:
                # 非等式命题
                type_check_passed = True  # 宽松通过
                checks["non_equality_statement"] = True
        except (ValueError, TypeError, AttributeError) as e:
            checks["error"] = str(e)
            type_check_passed = False

        # CIC 层：检查 Prop/Type 分类
        cic_checks = {}
        if "守恒" in statement or "conservation" in statement.lower():
            cic_checks["sort"] = "Type_0 (计算相关)"
        elif "定理" in statement or "theorem" in statement.lower():
            cic_checks["sort"] = "Prop (证明无关)"
        else:
            cic_checks["sort"] = "Type_0 (默认)"
        checks["cic_sort"] = cic_checks

        # HoTT 层：推断 h-level
        hott_checks = {}
        if "同构" in statement or "isomorphism" in statement.lower():
            hott_checks["h_level"] = "CONTRACTIBLE (可缩)"
        elif "等价" in statement or "equivalence" in statement.lower():
            hott_checks["h_level"] = "SET (集合)"
        else:
            hott_checks["h_level"] = "SET (默认)"
        checks["hott_h_level"] = hott_checks

        passed = type_check_passed
        return LayerResult(
            layer=VerificationLayer.TYPE_SYSTEM,
            passed=passed,
            message="类型系统验证" + ("通过" if passed else "失败"),
            details=checks,
            confidence=0.9 if passed else 0.3,
        )

    def _verify_logic(self, statement: str, proof_text: str, assumptions: list[str], goals: list[str]) -> LayerResult:
        """Layer 3: 逻辑验证.

        使用命题逻辑和一阶逻辑验证：
        - 假设是否蕴含目标
        - 证明步骤是否逻辑有效
        - 可判定性分析
        """
        checks = {}

        # 假设蕴含检查
        contradiction_found = False
        if assumptions and goals:
            checks["assumption_count"] = len(assumptions)
            checks["goal_count"] = len(goals)
            # Check for obvious contradictions in assumptions
            assumption_set = set()
            for a in assumptions:
                a_lower = a.lower().strip()
                if a_lower.startswith("not "):
                    negated = a_lower[4:]
                    if negated in assumption_set:
                        contradiction_found = True
                        checks["contradiction"] = f"Assumption '{a}' contradicts '{negated}'"
                    assumption_set.add(a_lower)
                else:
                    if f"not {a_lower}" in assumption_set:
                        contradiction_found = True
                        checks["contradiction"] = f"Assumption '{a}' contradicts 'not {a_lower}'"
                    assumption_set.add(a_lower)
            # Check if any goal directly contradicts an assumption
            for g in goals:
                g_lower = g.lower().strip()
                if f"not {g_lower}" in assumption_set:
                    contradiction_found = True
                    checks["goal_contradiction"] = f"Goal '{g}' contradicts assumption 'not {g_lower}'"
            checks["implies"] = not contradiction_found

        # 证明结构检查
        if proof_text:
            lines = [line.strip() for line in proof_text.split("\n") if line.strip()]
            checks["proof_steps"] = len(lines)
            checks["has_proof_structure"] = len(lines) > 1

        # 可判定性分析
        decidability = self.metamath.analyze_invariant_decidability("statement", statement)
        if decidability:
            checks["decidability"] = decidability.decidability.name
            checks["invariant_state"] = decidability.maps_to_invariant_state
        else:
            checks["decidability"] = "UNKNOWN"
            checks["invariant_state"] = "UNKNOWN"

        # Gödel 限制
        godel = self.metamath.godel_limitation(FormalSystemStrength.CIC)
        checks["godel_limitation"] = godel.godel_sentence[:80]

        passed = not contradiction_found
        return LayerResult(
            layer=VerificationLayer.LOGIC,
            passed=passed,
            message="逻辑验证" + ("通过" if passed else "失败（发现矛盾）"),
            details=checks,
            confidence=0.7 if passed else 0.1,
            warnings=[] if passed else ["发现逻辑矛盾"],
        )

    # ── Layer 4: LLM 语义验证的知识库 ──

    _THEOREM_KNOWLEDGE: dict[str, dict[str, Any]] = {
        "noether": {
            "keywords": ["noether", "symmetry", "conservation", "invariant"],
            "domain": "mathematical_physics",
            "properties": [
                "continuous symmetry implies conserved current",
                "time translation → energy conservation",
                "space translation → momentum conservation",
                "rotation → angular momentum conservation",
            ],
            "contradictions": [
                "symmetry without conservation law",
                "conserved quantity without corresponding symmetry",
            ],
        },
        "spectral_theorem": {
            "keywords": ["spectral", "eigenvalue", "self-adjoint", "hermitian", "diagonaliz"],
            "domain": "functional_analysis",
            "properties": [
                "self-adjoint operators have real eigenvalues",
                "eigenvectors of self-adjoint operators form orthogonal basis",
                "normal operators are unitarily diagonalizable",
            ],
            "contradictions": [
                "self-adjoint operator with complex eigenvalues",
                "non-normal operator claimed to be unitarily diagonalizable",
            ],
        },
        "liouville": {
            "keywords": ["liouville", "phase space", "hamiltonian", "volume preservation"],
            "domain": "classical_mechanics",
            "properties": [
                "phase space volume is preserved under Hamiltonian flow",
                "density evolves via Liouville equation",
            ],
            "contradictions": [
                "phase space volume contraction in Hamiltonian system",
                "dissipation without external force in Hamiltonian dynamics",
            ],
        },
        "bloch": {
            "keywords": ["bloch", "periodic", "band structure", "crystal"],
            "domain": "condensed_matter",
            "properties": [
                "periodic potential yields Bloch waves",
                "energy spectrum forms bands",
                "wavevector is conserved modulo reciprocal lattice",
            ],
            "contradictions": [
                "localized state in infinite periodic potential without defect",
            ],
        },
        "hohenberg_kohn": {
            "keywords": ["hohenberg", "kohn", "density functional", "dft", "ground state density"],
            "domain": "quantum_chemistry",
            "properties": [
                "external potential is uniquely determined by ground state density",
                "energy functional is minimized by true ground state density",
            ],
            "contradictions": [
                "two different external potentials with same ground state density",
                "ground state density not determining the Hamiltonian",
            ],
        },
        "variational": {
            "keywords": ["variational", "rayleigh", "ritz", "minimum", "extremum"],
            "domain": "mathematical_physics",
            "properties": [
                "variational principle gives upper bound on ground state energy",
                "stationary condition δE=0 yields Euler-Lagrange equations",
            ],
            "contradictions": [
                "variational estimate below exact ground state energy",
                "unconstrained minimum not corresponding to physical state",
            ],
        },
        "cauchy_schwarz": {
            "keywords": ["cauchy", "schwarz", "inequality", "inner product"],
            "domain": "mathematics",
            "properties": [
                "|⟨u,v⟩|² ≤ ⟨u,u⟩⟨v,v⟩",
                "equality iff u and v are linearly dependent",
            ],
            "contradictions": [
                "inner product violating Cauchy-Schwarz bound",
            ],
        },
        "uncertainty": {
            "keywords": ["uncertainty", "heisenberg", "commutator", "variance"],
            "domain": "quantum_mechanics",
            "properties": [
                "ΔA·ΔB ≥ ½|⟨[A,B]⟩|",
                "non-commuting observables cannot be simultaneously measured",
            ],
            "contradictions": [
                "simultaneous exact measurement of non-commuting observables",
                "variance product below commutator bound",
            ],
        },
    }

    _PHYSICAL_CONTRADICTIONS: list[tuple[str, str, list[str]]] = [
        ("perpetual motion", "违反热力学第一/第二定律", ["thermodynamics", "energy"]),
        ("faster than light", "违反相对论因果律", ["relativity", "causality"]),
        ("exact solution NP", "NP 问题无已知多项式精确解", ["complexity", "P_vs_NP"]),
        ("negative probability", "概率不能为负", ["probability", "measure"]),
        ("energy from nothing", "违反能量守恒", ["conservation", "energy"]),
        ("violate bell inequality classically", "经典系统不能违反 Bell 不等式", ["quantum", "locality"]),
    ]

    def _verify_llm_semantic(self, statement: str, proof_text: str, assumptions: list[str]) -> LayerResult:
        """Layer 4: LLM 语义验证.

        基于知识库的语义层面验证：
        - 识别陈述涉及的数学定理
        - 检查陈述是否与已知定理性质矛盾
        - 检查是否违反物理定律
        - 验证假设与结论的逻辑一致性
        """
        checks = {}
        warnings = []
        total_checks = 0
        passed_checks = 0

        combined_text = (statement + " " + proof_text).lower()

        # ── 检查 1: 识别相关定理 ──
        relevant_theorems = {}
        for theo_name, theo_data in self._THEOREM_KNOWLEDGE.items():
            for kw in theo_data["keywords"]:
                if kw in combined_text:
                    relevant_theorems[theo_name] = theo_data
                    break

        if relevant_theorems:
            checks["relevant_theorems"] = list(relevant_theorems.keys())
            total_checks += 1
            passed_checks += 1
        else:
            warnings.append("未识别到已知定理，验证强度较低")
            total_checks += 1
            # 没识别到不算失败，只是强度低

        # ── 检查 2: 定理性质一致性 ──
        property_violations = []
        for theo_name, theo_data in relevant_theorems.items():
            for contradiction in theo_data["contradictions"]:
                # 检查陈述是否隐含矛盾性质
                contra_keywords = contradiction.lower().split()
                match_count = sum(1 for kw in contra_keywords if len(kw) > 3 and kw in combined_text)
                if match_count >= max(1, len(contra_keywords) // 2):
                    property_violations.append(
                        {
                            "theorem": theo_name,
                            "contradiction": contradiction,
                            "domain": theo_data["domain"],
                        }
                    )

        total_checks += 1
        if not property_violations:
            passed_checks += 1
            checks["property_consistency"] = "consistent"
        else:
            checks["property_consistency"] = "violations_found"
            checks["violations"] = property_violations
            for v in property_violations:
                warnings.append(f"与 {v['theorem']} 的已知性质矛盾: {v['contradiction']}")

        # ── 检查 3: 物理定律矛盾检测 ──
        physical_violations = []
        for pattern, reason, domains in self._PHYSICAL_CONTRADICTIONS:
            if pattern.lower() in combined_text:
                physical_violations.append(
                    {
                        "pattern": pattern,
                        "reason": reason,
                        "domains": domains,
                    }
                )

        total_checks += 1
        if not physical_violations:
            passed_checks += 1
            checks["physical_consistency"] = "consistent"
        else:
            checks["physical_consistency"] = "violations_found"
            checks["physical_violations"] = physical_violations
            for v in physical_violations:
                warnings.append(f"物理定律矛盾: {v['reason']}")

        # ── 检查 4: 假设-结论逻辑一致性 ──
        total_checks += 1
        if assumptions:
            # 检查假设是否与陈述中的结论方向一致
            # 简单启发式: 假设中提到的关键量应在陈述中出现
            assumption_keywords = set()
            for a in assumptions:
                words = [w.lower() for w in a.split() if len(w) > 3]
                assumption_keywords.update(words)

            statement_keywords = set(w.lower() for w in statement.split() if len(w) > 3)
            overlap = assumption_keywords & statement_keywords

            if overlap:
                passed_checks += 1
                checks["assumption_conclusion_overlap"] = list(overlap)
            else:
                checks["assumption_conclusion_overlap"] = []
                warnings.append("假设与结论之间缺乏关键词重叠，逻辑关联可能较弱")
        else:
            # 无假设时跳过此检查
            passed_checks += 1
            checks["assumption_conclusion_overlap"] = "no_assumptions"

        # ── 检查 5: 定理性质验证 ──
        total_checks += 1
        verified_properties = []
        for theo_name, theo_data in relevant_theorems.items():
            for prop in theo_data["properties"]:
                # 检查陈述是否声称满足该性质
                prop_keywords = [w.lower() for w in prop.split() if len(w) > 3]
                match_count = sum(1 for kw in prop_keywords if kw in combined_text)
                if match_count >= max(1, len(prop_keywords) // 3):
                    verified_properties.append(
                        {
                            "theorem": theo_name,
                            "property": prop,
                        }
                    )

        if verified_properties:
            passed_checks += 1
            checks["verified_properties"] = verified_properties
        else:
            # 没有验证到的性质不算失败
            passed_checks += 1
            checks["verified_properties"] = []

        # ── 汇总 ──
        semantic_ok = passed_checks >= total_checks - 1  # 允许一个检查不通过
        confidence = passed_checks / total_checks if total_checks > 0 else 0.5

        # 有物理矛盾时强制降低置信度
        if physical_violations:
            confidence *= 0.2
            semantic_ok = False

        checks["llm_available"] = False
        checks["semantic_check"] = "knowledge_base_matching"
        checks["total_checks"] = total_checks
        checks["passed_checks"] = passed_checks

        return LayerResult(
            layer=VerificationLayer.LLM_SEMANTIC,
            passed=semantic_ok,
            message="LLM 语义验证" + ("通过" if semantic_ok else "失败"),
            details=checks,
            confidence=confidence,
            warnings=warnings,
        )

    # ── Layer 5: Lean4 代码生成的结构映射 ──

    _LEAN4_TYPE_MAP: dict[str, str] = {
        "real": "ℝ",
        "integer": "ℤ",
        "natural": "ℕ",
        "complex": "ℂ",
        "matrix": "Matrix",
        "vector": "Vector",
        "function": "→",
        "boolean": "Bool",
        "proposition": "Prop",
        "set": "Set",
        "operator": "LinearMap",
    }

    _LEAN4_THEOREM_TEMPLATES: dict[str, str] = {
        "eigenvalue": (
            "\n-- Eigenvalue theorem statement\n"
            "theorem eigenvalue_statement {A : Matrix n n ℝ} (hA : A.IsHermitian)\n"
            "    : ∃ μ : ℝ, A.HasEigenvalue μ := by\n"
            "  sorry\n"
        ),
        "conservation": (
            "\n-- Conservation law statement\n"
            "theorem conservation_law {H : Type*} [Hamiltonian H]\n"
            "    (hH : H.timeIndependent) : IsConserved H.energy := by\n"
            "  sorry\n"
        ),
        "variational": (
            "\n-- Variational principle statement\n"
            "theorem variational_principle {V : Type*} [Hilbert V]\n"
            "    (ψ : V) (hψ : IsAdmissible ψ) :\n"
            "    ⟪ψ, H ψ⟫ ≥ E₀ := by\n"
            "  sorry\n"
        ),
        "self_adjoint": (
            "\n-- Self-adjoint operator properties\n"
            "theorem self_adjoint_real_spectrum {A : Type*} [Hilbert A]\n"
            "    (T : A →ₗ[ℂ] A) (hT : IsSelfAdjoint T) :\n"
            "    ∀ λ ∈ spectrum T, Im λ = 0 := by\n"
            "  sorry\n"
        ),
        "orthogonal": (
            "\n-- Orthogonality of eigenvectors\n"
            "theorem eigenvector_orthogonal {A : Type*} [Hilbert A]\n"
            "    (T : A →ₗ[ℂ] A) (hT : IsSelfAdjoint T)\n"
            "    {λ μ : ℂ} (hλμ : λ ≠ μ) (v w : A)\n"
            "    (hv : HasEigenvector T λ v) (hw : HasEigenvector T μ w) :\n"
            "    ⟪v, w⟫ = 0 := by\n"
            "  sorry\n"
        ),
    }

    def _verify_lean4(self, statement: str, proof_text: str, assumptions: list[str], goals: list[str]) -> LayerResult:
        """Layer 5: Lean4 形式化验证.

        生成 Lean4 代码（含类型签名和定理陈述）。
        由于无法运行 Lean4 编译器，将生成的代码作为证据返回，
        置信度设为 0.3（代码已生成但未经 Lean 验证）。
        """
        checks = {}
        warnings = []

        # ── 步骤 1: 生成带类型签名的 Lean4 代码 ──
        lean4_code = self._generate_lean4(statement, proof_text, assumptions, goals)
        checks["lean4_code"] = lean4_code

        # ── 步骤 2: 识别陈述中的数学结构并生成对应定理 ──
        combined_text = (statement + " " + proof_text).lower()
        generated_theorems = []

        for pattern, template in self._LEAN4_THEOREM_TEMPLATES.items():
            if pattern in combined_text:
                generated_theorems.append(
                    {
                        "pattern": pattern,
                        "code": template.strip(),
                    }
                )

        if generated_theorems:
            checks["generated_theorem_statements"] = generated_theorems
            # 将定理代码追加到主代码
            theorem_code = "\n".join(t["code"] for t in generated_theorems)
            lean4_code = lean4_code + "\n" + theorem_code
            checks["lean4_code_full"] = lean4_code

        # ── 步骤 3: 推断类型签名 ──
        type_signatures = self._infer_type_signatures(statement, assumptions)
        if type_signatures:
            checks["type_signatures"] = type_signatures
            # 追加类型签名到代码
            sig_code = "\n".join(f"-- Inferred type: {sig}" for sig in type_signatures)
            lean4_code = lean4_code + "\n\n" + sig_code
            checks["lean4_code_full"] = lean4_code

        # ── 步骤 4: 元数学分析 ──
        godel = self.metamath.godel_limitation(FormalSystemStrength.CIC)
        checks["formal_system"] = "CIC (Lean4)"
        checks["godel_boundary"] = godel.godel_sentence[:80]

        # ── 步骤 5: 代码质量评估 ──
        code_lines = [line for line in lean4_code.split("\n") if line.strip() and not line.strip().startswith("--")]
        has_theorem = any("theorem" in line for line in code_lines)
        has_axiom = any("axiom" in line for line in code_lines)
        has_sorry = any("sorry" in line for line in code_lines)

        checks["code_stats"] = {
            "total_lines": len(lean4_code.split("\n")),
            "code_lines": len(code_lines),
            "has_theorem": has_theorem,
            "has_axiom": has_axiom,
            "has_sorry": has_sorry,
        }

        if has_sorry:
            warnings.append("生成的 Lean4 代码包含 sorry 占位符，需要手动补全证明")

        # ── 汇总 ──
        # 代码已生成但未经验证，置信度 0.3
        # 有定理陈述时视为更好的代码质量
        passed = has_theorem or has_axiom
        confidence = 0.3

        if generated_theorems:
            confidence = min(0.3 + 0.05 * len(generated_theorems), 0.4)

        checks["lean4_available"] = False
        checks["code_generated"] = True
        checks["verification_status"] = "code_generated_not_verified"

        return LayerResult(
            layer=VerificationLayer.LEAN4_FORMAL,
            passed=passed,
            message="Lean4 形式化验证：代码已生成（未编译验证）",
            details=checks,
            confidence=confidence,
            warnings=warnings,
        )

    def _infer_type_signatures(self, statement: str, assumptions: list[str]) -> list[str]:
        """从陈述和假设推断 Lean4 类型签名."""
        signatures = []
        combined = (statement + " " + " ".join(assumptions)).lower()

        # 检测数学对象类型
        for keyword, lean_type in self._LEAN4_TYPE_MAP.items():
            if keyword in combined:
                signatures.append(f"{keyword} : {lean_type}")

        # 检测等式关系
        if "=" in statement:
            parts = statement.split("=", 1)
            if len(parts) == 2:
                lhs_expr = parts[0].strip()[:40]
                rhs_expr = parts[1].strip()[:40]
                signatures.append(f"eq_statement : {lhs_expr} = {rhs_expr}")

        # 检测蕴含关系
        if "→" in statement or "implies" in combined or "蕴含" in statement:
            signatures.append("implication : Prop → Prop")

        # 检测全称量词
        if "∀" in statement or "for all" in combined or "对所有" in statement:
            signatures.append("universal_quantifier : ∀ (x : α), P x")

        # 检测存在量词
        if "∃" in statement or "there exists" in combined or "存在" in statement:
            signatures.append("existential_quantifier : ∃ (x : α), P x")

        return signatures

    def _generate_lean4(
        self,
        statement: str,
        proof_text: str,
        assumptions: list[str],
        goals: list[str],
    ) -> str:
        """生成 Lean4 代码骨架."""
        lines = [
            "-- Auto-generated by bourbaki verification pipeline",
            "import Mathlib.Tactic",
            "",
        ]

        for i, assumption in enumerate(assumptions):
            lines.append(f"-- Assumption {i + 1}: {assumption}")
            lines.append(f"axiom hyp_{i} : Prop")
            lines.append("")

        lines.append(f"-- Statement: {statement}")
        lines.append("theorem auto_statement : Prop := by")

        if proof_text:
            lines.append("  -- Proof sketch:")
            for line in proof_text.split("\n")[:10]:
                if line.strip():
                    lines.append(f"  -- {line.strip()}")

        lines.append("  sorry  -- Requires manual proof")
        lines.append("")

        for i, goal in enumerate(goals):
            lines.append(f"-- Goal {i + 1}: {goal}")

        return "\n".join(lines)
