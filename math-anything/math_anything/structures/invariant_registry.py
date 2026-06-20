"""预定义不变量组与不变量注册表.

按结构类型索引，每个类型关联一组 StructuralInvariant。
提供 query_invariants 函数用于按关键词查询不变量。
"""

from __future__ import annotations

from ._core import StructuralInvariant

# ── 预定义：谱问题的不变量 ──

SPECTRAL_SELF_ADJOINT_INVARIANTS = [
    StructuralInvariant(
        name="eigenvalues_real",
        expression="λ_i ∈ ℝ for all i",
        theorem="Spectral Theorem for Self-Adjoint Operators",
        condition="properties.get('operator_type') == 'self_adjoint'",
        affected_quantities=["eigenvalues", "observable"],
    ),
    StructuralInvariant(
        name="eigenvectors_orthogonal",
        expression="⟨φ_i, φ_j⟩ = δ_ij",
        theorem="Spectral Theorem (orthogonal diagonalization)",
        condition="properties.get('operator_type') == 'self_adjoint'",
        affected_quantities=["eigenvectors", "basis"],
    ),
    StructuralInvariant(
        name="spectral_resolution",
        expression="H = Σ λ_i P_i",
        theorem="Resolution of the Identity",
        condition="properties.get('operator_type') == 'self_adjoint'",
        affected_quantities=["operator", "spectral_decomposition"],
    ),
]

VARIATIONAL_INVARIANTS = [
    StructuralInvariant(
        name="ground_state_minimum",
        expression="E[ψ_0] ≤ E[ψ] for all admissible ψ",
        theorem="Variational Principle (Rayleigh-Ritz)",
        condition="properties.get('variational') == True",
        affected_quantities=["ground_state_energy", "wavefunction"],
    ),
    StructuralInvariant(
        name="energy_lower_bound",
        expression="E[ψ] ≥ E_0",
        theorem="Bounded below + variational → infimum exists",
        condition="properties.get('variational') == True and properties.get('bounded_below') == True",
        affected_quantities=["total_energy"],
    ),
]

HAMILTONIAN_INVARIANTS = [
    StructuralInvariant(
        name="phase_volume_conservation",
        expression="d/dt (∫ dΓ) = 0",
        theorem="Liouville's Theorem",
        condition="properties.get('hamiltonian') == True",
        affected_quantities=["phase_space_volume", "entropy"],
    ),
    StructuralInvariant(
        name="energy_conservation",
        expression="dH/dt = 0 (for time-independent H)",
        theorem="Noether's Theorem (time translation symmetry)",
        condition="properties.get('hamiltonian') == True and not properties.get('time_dependent')",
        affected_quantities=["total_energy"],
    ),
    StructuralInvariant(
        name="symplectic_structure",
        expression="ω = Σ dp_i ∧ dq_i is preserved",
        theorem="Symplectic Geometry",
        condition="properties.get('hamiltonian') == True",
        affected_quantities=["phase_space", "integration"],
    ),
]

CONSERVATION_LAW_INVARIANTS = [
    StructuralInvariant(
        name="mass_conservation",
        expression="∂ρ/∂t + ∇·(ρu) = 0",
        theorem="Continuity Equation (Noether: gauge invariance)",
        affected_quantities=["density", "flux"],
    ),
    StructuralInvariant(
        name="momentum_conservation_smooth",
        expression="d/dt ∫ ρu dV = 0 (no external forces)",
        theorem="Noether's Theorem (space translation symmetry)",
        condition="not properties.get('external_force')",
        affected_quantities=["momentum"],
    ),
    StructuralInvariant(
        name="rankine_hugoniot",
        expression="s[U] = [F(U)] across discontinuities",
        theorem="Weak solution jump condition",
        condition="properties.get('hyperbolic') == True",
        affected_quantities=["shock_speed", "jump_conditions"],
    ),
]


# ── 不变量注册表 ──
# 按结构类型索引，每个类型关联一组不变量

INVARIANT_REGISTRY: dict[str, list[StructuralInvariant]] = {
    "spectral_self_adjoint": SPECTRAL_SELF_ADJOINT_INVARIANTS,
    "variational": VARIATIONAL_INVARIANTS,
    "hamiltonian": HAMILTONIAN_INVARIANTS,
    "conservation_law": CONSERVATION_LAW_INVARIANTS,
}


def get_invariants(category: str) -> list[StructuralInvariant]:
    """获取预定义的不变量集."""
    return INVARIANT_REGISTRY.get(category, [])


def query_invariants(
    keyword: str | None = None,
    theorem: str | None = None,
    affected_quantity: str | None = None,
) -> list[StructuralInvariant]:
    """按关键词查询不变量.

    支持按 name/expression/theorem/affected_quantities 中的关键词过滤。

    Args:
        keyword: 通用关键词，匹配 name, expression, theorem
        theorem: 仅匹配 theorem 字段
        affected_quantity: 仅匹配 affected_quantities 中的条目

    Returns:
        匹配的不变量列表
    """
    results: list[StructuralInvariant] = []
    seen_names: set[str] = set()

    for inv_group in INVARIANT_REGISTRY.values():
        for inv in inv_group:
            if inv.name in seen_names:
                continue

            match = True

            if keyword:
                kw = keyword.lower()
                if not (kw in inv.name.lower() or kw in inv.expression.lower() or kw in inv.theorem.lower()):
                    match = False

            if theorem:
                if theorem.lower() not in inv.theorem.lower():
                    match = False

            if affected_quantity:
                if affected_quantity.lower() not in " ".join(inv.affected_quantities).lower():
                    match = False

            if match:
                results.append(inv)
                seen_names.add(inv.name)

    return results


__all__ = [
    "SPECTRAL_SELF_ADJOINT_INVARIANTS",
    "VARIATIONAL_INVARIANTS",
    "HAMILTONIAN_INVARIANTS",
    "CONSERVATION_LAW_INVARIANTS",
    "INVARIANT_REGISTRY",
    "get_invariants",
    "query_invariants",
]
