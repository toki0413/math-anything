r"""*-代数与 C*-代数结构。

StarAlgebra: 带对合 *: A → A 的代数
CStarAlgebra: 完备的 *-代数，满足 ∥A*A∥ = ∥A∥²
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._core import StructuralInvariant
from .base import AbstractMathematicalStructure, StructureFamily, StructureMetadata


@dataclass
class StarAlgebra(AbstractMathematicalStructure):
    r"""\*-代数：带对合 \*: A → A 的代数.

    (AB)\* = B\* A\*,  A\*\* = A,  1\* = 1.

    Attributes:
        is_unital: 是否有单位元 1
        is_commutative: 是否交换（AB = BA）
        dimension: 代数维数（有限维或 None 表示无穷维）
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ALGEBRA,
            name="*-Algebra",
            canonical_form="(AB)* = B*A*,  A** = A,  1* = 1",
            description="Algebra with an involution (anti-automorphism of order 2)",
        )
    )
    is_unital: bool = True
    is_commutative: bool = False
    dimension: int | None = None
    generators: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)

    def is_abelian(self) -> bool:
        """交换性判断（is_commutative 的数学别名）."""
        return self.is_commutative

    def verify_involution(self, elements: dict[str, list[float]]) -> tuple[bool, list[str]]:
        """验证对合性质: (AB)* = B*A* 对所有元素对成立.

        Args:
            elements: 元素名到其矩阵表示的映射，每个值展平为方阵的行优先列表

        Returns:
            (是否全部通过, 失败信息列表)
        """
        failures: list[str] = []
        # 把展平的列表还原成方阵
        mats: dict[str, np.ndarray] = {}
        for name, flat in elements.items():
            n = int(np.sqrt(len(flat)))
            if n * n != len(flat):
                failures.append(f"{name}: 长度 {len(flat)} 不是完全平方数，无法构成方阵")
                continue
            mats[name] = np.array(flat, dtype=complex).reshape(n, n)

        names = list(mats.keys())
        for i, a_name in enumerate(names):
            for b_name in names[i:]:
                A = mats[a_name]
                B = mats[b_name]
                lhs = (A @ B).conj().T  # (AB)*
                rhs = B.conj().T @ A.conj().T  # B*A*
                if not np.allclose(lhs, rhs):
                    failures.append(f"({a_name}{b_name})* ≠ {b_name}*{a_name}*")

        return (len(failures) == 0, failures)

    def center_description(self) -> str:
        """返回代数中心的描述."""
        if self.is_commutative:
            return "A (entire algebra)"
        return "Z(A) = {a ∈ A: [a,b]=0 for all b ∈ A}"

    @property
    def function_space(self) -> str:
        dim_str = f"dim {self.dimension}" if self.dimension else "∞-dim"
        return f"*-Algebra A ({dim_str})"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="involution_antiautomorphism",
                expression="(AB)* = B* A*  (involution is an anti-automorphism)",
                theorem="*-algebra axioms: involution reverses multiplication order",
                affected_quantities=["involution", "product"],
            ),
            StructuralInvariant(
                name="identity_star",
                expression="1* = 1  (unit is self-adjoint under involution)",
                theorem="Unit element is fixed by involution",
                condition="self.is_unital",
                affected_quantities=["identity", "involution"],
            ),
            StructuralInvariant(
                name="positive_elements",
                expression="A ≥ 0 iff A = B* B for some B ∈ A",
                theorem="Positive elements in a *-algebra are sums of hermitean squares",
                affected_quantities=["positivity", "order"],
            ),
        ]


@dataclass
class CStarAlgebra(StarAlgebra):
    r"""C\*-代数：完备的 \*-代数，满足 ∥A\*A∥ = ∥A∥².

    Gelfand-Naimark 定理：任何 C\*-代数等距 \*-同构于某个 Hilbert 空间上有界算子的闭 \*-子代数。

    Attributes:
        is_separable: 是否为可分 C*-代数
        has_approximate_identity: 是否有逼近单位元
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ALGEBRA,
            name="C*-Algebra",
            canonical_form="∥A*A∥ = ∥A∥²  (C*-identity)",
            description="Complete normed *-algebra satisfying the C*-identity",
        )
    )
    is_separable: bool = False
    has_approximate_identity: bool = True

    def verify_c_star_identity(self, elements: dict[str, list[list[float]]]) -> tuple[bool, list[str]]:
        """验证 C*-恒等式: ||A*A|| = ||A||² 对每个元素成立.

        Args:
            elements: 元素名到矩阵的映射

        Returns:
            (是否全部通过, 失败信息列表)
        """
        failures: list[str] = []
        for name, mat_data in elements.items():
            A = np.array(mat_data, dtype=complex)
            A_star_A = A.conj().T @ A
            norm_A_star_A = np.linalg.norm(A_star_A, ord=2)  # 算子范数
            norm_A = np.linalg.norm(A, ord=2)
            if not np.isclose(norm_A_star_A, norm_A**2):
                failures.append(f"{name}: ||A*A|| = {norm_A_star_A:.6f} ≠ ||A||² = {norm_A**2:.6f}")
        return (len(failures) == 0, failures)

    def spectrum_of_element(self, A: list[list[float]]) -> list[complex]:
        """计算元素的谱（特征值）.

        Args:
            A: 方阵表示

        Returns:
            特征值列表
        """
        return list(np.linalg.eigvals(np.array(A, dtype=complex)))

    def gelfand_transform_description(self) -> str:
        """Gelfand 变换描述."""
        if self.is_commutative:
            return "A ≅ C_0(σ(A)) via â(φ) = φ(a)"
        return "Noncommutative: no Gelfand transform; use spectral theorem for normal elements"

    def functional_calculus(self, A: list[list[float]], f_name: str) -> str:
        """连续函数演算: 返回 f(A) 的公式描述.

        Args:
            A: 元素的矩阵表示
            f_name: 函数名（如 "exp", "sqrt"）

        Returns:
            函数演算的公式字符串
        """
        if f_name == "exp":
            return "exp(A) = Σ A^n/n!"
        if f_name == "sqrt":
            return "√A via spectral decomposition: A = UDU* → √A = U√D U*"
        return "f(A) defined via continuous functional calculus on σ(A)"

    @property
    def function_space(self) -> str:
        return "C*-Algebra A (bounded operators on Hilbert space)"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="c_star_identity",
                    expression="∥A*A∥ = ∥A∥²  (norm uniquely determined by *-algebra structure)",
                    theorem="C*-identity: the norm is spectral — ∥A∥² = sup{abs:λabs:: λ ∈ σ(A*A)}",
                    affected_quantities=["norm", "spectrum"],
                ),
                StructuralInvariant(
                    name="gelfand_transform",
                    expression="If A commutative: A ≅ C_0(σ(A)) via Gelfand transform a ↦ â(φ) = φ(a)",
                    theorem="Gelfand-Naimark Theorem: commutative C*-algebra ≅ C_0(X)",
                    condition="self.is_commutative",
                    affected_quantities=["spectrum", "gelfand_transform"],
                ),
                StructuralInvariant(
                    name="continuous_functional_calculus",
                    expression="For A = A*: f(A) defined for all f ∈ C(σ(A))",
                    theorem="Continuous functional calculus for normal elements",
                    affected_quantities=["normal_operator", "spectrum"],
                ),
            ]
        )
        return invariants


__all__ = ["StarAlgebra", "CStarAlgebra"]
