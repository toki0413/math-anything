"""连续介质力学几何结构。

变形映射、Cauchy-Green 张量、Green-Lagrange 应变、极分解。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import AbstractMathematicalStructure, StructureFamily, StructureMetadata
from .properties import StructuralInvariant


@dataclass
class DeformationMapping(AbstractMathematicalStructure):
    """变形映射：F = Tφ: T_X B → T_{φ(X)} S.

    两点张量（two-point tensor），变形梯度。

    Attributes:
        body_dim: 参考构形 B 的维数
        spatial_dim: 空间构形 S 的维数
        det_F: det F 的值
        orientation_preserving: det F > 0 是否成立
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Deformation Mapping",
            canonical_form="F = Tφ: T_X B → T_{φ(X)} S  (deformation gradient)",
            description="Tangent map of deformation — two-point tensor linking reference and spatial configurations",
        )
    )
    body_dim: int = 3
    spatial_dim: int = 3
    det_F: float = 1.0
    orientation_preserving: bool = True

    @property
    def function_space(self) -> str:
        return f"Two-point tensor field on {self.body_dim}D→{self.spatial_dim}D deformation"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = [
            StructuralInvariant(
                name="orientation_preserving",
                expression="det F > 0  (deformation preserves orientation, no material interpenetration)",
                theorem="Orientation preservation: det F > 0 for physically admissible deformations",
                affected_quantities=["deformation_gradient", "jacobian"],
            ),
        ]
        if self.body_dim == 3 and self.spatial_dim == 3:
            invariants.append(
                StructuralInvariant(
                    name="principal_invariants_of_C",
                    expression="I_C = tr(C),  II_C = ½[(tr C)² - tr(C²)],  III_C = det C = (det F)²",
                    theorem="Principal invariants of right Cauchy-Green tensor C = F^T F",
                    affected_quantities=["stretch", "volume_change"],
                )
            )
        return invariants


@dataclass
class RightCauchyGreen(DeformationMapping):
    """右 Cauchy-Green 张量：C = F^T F.

    参考构形中的度量。对称正定。
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Right Cauchy-Green Tensor",
            canonical_form="C = F^T F  (pullback of spatial metric to reference configuration)",
            description="Right Cauchy-Green deformation tensor in material description",
        )
    )

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="C_symmetric_positive_definite",
                    expression="C = C^T,  v^T C v > 0 for all v ≠ 0",
                    theorem="C = F^T F is symmetric positive-definite (pullback of Euclidean metric)",
                    affected_quantities=["right_cauchy_green"],
                ),
            ]
        )
        return invariants


@dataclass
class GreenLagrangeStrain(DeformationMapping):
    """Green-Lagrange 应变：E = ½(C - I).

    描述相对于参考构形的有限应变。
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Green-Lagrange Strain",
            canonical_form="E = ½(C - I) = ½(F^T F - I)",
            description="Green-Lagrange strain tensor — finite strain in reference configuration",
        )
    )

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="e_vanishes_for_rigid_motion",
                    expression="F ∈ SO(3) ⇒ E = 0  (strain zero for rigid body rotation)",
                    theorem="Green-Lagrange strain vanishes for rigid body motions (objectivity)",
                    affected_quantities=["strain", "rigid_body_motion"],
                ),
                StructuralInvariant(
                    name="small_strain_approximation",
                    expression="|∇u| ≪ 1 ⇒ E ≈ ε = ½(∇u + (∇u)^T)",
                    theorem="Linearization: E → infinitesimal strain ε under small displacement gradient",
                    affected_quantities=["strain", "linearization"],
                ),
            ]
        )
        return invariants


@dataclass
class PolarDecomposition(DeformationMapping):
    """极分解：F = RU = VR.

    R ∈ SO(3)：旋转张量（正交）
    U：右伸长张量（对称正定，参考构形）
    V：左伸长张量（对称正定，空间构形）

    唯一分解！
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.GEOMETRY,
            name="Polar Decomposition",
            canonical_form="F = RU = VR,  R∈SO(3), U,V symmetric positive-definite",
            description="Unique polar decomposition of deformation gradient into rotation and stretch",
        )
    )

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="polar_decomposition_uniqueness",
                    expression="∃! R ∈ SO(3), U symmetric positive-definite: F = RU  (if det F > 0)",
                    theorem="Polar decomposition theorem (Cauchy) — unique factorization into rotation and stretch",
                    affected_quantities=["deformation_gradient", "rotation", "stretch"],
                ),
                StructuralInvariant(
                    name="U_from_C",
                    expression="U = √C = √(F^T F)  (right stretch from right Cauchy-Green)",
                    theorem="U is the unique symmetric positive-definite square root of C = F^T F",
                    affected_quantities=["right_stretch", "right_cauchy_green"],
                ),
            ]
        )
        return invariants


class DeformationGradient:
    """Compute deformation gradient F = ∂x/∂X from deformation mapping."""

    def __init__(self, F: np.ndarray):
        """
        Args:
            F: Deformation gradient tensor (dim x dim)
        """
        self.F = np.asarray(F, dtype=float)
        self.dim = self.F.shape[0]

    def right_cauchy_green(self) -> np.ndarray:
        """C = F^T F."""
        return self.F.T @ self.F  # type: ignore[no-any-return]

    def left_cauchy_green(self) -> np.ndarray:
        """B = F F^T."""
        return self.F @ self.F.T  # type: ignore[no-any-return]

    def green_lagrange_strain(self) -> np.ndarray:
        """E = (C - I) / 2."""
        C = self.right_cauchy_green()
        return 0.5 * (C - np.eye(self.dim))

    def jacobian(self) -> float:
        """J = det(F)."""
        return float(np.linalg.det(self.F))

    def is_incompressible(self, tol: float = 1e-10) -> bool:
        """Check if deformation is volume-preserving (J ≈ 1)."""
        return abs(self.jacobian() - 1.0) < tol

    def principal_stretches(self) -> np.ndarray:
        """Compute principal stretches (square roots of eigenvalues of C)."""
        C = self.right_cauchy_green()
        return np.sqrt(np.linalg.eigvalsh(C))

    def polar_decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """F = R U (right polar decomposition).

        Returns:
            R: Rotation tensor
            U: Right stretch tensor
        """
        C = self.right_cauchy_green()
        evals, evecs = np.linalg.eigh(C)
        U = evecs @ np.diag(np.sqrt(np.maximum(evals, 0))) @ evecs.T
        R = self.F @ np.linalg.inv(U)
        return R, U

    def cauchy_stress(self, lame_lambda: float, lame_mu: float) -> np.ndarray:
        """Compute Cauchy stress for Saint-Venant Kirchhoff material.

        σ = (1/J) F S F^T where S = λ tr(E)I + 2μE
        """
        E = self.green_lagrange_strain()
        S = lame_lambda * np.trace(E) * np.eye(self.dim) + 2 * lame_mu * E
        J = self.jacobian()
        if abs(J) < 1e-15:
            return np.zeros((self.dim, self.dim))
        return (1.0 / J) * self.F @ S @ self.F.T  # type: ignore[no-any-return]

    def von_mises_stress(self, lame_lambda: float, lame_mu: float) -> float:
        """Compute von Mises equivalent stress."""
        sigma = self.cauchy_stress(lame_lambda, lame_mu)
        # Deviatoric stress
        hydro = np.trace(sigma) / self.dim
        s = sigma - hydro * np.eye(self.dim)
        # von Mises
        if self.dim == 3:
            return float(np.sqrt(1.5 * np.sum(s * s)))
        return float(np.sqrt(2.0 * np.sum(s * s)))
