"""谱问题结构家族。

SpectralProblem = Hψ = λψ 及其所有变体。

谱问题是计算材料科学中最基础的数学结构之一：
  - DFT: 非线性自伴算子的特征值问题
  - 模态分析: 广义线性特征值问题 Kφ = λMφ
  - 量子化学: Fock 算子的特征值问题
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .base import AbstractMathematicalStructure, StructureFamily, StructureMetadata
from .properties import (
    OperatorType,
    SpectrumType,
    StructuralInvariant,
    get_invariants,
)


@dataclass
class SpectralProblem(AbstractMathematicalStructure):
    """谱问题的抽象定义：Hψ = λψ.

    所有特征值/特征函数问题的母类。

    Attributes:
        operator_type: 算子的数学类型（自伴/非自伴/正定）
        spectrum_type: 谱的拓扑类型（离散/连续/能带）
        hilbert_space_dim: Hilbert 空间的物理维度
        symmetry_groups: 对称群列表
        variational: 是否从变分原理导出
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.SPECTRAL,
            name="Spectral Problem",
            canonical_form="H ψ = λ ψ",
            description="Find eigenvalues and eigenfunctions of an operator",
        )
    )
    operator_type: OperatorType = OperatorType.SELF_ADJOINT
    spectrum_type: SpectrumType = SpectrumType.PURE_POINT
    hilbert_space_dim: int = 3
    variational: bool = False
    bounded_below: bool = False
    symmetry_groups: list[str] = field(default_factory=list)

    @property
    def function_space(self) -> str:
        return f"L²(ℝ^{self.hilbert_space_dim})"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants: list[StructuralInvariant] = []
        props = self._property_dict()

        # 从注册表加载不变量
        if self.operator_type == OperatorType.SELF_ADJOINT:
            for inv in get_invariants("spectral_self_adjoint"):
                if inv.is_active(props):
                    invariants.append(inv)

        if self.variational:
            for inv in get_invariants("variational"):
                if inv.is_active(props):
                    invariants.append(inv)

        return invariants

    def _property_dict(self) -> dict[str, Any]:
        return {
            "operator_type": str(self.operator_type),
            "variational": self.variational,
            "bounded_below": self.bounded_below,
        }


@dataclass
class LinearEigenvalueProblem(SpectralProblem):
    """线性谱问题：算子不依赖特征函数.

    实例：
      - 模态分析：Kφ = λ Mφ（广义特征值）
      - Hartree-Fock：F ψ = ε ψ（Fock 算子固定）
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.SPECTRAL,
            name="Linear Eigenvalue Problem",
            canonical_form="H ψ = λ ψ  (H independent of ψ)",
            description="Standard eigenvalue problem where operator does not depend on eigenfunctions",
        )
    )
    generalized: bool = False  # True = Kφ = λMφ, False = Hψ = εψ


@dataclass
class NonlinearEigenvalueProblem(SpectralProblem):
    """非线性谱问题：算子依赖于特征函数.

    实例：
      - Kohn-Sham DFT：H[n] ψ = ε ψ（H 依赖密度 n = Σabs:ψabs:²）
      - 耦合簇：指数型非线性参数化
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.SPECTRAL,
            name="Nonlinear Eigenvalue Problem",
            canonical_form="H[ψ] ψ = ε ψ  (H depends on ψ)",
            description="Nonlinear eigenvalue problem where operator depends on eigenfunction",
        )
    )
    nonlinearity_source: str = ""  # "density_dependent", "occupation_dependent", etc.
    self_consistency_required: bool = True

    @property
    def additional_invariants(self) -> list[StructuralInvariant]:
        return [
            StructuralInvariant(
                name="brouwer_fixed_point",
                expression="F[n] has a fixed point under SCF iteration (Banach conditions)",
                theorem="Banach Fixed Point Theorem (requires contraction mapping)",
                condition="self.self_consistency_required",
                affected_quantities=["convergence", "density"],
            ),
            StructuralInvariant(
                name="hohenberg_kohn_uniqueness",
                expression="V_ext(r) is a unique functional of n(r) (up to constant)",
                theorem="Hohenberg-Kohn Theorem",
                condition="self.nonlinearity_source == 'density_dependent'",
                affected_quantities=["external_potential", "electron_density"],
            ),
        ]

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(self.additional_invariants)
        return invariants


@dataclass
class SelfConsistentProblem(NonlinearEigenvalueProblem):
    """自洽场问题：DFT 和 HF 的标准形式.

    完整的 Kohn-Sham 循环：
    n_in → construct H[n_in] → solve Hψ = εψ → compute n_out → check abs:n_out - n_inabs: < tol
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.SPECTRAL,
            name="Self-Consistent Field Problem",
            canonical_form="H[n] ψ = ε ψ, with n = Σ f_i abs:ψ_iabs:²",
            description="Self-consistent field iteration: density-dependent Hamiltonian solved iteratively",
        )
    )

    @property
    def scf_cycle_invariants(self) -> list[StructuralInvariant]:
        """SCF 循环特有的不变量."""
        return [
            StructuralInvariant(
                name="charge_conservation",
                expression="∫ n(r) dr = N_e",
                theorem="Hohenberg-Kohn Theorem (electron number)",
                affected_quantities=["electron_number", "density"],
            ),
            StructuralInvariant(
                name="aufbau_principle",
                expression="Occupy lowest N_e eigenvalues",
                theorem="Variational Principle",
                condition="self.variational",
                affected_quantities=["occupations", "fermi_level"],
            ),
        ]

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(self.scf_cycle_invariants)
        return invariants


class EigenvalueSolver:
    """Numerical eigenvalue solver for mathematical structures."""

    def __init__(self, matrix: np.ndarray):
        """Initialize with a matrix to find eigenvalues of."""
        self.matrix = np.asarray(matrix, dtype=float)
        self.dim = self.matrix.shape[0]

    def eigenvalues(self) -> np.ndarray:
        """Compute all eigenvalues."""
        return np.linalg.eigvals(self.matrix)

    def eigenvectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors."""
        return np.linalg.eig(self.matrix)

    def spectral_gap(self) -> float:
        """Compute the spectral gap (difference between two smallest |eigenvalues|)."""
        evals = np.sort(np.abs(self.eigenvalues()))
        if len(evals) < 2:
            return 0.0
        return float(evals[1] - evals[0])

    def condition_number(self) -> float:
        """Compute the condition number from eigenvalues."""
        evals = np.abs(self.eigenvalues())
        evals = evals[evals > 1e-15]
        if len(evals) == 0:
            return float("inf")
        return float(np.max(evals) / np.min(evals))

    def is_positive_definite(self) -> bool:
        """Check if all eigenvalues are positive."""
        return bool(np.all(self.eigenvalues().real > 0))

    def is_self_adjoint(self, tol: float = 1e-10) -> bool:
        """Check if the matrix is Hermitian/self-adjoint."""
        return bool(np.allclose(self.matrix, self.matrix.conj().T, atol=tol))


class SelfConsistentSolver:
    """Self-consistent field iteration solver.

    Solves problems of the form: H[n]ψ = εψ, n = |ψ|²
    where the Hamiltonian depends on its own eigenstate.
    """

    def __init__(
        self,
        hamiltonian_builder: Callable,
        n_states: int = 1,
        mixing: float = 0.3,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        """
        Args:
            hamiltonian_builder: Function that takes density matrix and returns Hamiltonian
            n_states: Number of states to track
            mixing: Density mixing parameter (0 < α ≤ 1)
            max_iter: Maximum SCF iterations
            tol: Convergence tolerance on density change
        """
        self.hamiltonian_builder = hamiltonian_builder
        self.n_states = n_states
        self.mixing = mixing
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, initial_density: np.ndarray | None = None) -> dict:
        """Run the SCF iteration.

        Returns:
            Dict with converged, iterations, eigenvalues, density, convergence_history
        """
        dim = self.hamiltonian_builder(np.eye(2)).shape[0]  # Infer dimension
        if initial_density is not None:
            density = initial_density.copy()
        else:
            density = np.eye(dim) / dim

        convergence_history = []
        evals = np.array([])
        for iteration in range(self.max_iter):
            # Build Hamiltonian from current density
            H = self.hamiltonian_builder(density)

            # Solve eigenvalue problem
            evals, evecs = np.linalg.eigh(H)

            # Build new density from lowest n_states
            new_density = np.zeros_like(density)
            for i in range(min(self.n_states, dim)):
                new_density += np.outer(evecs[:, i], evecs[:, i].conj())

            # Mix densities
            density_new = self.mixing * new_density + (1 - self.mixing) * density

            # Check convergence
            delta = np.linalg.norm(density_new - density)
            convergence_history.append(delta)

            if delta < self.tol:
                return {
                    "converged": True,
                    "iterations": iteration + 1,
                    "eigenvalues": evals.tolist(),
                    "density_change": delta,
                    "convergence_history": convergence_history,
                }

            density = density_new

        return {
            "converged": False,
            "iterations": self.max_iter,
            "eigenvalues": evals.tolist(),
            "density_change": delta,
            "convergence_history": convergence_history,
        }
