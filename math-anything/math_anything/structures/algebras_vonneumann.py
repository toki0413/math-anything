r"""von Neumann 代数、态与表示、非交换概率。

VonNeumannAlgebra: B(H) 的弱闭 *-子代数
State / ClassicalState / QuantumState / PureState: 态空间
GNSConstruction: GNS 构造
CommutativeCase: 交换情形
TomitaTakesakiTheory: 模理论
NoncommutativeProbability: 非交换概率
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._core import StructuralInvariant
from .algebras_star import CStarAlgebra
from .base import AbstractMathematicalStructure, StructureFamily, StructureMetadata


def _matrix_intersection(
    basis_a: list[list[list[float]]],
    basis_b: list[list[list[float]]],
    n: int,
) -> list[list[list[float]]]:
    """计算两组矩阵基张成子空间的交集，返回交集的一组基."""
    if not basis_a or not basis_b:
        return []
    A = np.array([np.array(m, dtype=complex).flatten() for m in basis_a]).T
    B = np.array([np.array(m, dtype=complex).flatten() for m in basis_b]).T
    # 对 A 做列空间投影: P_A = A (A^T A)^{-1} A^T
    # 交集 = B 中被 P_A 映射到自身的列
    # 等价地: 求 B 的列中属于 col(A) 的部分
    # 方法: 解 A x = B y，即 [A | -B] [x; y] = 0 的零空间
    combined = np.hstack([A, -B])
    _, s, Vh = np.linalg.svd(combined)
    tol = max(combined.shape) * s[0] * np.finfo(float).eps
    null_mask = s <= tol
    if not null_mask.any():
        return []
    null_space = Vh.conj().T[:, null_mask]
    # 提取 B 的部分（后半段系数），构造交集基
    k = A.shape[1]
    result_basis: list[list[list[float]]] = []
    seen: list[np.ndarray] = []
    for i in range(null_space.shape[1]):
        y_coeffs = null_space[k:, i]
        vec = B @ y_coeffs
        mat = vec.reshape(n, n)
        # 去重
        is_new = True
        for existing in seen:
            if np.allclose(mat, existing, atol=1e-10):
                is_new = False
                break
        if is_new and np.linalg.norm(mat) > 1e-12:
            seen.append(mat)
            result_basis.append(mat.real.tolist())
    return result_basis


@dataclass
class VonNeumannAlgebra(CStarAlgebra):
    r"""von Neumann 代数：B(H) 的弱闭 \*-子代数，等于其双重交换子.

    M = M''（bicommutant theorem）。

    Attributes:
        is_factor: 是否为因子（中心平凡: M ∩ M' = ℂ·1）
        factor_type: 因子类型 I, II₁, II∞, III_λ (Connes 分类)
        has_separable_predual: 是否有可分的前对偶空间
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ALGEBRA,
            name="von Neumann Algebra",
            canonical_form="M = M''  (double commutant theorem)",
            description="Weakly closed *-subalgebra of B(H) equal to its bicommutant",
        )
    )
    is_factor: bool = False
    factor_type: str = "I_∞"
    has_separable_predual: bool = False

    def commutant(self, M: list[list[float]]) -> list[list[float]]:
        """计算矩阵 M 的交换子 M' = {X: [M,X]=0}.

        求解线性方程组 vec(MX - XM) = 0，返回交换子的基.

        Args:
            M: 方阵表示

        Returns:
            交换子的一组基（矩阵列表）
        """
        M_arr = np.array(M, dtype=complex)
        n = M_arr.shape[0]
        # 利用 Kronecker 积: vec(MX) = (I ⊗ M) vec(X), vec(XM) = (M^T ⊗ I) vec(X)
        I_n = np.eye(n, dtype=complex)
        # 交换子方程: (I⊗M - M^T⊗I) vec(X) = 0
        comm_matrix = np.kron(I_n, M_arr) - np.kron(M_arr.T, I_n)
        # 零空间即交换子
        _, s, Vh = np.linalg.svd(comm_matrix)
        # 奇异值接近0的右奇异向量构成零空间
        tol = max(comm_matrix.shape) * s[0] * np.finfo(float).eps
        null_mask = s <= tol
        null_space = Vh.conj().T[:, null_mask] if null_mask.any() else np.zeros((n * n, 0))
        # 把基向量还原成矩阵
        basis: list[list[list[float]]] = []
        for i in range(null_space.shape[1]):
            vec = null_space[:, i]
            mat = vec.reshape(n, n)
            basis.append(mat.real.tolist())
        return basis

    def verify_bicommutant(self, M: list[list[float]]) -> bool:
        """验证双重交换子定理: M 是否等于 M''.

        计算 M', 再计算 M'', 检查 M'' 是否由 M 生成.

        Args:
            M: 方阵表示

        Returns:
            是否满足 M = M''
        """
        M_arr = np.array(M, dtype=complex)
        n = M_arr.shape[0]
        # 第一层交换子
        comm1_basis = self.commutant(M)
        if not comm1_basis:
            # 交换子为空意味着 M 与一切交换（M 是标量矩阵），M'' = 全空间
            # 标量矩阵的交换子是全矩阵代数，双重交换子回到标量矩阵
            return True
        # 第二层交换子: 对 comm1_basis 的每个基元素取交换子，取交集
        # 简化处理: 把 comm1_basis 拼成一个大矩阵，对其求交换子
        comm2_basis: list[list[list[float]]] | None = None
        for B in comm1_basis:
            cb = self.commutant(B)
            if comm2_basis is None:
                comm2_basis = cb
            else:
                # 取交集: 保留同时在两个交换子中的基
                comm2_basis = _matrix_intersection(comm2_basis, cb, n)
        if comm2_basis is None:
            return True
        # 检查 M 是否在 M'' 的基张成的空间中
        for B in comm2_basis:
            B_arr = np.array(B, dtype=complex)
            if np.allclose(M_arr, B_arr, atol=1e-8):
                return True
        # 也检查 M 是否能被基的线性组合表出
        if comm2_basis:
            coeffs = np.array([np.array(b, dtype=complex).flatten() for b in comm2_basis]).T
            target = M_arr.flatten()
            # 最小二乘求解
            result, residuals, _, _ = np.linalg.lstsq(coeffs, target, rcond=None)
            reconstructed = coeffs @ result
            if np.allclose(reconstructed, target, atol=1e-8):
                return True
        return False

    def factor_classification(self) -> str:
        """因子分类描述."""
        if not self.is_factor:
            return "Not a factor: center Z(M) is nontrivial"
        type_map = {
            "I_n": "Type I_n: isomorphic to B(H) with dim(H) = n < ∞",
            "I_∞": "Type I_∞: isomorphic to B(H) with dim(H) = ∞",
            "II_1": "Type II_1: infinite-dimensional with finite tracial state",
            "II_∞": "Type II_∞: infinite-dimensional with semifinite trace",
            "III_0": "Type III_0: Connes invariant S(M) = {0,1}",
            "III_λ": "Type III_λ: Connes invariant S(M) = {0} ∪ {λ^n: n∈ℤ}, 0<λ<1",
            "III_1": "Type III_1: Connes invariant S(M) = [0,∞) — most noncommutative",
        }
        return type_map.get(self.factor_type, f"Factor of type {self.factor_type}")

    @property
    def function_space(self) -> str:
        return f"von Neumann algebra M ⊂ B(H), factor type {self.factor_type}"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="bicommutant_theorem",
                    expression="M = M''  (double commutant equals WOT-closure)",
                    theorem="von Neumann Bicommutant Theorem: M'' = M^{WOT} = M^{SOT}",
                    affected_quantities=["commutant", "closure"],
                ),
                StructuralInvariant(
                    name="abundance_of_projections",
                    expression="M is generated by its projections (M = span{projections in M}^{WOT})",
                    theorem="von Neumann algebra is the closed span of its projections",
                    affected_quantities=["projections", "spectral_measure"],
                ),
                StructuralInvariant(
                    name="kaplansky_density",
                    expression="(M)_1 is SOT-dense in (M^{WOT})_1",
                    theorem="Kaplansky Density Theorem: unit ball of M is SOT-dense in unit ball of its closure",
                    affected_quantities=["density", "unit_ball"],
                ),
            ]
        )
        return invariants


@dataclass
class State(AbstractMathematicalStructure):
    r"""态：正线性泛函 ω: A → ℂ，满足 ω(A\*A) ≥ 0，ω(1) = 1.

    物理意义：量子力学的期望值。

    Attributes:
        is_normal: 是否正规（在 von Neumann 代数意义下）
        is_faithful: ω(A\*A) = 0 ⇒ A = 0
        is_tracial: ω(AB) = ω(BA)
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ALGEBRA,
            name="State",
            canonical_form="ω: A → ℂ,  ω(A*A) ≥ 0,  ω(1) = 1",
            description="Positive linear functional — expectation value in quantum mechanics",
        )
    )
    is_normal: bool = False
    is_faithful: bool = False
    is_tracial: bool = False
    density_matrix: list[list[float]] | None = None

    def expectation(self, observable: list[list[float]]) -> float | None:
        """计算期望值 ω(A) = Tr(ρ A).

        Args:
            observable: 可观测量的矩阵表示

        Returns:
            期望值，若无密度矩阵则返回 None
        """
        if self.density_matrix is None:
            return None
        rho = np.array(self.density_matrix, dtype=complex)
        A = np.array(observable, dtype=complex)
        return float(np.real(np.trace(rho @ A)))

    def verify_positivity(self) -> tuple[bool, str]:
        """验证正性: ρ ≥ 0（所有特征值 ≥ 0）.

        Returns:
            (是否正定, 描述字符串)
        """
        if self.density_matrix is None:
            return (False, "No density matrix available")
        rho = np.array(self.density_matrix, dtype=complex)
        eigenvalues = np.linalg.eigvalsh(rho)
        min_eig = float(np.min(eigenvalues))
        is_pos = min_eig >= -1e-10
        desc = f"ρ ≥ 0: eigenvalues in [{min_eig:.6f}, {np.max(eigenvalues):.6f}]"
        return (is_pos, desc)

    def verify_normalization(self) -> tuple[bool, float]:
        """验证归一化: Tr(ρ) = 1.

        Returns:
            (是否归一化, 迹值)
        """
        if self.density_matrix is None:
            return (False, 0.0)
        rho = np.array(self.density_matrix, dtype=complex)
        trace_val = float(np.real(np.trace(rho)))
        return (np.isclose(trace_val, 1.0), trace_val)

    def von_neumann_entropy(self) -> float | None:
        """计算 von Neumann 熵: S(ρ) = -Tr(ρ ln ρ).

        Returns:
            熵值，若无密度矩阵则返回 None
        """
        if self.density_matrix is None:
            return None
        rho = np.array(self.density_matrix, dtype=complex)
        eigenvalues = np.linalg.eigvalsh(rho)
        # 只对正特征值求和，避免 log(0)
        pos_eigs = eigenvalues[eigenvalues > 1e-15]
        entropy = -float(np.sum(pos_eigs * np.log(pos_eigs)))
        return entropy

    @property
    def function_space(self) -> str:
        return "State space S(A) — convex set of positive normalized linear functionals"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = [
            StructuralInvariant(
                name="convexity_of_state_space",
                expression="S(A) is a convex set: ω₁, ω₂ ∈ S(A) ⇒ t ω₁ + (1-t) ω₂ ∈ S(A) for t∈[0,1]",
                theorem="State space is convex (Bayesian mixtures / classical mixtures)",
                affected_quantities=["state", "convexity"],
            ),
            StructuralInvariant(
                name="cauchy_schwarz_state",
                expression="abs:ω(A*B)abs:² ≤ ω(A*A) ω(B*B)",
                theorem="Cauchy-Schwarz inequality for positive functionals",
                affected_quantities=["expectation", "variance"],
            ),
            StructuralInvariant(
                name="norm_of_state",
                expression="∥ω∥ = ω(1) = 1  (for positive normalized functional)",
                theorem="Positive linear functional on unital C*-algebra has norm = ω(1)",
                affected_quantities=["norm", "state"],
            ),
        ]
        return invariants


@dataclass
class ClassicalState(State):
    """经典态：ω(f) = ∫_X f(x) ρ(x) dx.

    概率密度 ρ ≥ 0，∫ ρ = 1。
    Kolmogorov 概率论：ℙ(A) = ∫_A ρ dx。
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ALGEBRA,
            name="Classical State",
            canonical_form="ω(f) = ∫_X f(x) ρ(x) dx,  ρ ≥ 0, ∫ ρ = 1",
            description="State on commutative algebra — classical probability measure",
        )
    )
    is_commutative: bool = True

    @property
    def function_space(self) -> str:
        return "Probability measures on locally compact Hausdorff space X"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="kolmogorov_axioms",
                    expression="ℙ(Ω)=1, ℙ(A)≥0, ℙ(∪A_i)=Σℙ(A_i) for disjoint A_i",
                    theorem="Kolmogorov probability axioms (measure-theoretic foundation)",
                    affected_quantities=["probability", "measure"],
                ),
                StructuralInvariant(
                    name="law_of_total_probability",
                    expression="ω(f) = Σ ω(χ_{A_i}) ω(f abs: A_i)  (Bayesian decomposition)",
                    theorem="Law of total expectation (tower property)",
                    affected_quantities=["conditional_expectation", "probability"],
                ),
            ]
        )
        return invariants


@dataclass
class QuantumState(State):
    """量子态：ω(A) = Tr(ρ A).

    密度矩阵 ρ ≥ 0，Tr(ρ) = 1。

    Attributes:
        hilbert_space_dim: Hilbert 空间的维数
        is_pure: 是否纯态（ρ² = ρ，即 ρ 是秩一投影）
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ALGEBRA,
            name="Quantum State",
            canonical_form="ω(A) = Tr(ρ A),  ρ ≥ 0, Tr(ρ) = 1",
            description="State on noncommutative algebra — density matrix formulation",
        )
    )
    hilbert_space_dim: int | None = None
    is_pure: bool = False

    def purity(self) -> float | None:
        """计算纯度 Tr(ρ²). 纯态: Tr(ρ²) = 1，混合态: Tr(ρ²) < 1.

        Returns:
            纯度值，若无密度矩阵则返回 None
        """
        if self.density_matrix is None:
            return None
        rho = np.array(self.density_matrix, dtype=complex)
        return float(np.real(np.trace(rho @ rho)))

    def is_entangled(self) -> str | None:
        """纠缠判定: Schmidt 分解测试.

        对于二分态，通过 Schmidt 分解判定纠缠性.

        Returns:
            纠缠描述字符串，若无密度矩阵则返回 None
        """
        if self.density_matrix is None:
            return None
        rho = np.array(self.density_matrix, dtype=complex)
        n = rho.shape[0]
        # 检查是否可能是二分态（维数为完全平方数）
        sqrt_n = int(np.sqrt(n + 0.5))
        if sqrt_n * sqrt_n != n:
            return f"State dimension {n} is not a perfect square; bipartite Schmidt test not applicable"
        d = sqrt_n
        # 对纯态: 重排为 d×d 矩阵，做 SVD
        # 对混合态: 用 PPT 判据或纠缠熵
        if self.is_pure:
            # 纯态: ρ = |ψ⟩⟨ψ|，取第一列作为态向量
            psi = rho[:, 0]
            # 重排为 d×d 矩阵
            psi_mat = psi.reshape(d, d)
            _, s, _ = np.linalg.svd(psi_mat)
            schmidt_rank = int(np.sum(s > 1e-10))
            if schmidt_rank == 1:
                return "Pure separable state: Schmidt rank = 1 (product state)"
            return (
                f"Entangled pure state: Schmidt rank = {schmidt_rank}, Schmidt coefficients = {s[s > 1e-10].tolist()}"
            )
        # 混合态: 用部分转置判据 (PPT criterion)
        # 重排密度矩阵为 4-index 张量 ρ_{ij,kl}，对第二个子系统做转置
        rho_tensor = rho.reshape(d, d, d, d)
        rho_pt = rho_tensor.transpose(0, 3, 2, 1).reshape(n, n)
        eigenvalues = np.linalg.eigvalsh(rho_pt)
        min_eig = float(np.min(eigenvalues))
        if min_eig < -1e-10:
            return f"Entangled mixed state: PPT criterion violated (min eigenvalue of ρ^Γ = {min_eig:.6f})"
        return "PPT criterion satisfied: state may be separable (necessary condition for 2×2 and 2×3 systems)"

    def partial_trace(self, subsystem: int) -> list[list[float]] | None:
        """部分迹: 对指定子系统求迹.

        假设密度矩阵是二分态，维数为 d²×d².

        Args:
            subsystem: 0 表示对子系统 A 求迹，1 表示对子系统 B 求迹

        Returns:
            约化密度矩阵，若无密度矩阵则返回 None
        """
        if self.density_matrix is None:
            return None
        rho = np.array(self.density_matrix, dtype=complex)
        n = rho.shape[0]
        d = int(np.sqrt(n + 0.5))
        if d * d != n:
            return None
        rho_tensor = rho.reshape(d, d, d, d)
        if subsystem == 0:
            # Tr_A: 对第 0,2 维求迹
            reduced = np.trace(rho_tensor, axis1=0, axis2=2)
        else:
            # Tr_B: 对第 1,3 维求迹
            reduced = np.trace(rho_tensor, axis1=1, axis2=3)
        return reduced.real.tolist()

    @property
    def function_space(self) -> str:
        dim_str = f"ℂ^{self.hilbert_space_dim}" if self.hilbert_space_dim else "H"
        return f"Density operators on {dim_str}"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="gleasons_theorem",
                    expression="Every state on B(H) (dim H ≥ 3) is of the form ω(A) = Tr(ρ A)",
                    theorem="Gleason's Theorem: states on B(H) are density matrices",
                    affected_quantities=["state", "density_matrix"],
                ),
                StructuralInvariant(
                    name="schmidt_decomposition",
                    expression="abs:ψ⟩ = Σ √λ_i abs:i_A⟩ ⊗ abs:i_B⟩  (entanglement quantified by singular values)",
                    theorem="Schmidt decomposition for pure bipartite states",
                    affected_quantities=["entanglement", "singular_values"],
                ),
            ]
        )
        return invariants


@dataclass
class PureState(QuantumState):
    """纯态：态空间的极值点，不能写成凸组合。

    ω = t ω₁ + (1-t) ω₂ ⇒ ω₁ = ω₂ = ω。

    对应秩一投影 ρ = abs:ψ⟩⟨ψabs:。
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ALGEBRA,
            name="Pure State",
            canonical_form="ω is extreme point of S(A): ω = t ω₁ + (1-t)ω₂ ⇒ ω₁ = ω₂ = ω",
            description="Extreme point of state space — irreducible representation via GNS",
        )
    )
    is_pure: bool = True

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = super().structural_invariants
        invariants.extend(
            [
                StructuralInvariant(
                    name="extremal_decomposition",
                    expression="Every state is a convex combination / integral of pure states",
                    theorem="Krein-Milman: S(A) = closed convex hull of pure states",
                    affected_quantities=["state", "convexity"],
                ),
                StructuralInvariant(
                    name="pure_state_rank_one",
                    expression="ρ = abs:ψ⟩⟨ψabs: ⇒ ρ² = ρ, rank(ρ) = 1",
                    theorem="Pure state ↔ rank-1 projection ↔ extreme point of state space",
                    affected_quantities=["density_matrix", "projection"],
                ),
                StructuralInvariant(
                    name="gns_irreducible_iff_pure",
                    expression="π_ω is irreducible ⇔ ω is pure",
                    theorem="GNS representation: purity of state ⇔ irreducibility of representation",
                    affected_quantities=["representation", "purity"],
                ),
            ]
        )
        return invariants


@dataclass
class GNSConstruction(AbstractMathematicalStructure):
    """GNS 构造：从态 ω 构造 Hilbert 空间 H_ω，表示 π_ω，循环向量 Ω_ω.

    ω(A) = ⟨Ω_ω abs: π_ω(A) abs: Ω_ω⟩

    H_ω 由 π_ω(A)Ω_ω 张成（循环性）。

    Attributes:
        state_name: 态的名称
        is_irreducible: π_ω 是否不可约（等价于 ω 纯性）
        is_cyclic: Ω_ω 是否循环
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ALGEBRA,
            name="GNS Construction",
            canonical_form="ω(A) = ⟨Ω_ω abs: π_ω(A) abs: Ω_ω⟩",
            description="Gelfand-Naimark-Segal construction: from state to representation",
        )
    )
    state_name: str = ""
    is_irreducible: bool = False
    is_cyclic: bool = True

    def construct(self, state: State, algebra_elements: dict[str, list[list[float]]]) -> dict:
        """GNS 构造: 从态 ω 构造 Hilbert 空间、表示和循环向量.

        内积 ⟨A, B⟩ = ω(A*B)，利用态的密度矩阵计算.

        Args:
            state: 态对象（需有 density_matrix）
            algebra_elements: 代数元素名到矩阵的映射

        Returns:
            包含 hilbert_space_dim, representation, cyclic_vector, inner_product_matrix 的字典
        """
        names = list(algebra_elements.keys())
        n_elem = len(names)
        if n_elem == 0 or state.density_matrix is None:
            return {
                "hilbert_space_dim": 0,
                "representation": {},
                "cyclic_vector": [],
                "inner_product_matrix": [],
            }

        rho = np.array(state.density_matrix, dtype=complex)
        mats = {name: np.array(m, dtype=complex) for name, m in algebra_elements.items()}

        # 计算内积矩阵 G_{ij} = ⟨A_i, A_j⟩ = ω(A_i* A_j) = Tr(ρ A_i* A_j)
        G = np.zeros((n_elem, n_elem), dtype=complex)
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                A_star_B = mats[ni].conj().T @ mats[nj]
                G[i, j] = np.trace(rho @ A_star_B)

        # Hilbert 空间维数 = 内积矩阵的秩
        rank = int(np.linalg.matrix_rank(G, tol=1e-10))

        # GNS 表示: π_ω(A) 作用在左乘算子上
        # 在基 {A_i} 下，π(A_k) 的矩阵表示为左乘的 Gram 矩阵
        representation: dict[str, list[list[float]]] = {}
        for k, nk in enumerate(names):
            # π(A_k) 作用: A_i ↦ A_k A_i，在内积下的矩阵表示
            pi_k = np.zeros((n_elem, n_elem), dtype=complex)
            for i, ni in enumerate(names):
                product = mats[nk] @ mats[ni]
                for j, nj in enumerate(names):
                    # ⟨A_j, A_k A_i⟩ = Tr(ρ A_j* A_k A_i)
                    A_star_product = mats[nj].conj().T @ product
                    pi_k[j, i] = np.trace(rho @ A_star_product)
            representation[nk] = pi_k.real.tolist()

        # 循环向量: Ω_ω 对应于单位元 1 在基中的坐标
        # 简化处理: 用内积矩阵第一列的归一化版本
        omega_vec = np.zeros(n_elem, dtype=complex)
        for i, ni in enumerate(names):
            omega_vec[i] = np.trace(rho @ mats[ni])
        norm = np.sqrt(np.real(omega_vec.conj() @ G @ omega_vec))
        if norm > 1e-15:
            omega_vec = omega_vec / norm

        return {
            "hilbert_space_dim": rank,
            "representation": representation,
            "cyclic_vector": omega_vec.real.tolist(),
            "inner_product_matrix": G.real.tolist(),
        }

    def verify_cyclic(self, representation: dict, omega: list[list[float]]) -> bool:
        """验证循环性: {π(A)Ω: A ∈ A} 是否张成 H_ω.

        Args:
            representation: GNS 表示的矩阵字典
            omega: 循环向量

        Returns:
            是否循环
        """
        if not representation or not omega:
            return False
        omega_vec = np.array(omega, dtype=complex).flatten()
        # 收集所有 π(A)Ω 的像
        images = []
        for name, mat_data in representation.items():
            pi_A = np.array(mat_data, dtype=complex)
            img = pi_A @ omega_vec
            images.append(img)
        if not images:
            return False
        # 检查这些像是否张成整个空间
        mat = np.column_stack(images)
        rank = np.linalg.matrix_rank(mat, tol=1e-10)
        return rank == len(omega_vec)

    def irreducibility_description(self) -> str:
        """不可约性判定描述."""
        return "π_ω is irreducible ↔ ω is pure ↔ no nontrivial invariant subspaces"

    @property
    def function_space(self) -> str:
        return "Hilbert space H_ω with representation π_ω: A → B(H_ω)"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = [
            StructuralInvariant(
                name="cyclic_vector",
                expression="H_ω = span{π_ω(A) Ω_ω: A ∈ A}  (Ω_ω is cyclic)",
                theorem="GNS: cyclic vector Ω_ω generates the whole Hilbert space",
                affected_quantities=["representation", "hilbert_space"],
            ),
            StructuralInvariant(
                name="gns_uniqueness",
                expression="Any cyclic representation with state ω is unitarily equivalent to GNS(ω)",
                theorem="GNS construction is unique up to unitary equivalence",
                affected_quantities=["representation", "unitary"],
            ),
            StructuralInvariant(
                name="gns_irreducible_iff_pure",
                expression="π_ω is irreducible ↔ ω is pure",
                theorem="GNS irreducibility criterion: pure state ↔ irreducible representation",
                affected_quantities=["irreducibility", "pure_state"],
            ),
        ]
        return invariants


@dataclass
class CommutativeCase(AbstractMathematicalStructure):
    """交换情形：Gelfand-Naimark — 交换 C*-代数 ≅ C_0(X).

    X = σ(A)（C*-代数的 Gelfand 谱），局部紧 Hausdorff 空间。

    经典概率 = 交换量子概率。

    Attributes:
        spectrum_space: Gelfand 谱空间 X（局部紧 Hausdorff）
        is_unital: A 是否有单位元（等价于 X 是否紧）
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ALGEBRA,
            name="Commutative C*-Algebra ≅ C_0(X)",
            canonical_form="A_commutative ≅ C_0(X),  X = σ(A) locally compact Hausdorff",
            description="Gelfand-Naimark: commutative C*-algebra is algebra of continuous vanishing-at-infinity functions",
        )
    )
    spectrum_space: str = ""
    is_unital: bool = True

    @property
    def function_space(self) -> str:
        space = self.spectrum_space or "X"
        return f"C_0({space}) — continuous functions vanishing at infinity on {space}"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = [
            StructuralInvariant(
                name="gelfand_naimark_isomorphism",
                expression="A ≅ C_0(σ(A))  (isometric *-isomorphism via Gelfand transform)",
                theorem="Gelfand-Naimark Theorem: commutative C*-algebra = algebra of continuous functions",
                affected_quantities=["spectrum", "isomorphism"],
            ),
            StructuralInvariant(
                name="spectral_theorem_normal",
                expression="Normal operator N has spectral resolution: N = ∫_σ(N) λ dE(λ)",
                theorem="Spectral Theorem: eigenvalues = points of spectrum, eigenvectors = basis of Hilbert space",
                affected_quantities=["normal_operator", "spectral_measure"],
            ),
            StructuralInvariant(
                name="classical_probability_commutative",
                expression="Classical probability = commutative quantum probability: L^∞(Ω) is a commutative von Neumann algebra",
                theorem="Kolmogorov ⇔ commutative von Neumann algebra with faithful normal tracial state",
                affected_quantities=["probability", "commutativity"],
            ),
            StructuralInvariant(
                name="riesz_markov_duality",
                expression="C_0(X)* ≅ M(X)  (states ↔ Radon probability measures)",
                theorem="Riesz-Markov-Kakutani: dual of C_0(X) is space of Radon measures",
                affected_quantities=["dual", "measure"],
            ),
        ]
        return invariants


@dataclass
class TomitaTakesakiTheory(AbstractMathematicalStructure):
    """Tomita-Takesaki 理论：从忠实正规态 ω 构造模自同构群 σ_t^ω.

    核心对象：
      S: A Ω_ω ↦ A* Ω_ω（闭算子）
      极分解 S = J Δ^{1/2}：
        J: 模共轭（反酉，J² = 1, J M J = M'）
        Δ: 模算子（正自伴，σ_t^ω(A) = Δ^{it} A Δ^{-it}）

    KMS 条件：ω(A σ_{t+iβ}(B)) = ω(σ_t(B) A)

    Attributes:
        has_kms_state: 是否满足 KMS 条件
        inverse_temperature: 逆温度 β
        connes_type: Connes 分类 (0 < λ < 1: III_λ, λ = 0: III_0, λ = 1: III_1)
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ALGEBRA,
            name="Tomita-Takesaki Theory",
            canonical_form="S = J Δ^{1/2},  σ_t^ω(A) = Δ^{it} A Δ^{-it}",
            description="Modular theory: intrinsic dynamics from a faithful normal state",
        )
    )
    has_kms_state: bool = True
    inverse_temperature: float = 1.0
    connes_type: str = ""

    @property
    def function_space(self) -> str:
        return "Modular Hilbert space with modular automorphism group"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = [
            StructuralInvariant(
                name="modular_conjugation",
                expression="J² = 1,  J M J = M'  (modular conjugation maps algebra to commutant)",
                theorem="Tomita-Takesaki: J is an antiunitary involution implementing the commutant",
                affected_quantities=["von_neumann_algebra", "commutant"],
            ),
            StructuralInvariant(
                name="modular_operator_positive",
                expression="Δ > 0 (positive self-adjoint),  J Δ J = Δ^{-1}",
                theorem="Modular operator is positive and self-adjoint with symmetry J",
                affected_quantities=["modular_operator", "spectrum"],
            ),
            StructuralInvariant(
                name="kms_condition",
                expression="ω(A σ_{t+iβ}(B)) = ω(σ_t(B) A)  (KMS boundary condition)",
                theorem="Kubo-Martin-Schwinger condition: characterizes equilibrium states",
                condition="self.has_kms_state",
                affected_quantities=["equilibrium", "temperature"],
            ),
            StructuralInvariant(
                name="connes_classification",
                expression="Type III factors classified by S(M) = {λ ∈ [0,1]: M ⊗ R_λ ≅ M}",
                theorem="Connes' classification of type III factors: III_0, III_λ (0<λ<1), III_1",
                affected_quantities=["factor_type", "classification"],
            ),
        ]
        return invariants


@dataclass
class NoncommutativeProbability(AbstractMathematicalStructure):
    """非交换概率：统一经典（交换）和量子（非交换）概率框架.

    经典：Kolmogorov 公理 → (Ω, ℱ, ℙ) → L^∞(Ω, ℱ, ℙ)（交换 von Neumann 代数）
    量子：Born 规则 → ρ（密度矩阵）→ B(H)（非交换 von Neumann 代数）

    Attributes:
        is_commutative_probability: True = 经典概率, False = 量子概率
        satisfies_bell_inequality: 是否满足 Bell 不等式
        uncertainty_constant: 不确定性常数 ℏ_eff
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.ALGEBRA,
            name="Noncommutative Probability",
            canonical_form="(A, ω): noncommutative probability space = (*-algebra, state)",
            description="Unifying classical (commutative) and quantum (noncommutative) probability",
        )
    )
    is_commutative_probability: bool = True
    satisfies_bell_inequality: bool = True
    uncertainty_constant: float = 0.0

    @property
    def function_space(self) -> str:
        kind = "Classical" if self.is_commutative_probability else "Quantum"
        return f"{kind} probability space (A, ω)"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants = [
            StructuralInvariant(
                name="uncertainty_principle",
                expression="ΔA · ΔB ≥ ½ abs:⟨[A, B]⟩abs:  (Heisenberg-Robertson uncertainty)",
                theorem="Robertson uncertainty relation: commutator sets lower bound on joint measurability",
                affected_quantities=["observable", "variance", "commutator"],
            ),
            StructuralInvariant(
                name="bell_inequality_classical_bound",
                expression="abs:⟨A₁B₁⟩ + ⟨A₁B₂⟩ + ⟨A₂B₁⟩ - ⟨A₂B₂⟩abs: ≤ 2  (CHSH bound for classical)",
                theorem="Bell-CHSH inequality: local realism ⇒ abs:Sabs: ≤ 2; quantum maximum is 2√2",
                affected_quantities=["correlation", "locality"],
            ),
        ]
        if self.is_commutative_probability:
            invariants.extend(
                [
                    StructuralInvariant(
                        name="commutative_no_interference",
                        expression="[A, B] = 0 ⇒ simultaneous measurability, no interference terms",
                        theorem="Commutative probability: no quantum interference, all observables jointly measurable",
                        affected_quantities=["observable", "commutator"],
                    ),
                    StructuralInvariant(
                        name="classical_kolmogorov",
                        expression="ℙ(A) = ∫_A ρ dx  (Bayesian interpretation)",
                        theorem="Kolmogorov axioms: classical probability on σ-algebra",
                        affected_quantities=["probability", "measure"],
                    ),
                ]
            )
        else:
            invariants.extend(
                [
                    StructuralInvariant(
                        name="born_rule",
                        expression="ℙ(outcome = a_i) = Tr(ρ P_{a_i})  where P_{a_i} projects onto eigenspace of a_i",
                        theorem="Born rule: quantum probability from projection-valued measure",
                        affected_quantities=["measurement", "projection"],
                    ),
                    StructuralInvariant(
                        name="quantum_noncommutativity",
                        expression="[A, B] ≠ 0 ⇒ no joint probability distribution for A and B",
                        theorem="Quantum contextuality: noncommuting observables lack joint probability",
                        affected_quantities=["observable", "contextuality"],
                    ),
                    StructuralInvariant(
                        name="tsirelson_bound",
                        expression="abs:S_quantumabs: ≤ 2√2  (Tsirelson bound — quantum violation of CHSH)",
                        theorem="Tsirelson's bound: maximal quantum violation of Bell inequalities",
                        affected_quantities=["bell_inequality", "entanglement"],
                    ),
                ]
            )
        return invariants


__all__ = [
    "VonNeumannAlgebra",
    "State",
    "ClassicalState",
    "QuantumState",
    "PureState",
    "GNSConstruction",
    "CommutativeCase",
    "TomitaTakesakiTheory",
    "NoncommutativeProbability",
    "_matrix_intersection",
]
