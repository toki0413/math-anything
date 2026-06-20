"""Gaussian 量子化学数学结构提取器。

从 Gaussian 量子化学计算中提取数学结构，映射到 Math Schema v1.0。
核心: Hartree-Fock / post-HF 方法，Gaussian 基组离散化。
"""

from typing import Any, Dict, List, Optional

from math_anything.schemas import (
    BoundaryCondition,
    ComputationalEdge,
    ComputationalGraph,
    ComputationalNode,
    GoverningEquation,
    MathematicalObject,
    NumericalMethod,
    Solver,
    UpdateMode,
)

from engines.base import BaseEngineExtractor


class GaussianExtractor(BaseEngineExtractor):
    """Gaussian 量子化学数学结构提取器。

    支持 Hartree-Fock、DFT、MP2、CCSD(T) 等方法，
    使用 Gaussian 基组 (GTO) 进行离散化。

    参数:
        method: 计算方法 (hf, dft, mp2, ccsd, ccsdt, cisd)
        basis: 基组名称 (sto-3g, 6-31g*, cc-pvdz, ...)
        charge: 分子电荷
        multiplicity: 自旋多重度
        n_atoms: 原子数
    """

    def __init__(
        self,
        method: str = "hf",
        basis: str = "sto-3g",
        charge: int = 0,
        multiplicity: int = 1,
        n_atoms: int = 0,
    ):
        self.method = method
        self.basis = basis
        self.charge = charge
        self.multiplicity = multiplicity
        self.n_atoms = n_atoms

    @property
    def engine_name(self) -> str:
        return "gaussian"

    @property
    def extractor_version(self) -> str:
        return "0.2.0"

    def extract(
        self, files: Dict[str, str], options: Optional[Dict[str, Any]] = None
    ) -> "MathSchema":
        """从 Gaussian 文件提取数学结构。

        Args:
            files: 源文件字典，如 {"input": "molecule.gjf"}
            options: 可选参数，覆盖构造函数中的 method/basis 等

        Returns:
            MathSchema 对象
        """
        options = options or {}
        if "method" in options:
            self.method = options["method"]
        if "basis" in options:
            self.basis = options["basis"]
        if "charge" in options:
            self.charge = options["charge"]
        if "multiplicity" in options:
            self.multiplicity = options["multiplicity"]
        if "n_atoms" in options:
            self.n_atoms = options["n_atoms"]

        source_files = {"input": list(files.values())}
        return self.build_schema(source_files)

    # ------------------------------------------------------------------
    # 控制方程
    # ------------------------------------------------------------------

    def _extract_governing_equations(self) -> List[GoverningEquation]:
        """提取 Roothaan-Hall 方程和 Fock 算子定义。"""
        equations = []

        # Roothaan-Hall 方程 (核心)
        equations.append(
            GoverningEquation(
                id="roothaan_hall",
                type="generalized_eigenvalue_problem",
                name="Roothaan-Hall Equations",
                mathematical_form="F C = S C ε",
                variables=[
                    "fock_matrix",
                    "coefficient_matrix",
                    "overlap_matrix",
                    "orbital_energies",
                ],
                parameters={
                    "form": "generalized_eigenvalue",
                    "self_consistent": True,
                    "method": self.method,
                },
                description="广义特征值问题: Fock 矩阵在 AO 基下的对角化",
            )
        )

        # Fock 算子定义
        is_dft = self.method.lower() not in ("hf", "mp2", "ccsd", "ccsdt", "cisd")
        xc_term = " + V_xc_μν" if is_dft else ""
        exchange_factor = "0.0" if is_dft else "1.0"
        equations.append(
            GoverningEquation(
                id="fock_operator",
                type="operator_definition",
                name="Fock Operator",
                mathematical_form=(
                    "F_μν = H_core_μν + Σ_λσ P_λσ [(μν|λσ) - c_x/2 (μλ|νσ)]"
                    + xc_term
                ),
                variables=[
                    "fock_matrix",
                    "core_hamiltonian",
                    "density_matrix",
                    "two_electron_integrals",
                ],
                parameters={
                    "hf_exchange_factor": exchange_factor,
                    "method": self.method,
                },
                description="Fock 算子: 单电子 + 双电子 Coulomb + 交换贡献"
                + (" + 交换关联势" if is_dft else ""),
            )
        )

        # 密度矩阵
        equations.append(
            GoverningEquation(
                id="density_matrix",
                type="density_construction",
                name="Density Matrix",
                mathematical_form="P_μν = 2 Σ_i^occ C_μi C_νi*",
                variables=["density_matrix", "coefficient_matrix"],
                parameters={"form": "sum_over_occupied"},
                description="闭壳层 AO 密度矩阵由占据 MO 系数构造",
            )
        )

        # 双电子积分
        equations.append(
            GoverningEquation(
                id="two_electron_integrals",
                type="integral_equation",
                name="Two-Electron Repulsion Integrals",
                mathematical_form="(μν|λσ) = ∫∫ χ_μ(r1) χ_ν(r1) r12⁻¹ χ_λ(r2) χ_σ(r2) dr1 dr2",
                variables=["basis_function", "eri"],
                parameters={"basis_type": "gaussian_type_orbitals"},
                description="Gaussian 基函数上的双电子排斥积分",
            )
        )

        # 基组展开
        equations.append(
            GoverningEquation(
                id="gto_expansion",
                type="expansion_equation",
                name="Gaussian Basis Set Expansion",
                mathematical_form=(
                    "χ_μ(r) = Σ_k d_μk (x-A_x)^a (y-A_y)^b (z-A_z)^c exp(-α_k|r-A|²)"
                ),
                variables=[
                    "basis_function",
                    "contraction_coefficient",
                    "exponent",
                    "angular_momentum",
                ],
                parameters={
                    "basis": self.basis,
                    "basis_type": "contracted_gto",
                },
                description=f"分子轨道在 {self.basis} 基组下的收缩 Gaussian 展开",
            )
        )

        # post-HF 方程
        if self.method.lower() in ("mp2", "ccsd", "ccsdt", "cisd"):
            equations.extend(self._build_post_hf_equations(self.method.lower()))

        return equations

    def _build_post_hf_equations(self, method: str) -> List[GoverningEquation]:
        """构建 post-HF 电子相关方程。"""
        equations = []

        if method == "mp2":
            equations.append(
                GoverningEquation(
                    id="mp2_correlation",
                    type="energy_correction",
                    name="MP2 Correlation Energy",
                    mathematical_form="E_MP2 = Σ_{ijab} |(ia||jb)|² / (ε_i + ε_j - ε_a - ε_b)",
                    variables=["orbital_energies", "eri", "correlation_energy"],
                    parameters={"order": 2, "reference": "hartree_fock"},
                    description="二阶 Møller-Plesset 微扰理论相关能修正",
                )
            )

        if method in ("ccsd", "ccsdt"):
            t_ops = "T_1 + T_2" if method == "ccsd" else "T_1 + T_2 + T_3"
            equations.append(
                GoverningEquation(
                    id="coupled_cluster",
                    type="energy_correction",
                    name=f"CCSD{'(T)' if method == 'ccsdt' else ''} Amplitude Equations",
                    mathematical_form=f"⟨Φ_μ|e⁻ᵀ H eᵀ|Φ₀⟩ = 0  (T = {t_ops})",
                    variables=["t_amplitudes", "cluster_operator", "hamiltonian"],
                    parameters={
                        "truncation": method,
                        "reference": "hartree_fock",
                    },
                    description="耦合簇振幅方程，通过指数算子参数化相关波函数",
                )
            )

        if method == "cisd":
            equations.append(
                GoverningEquation(
                    id="ci_expansion",
                    type="eigenvalue_problem",
                    name="CISD",
                    mathematical_form="H C = E C  (CI 空间: {Φ₀, Φ_i^a, Φ_{ij}^{ab}})",
                    variables=["hamiltonian_matrix", "ci_coefficients", "ci_energies"],
                    parameters={"excitation_level": "cisd", "size_consistent": False},
                    description="组态相互作用 (单双激发) 精确对角化",
                )
            )

        return equations

    # ------------------------------------------------------------------
    # 边界条件
    # ------------------------------------------------------------------

    def _extract_boundary_conditions(self) -> List[BoundaryCondition]:
        """提取 Born-Oppenheimer 近似和基组截断。"""
        bcs = []

        # Born-Oppenheimer 近似
        bcs.append(
            BoundaryCondition(
                id="born_oppenheimer",
                type="approximation",
                domain={
                    "geometric_region": "nuclear_coordinates",
                    "entity_type": "clamped_nuclei",
                },
                mathematical_object=MathematicalObject(
                    field="nuclear_wavefunction",
                    tensor_rank=0,
                    tensor_form="Ψ_total ≈ Ψ_elec(r; R) · Ψ_nuc(R)",
                ),
                software_implementation={
                    "command": "route_card",
                    "parameters": {"approximation": "born_oppenheimer"},
                },
                dual_role={
                    "is_boundary_condition": True,
                    "is_external_drive": False,
                    "note": "原子核坐标作为参数固定，电子波函数在固定核势场中求解",
                },
            )
        )

        # 基组截断
        bcs.append(
            BoundaryCondition(
                id="basis_truncation",
                type="discretization_boundary",
                domain={
                    "geometric_region": "hilbert_space",
                    "entity_type": "finite_basis",
                },
                mathematical_object=MathematicalObject(
                    field="basis_set",
                    tensor_rank=0,
                    tensor_form=f"span{{χ_μ}}, μ=1..N_basis ({self.basis})",
                ),
                software_implementation={
                    "command": "route_card",
                    "parameters": {"basis": self.basis},
                },
                dual_role={
                    "is_boundary_condition": True,
                    "is_external_drive": False,
                    "note": "完备 Hilbert 空间截断为有限基组张成的子空间",
                },
            )
        )

        return bcs

    # ------------------------------------------------------------------
    # 本构关系
    # ------------------------------------------------------------------

    def _extract_constitutive_relations(self) -> List[Dict[str, Any]]:
        """提取交换关联泛函和电子-核吸引势。"""
        relations = []

        is_dft = self.method.lower() not in ("hf", "mp2", "ccsd", "ccsdt", "cisd")

        if is_dft:
            relations.append({
                "type": "exchange_correlation",
                "name": self.method.upper(),
                "mathematical_form": "E_xc[n] = ∫ ε_xc(n(r), ∇n(r)) n(r) dr",
                "parameters": {"functional_type": "dft"},
                "description": "交换关联泛函: 密度泛函理论核心近似",
            })
        else:
            relations.append({
                "type": "exact_exchange",
                "name": "Hartree-Fock Exchange",
                "mathematical_form": "E_x = -1/2 Σ_{ij} (ii|jj) / (ε_i + ε_j)",
                "parameters": {"functional_type": "hf"},
                "description": "Hartree-Fock 精确交换: 无近似交换能",
            })

        # 电子-核吸引势
        relations.append({
            "type": "electron_nuclear_attraction",
            "name": "Nuclear Attraction",
            "mathematical_form": "V_ne = -Σ_A Σ_i Z_A / |r_i - R_A|",
            "parameters": {"n_atoms": self.n_atoms},
            "description": "电子与原子核之间的 Coulomb 吸引势",
        })

        # Mulliken 布居分析
        relations.append({
            "type": "population_analysis",
            "name": "Mulliken Population",
            "mathematical_form": "q_A = Z_A - Σ_{μ∈A} (PS)_μμ",
            "parameters": {"method": "mulliken"},
            "description": "Mulliken 布居分析: 原子电荷由密度矩阵和重叠矩阵计算",
        })

        return relations

    # ------------------------------------------------------------------
    # 数值方法
    # ------------------------------------------------------------------

    def _extract_numerical_method(self) -> NumericalMethod:
        """提取 SCF 迭代和 DIIS 收敛加速。"""
        method = NumericalMethod()

        method.discretization.space_discretization = "gaussian_basis_set"
        method.discretization.time_integrator = "scf_iteration"
        method.discretization.order = 1

        method.solver = Solver(
            algorithm="scf_diagonalization",
            convergence_criterion="density_matrix_residual",
            tolerance=1e-7,
            max_iterations=128,
        )

        method.parallelization = {
            "diis_acceleration": True,
            "integral_direct": True,
            "thread_parallel": True,
        }

        return method

    # ------------------------------------------------------------------
    # 计算图
    # ------------------------------------------------------------------

    def _extract_computational_graph(self) -> ComputationalGraph:
        """提取 SCF 循环计算图。"""
        graph = ComputationalGraph()

        # 积分计算节点
        graph.add_node(ComputationalNode(
            id="integral_evaluation",
            type="construct",
            math_semantics={
                "operator_type": "two_electron_integral_generation",
                "updates": {"target": "ERI", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # 初始猜测
        graph.add_node(ComputationalNode(
            id="initial_guess",
            type="initialize",
            math_semantics={
                "operator_type": "core_hamiltonian_guess",
                "updates": {"target": "P", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # Fock 矩阵构建
        graph.add_node(ComputationalNode(
            id="fock_build",
            type="construct",
            math_semantics={
                "operator_type": "fock_matrix_construction",
                "updates": {"target": "F", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # 对角化
        graph.add_node(ComputationalNode(
            id="diagonalization",
            type="eigensolve",
            math_semantics={
                "operator_type": "generalized_eigensolver",
                "updates": {"target": "C", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # 密度矩阵构建
        graph.add_node(ComputationalNode(
            id="density_matrix",
            type="update",
            math_semantics={
                "operator_type": "density_matrix_construction",
                "updates": {"target": "P", "mode": UpdateMode.IMPLICIT_LOOP.value},
                "convergence": {
                    "required": True,
                    "criterion": "rms_density_change < 1e-8",
                    "max_iterations": 128,
                },
            },
        ))

        # 收敛检查
        graph.add_node(ComputationalNode(
            id="convergence_check",
            type="convergence_check",
            math_semantics={
                "operator_type": "residual_evaluation",
                "updates": {"target": "delta_P", "mode": UpdateMode.IMPLICIT_LOOP.value},
                "convergence": {
                    "required": True,
                    "criterion": "rms_density_change < 1e-8",
                },
            },
        ))

        # 能量计算
        graph.add_node(ComputationalNode(
            id="energy_evaluation",
            type="compute",
            math_semantics={
                "operator_type": "electronic_energy",
                "updates": {"target": "E_total", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # 边: 积分 → Fock 构建
        graph.add_edge(ComputationalEdge(
            from_node="integral_evaluation", to_node="fock_build",
            data_type="two_electron_integrals", dependency="construct",
        ))

        # 边: 初始猜测 → Fock 构建
        graph.add_edge(ComputationalEdge(
            from_node="initial_guess", to_node="fock_build",
            data_type="density_matrix", dependency="construct",
        ))

        # 边: Fock → 对角化
        graph.add_edge(ComputationalEdge(
            from_node="fock_build", to_node="diagonalization",
            data_type="fock_matrix", dependency="solve",
        ))

        # 边: 对角化 → 密度矩阵
        graph.add_edge(ComputationalEdge(
            from_node="diagonalization", to_node="density_matrix",
            data_type="molecular_orbitals", dependency="compute",
        ))

        # 边: 密度矩阵 → 收敛检查
        graph.add_edge(ComputationalEdge(
            from_node="density_matrix", to_node="convergence_check",
            data_type="density_matrix", dependency="compute",
        ))

        # 边: 收敛检查 → Fock 构建 (SCF 循环)
        graph.add_edge(ComputationalEdge(
            from_node="convergence_check", to_node="fock_build",
            data_type="convergence_status", dependency="feedback",
        ))

        # 边: 收敛 → 能量
        graph.add_edge(ComputationalEdge(
            from_node="convergence_check", to_node="energy_evaluation",
            data_type="converged_density", dependency="compute",
        ))

        graph.execution_topology = {
            "schedule": "self_consistent_loop",
            "implicit_loops": [{
                "loop_id": "scf_cycle",
                "nested_in": "density_matrix",
                "convergence_guarantee": "user_controlled",
                "max_iterations_source": "scf_maxcycle",
            }],
        }

        return graph

    # ------------------------------------------------------------------
    # 守恒性质
    # ------------------------------------------------------------------

    def _extract_conservation_properties(self) -> Dict[str, Any]:
        """提取总能量守恒和粒子数守恒。"""
        return {
            "total_energy": {
                "preserved": True,
                "mechanism": "variational_principle",
                "note": "SCF 收敛后总能量满足变分下界",
            },
            "particle_number": {
                "preserved": True,
                "mechanism": "electron_count_constraint",
                "note": f"电子数 = Σ P_μν S_μν = N_elec (charge={self.charge}, mult={self.multiplicity})",
            },
        }

    # ------------------------------------------------------------------
    # 原始符号
    # ------------------------------------------------------------------

    def _extract_raw_symbols(self) -> Dict[str, Any]:
        """提取原始数学符号。"""
        return {
            "F": "Fock 矩阵",
            "C": "分子轨道系数矩阵",
            "S": "重叠矩阵",
            "ε": "轨道能量",
            "P": "密度矩阵",
            "E_total": "总能量",
            "method": self.method,
            "basis": self.basis,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "n_atoms": self.n_atoms,
        }
