"""Quantum ESPRESSO DFT 数学结构提取器。

从 Quantum ESPRESSO 密度泛函理论计算中提取数学结构，映射到 Math Schema v1.0。
核心: Kohn-Sham 方程、平面波展开、赝势、SCF 循环。
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


class QuantumEspressoExtractor(BaseEngineExtractor):
    """Quantum ESPRESSO DFT 数学结构提取器。

    QE 是开源 DFT 计算软件，使用平面波基组和赝势方法。
    支持 NC、Ultrasoft、PAW 赝势和多种交换关联泛函。

    参数:
        ecutwfc: 波函数截断能 (Ry)
        ecutrho: 电荷密度截断能 (Ry)
        k_points: k 点设置，如 {"grid": [4,4,4], "mode": "automatic"}
        pseudopotential: 赝势类型 (nc, ultrasoft, paw)
        n_atoms: 原子数
    """

    def __init__(
        self,
        ecutwfc: float = 0.0,
        ecutrho: float = 0.0,
        k_points: Optional[Dict[str, Any]] = None,
        pseudopotential: str = "nc",
        n_atoms: int = 0,
    ):
        self.ecutwfc = ecutwfc
        self.ecutrho = ecutrho
        self.k_points = k_points or {"grid": [1, 1, 1], "mode": "gamma"}
        self.pseudopotential = pseudopotential
        self.n_atoms = n_atoms

    @property
    def engine_name(self) -> str:
        return "quantum_espresso"

    @property
    def extractor_version(self) -> str:
        return "0.2.0"

    def extract(
        self, files: Dict[str, str], options: Optional[Dict[str, Any]] = None
    ) -> "MathSchema":
        """从 QE 文件提取数学结构。

        Args:
            files: 源文件字典，如 {"input": "pw.in"}
            options: 可选参数，覆盖构造函数中的设置

        Returns:
            MathSchema 对象
        """
        options = options or {}
        if "ecutwfc" in options:
            self.ecutwfc = options["ecutwfc"]
        if "ecutrho" in options:
            self.ecutrho = options["ecutrho"]
        if "k_points" in options:
            self.k_points = options["k_points"]
        if "pseudopotential" in options:
            self.pseudopotential = options["pseudopotential"]
        if "n_atoms" in options:
            self.n_atoms = options["n_atoms"]

        source_files = {"input": list(files.values())}
        return self.build_schema(source_files)

    # ------------------------------------------------------------------
    # 控制方程
    # ------------------------------------------------------------------

    def _extract_governing_equations(self) -> List[GoverningEquation]:
        """提取 Kohn-Sham 方程和平面波展开。"""
        equations = []

        # Kohn-Sham 方程
        equations.append(
            GoverningEquation(
                id="kohn_sham",
                type="eigenvalue_problem",
                name="Kohn-Sham Equations",
                mathematical_form="H_KS ψ_nk = ε_nk ψ_nk",
                variables=["wavefunction", "eigenvalue", "ks_hamiltonian"],
                parameters={
                    "form": "nonlinear_eigenvalue",
                    "self_consistent": True,
                },
                description="Kohn-Sham 本征值问题: DFT 核心方程",
            )
        )

        # Kohn-Sham 算子展开
        equations.append(
            GoverningEquation(
                id="ks_hamiltonian",
                type="operator_definition",
                name="Kohn-Sham Hamiltonian",
                mathematical_form="H_KS = -ℏ²∇²/2m + V_ext(r) + V_H[rho](r) + V_xc[rho](r)",
                variables=[
                    "kinetic_energy",
                    "external_potential",
                    "hartree_potential",
                    "exchange_correlation_potential",
                ],
                parameters={"self_consistent": True},
                description="Kohn-Sham 哈密顿量: 动能 + 外势 + Hartree 势 + 交换关联势",
            )
        )

        # 平面波展开
        equations.append(
            GoverningEquation(
                id="plane_wave_expansion",
                type="expansion_equation",
                name="Plane Wave Expansion",
                mathematical_form="ψ_nk(r) = Σ_G c_{nk+G} exp(i(k+G)·r)",
                variables=["wavefunction", "plane_wave_coefficients", "reciprocal_lattice"],
                parameters={
                    "ecutwfc": self.ecutwfc,
                    "basis_type": "plane_wave",
                },
                description=f"波函数在平面波基组下的展开 (ecutwfc={self.ecutwfc} Ry)",
            )
        )

        # 电荷密度
        equations.append(
            GoverningEquation(
                id="charge_density",
                type="density_construction",
                name="Charge Density Construction",
                mathematical_form="ρ(r) = Σ_nk f_nk |ψ_nk(r)|²",
                variables=["density", "occupation", "wavefunction"],
                parameters={"occupation": "smearing"},
                description="电荷密度由占据态波函数构造",
            )
        )

        # Hartree 势 (Poisson 方程)
        equations.append(
            GoverningEquation(
                id="poisson",
                type="partial_differential_equation",
                name="Poisson Equation (Hartree Potential)",
                mathematical_form="∇²V_H(r) = -4π ρ(r)",
                variables=["hartree_potential", "electron_density"],
                parameters={"form": "poisson_equation"},
                description="Hartree 势由电荷密度通过 Poisson 方程求解",
            )
        )

        return equations

    # ------------------------------------------------------------------
    # 边界条件
    # ------------------------------------------------------------------

    def _extract_boundary_conditions(self) -> List[BoundaryCondition]:
        """提取周期性边界条件 (Bloch 定理) 和赝势近似。"""
        bcs = []

        # 周期性边界条件 (Bloch 定理)
        kgrid = self.k_points.get("grid", [1, 1, 1])
        kmode = self.k_points.get("mode", "gamma")
        bcs.append(
            BoundaryCondition(
                id="bloch_periodicity",
                type="quasi_periodic",
                domain={
                    "geometric_region": "unit_cell",
                    "entity_type": "crystal_lattice",
                },
                mathematical_object=MathematicalObject(
                    field="bloch_wavefunction",
                    tensor_rank=1,
                    tensor_form="ψ_nk(r+R) = exp(ik·R) ψ_nk(r)",
                ),
                software_implementation={
                    "command": "K_POINTS",
                    "parameters": {"grid": kgrid, "mode": kmode},
                },
                dual_role={
                    "is_boundary_condition": True,
                    "is_external_drive": False,
                    "note": "Bloch 定理: 周期性势场中波函数的准周期性",
                },
                equivalent_formulations=[
                    {
                        "type": "reciprocal_space_sampling",
                        "form": f"Monkhorst-Pack grid: {kgrid}",
                    },
                ],
            )
        )

        # 赝势近似
        ps_form = {
            "nc": "V_ps(r) = V_local(r) + Σ_l |p_l⟩ ΔE_l ⟨p_l|  (norm-conserving)",
            "ultrasoft": "V_ps(r) = V_local(r) + Σ_ij |β_i⟩ D_ij ⟨β_j|  (ultrasoft)",
            "paw": "V_ps(r) = V_local(r) + Σ_ij |p_i⟩ D_ij ⟨p_j|  (PAW)",
        }.get(self.pseudopotential.lower(), f"V_ps ({self.pseudopotential})")

        bcs.append(
            BoundaryCondition(
                id="pseudopotential_approximation",
                type="approximation",
                domain={
                    "geometric_region": "core_region",
                    "entity_type": "ionic_cores",
                },
                mathematical_object=MathematicalObject(
                    field="pseudopotential",
                    tensor_rank=0,
                    tensor_form=ps_form,
                ),
                software_implementation={
                    "command": "ATOMIC_SPECIES",
                    "parameters": {"pseudo_type": self.pseudopotential},
                },
                dual_role={
                    "is_boundary_condition": True,
                    "is_external_drive": False,
                    "note": "赝势近似: 用光滑有效势替代核区全电子势，减少平面波需求",
                },
            )
        )

        return bcs

    # ------------------------------------------------------------------
    # 本构关系
    # ------------------------------------------------------------------

    def _extract_constitutive_relations(self) -> List[Dict[str, Any]]:
        """提取交换关联泛函和 Hartree 势。"""
        relations = []

        # 交换关联泛函
        relations.append({
            "type": "exchange_correlation",
            "name": "XC Functional",
            "mathematical_form": "V_xc(r) = δE_xc[ρ]/δρ(r)",
            "parameters": {
                "functional_type": "gga",
                "common_choices": ["PBE", "PBEsol", "BLYP", "LDA", "SCAN", "HSE"],
            },
            "description": "交换关联势: 交换关联泛函对密度的泛函导数",
        })

        # Hartree 势
        relations.append({
            "type": "hartree_potential",
            "name": "Hartree Potential",
            "mathematical_form": "V_H(r) = ∫ ρ(r') / |r - r'| dr'",
            "parameters": {"method": "poisson_solver"},
            "description": "Hartree 势: 电子间经典 Coulomb 排斥的平均场",
        })

        # 赝势局域部分
        relations.append({
            "type": "pseudopotential",
            "name": f"Pseudopotential ({self.pseudopotential.upper()})",
            "mathematical_form": "V_eff = V_local + Σ_ij |β_i⟩ D_ij ⟨β_j| + Σ_ij |p_i⟩ D_ij ⟨p_j|",
            "parameters": {"pseudo_type": self.pseudopotential},
            "description": "赝势: 局域势 + 非局域投影算子",
        })

        return relations

    # ------------------------------------------------------------------
    # 数值方法
    # ------------------------------------------------------------------

    def _extract_numerical_method(self) -> NumericalMethod:
        """提取平面波基组、FFT、迭代对角化。"""
        method = NumericalMethod()

        method.discretization.space_discretization = "plane_wave_basis"
        method.discretization.time_integrator = "scf_iteration"
        method.discretization.order = 1

        if self.ecutwfc > 0:
            method.discretization.parameters = {
                "ecutwfc": self.ecutwfc,
                "ecutrho": self.ecutrho if self.ecutrho > 0 else 4 * self.ecutwfc,
            }

        method.solver = Solver(
            algorithm="davidson_diagonalization",
            convergence_criterion="energy_residual",
            tolerance=1e-6,
            max_iterations=100,
        )

        method.parallelization = {
            "fft_parallel": True,
            "kpoint_parallel": True,
            "band_parallel": True,
            "plane_wave_parallel": True,
        }

        return method

    # ------------------------------------------------------------------
    # 计算图
    # ------------------------------------------------------------------

    def _extract_computational_graph(self) -> ComputationalGraph:
        """提取 SCF 循环计算图 (含 FFT 节点)。"""
        graph = ComputationalGraph()

        # 电荷密度混合
        graph.add_node(ComputationalNode(
            id="density_mixing",
            type="density_mixing",
            math_semantics={
                "operator_type": "pulay_mixing",
                "updates": {"target": "ρ", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # FFT: 实空间 → 倒空间
        graph.add_node(ComputationalNode(
            id="fft_forward",
            type="transform",
            math_semantics={
                "operator_type": "fft_r_to_g",
                "updates": {"target": "ρ(G)", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # Hartree 势计算 (倒空间)
        graph.add_node(ComputationalNode(
            id="hartree_potential",
            type="construct",
            math_semantics={
                "operator_type": "poisson_solver_gspace",
                "updates": {"target": "V_H(G)", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # 交换关联势计算 (实空间)
        graph.add_node(ComputationalNode(
            id="xc_potential",
            type="construct",
            math_semantics={
                "operator_type": "xc_functional_evaluation",
                "updates": {"target": "V_xc(r)", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # Kohn-Sham 哈密顿量构建
        graph.add_node(ComputationalNode(
            id="ks_hamiltonian",
            type="construct",
            math_semantics={
                "operator_type": "ks_hamiltonian_construction",
                "updates": {"target": "H_KS", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # 迭代对角化
        graph.add_node(ComputationalNode(
            id="diagonalization",
            type="eigensolver",
            math_semantics={
                "operator_type": "davidson",
                "updates": {"target": "ψ_nk", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # 电荷密度更新
        graph.add_node(ComputationalNode(
            id="density_update",
            type="density_computation",
            math_semantics={
                "operator_type": "sum_occupied_states",
                "updates": {"target": "ρ_new", "mode": UpdateMode.EXPLICIT_UPDATE.value},
                "convergence": {
                    "required": True,
                    "criterion": "energy_change < conv_thr",
                    "max_iterations": 100,
                },
            },
        ))

        # SCF 收敛检查
        graph.add_node(ComputationalNode(
            id="scf_convergence",
            type="convergence_check",
            math_semantics={
                "operator_type": "residual_evaluation",
                "updates": {"target": "ΔE", "mode": UpdateMode.IMPLICIT_LOOP.value},
                "convergence": {
                    "required": True,
                    "criterion": "energy_change < conv_thr",
                },
            },
        ))

        # 边: 密度混合 → FFT
        graph.add_edge(ComputationalEdge(
            from_node="density_mixing", to_node="fft_forward",
            data_type="density_real", dependency="transform",
        ))

        # 边: FFT → Hartree 势
        graph.add_edge(ComputationalEdge(
            from_node="fft_forward", to_node="hartree_potential",
            data_type="density_reciprocal", dependency="construct",
        ))

        # 边: 密度混合 → XC 势
        graph.add_edge(ComputationalEdge(
            from_node="density_mixing", to_node="xc_potential",
            data_type="density_real", dependency="construct",
        ))

        # 边: Hartree 势 + XC 势 → KS 哈密顿量
        graph.add_edge(ComputationalEdge(
            from_node="hartree_potential", to_node="ks_hamiltonian",
            data_type="hartree_potential", dependency="construct",
        ))
        graph.add_edge(ComputationalEdge(
            from_node="xc_potential", to_node="ks_hamiltonian",
            data_type="xc_potential", dependency="construct",
        ))

        # 边: KS 哈密顿量 → 对角化
        graph.add_edge(ComputationalEdge(
            from_node="ks_hamiltonian", to_node="diagonalization",
            data_type="hamiltonian", dependency="solve",
        ))

        # 边: 对角化 → 密度更新
        graph.add_edge(ComputationalEdge(
            from_node="diagonalization", to_node="density_update",
            data_type="wavefunctions", dependency="compute",
        ))

        # 边: 密度更新 → 收敛检查
        graph.add_edge(ComputationalEdge(
            from_node="density_update", to_node="scf_convergence",
            data_type="density", dependency="compute",
        ))

        # 边: 收敛检查 → 密度混合 (SCF 循环)
        graph.add_edge(ComputationalEdge(
            from_node="scf_convergence", to_node="density_mixing",
            data_type="convergence_status", dependency="feedback",
        ))

        graph.execution_topology = {
            "schedule": "scf_loop",
            "implicit_loops": [
                {
                    "loop_id": "scf_cycle",
                    "nested_in": "density_update",
                    "convergence_guarantee": "none_for_general_systems",
                    "max_iterations_source": "electron_maxstep",
                },
            ],
        }

        return graph

    # ------------------------------------------------------------------
    # 守恒性质
    # ------------------------------------------------------------------

    def _extract_conservation_properties(self) -> Dict[str, Any]:
        """提取总能量变分下界和电荷守恒。"""
        return {
            "energy_variational": {
                "preserved": True,
                "mechanism": "hohenberg_kohn_theorem",
                "note": "基态密度使能量泛函取极小值 (变分原理)",
            },
            "charge_conservation": {
                "preserved": True,
                "mechanism": "poisson_solver",
                "note": "总电子数守恒: ∫ρ(r)dr = N_electrons",
            },
        }

    # ------------------------------------------------------------------
    # 原始符号
    # ------------------------------------------------------------------

    def _extract_raw_symbols(self) -> Dict[str, Any]:
        """提取原始数学符号。"""
        return {
            "H_KS": "Kohn-Sham 哈密顿量",
            "ψ_nk": "Bloch 波函数",
            "ε_nk": "Kohn-Sham 本征值",
            "ρ": "电子密度",
            "V_H": "Hartree 势",
            "V_xc": "交换关联势",
            "G": "倒格矢",
            "k": "Bloch 波矢",
            "ecutwfc": self.ecutwfc,
            "ecutrho": self.ecutrho,
            "k_points": self.k_points,
            "pseudopotential": self.pseudopotential,
            "n_atoms": self.n_atoms,
        }
