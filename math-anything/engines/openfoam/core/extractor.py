"""OpenFOAM CFD 数学结构提取器。

从 OpenFOAM 计算流体力学仿真中提取数学结构，映射到 Math Schema v1.0。
核心: Navier-Stokes 守恒律、湍流建模、有限体积离散化。
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


class OpenFOAMExtractor(BaseEngineExtractor):
    """OpenFOAM CFD 数学结构提取器。

    OpenFOAM 使用有限体积法在任意多面体网格上求解守恒律。
    支持 RANS、LES、DNS 湍流模型和 SIMPLE/PISO 压力-速度耦合。

    参数:
        solver_type: 求解器类型 (simpleFoam, pisoFoam, pimpleFoam, ...)
        turbulence_model: 湍流模型 (kEpsilon, kOmega, smagorinsky, dns, ...)
        reynolds_number: 雷诺数
        mesh_type: 网格类型 (polyhedral, hexahedral, tetrahedral)
    """

    def __init__(
        self,
        solver_type: str = "simpleFoam",
        turbulence_model: str = "kEpsilon",
        reynolds_number: float = 0.0,
        mesh_type: str = "polyhedral",
    ):
        self.solver_type = solver_type
        self.turbulence_model = turbulence_model
        self.reynolds_number = reynolds_number
        self.mesh_type = mesh_type

    @property
    def engine_name(self) -> str:
        return "openfoam"

    @property
    def extractor_version(self) -> str:
        return "0.2.0"

    def extract(
        self, files: Dict[str, str], options: Optional[Dict[str, Any]] = None
    ) -> "MathSchema":
        """从 OpenFOAM 文件提取数学结构。

        Args:
            files: 源文件字典，如 {"fvSchemes": "...", "fvSolution": "..."}
            options: 可选参数，覆盖构造函数中的设置

        Returns:
            MathSchema 对象
        """
        options = options or {}
        if "solver_type" in options:
            self.solver_type = options["solver_type"]
        if "turbulence_model" in options:
            self.turbulence_model = options["turbulence_model"]
        if "reynolds_number" in options:
            self.reynolds_number = options["reynolds_number"]
        if "mesh_type" in options:
            self.mesh_type = options["mesh_type"]

        source_files = {"input": list(files.values())}
        return self.build_schema(source_files)

    # ------------------------------------------------------------------
    # 控制方程
    # ------------------------------------------------------------------

    def _extract_governing_equations(self) -> List[GoverningEquation]:
        """提取 Navier-Stokes 方程和连续性方程。"""
        equations = []

        # 连续性方程 (不可压)
        is_compressible = "rho" in self.solver_type.lower() or "buoyant" in self.solver_type.lower()
        if is_compressible:
            cont_form = "∂ρ/∂t + ∇·(ρu) = 0"
            cont_params = {"type": "compressible"}
        else:
            cont_form = "∇·u = 0"
            cont_params = {"type": "incompressible"}

        equations.append(
            GoverningEquation(
                id="continuity",
                type="conservation_law",
                name="Mass Conservation (Continuity)",
                mathematical_form=cont_form,
                variables=["velocity", "density"],
                parameters=cont_params,
                description="质量守恒: 连续性方程",
            )
        )

        # 动量方程 (Navier-Stokes)
        if is_compressible:
            mom_form = "∂(ρu)/∂t + ∇·(ρuu) = -∇p + ∇·τ + ρg"
        else:
            mom_form = "∂u/∂t + ∇·(uu) = -∇p/ρ + ∇·(ν_eff ∇u) + f"

        equations.append(
            GoverningEquation(
                id="navier_stokes",
                type="partial_differential_equation",
                name="Navier-Stokes Momentum Equation",
                mathematical_form=mom_form,
                variables=["velocity", "pressure", "effective_viscosity", "density"],
                parameters={
                    "type": "navier_stokes",
                    "form": "conservation",
                    "compressible": is_compressible,
                },
                description="Navier-Stokes 动量守恒方程",
            )
        )

        # 能量方程 (可压)
        if is_compressible:
            equations.append(
                GoverningEquation(
                    id="energy",
                    type="conservation_law",
                    name="Energy Conservation",
                    mathematical_form="∂(ρe)/∂t + ∇·(ρue) = -p∇·u + ∇·(k_eff ∇T) + τ:∇u + S_e",
                    variables=["energy", "temperature", "velocity", "thermal_conductivity"],
                    parameters={"type": "compressible"},
                    description="能量守恒: 对流 + 扩散 + 压力功 + 粘性耗散",
                )
            )

        # 湍流模型方程
        turb = self.turbulence_model.lower()
        if turb in ("kepsilon", "k-epsilon", "kepsilon"):
            equations.extend(self._build_kepsilon_equations())
        elif turb in ("komega", "k-omega", "komegasst", "k-omega-sst"):
            equations.extend(self._build_komega_equations())
        elif turb in ("smagorinsky",):
            equations.append(
                GoverningEquation(
                    id="les_filter",
                    type="filter_operation",
                    name="LES Spatial Filter",
                    mathematical_form="ū(x) = ∫ G(x, x') u(x') dx'",
                    variables=["filtered_velocity", "filter_kernel"],
                    parameters={"model": "smagorinsky", "filter_type": "top_hat"},
                    description="LES 大涡模拟空间滤波: 分离大尺度和小尺度涡",
                )
            )

        return equations

    def _build_kepsilon_equations(self) -> List[GoverningEquation]:
        """构建 k-ε 湍流模型方程。"""
        return [
            GoverningEquation(
                id="k_transport",
                type="transport_equation",
                name="Turbulent Kinetic Energy (k) Transport",
                mathematical_form="∂k/∂t + u·∇k = ∇·((ν+ν_t/σ_k)∇k) + P_k - ε",
                variables=["turbulent_kinetic_energy", "velocity", "dissipation_rate"],
                parameters={"model": "kEpsilon", "sigma_k": 1.0},
                description="湍动能 k 输运方程: 产生 - 耗散平衡",
            ),
            GoverningEquation(
                id="epsilon_transport",
                type="transport_equation",
                name="Dissipation Rate (ε) Transport",
                mathematical_form="∂ε/∂t + u·∇ε = ∇·((ν+ν_t/σ_ε)∇ε) + C₁ε/k P_k - C₂ε²/k",
                variables=["dissipation_rate", "turbulent_kinetic_energy", "velocity"],
                parameters={"model": "kEpsilon", "C1": 1.44, "C2": 1.92, "sigma_eps": 1.3},
                description="耗散率 ε 输运方程: 湍流长度尺度演化",
            ),
        ]

    def _build_komega_equations(self) -> List[GoverningEquation]:
        """构建 k-ω 湍流模型方程。"""
        return [
            GoverningEquation(
                id="k_transport",
                type="transport_equation",
                name="Turbulent Kinetic Energy (k) Transport",
                mathematical_form="∂k/∂t + u·∇k = ∇·((ν+ν_t/σ_k)∇k) + P_k - β* k ω",
                variables=["turbulent_kinetic_energy", "velocity", "specific_dissipation"],
                parameters={"model": "kOmega", "sigma_k": 0.5, "beta_star": 0.09},
                description="k-ω 模型湍动能输运方程",
            ),
            GoverningEquation(
                id="omega_transport",
                type="transport_equation",
                name="Specific Dissipation Rate (ω) Transport",
                mathematical_form="∂ω/∂t + u·∇ω = ∇·((ν+ν_t/σ_ω)∇ω) + α P_k/k - β ω²",
                variables=["specific_dissipation", "turbulent_kinetic_energy", "velocity"],
                parameters={"model": "kOmega", "alpha": 0.52, "beta": 0.072},
                description="比耗散率 ω 输运方程",
            ),
        ]

    # ------------------------------------------------------------------
    # 边界条件
    # ------------------------------------------------------------------

    def _extract_boundary_conditions(self) -> List[BoundaryCondition]:
        """提取 CFD 边界条件 (no-slip, inlet/outlet, periodic)。"""
        bcs = []

        # 无滑移壁面
        bcs.append(
            BoundaryCondition(
                id="no_slip_wall",
                type="dirichlet",
                domain={
                    "geometric_region": "wall_boundaries",
                    "entity_type": "wall_patch",
                },
                mathematical_object=MathematicalObject(
                    field="velocity",
                    tensor_rank=1,
                    tensor_form="u|_wall = 0",
                ),
                software_implementation={
                    "command": "boundaryField",
                    "parameters": {"type": "fixedValue", "value": "uniform (0 0 0)"},
                },
                dual_role={
                    "is_boundary_condition": True,
                    "is_external_drive": False,
                    "note": "粘性流体壁面无滑移条件",
                },
            )
        )

        # 入口边界
        bcs.append(
            BoundaryCondition(
                id="inlet",
                type="dirichlet",
                domain={
                    "geometric_region": "inlet_patch",
                    "entity_type": "inlet_boundary",
                },
                mathematical_object=MathematicalObject(
                    field="velocity",
                    tensor_rank=1,
                    tensor_form="u|_inlet = U_in(t)",
                ),
                software_implementation={
                    "command": "boundaryField",
                    "parameters": {"type": "fixedValue", "value": "uniform U_in"},
                },
                dual_role={
                    "is_boundary_condition": True,
                    "is_external_drive": True,
                    "note": "入口速度驱动流动",
                },
            )
        )

        # 出口边界
        bcs.append(
            BoundaryCondition(
                id="outlet",
                type="neumann",
                domain={
                    "geometric_region": "outlet_patch",
                    "entity_type": "outlet_boundary",
                },
                mathematical_object=MathematicalObject(
                    field="pressure",
                    tensor_rank=0,
                    tensor_form="∂p/∂n|_outlet = 0",
                ),
                software_implementation={
                    "command": "boundaryField",
                    "parameters": {"type": "zeroGradient"},
                },
                dual_role={
                    "is_boundary_condition": True,
                    "is_external_drive": False,
                    "note": "出口零梯度条件: 充分发展流动",
                },
            )
        )

        # 周期性边界
        bcs.append(
            BoundaryCondition(
                id="periodic",
                type="periodic",
                domain={
                    "geometric_region": "cyclic_patches",
                    "entity_type": "cyclic_boundary",
                },
                mathematical_object=MathematicalObject(
                    field="velocity_pressure",
                    tensor_rank=0,
                    tensor_form="u(x) = u(x + L), p(x) = p(x + L)",
                ),
                software_implementation={
                    "command": "boundaryField",
                    "parameters": {"type": "cyclic"},
                },
                dual_role={
                    "is_boundary_condition": True,
                    "is_external_drive": False,
                    "note": "周期性边界: 模拟无限域中的周期结构",
                },
            )
        )

        return bcs

    # ------------------------------------------------------------------
    # 本构关系
    # ------------------------------------------------------------------

    def _extract_constitutive_relations(self) -> List[Dict[str, Any]]:
        """提取 Newton 流体本构和湍流闭合。"""
        relations = []

        # Newton 流体本构
        relations.append({
            "type": "constitutive",
            "name": "Newtonian Fluid",
            "mathematical_form": "τ = μ(∇u + ∇uᵀ) - 2/3 μ(∇·u)I",
            "parameters": {
                "fluid_type": "newtonian",
                "compressible": "rho" in self.solver_type.lower(),
            },
            "description": "Newton 流体: 应力与应变率线性关系",
        })

        # 湍流闭合
        turb = self.turbulence_model.lower()
        if turb in ("kepsilon", "k-epsilon"):
            relations.append({
                "type": "turbulence_closure",
                "name": "Boussinesq Eddy Viscosity (k-ε)",
                "mathematical_form": "ν_t = C_μ k²/ε,  τ_t = -2/3 ρk I + 2ρν_t S",
                "parameters": {"C_mu": 0.09, "model": "kEpsilon"},
                "description": "Boussinesq 假设: 涡粘性由湍动能和耗散率决定",
            })
        elif turb in ("komega", "k-omega", "komegasst", "k-omega-sst"):
            relations.append({
                "type": "turbulence_closure",
                "name": "Boussinesq Eddy Viscosity (k-ω)",
                "mathematical_form": "ν_t = k/ω,  τ_t = -2/3 ρk I + 2ρν_t S",
                "parameters": {"model": "kOmega"},
                "description": "k-ω 模型涡粘性闭合",
            })
        elif turb in ("smagorinsky",):
            relations.append({
                "type": "turbulence_closure",
                "name": "Smagorinsky SGS Model",
                "mathematical_form": "ν_sgs = (C_s Δ)² |S̄|,  |S̄| = √(2 S̄:S̄)",
                "parameters": {"C_s": 0.167, "model": "smagorinsky"},
                "description": "Smagorinsky 亚格子模型: SGS 粘性由滤波应变率计算",
            })

        return relations

    # ------------------------------------------------------------------
    # 数值方法
    # ------------------------------------------------------------------

    def _extract_numerical_method(self) -> NumericalMethod:
        """提取 FVM 离散和 SIMPLE/PISO 算法。"""
        method = NumericalMethod()

        # 有限体积离散
        method.discretization.space_discretization = f"finite_volume_{self.mesh_type}"
        solver_lower = self.solver_type.lower()
        if "simple" in solver_lower:
            method.discretization.time_integrator = "steady_state"
        elif "piso" in solver_lower:
            method.discretization.time_integrator = "implicit_euler"
        elif "pimple" in solver_lower:
            method.discretization.time_integrator = "crank_nicolson"
        else:
            method.discretization.time_integrator = "implicit_euler"

        method.discretization.order = 2

        # 压力-速度耦合算法
        if "simple" in solver_lower:
            algo = "SIMPLE"
            conv_criterion = "residual_norm"
        elif "piso" in solver_lower:
            algo = "PISO"
            conv_criterion = "continuity_residual"
        elif "pimple" in solver_lower:
            algo = "PIMPLE"
            conv_criterion = "residual_norm"
        else:
            algo = solver_lower
            conv_criterion = "residual"

        method.solver = Solver(
            algorithm=algo,
            convergence_criterion=conv_criterion,
            tolerance=1e-6,
            max_iterations=1000,
        )

        method.parallelization = {
            "domain_decomposition": True,
            "mpi_parallel": True,
            "gauss_theorem": True,
        }

        return method

    # ------------------------------------------------------------------
    # 计算图
    # ------------------------------------------------------------------

    def _extract_computational_graph(self) -> ComputationalGraph:
        """提取 FVM 计算图: 网格→系数→线性系统→解→更新。"""
        graph = ComputationalGraph()

        # 网格构建
        graph.add_node(ComputationalNode(
            id="mesh_construction",
            type="construct",
            math_semantics={
                "operator_type": "mesh_generation",
                "updates": {"target": "mesh", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # 系数矩阵组装
        graph.add_node(ComputationalNode(
            id="coefficient_assembly",
            type="construct",
            math_semantics={
                "operator_type": "fvm_discretization",
                "updates": {"target": "A, b", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # 动量预测
        graph.add_node(ComputationalNode(
            id="momentum_predictor",
            type="solve",
            math_semantics={
                "operator_type": "momentum_predictor",
                "updates": {"target": "U", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # 压力修正
        graph.add_node(ComputationalNode(
            id="pressure_correction",
            type="solve",
            math_semantics={
                "operator_type": "pressure_poisson",
                "updates": {"target": "p", "mode": UpdateMode.IMPLICIT_LOOP.value},
                "convergence": {
                    "required": True,
                    "criterion": "mass_conservation_residual",
                },
            },
        ))

        # 速度修正
        graph.add_node(ComputationalNode(
            id="velocity_correction",
            type="update",
            math_semantics={
                "operator_type": "flux_correction",
                "updates": {"target": "U", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # 湍流求解
        graph.add_node(ComputationalNode(
            id="turbulence_solve",
            type="solve",
            math_semantics={
                "operator_type": "turbulence_transport",
                "updates": {"target": "k, ε", "mode": UpdateMode.EXPLICIT_UPDATE.value},
            },
        ))

        # 边: 网格 → 系数组装
        graph.add_edge(ComputationalEdge(
            from_node="mesh_construction", to_node="coefficient_assembly",
            data_type="mesh_geometry", dependency="construct",
        ))

        # 边: 系数组装 → 动量预测
        graph.add_edge(ComputationalEdge(
            from_node="coefficient_assembly", to_node="momentum_predictor",
            data_type="momentum_coefficients", dependency="solve",
        ))

        # 边: 动量预测 → 压力修正
        graph.add_edge(ComputationalEdge(
            from_node="momentum_predictor", to_node="pressure_correction",
            data_type="momentum_flux", dependency="construct",
        ))

        # 边: 压力修正 → 速度修正
        graph.add_edge(ComputationalEdge(
            from_node="pressure_correction", to_node="velocity_correction",
            data_type="pressure_gradient", dependency="compute",
        ))

        # 边: 速度修正 → 湍流求解
        graph.add_edge(ComputationalEdge(
            from_node="velocity_correction", to_node="turbulence_solve",
            data_type="velocity_field", dependency="compute",
        ))

        # 边: 湍流求解 → 系数组装 (反馈: 涡粘性更新系数)
        graph.add_edge(ComputationalEdge(
            from_node="turbulence_solve", to_node="coefficient_assembly",
            data_type="eddy_viscosity", dependency="feedback",
        ))

        # 边: 压力修正 → 系数组装 (SIMPLE 循环)
        graph.add_edge(ComputationalEdge(
            from_node="pressure_correction", to_node="coefficient_assembly",
            data_type="corrected_pressure", dependency="feedback",
        ))

        graph.execution_topology = {
            "schedule": "sequential_per_timestep",
            "implicit_loops": [
                {
                    "loop_id": "pressure_velocity_coupling",
                    "nested_in": "pressure_correction",
                    "convergence_guarantee": "user_controlled",
                    "max_iterations_source": "fvSolution",
                },
            ],
        }

        return graph

    # ------------------------------------------------------------------
    # 守恒性质
    # ------------------------------------------------------------------

    def _extract_conservation_properties(self) -> Dict[str, Any]:
        """提取质量、动量、能量守恒。"""
        props = {
            "mass": {
                "preserved": True,
                "mechanism": "continuity_equation",
                "note": "FVM 天然满足局部质量守恒 (Gauss 定理)",
            },
            "momentum": {
                "preserved": True,
                "mechanism": "navier_stokes_equation",
                "note": "动量守恒由 Navier-Stokes 方程保证",
            },
        }

        is_compressible = "rho" in self.solver_type.lower() or "buoyant" in self.solver_type.lower()
        if is_compressible:
            props["energy"] = {
                "preserved": True,
                "mechanism": "energy_equation",
                "note": "可压流动能量守恒",
            }

        if self.reynolds_number > 0:
            props["reynolds_number"] = {
                "preserved": True,
                "mechanism": "dimensional_analysis",
                "note": f"Re = {self.reynolds_number} (层流→湍流转捩 Re~2300)",
            }

        return props

    # ------------------------------------------------------------------
    # 原始符号
    # ------------------------------------------------------------------

    def _extract_raw_symbols(self) -> Dict[str, Any]:
        """提取原始数学符号。"""
        return {
            "u": "速度矢量",
            "p": "压力",
            "ρ": "密度",
            "μ": "动力粘度",
            "ν": "运动粘度",
            "k": "湍动能",
            "ε": "耗散率",
            "ν_t": "涡粘性",
            "solver_type": self.solver_type,
            "turbulence_model": self.turbulence_model,
            "reynolds_number": self.reynolds_number,
            "mesh_type": self.mesh_type,
        }
