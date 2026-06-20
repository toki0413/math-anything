"""结构-提取器桥接模块。

将现有的 EnhancedMathSchema 提取结果映射到新的数学结构类型系统，
并同时填充知识图谱。

这个模块使得新的结构系统与旧的提取器兼容共存。
"""

from __future__ import annotations

from typing import Any

from .categories.builder import KnowledgeGraphBuilder
from .categories.engine import CategoryEngine
from .categories.graph import MathKnowledgeGraph
from .dimensional.scaling_group import BuckinghamPiEngine, FluidDimensionAnalyzer, QMDimensionAnalyzer
from .morphisms.approximations import (
    BornOppenheimerApproximation,
    ClassicalLimitMorphism,
    ExchangeCorrelationApproximation,
    ForceFieldMorphism,
    IncompressibilityMorphism,
    KohnShamMapping,
    PlaneWaveTruncation,
    SCFIterationMorphism,
)
from .morphisms.discretizations import (
    FEMDiscretization,
    FVMDiscretization,
    ParticleDiscretization,
    SpectralDiscretization,
)
from .structures import (
    ConservedQuantity,
    HamiltonianSystem,
    NavierStokesProblem,
    SelfConsistentProblem,
    VariationalMinimizationProblem,
)


class StructureBridge:
    """将 EnhancedMathSchema → 数学结构 + 知识图谱.

    使用示例:
        bridge = StructureBridge()

        # 从 VASP 提取结果构建结构
        struct = bridge.build_from_vasp_schema(enhanced_schema)

        # 获取知识图谱
        kg = bridge.knowledge_graph

        # 查询
        results = kg.query("ENCUT")
    """

    def __init__(self, kg_root: str = "~/.math-anything/kg"):
        from pathlib import Path

        self.kg = MathKnowledgeGraph(Path(kg_root).expanduser())
        self.builder = KnowledgeGraphBuilder(self.kg)
        self.engine = CategoryEngine()
        self.pi_engine = BuckinghamPiEngine()

        # 注册所有已知态射
        self._register_known_morphisms()

    def _register_known_morphisms(self) -> None:
        """注册所有已知态射到范畴引擎."""
        morphisms = [
            BornOppenheimerApproximation(),
            KohnShamMapping(),
            PlaneWaveTruncation(),
            SCFIterationMorphism(),
            ExchangeCorrelationApproximation(),
            ClassicalLimitMorphism(),
            ForceFieldMorphism(),
            IncompressibilityMorphism(),
            SpectralDiscretization(),
            FEMDiscretization(),
            FVMDiscretization(),
            ParticleDiscretization(),
        ]
        for m in morphisms:
            self.engine.register_morphism(m)

    # ── VASP / DFT ──

    def build_from_vasp(self, params: dict[str, Any]) -> dict[str, Any]:
        """为 VASP DFT 计算构建完整的结构表示."""
        params.get("ENCUT", 520)
        params.get("ISMEAR", 1)

        # 创建 Kohn-Sham 结构
        struct = SelfConsistentProblem(
            operator_type="self_adjoint",
            spectrum_type="band" if params.get("kpoints") else "pure_point",
            variational=True,
            bounded_below=True,
            nonlinearity_source="density_dependent",
        )

        sid = self.builder.build_from_structure(struct)
        self.builder.build_from_engine("VASP", struct.name, params)

        # 注册态射链
        self.engine.link("born_oppenheimer", "FullManyBody", "ElectronicSchrodinger")
        self.engine.link("kohn_sham", "ElectronicSchrodinger", "KohnSham_Full")
        self.engine.link("plane_wave_truncation", "KohnSham_Full", "KohnSham_Truncated")
        self.engine.link("scf_iteration", "KohnSham_Truncated", "KohnSham_Converged")

        # 构建态射链到 KG
        for morph_name in ["born_oppenheimer", "kohn_sham", "plane_wave_truncation", "scf_iteration"]:
            m = self.engine.morphisms.get(morph_name)
            if m:
                self.builder.build_from_morphism(m, struct.name, struct.name)

        # 维度分析
        qm_analyzer = QMDimensionAnalyzer()
        pi_groups = qm_analyzer.analyze_dft(params)
        if pi_groups:
            self.builder.build_from_pi_groups([pg.to_dict() for pg in pi_groups], sid)

        return {
            "structure": struct.to_dict(),
            "knowledge_graph_stats": self.kg.stats(),
            "pi_groups": [pg.to_dict() for pg in pi_groups],
        }

    # ── LAMMPS / MD ──

    def build_from_lammps(self, params: dict[str, Any]) -> dict[str, Any]:
        """为 LAMMPS MD 构建完整的结构表示."""
        struct = HamiltonianSystem(
            phase_space_dim=3 * params.get("n_atoms", 1000),
            symplectic=True,
            reversible=True,
        )

        self.builder.build_from_structure(struct)
        self.builder.build_from_engine("LAMMPS", struct.name, params)

        self.engine.link("classical_limit", "QuantumDynamics", "ClassicalDynamics")
        self.engine.link("force_field", "AbInitioPotential", "EmpiricalForceField")

        for morph_name in ["classical_limit", "force_field"]:
            m = self.engine.morphisms.get(morph_name)
            if m:
                self.builder.build_from_morphism(m, struct.name, struct.name)

        return {
            "structure": struct.to_dict(),
            "knowledge_graph_stats": self.kg.stats(),
        }

    # ── CFD / Navier-Stokes ──

    def build_from_cfd(self, params: dict[str, Any]) -> dict[str, Any]:
        """为 CFD 仿真构建结构表示.

        可处理 OpenFOAM、ANSYS Fluent 等引擎。
        """
        regime = params.get("regime", "incompressible")
        turb_model = params.get("turbulence_model", "none")

        struct = NavierStokesProblem(
            regime=regime,
            turbulence_model=turb_model,
            include_energy=params.get("include_energy", False),
            include_gravity=params.get("include_gravity", False),
            include_surface_tension=params.get("include_surface_tension", False),
            reynolds_number=params.get("Re"),
            mach_number=params.get("Ma"),
            conserved_variables=[
                ConservedQuantity("mass", "ρ", {"M": 1, "L": -3}),
                ConservedQuantity("momentum", "ρu", {"M": 1, "L": -2, "T": -1}),
            ],
            has_diffusion=params.get("viscous", True),
        )

        sid = self.builder.build_from_structure(struct)
        engine_name = params.get("engine", "OpenFOAM").upper()
        self.builder.build_from_engine(engine_name, struct.name, params)

        # 态射链：根据 regime 和 turb_model 注册
        if "compressible" not in regime:
            self.engine.link("incompressibility", "CompressibleNS", "IncompressibleNS")
            m = self.engine.morphisms.get("incompressibility")
            if m:
                self.builder.build_from_morphism(m, "CompressibleNS", struct.name)

        # 维度分析
        fluid_analyzer = FluidDimensionAnalyzer()
        pi_groups = fluid_analyzer.analyze_ns(params)
        if pi_groups:
            self.builder.build_from_pi_groups([pg.to_dict() for pg in pi_groups], sid)

        return {
            "structure": struct.to_dict(),
            "knowledge_graph_stats": self.kg.stats(),
            "pi_groups": [pg.to_dict() for pg in pi_groups],
            "named_pi_groups": struct.named_pi_groups,
        }

    # ── FEM / 结构力学 ──

    def build_from_fem(self, params: dict[str, Any]) -> dict[str, Any]:
        """为 FEM 结构力学构建结构表示."""
        nonlinear = params.get("nonlinear", False)
        struct = VariationalMinimizationProblem(
            principle="minimum",
            convex=not nonlinear,
            material_nonlinear=nonlinear,
            geometric_nonlinear=nonlinear,
        )

        self.builder.build_from_structure(struct)
        engine_name = params.get("engine", "Abaqus").upper()
        self.builder.build_from_engine(engine_name, struct.name, params)

        self.engine.link("fem_discretization", "ContinuumMechanics", "DiscreteFEM")

        return {
            "structure": struct.to_dict(),
            "knowledge_graph_stats": self.kg.stats(),
        }

    # ── 查询接口 ──

    @property
    def knowledge_graph(self) -> MathKnowledgeGraph:
        return self.kg

    def query(self, text: str) -> dict:
        from .categories.query import GraphQueryEngine

        qe = GraphQueryEngine(self.kg)
        return qe.query(text)

    def impact_analysis(self, param: str) -> dict:
        from .categories.query import GraphQueryEngine

        qe = GraphQueryEngine(self.kg)
        return qe.impact(param)

    def root_cause(self, problem: str) -> dict:
        from .categories.query import GraphQueryEngine

        qe = GraphQueryEngine(self.kg)
        return qe.root_cause(problem)

    def cumulative_loss(self, from_struct: str, to_struct: str) -> dict:
        return self.engine.cumulative_loss(from_struct, to_struct)

    # ── 约束分析接口 ──

    def analyze_constraints(
        self,
        structure,
        params: dict[str, Any],
        morphism_chain: list[str] | None = None,
    ) -> dict[str, Any]:
        """完整约束分析：不变量评估 + 传播 + 边界检查 + 风险.

        Args:
            structure: 数学结构实例
            params: 当前参数
            morphism_chain: 态射链名称列表（可选）

        Returns:
            {
                "interior": {invariant_name: status},
                "boundary_risks": [{invariant, risk_score, recommendation}],
                "propagation_chain": PropagationChain (如果提供),
                "weakening_suggestions": [{invariant, rule}],
                "boundary_state": BoundaryState.to_dict(),
            }
        """
        from .constraints import (
            BoundaryState,
            ConstraintPropagation,
            InvariantStatus,
            LearnedInvariant,
            from_structural_invariant,
        )

        # 1. 从结构不变量创建 LearnedInvariant
        learned: list[LearnedInvariant] = []
        for inv in structure.structural_invariants:
            li = from_structural_invariant(inv)
            learned.append(li)

        # 2. 评估每个不变量
        interior_invs = []
        boundary_invs = []
        violation_count = 0

        for li in learned:
            status = li.evaluate(params)
            if status == InvariantStatus.SATISFIED:
                interior_invs.append(li)
            elif status in (InvariantStatus.WEAKENED, InvariantStatus.CONDITIONAL):
                boundary_invs.append((li, 0.3))
            elif status == InvariantStatus.VIOLATED:
                boundary_invs.append((li, 0.8))
                violation_count += 1

        # 3. 构建初始边界状态
        state = BoundaryState(
            interior_invariants=interior_invs,
            boundary_invariants=boundary_invs,
        )

        # 4. 态射链上的约束传播（如果提供）
        propagation_chain = None
        if morphism_chain:
            propagator = ConstraintPropagation()
            morphisms = [self.engine.morphisms.get(name) for name in morphism_chain if name in self.engine.morphisms]
            sources = ["source"] * len(morphisms)
            targets = ["target"] * len(morphisms)
            propagation_chain = propagator.propagate_chain(
                learned,
                morphisms,
                sources,
                targets,
            )

        # 5. 风险评估
        risks = state.risk_assessment(params)

        # 6. 弱化建议
        weakening_suggestions = []
        for inv, _ in boundary_invs:
            if inv.weakening_rules:
                weakening_suggestions.append(
                    {
                        "invariant": inv.name,
                        "available_rules": [
                            {"name": r.name, "consequence": r.consequence} for r in inv.weakening_rules
                        ],
                    }
                )

        return {
            "interior": {inv.name: "SATISFIED" for inv in interior_invs},
            "boundary_count": len(boundary_invs),
            "violation_count": violation_count,
            "risks": [
                {
                    "invariant": r.invariant.name,
                    "risk_score": r.risk_score,
                    "status": r.status.value,
                    "recommendation": r.recommendation,
                }
                for r in risks
            ],
            "propagation": (
                {
                    "chain": propagation_chain.chain,
                    "final_state": {k: v.value for k, v in propagation_chain.final_state.items()},
                    "preserved": len(propagation_chain.preserved_invariants),
                    "weakened": len(propagation_chain.weakened_invariants),
                    "lost": len(propagation_chain.lost_invariants),
                }
                if propagation_chain
                else None
            ),
            "weakening_suggestions": weakening_suggestions,
            "boundary_state": state.to_dict(),
        }

    def get_summary(self) -> dict[str, Any]:
        return {
            "knowledge_graph": self.kg.stats(),
            "category_engine": self.engine.to_dict(),
        }
