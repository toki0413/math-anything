"""Math Proposition Generator - Translate physical models to LLM-solvable mathematical tasks.

This module implements the core Translate functionality of Math Anything:
1. Extract mathematical structures from computational software
2. Generate formal mathematical propositions and proof tasks
3. Output LLM-native structured data for symbolic reasoning

Example:
    >>> from math_anything import MathAnything, PropositionGenerator
    >>> ma = MathAnything()
    >>> result = ma.extract_file("lammps", "equil.lmp")
    >>> gen = PropositionGenerator()
    >>> propositions = gen.translate(result.schema)
    >>> for task in propositions.proof_tasks:
    ...     print(task.llm_prompt)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskType(Enum):
    """Types of mathematical tasks that can be generated."""
    PROOF = "proof"
    VALIDATION = "validation"
    CONSISTENCY_CHECK = "consistency_check"
    COMPARISON = "comparison"
    ERROR_ANALYSIS = "error_analysis"
    CONVERGENCE_ANALYSIS = "convergence_analysis"
    STABILITY_ANALYSIS = "stability_analysis"
    WELL_POSEDNESS = "well_posedness"


@dataclass
class MathematicalTask:
    """A mathematical task suitable for LLM reasoning.

    Attributes:
        id: Unique task identifier
        type: Task type (proof, validation, etc.)
        name: Human-readable task name
        statement: Formal mathematical statement
        assumptions: List of assumptions
        goals: What needs to be proven/verified
        llm_prompt: Complete prompt for LLM consumption
        difficulty: Estimated difficulty level (1-5)
        references: Related equations or theorems
    """

    id: str
    type: TaskType
    name: str
    statement: str
    assumptions: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    llm_prompt: str = ""
    difficulty: int = 3
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "statement": self.statement,
            "assumptions": self.assumptions,
            "goals": self.goals,
            "difficulty": self.difficulty,
            "references": self.references,
        }


@dataclass
class MathematicalPropositions:
    """Collection of mathematical propositions extracted from a model.

    Attributes:
        core_problem: The fundamental mathematical problem
        proof_tasks: Tasks requiring formal proof
        validation_tasks: Tasks requiring numerical validation
        consistency_checks: Consistency verification tasks
        comparison_tasks: Cross-model comparison tasks
        error_analysis: Error estimation tasks
    """

    core_problem: Dict[str, Any] = field(default_factory=dict)
    proof_tasks: List[MathematicalTask] = field(default_factory=list)
    validation_tasks: List[MathematicalTask] = field(default_factory=list)
    consistency_checks: List[MathematicalTask] = field(default_factory=list)
    comparison_tasks: List[MathematicalTask] = field(default_factory=list)
    error_analysis: List[MathematicalTask] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_problem": self.core_problem,
            "proof_tasks": [t.to_dict() for t in self.proof_tasks],
            "validation_tasks": [t.to_dict() for t in self.validation_tasks],
            "consistency_checks": [t.to_dict() for t in self.consistency_checks],
            "comparison_tasks": [t.to_dict() for t in self.comparison_tasks],
            "error_analysis": [t.to_dict() for t in self.error_analysis],
        }

    def all_tasks(self) -> List[MathematicalTask]:
        """Return all tasks in a single list."""
        return (
            self.proof_tasks
            + self.validation_tasks
            + self.consistency_checks
            + self.comparison_tasks
            + self.error_analysis
        )


class PropositionGenerator:
    """Generate mathematical propositions from extracted schemas.

    This is the core Translate component of Math Anything. It converts
    structured mathematical models into formal propositions and proof tasks
    that LLMs can reason about symbolically.

    Example:
        >>> gen = PropositionGenerator()
        >>> props = gen.translate(schema)
        >>> print(f"Generated {len(props.proof_tasks)} proof tasks")
    """

    def __init__(self):
        self.task_counter = 0

    def _next_id(self, prefix: str) -> str:
        """Generate next task ID."""
        self.task_counter += 1
        return f"{prefix}_{self.task_counter}"

    def translate(self, schema: Dict[str, Any]) -> MathematicalPropositions:
        """Translate a mathematical schema into propositions.

        This is the main entry point for the Translate functionality.
        It analyzes the schema and generates appropriate mathematical tasks.

        Args:
            schema: Mathematical schema from extract() or extract_file()

        Returns:
            MathematicalPropositions containing all generated tasks
        """
        self.task_counter = 0
        propositions = MathematicalPropositions()

        math_struct = schema.get("mathematical_structure", {})
        approximations = schema.get("approximations", [])
        discretization = schema.get("discretization_scheme", {})
        variable_deps = schema.get("variable_dependencies", [])
        solution = schema.get("solution_strategy", {})
        decoding = schema.get("mathematical_decoding", {})

        propositions.core_problem = self._generate_core_problem(math_struct, decoding)
        propositions.proof_tasks = self._generate_proof_tasks(approximations)
        propositions.validation_tasks = self._generate_validation_tasks(discretization, solution)
        propositions.consistency_checks = self._generate_consistency_checks(variable_deps, math_struct)
        propositions.error_analysis = self._generate_error_analysis(approximations, discretization)

        return propositions

    def translate_comparison(
        self, schema_a: Dict[str, Any], schema_b: Dict[str, Any]
    ) -> MathematicalPropositions:
        """Generate comparison propositions between two models.

        Args:
            schema_a: First model schema
            schema_b: Second model schema

        Returns:
            MathematicalPropositions with comparison tasks
        """
        self.task_counter = 0
        propositions = MathematicalPropositions()

        math_a = schema_a.get("mathematical_structure", {})
        math_b = schema_b.get("mathematical_structure", {})

        propositions.comparison_tasks = self._generate_comparison_tasks(math_a, math_b)

        return propositions

    def _generate_core_problem(
        self, math_struct: Dict[str, Any], decoding: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate the core problem statement."""
        problem_type = math_struct.get("problem_type", "unknown")
        canonical_form = math_struct.get("canonical_form", "N/A")
        properties = math_struct.get("properties", {})

        core = decoding.get("core_problem", {})
        equation = core.get("equation", canonical_form)
        physical_meaning = core.get("physical_meaning", "N/A")

        return {
            "type": problem_type,
            "canonical_form": canonical_form,
            "equation": equation,
            "physical_meaning": physical_meaning,
            "properties": properties,
            "proposition": (
                f"命题: 给定计算模型求解的是 {problem_type} 问题, "
                f"其规范形式为 {canonical_form}. "
                f"证明此问题的适定性 (well-posedness)."
            ),
        }

    def _generate_proof_tasks(self, approximations: List[Dict[str, Any]]) -> List[MathematicalTask]:
        """Generate proof tasks for each approximation."""
        tasks = []

        for i, approx in enumerate(approximations):
            name = approx.get("name", f"approximation_{i}")
            form = approx.get("mathematical_form", "N/A")
            consequence = approx.get("consequence", "N/A")
            basis = approx.get("theoretical_basis", "N/A")
            affected = approx.get("affected_quantities", [])

            task = MathematicalTask(
                id=self._next_id("proof"),
                type=TaskType.PROOF,
                name=f"近似有效性证明: {name}",
                statement=f"证明近似 '{name}' 在适当条件下是有效的",
                assumptions=[
                    f"理论依据: {basis}",
                    f"数学形式: {form}",
                ],
                goals=[
                    "1. 给出近似成立的充分必要条件",
                    "2. 估计近似引入的误差上界",
                    "3. 确定近似的适用范围",
                    f"4. 分析对 {', '.join(affected)} 的影响",
                ],
                llm_prompt=f"""【数学证明任务】
名称: {name}
类型: 近似有效性证明

数学形式:
  {form}

理论依据:
  {basis}

预期后果:
  {consequence}

影响量:
  {', '.join(affected)}

请完成以下证明:
1. 给出此近似成立的数学条件 (充分/必要条件)
2. 估计误差: ||精确解 - 近似解|| ≤ ?
3. 确定适用范围 (参数空间中的有效区域)
4. 讨论此近似对物理量的定量影响

请使用严格的数学语言, 必要时引用相关定理.""",
                difficulty=4,
                references=[form, basis],
            )
            tasks.append(task)

        return tasks

    def _generate_validation_tasks(
        self, discretization: Dict[str, Any], solution: Dict[str, Any]
    ) -> List[MathematicalTask]:
        """Generate validation tasks for numerical methods."""
        tasks = []

        if discretization:
            method = discretization.get("method", "unknown")
            meaning = discretization.get("mathematical_meaning", "N/A")
            conv_order = discretization.get("convergence_order", "N/A")
            basis = discretization.get("basis_type", "N/A")
            completeness = discretization.get("completeness", "N/A")

            task = MathematicalTask(
                id=self._next_id("val"),
                type=TaskType.CONVERGENCE_ANALYSIS,
                name=f"收敛性验证: {method}",
                statement=f"验证离散化方法 '{method}' 的收敛性",
                assumptions=[
                    f"离散化方法: {method}",
                    f"数学含义: {meaning}",
                    f"基函数: {basis}",
                    f"完备性: {completeness}",
                ],
                goals=[
                    "1. 证明数值格式的稳定性",
                    "2. 证明数值格式的相容性",
                    "3. 由Lax等价定理推导收敛性",
                    f"4. 验证收敛阶: {conv_order}",
                ],
                llm_prompt=f"""【数值验证任务】
名称: {method} 收敛性分析
类型: 收敛性验证

离散化方案:
  方法: {method}
  数学含义: {meaning}
  基函数: {basis}
  完备性: {completeness}
  收敛阶: {conv_order}

请完成以下分析:
1. 稳定性分析: 证明格式满足 von Neumann 稳定性条件
2. 相容性分析: 证明截断误差趋于零
3. 收敛性证明: 应用 Lax 等价定理
4. 收敛阶验证: 数值实验验证 {conv_order}

请给出严格的数学推导和误差估计.""",
                difficulty=5,
                references=[method, meaning],
            )
            tasks.append(task)

        if solution:
            method = solution.get("method", "unknown")
            form = solution.get("mathematical_form", "N/A")
            stability = solution.get("stability_requirement", "N/A")

            task = MathematicalTask(
                id=self._next_id("val"),
                type=TaskType.STABILITY_ANALYSIS,
                name=f"稳定性分析: {method}",
                statement=f"分析求解方法 '{method}' 的稳定性",
                assumptions=[
                    f"求解方法: {method}",
                    f"数学形式: {form}",
                ],
                goals=[
                    "1. 确定稳定性条件",
                    "2. 分析数值耗散和色散",
                    "3. 评估长期行为",
                ],
                llm_prompt=f"""【稳定性分析任务】
名称: {method} 稳定性分析
类型: 稳定性验证

求解策略:
  方法: {method}
  数学形式: {form}
  稳定性要求: {stability}

请完成以下分析:
1. 稳定性条件: 给出显式/隐式格式的稳定性判据
2. 数值耗散: 分析振幅误差的增长
3. 数值色散: 分析相位误差
4. 长期稳定性: 评估能量/动量守恒

请给出特征值分析和数值实验验证.""",
                difficulty=4,
                references=[method, form],
            )
            tasks.append(task)

        return tasks

    def _generate_consistency_checks(
        self, variable_deps: List[Dict[str, Any]], math_struct: Dict[str, Any]
    ) -> List[MathematicalTask]:
        """Generate consistency check tasks."""
        tasks = []

        for i, dep in enumerate(variable_deps):
            relation = dep.get("relation", "N/A")
            math_form = dep.get("mathematical_form", "N/A")
            circular = dep.get("circular", False)
            physical = dep.get("physical_interpretation", "N/A")

            task = MathematicalTask(
                id=self._next_id("cons"),
                type=TaskType.CONSISTENCY_CHECK,
                name=f"一致性检查: {relation}",
                statement=f"验证变量依赖关系 '{relation}' 的数学一致性",
                assumptions=[
                    f"关系式: {relation}",
                    f"数学形式: {math_form}",
                    f"物理解释: {physical}",
                ],
                goals=[
                    "1. 检查量纲一致性",
                    "2. 检查循环依赖",
                    "3. 验证物理意义",
                ],
                llm_prompt=f"""【一致性检查任务】
名称: {relation}
类型: 数学一致性验证

依赖关系:
  关系式: {relation}
  数学形式: {math_form}
  物理解释: {physical}
  循环依赖: {'是' if circular else '否'}

请完成以下检查:
1. 量纲分析: 验证等式两边的量纲是否匹配
2. 循环依赖: {'分析循环依赖的收敛性' if circular else '确认无循环依赖'}
3. 物理意义: 验证数学关系是否符合物理直觉
4. 边界情况: 讨论极端参数下的行为

请给出详细的数学推导和物理解释.""",
                difficulty=3,
                references=[relation, math_form],
            )
            tasks.append(task)

        return tasks

    def _generate_error_analysis(
        self, approximations: List[Dict[str, Any]], discretization: Dict[str, Any]
    ) -> List[MathematicalTask]:
        """Generate error analysis tasks."""
        tasks = []

        if approximations:
            task = MathematicalTask(
                id=self._next_id("err"),
                type=TaskType.ERROR_ANALYSIS,
                name="累积误差估计",
                statement="估计所有近似引入的累积误差",
                assumptions=[
                    f"涉及 {len(approximations)} 个近似层次",
                ],
                goals=[
                    "1. 估计每个近似的局部误差",
                    "2. 分析误差的传播和累积",
                    "3. 给出总误差上界",
                ],
                llm_prompt=f"""【误差分析任务】
名称: 累积误差估计
类型: 误差分析

涉及的近似层次 ({len(approximations)} 个):
"""
                + "\n".join(
                    f"  {i+1}. {a.get('name', 'unknown')}: {a.get('mathematical_form', 'N/A')}"
                    for i, a in enumerate(approximations)
                )
                + f"""

请完成以下分析:
1. 局部误差: 估计每个近似的误差上界 ε_i
2. 误差传播: 分析误差如何通过方程传播
3. 累积误差: 估计总误差 ||E_total|| ≤ Σ ε_i
4. 敏感性分析: 确定对结果影响最大的近似

请给出定量误差估计和数值验证.""",
                difficulty=5,
                references=[a.get("name", "") for a in approximations],
            )
            tasks.append(task)

        return tasks

    def _generate_comparison_tasks(
        self, math_a: Dict[str, Any], math_b: Dict[str, Any]
    ) -> List[MathematicalTask]:
        """Generate cross-model comparison tasks."""
        tasks = []

        type_a = math_a.get("problem_type", "unknown")
        type_b = math_b.get("problem_type", "unknown")
        form_a = math_a.get("canonical_form", "N/A")
        form_b = math_b.get("canonical_form", "N/A")
        props_a = math_a.get("properties", {})
        props_b = math_b.get("properties", {})

        task = MathematicalTask(
            id=self._next_id("comp"),
            type=TaskType.COMPARISON,
            name="跨尺度模型比较",
            statement="证明两个模型在不同尺度上的数学联系",
            assumptions=[
                f"模型A ({type_a}): {form_a}",
                f"模型B ({type_b}): {form_b}",
            ],
            goals=[
                "1. 识别两个模型的共同数学结构",
                "2. 建立尺度间的数学桥梁",
                "3. 证明一致性或收敛关系",
            ],
            llm_prompt=f"""【跨尺度比较任务】
名称: 跨尺度模型比较
类型: 数学等价性/一致性证明

模型A (微观/原子尺度):
  问题类型: {type_a}
  规范形式: {form_a}
  性质: {props_a}

模型B (宏观/连续介质尺度):
  问题类型: {type_b}
  规范形式: {form_b}
  性质: {props_b}

请完成以下分析:
1. 共同结构: 识别两个模型共享的数学结构
2. 尺度桥梁: 建立从微观到宏观的数学映射
   (例如: 统计力学 → 连续介质力学)
3. 极限关系: 证明当 N→∞ 或 Δx→0 时, 模型A收敛到模型B
4. 参数映射: 给出微观参数到宏观参数的转换关系
5. 误差估计: 估计多尺度耦合引入的误差

请给出严格的数学推导和物理意义解释.""",
            difficulty=5,
            references=[form_a, form_b],
        )
        tasks.append(task)

        return tasks
