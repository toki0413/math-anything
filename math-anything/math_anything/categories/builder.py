"""知识图谱构建器。

从各种来源（提取、比较、不变量分析）构建和更新知识图谱。

原则：
  - 增量构建：每次操作都添加新节点/边，不过度写入
  - 去重：同名+同类型的实体自动合并（通过 mentions 和 confidence 增量）
  - 无侵入：构建 KG 不影响现有提取流程
"""

from __future__ import annotations

from typing import Any

from ..structures import AbstractMathematicalStructure
from ..structures.properties import StructuralInvariant
from .graph import MathKnowledgeGraph, node_id


class KnowledgeGraphBuilder:
    """从 math-anything 的输出构建知识图谱."""

    def __init__(self, kg: MathKnowledgeGraph):
        self.kg = kg

    # ── 从结构对象构建 ──

    def build_from_structure(self, structure: AbstractMathematicalStructure) -> str:
        """注册一个数学结构到 KG.

        Returns:
            结构节点的 ID.
        """
        sid = self.kg.add_entity(
            label=structure.name,
            entity_type="structure",
            source="structure_builder",
            confidence=0.95,
            canonical_form=structure.canonical_form,
            family=str(structure.family),
            function_space=structure.function_space,
            dimensional_rank=structure.dimensional_rank,
        )

        # 注册结构的不变量
        for inv in structure.structural_invariants:
            inv_id = self._add_invariant(inv)
            self.kg.add_relation(sid, "has_invariant", inv_id, source="structure_builder")

        return sid

    def _add_invariant(self, inv: StructuralInvariant) -> str:
        """添加一个不变量节点."""
        return self.kg.add_entity(
            label=inv.name,
            entity_type="invariant",
            source="structure_builder",
            confidence=1.0,
            expression=inv.expression,
            theorem=inv.theorem,
            condition=inv.condition or "",
            severity=inv.severity,
        )

    # ── 从引擎实例构建 ──

    def build_from_engine(self, engine_name: str, structure_name: str, params: dict[str, Any]) -> str:
        """注册一个引擎实例及其与数学结构的关系.

        创建：Engine 节点 --instantiated_by→ Structure 节点
              各 Parameter 节点 --controls→ 相关的 Discretization/Approximation 节点

        Returns:
            引擎节点的 ID.
        """
        eid = self.kg.add_entity(
            label=engine_name.upper(),
            entity_type="engine",
            source="engine_builder",
            confidence=1.0,
            structure=structure_name,
        )

        sid = node_id(structure_name, "structure")
        if sid in self.kg._graph:
            self.kg.add_relation(eid, "instantiates", sid, source="engine_builder")

        # 注册参数
        for param_name, param_value in params.items():
            pid = self.kg.add_entity(
                label=param_name,
                entity_type="parameter",
                source="engine_builder",
                confidence=0.9,
                value=str(param_value),
                engine=engine_name.upper(),
            )
            self.kg.add_relation(eid, "has_parameter", pid, source="engine_builder")

        return eid

    # ── 从态射构建 ──

    def build_from_morphism(
        self,
        morphism: Any,
        source_struct_id: str,
        target_struct_id: str,
    ) -> str:
        """注册一个态射实例及其与源/目标结构的关系.

        Returns:
            态射链接的 ID（虚拟的，实际存储在边上）.
        """
        mid = self.kg.add_entity(
            label=morphism.name,
            entity_type="morphism",
            source="morphism_builder",
            confidence=0.85,
            category=morphism.category,
            mathematical_form=morphism.mathematical_form,
            kernel=morphism.kernel_description,
            condition=morphism.condition,
        )

        # 源 → 态射 → 目标
        self.kg.add_relation(source_struct_id, "source_of", mid, source="morphism_builder")
        self.kg.add_relation(mid, "maps_to", target_struct_id, source="morphism_builder")

        # 保持的不变量
        for inv_name in morphism.invariants_kept:
            inv_id = node_id(inv_name, "invariant")
            if inv_id not in self.kg._graph:
                self.kg.add_entity(
                    label=inv_name,
                    entity_type="invariant",
                    source="morphism_builder",
                    confidence=0.7,
                )
            self.kg.add_relation(mid, "keeps", inv_id, source="morphism_builder")

        # 丢失的不变量
        for inv_name in morphism.invariants_lost:
            inv_id = node_id(inv_name, "invariant")
            if inv_id not in self.kg._graph:
                self.kg.add_entity(
                    label=inv_name,
                    entity_type="invariant",
                    source="morphism_builder",
                    confidence=0.7,
                )
            self.kg.add_relation(mid, "loses", inv_id, source="morphism_builder")

        return mid

    # ── 从 π 群构建 ──

    def build_from_pi_groups(self, pi_groups: list[dict], structure_id: str) -> None:
        """添加 Buckingham π 群节点."""
        for pg in pi_groups:
            pi_id = self.kg.add_entity(
                label=pg.get("name", f"π_{pg.get('pi_id', '?')}"),
                entity_type="pi_group",
                source="dimensional_analysis",
                confidence=1.0,
                expression=pg.get("expression", ""),
                physical_meaning=pg.get("physical_meaning", ""),
            )
            self.kg.add_relation(
                structure_id,
                "has_dimensionless_group",
                pi_id,
                source="dimensional_analysis",
            )
            for var_name in pg.get("variables", {}):
                param_id = node_id(var_name, "parameter")
                if param_id in self.kg._graph:
                    self.kg.add_relation(
                        pi_id,
                        "involves",
                        param_id,
                        source="dimensional_analysis",
                    )

    # ── 从比较报告构建 ──

    def build_from_cross_engine(
        self,
        engine_a: str,
        engine_b: str,
        param_mappings: list[dict],
    ) -> None:
        """从跨引擎比较结果建立等价边."""
        for mapping in param_mappings:
            param_a_id = node_id(mapping["source"], "parameter")
            param_b_id = node_id(mapping["target"], "parameter")

            # 确保节点存在
            if param_a_id not in self.kg._graph:
                self.kg.add_entity(mapping["source"], "parameter", source="cross_engine")
            if param_b_id not in self.kg._graph:
                self.kg.add_entity(mapping["target"], "parameter", source="cross_engine")

            self.kg.add_relation(
                param_a_id,
                "equivalent_to",
                param_b_id,
                source="cross_engine",
                meaning=mapping.get("meaning", ""),
                engine_a=engine_a.upper(),
                engine_b=engine_b.upper(),
            )
