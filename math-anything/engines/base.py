"""引擎提取器基类，定义通用提取管线。

所有引擎提取器（VASP、LAMMPS、Abaqus、Ansys 等）共享相同的提取管线：
  _extract_mathematical_model() -> _extract_governing_equations()
                                + _extract_boundary_conditions()
                                + _extract_constitutive_relations()
  _extract_numerical_method()
  _extract_computational_graph()
  _extract_conservation_properties()
  _extract_raw_symbols()

子类只需实现引擎特定的抽象方法，build_schema() 负责组装完整 MathSchema。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from math_anything.schemas import (
    BoundaryCondition,
    ComputationalGraph,
    GoverningEquation,
    MathematicalModel,
    MathSchema,
    MetaInfo,
    NumericalMethod,
)


class BaseEngineExtractor(ABC):
    """引擎提取器抽象基类。

    子类必须实现:
      - engine_name (属性): 引擎名称，如 'vasp', 'lammps'
      - _extract_governing_equations(): 控制方程
      - _extract_boundary_conditions(): 边界条件
      - _extract_constitutive_relations(): 本构关系
      - _extract_numerical_method(): 数值方法
      - _extract_computational_graph(): 计算图
      - _extract_conservation_properties(): 守恒性质
      - _extract_raw_symbols(): 原始符号

    子类可选覆写:
      - extractor_version: 提取器版本号
      - _extract_mathematical_model(): 数学模型组装逻辑
      - _enrich_schema(): 向 schema 添加额外字段
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """引擎名称，如 'vasp', 'lammps', 'abaqus', 'ansys'。"""

    @property
    def extractor_version(self) -> str:
        """提取器版本号，子类可覆写。"""
        return "0.1.0"

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def build_schema(
        self,
        source_files: Dict[str, List[str]],
    ) -> MathSchema:
        """组装完整的 MathSchema。

        Args:
            source_files: 源文件路径字典，如 {"input": ["INCAR"], "log": ["log.lammps"]}

        Returns:
            完整的 MathSchema 对象
        """
        schema = MathSchema(
            schema_version="1.0.0",
            meta=MetaInfo(
                extracted_by=f"math-anything-{self.engine_name}",
                extractor_version=self.extractor_version,
                source_files=source_files,
            ),
            mathematical_model=self._extract_mathematical_model(),
            numerical_method=self._extract_numerical_method(),
            computational_graph=self._extract_computational_graph(),
            conservation_properties=self._extract_conservation_properties(),
            raw_symbols=self._extract_raw_symbols(),
        )

        # 子类可覆写以添加 symbolic_constraints 等额外字段
        self._enrich_schema(schema)

        return schema

    # ------------------------------------------------------------------
    # 数学模型管线（默认实现）
    # ------------------------------------------------------------------

    def _extract_mathematical_model(self) -> MathematicalModel:
        """提取数学模型，默认管线：控制方程 + 边界条件 + 本构关系。

        子类如需添加 initial_conditions / parameter_relationships，
        可覆写此方法或调用 super() 后追加。
        """
        model = MathematicalModel()
        model.governing_equations = self._extract_governing_equations()
        model.boundary_conditions = self._extract_boundary_conditions()
        model.constitutive_relations = self._extract_constitutive_relations()
        return model

    # ------------------------------------------------------------------
    # 抽象方法：子类必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract_governing_equations(self) -> List[GoverningEquation]:
        """提取控制方程。"""

    @abstractmethod
    def _extract_boundary_conditions(self) -> List[BoundaryCondition]:
        """提取边界条件。"""

    @abstractmethod
    def _extract_constitutive_relations(self) -> List[Dict[str, Any]]:
        """提取本构关系。"""

    @abstractmethod
    def _extract_numerical_method(self) -> NumericalMethod:
        """提取数值方法。"""

    @abstractmethod
    def _extract_computational_graph(self) -> ComputationalGraph:
        """提取计算图。"""

    @abstractmethod
    def _extract_conservation_properties(self) -> Dict[str, Any]:
        """提取守恒性质。"""

    @abstractmethod
    def _extract_raw_symbols(self) -> Dict[str, Any]:
        """提取原始符号。"""

    # ------------------------------------------------------------------
    # 可选覆写
    # ------------------------------------------------------------------

    def _enrich_schema(self, schema: MathSchema) -> None:
        """向 schema 添加额外字段，子类可覆写。

        典型用途：添加 symbolic_constraints、parameter_relationships 等。
        """
        pass
