"""Insight engine — explain the mathematics behind simulation inputs.

Transforms extracted MathSchema into human-readable mathematical narratives
for daily computational materials research.
"""

from .abaqus_insight import AbaqusInsightEngine
from .ansys_insight import AnsysInsightEngine
from .base import InsightEngine, explain_schema
from .comsol_insight import ComsolInsightEngine
from .lammps_insight import LammpsInsightEngine
from .qe_insight import QuantumEspressoInsightEngine
from .vasp_insight import VaspInsightEngine

__all__ = ["InsightEngine", "explain_schema", "VaspInsightEngine", "LammpsInsightEngine"]
