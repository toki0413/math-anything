"""Check engine — pre-flight parameter consistency validation.

Catches mathematically or physically inconsistent simulation
settings before wasting computational resources.
"""

from .abaqus_check import AbaqusCheckEngine
from .ansys_check import AnsysCheckEngine
from .base import CheckEngine, check_schema
from .comsol_check import ComsolCheckEngine
from .lammps_check import LammpsCheckEngine
from .qe_check import QuantumEspressoCheckEngine
from .vasp_check import VaspCheckEngine

__all__ = ["CheckEngine", "check_schema", "VaspCheckEngine", "LammpsCheckEngine"]
