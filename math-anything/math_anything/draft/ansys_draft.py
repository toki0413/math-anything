"""ANSYS draft engine.

Leverages FEM domain template; only ANSYS-specific overrides needed.
"""

from ..schemas import MathSchema
from ..templates import FEMDraftTemplate, FEMParamExtractor
from .base import DraftEngine


class AnsysDraftEngine(DraftEngine):
    """Generate publication methodology for ANSYS FEM simulations."""

    @property
    def engine_name(self) -> str:
        return "ansys"

    def generate(self, schema: MathSchema, fmt: str = "markdown") -> str:
        params = FEMParamExtractor.extract(schema, engine="ansys")
        tpl = FEMDraftTemplate(params)
        tpl.software_name = "ANSYS Mechanical APDL"
        tpl.element_library_name = "ANSYS Element Library"
        return tpl.to_draft_text(fmt=fmt)
