"""ANSYS insight engine.

Leverages FEM domain template; only ANSYS-specific overrides needed.
"""

from typing import List

from ..schemas import MathSchema
from ..templates import FEMInsightTemplate, FEMParamExtractor
from .base import InsightBlock, InsightEngine


class AnsysInsightEngine(InsightEngine):
    """Generate mathematical insights for ANSYS FEM calculations."""

    @property
    def engine_name(self) -> str:
        return "ansys"

    def generate(self, schema: MathSchema) -> List[InsightBlock]:
        params = FEMParamExtractor.extract(schema, engine="ansys")
        tpl = FEMInsightTemplate(params)
        tpl.software_name = "ANSYS Mechanical APDL"
        tpl.element_library_name = "ANSYS Element Library"

        blocks = tpl.to_insight_blocks()

        # ANSYS-specific: element technology insight
        elements = params.get("elements", [])
        if elements:
            elem = elements[0].upper()
            elem_desc = self._describe_ansys_element(elem)
            if elem_desc:
                blocks.append(
                    InsightBlock(
                        title="ANSYS Element Technology",
                        content=elem_desc,
                        level="math",
                    )
                )

        return blocks

    def _describe_ansys_element(self, elem: str) -> str:
        """ANSYS-specific element descriptions."""
        knowledge = {
            "SOLID185": (
                "SOLID185 is an 8-node structural solid element with three degrees of freedom per node "
                "(translations in x, y, z). It supports plasticity, hyperelasticity, stress stiffening, "
                "creep, and large deflection. Full integration (2x2x2 Gauss) is used by default. "
                "The element employs the B-bar method (selective reduced integration) for "
                "near-incompressible materials to avoid volumetric locking."
            ),
            "SOLID186": (
                "SOLID186 is a 20-node quadratic structural solid element. "
                "Higher-order interpolation provides superior accuracy for bending and stress gradients. "
                "Compatible with SOLID187 (10-node tet) for pyramid transitions."
            ),
            "SOLID187": (
                "SOLID187 is a 10-node quadratic tetrahedral element. "
                "Appropriate for automatic tetrahedral meshing of complex geometries. "
                "Avoids shear locking through quadratic shape functions."
            ),
            "SOLID65": (
                "SOLID65 is an 8-node reinforced concrete solid with smeared cracking and crushing capability. "
                "Includes rebar modeling with independent material properties."
            ),
            "BEAM188": (
                "BEAM188 is a 2-node 3-D beam based on Timoshenko beam theory. "
                "Includes shear deformation effects and warping degrees of freedom."
            ),
            "SHELL181": (
                "SHELL181 is a 4-node shell with 6 DOF per node, suitable for thin to moderately thick shells. "
                "Uses first-order shear deformation theory (Mindlin-Reissner)."
            ),
            "LINK180": (
                "LINK180 is a 3-D truss element with axial tension/compression only. "
                "Used for cable and pin-jointed structures."
            ),
            "PLANE182": (
                "PLANE182 is a 4-node 2-D structural solid with plane stress, plane strain, or axisymmetric options."
            ),
            "PLANE183": (
                "PLANE183 is an 8-node quadratic 2-D structural solid. Superior for curved boundaries and bending."
            ),
        }
        return knowledge.get(elem, "")
