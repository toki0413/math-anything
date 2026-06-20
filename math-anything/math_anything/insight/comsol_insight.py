"""COMSOL insight engine.

Leverages FEM domain template; adds multiphysics-specific insights.
"""

from typing import List

from ..schemas import MathSchema
from ..templates import FEMInsightTemplate, FEMParamExtractor
from .base import InsightBlock, InsightEngine


class ComsolInsightEngine(InsightEngine):
    """Generate mathematical insights for COMSOL Multiphysics simulations."""

    @property
    def engine_name(self) -> str:
        return "comsol"

    def generate(self, schema: MathSchema) -> List[InsightBlock]:
        params = FEMParamExtractor.extract(schema, engine="comsol")
        tpl = FEMInsightTemplate(params)
        tpl.software_name = "COMSOL Multiphysics"
        tpl.element_library_name = "COMSOL Lagrange Elements"

        blocks = tpl.to_insight_blocks()

        # COMSOL-specific: multiphysics coupling insight
        physics = params.get("physics_type", "solid_mechanics")
        if "_" in physics and physics != "solid_mechanics":
            blocks.append(
                InsightBlock(
                    title="Multiphysics Coupling",
                    content=(
                        "COMSOL solves a coupled system of PDEs using a monolithic approach. "
                        "The Jacobian matrix includes cross-derivatives between physics variables, "
                        "enabling strong coupling without operator splitting. "
                        "For thermomechanics, the thermal expansion strain epsilon_th = alpha Delta T I "
                        "enters the mechanical equilibrium as a eigenstrain contribution."
                    ),
                    level="math",
                )
            )

        # Element technology
        mesh = params.get("mesh", {})
        elem_type = mesh.get("element_type", "")
        if "tetrahedral" in elem_type.lower():
            blocks.append(
                InsightBlock(
                    title="Mesh Generation",
                    content=(
                        "Tetrahedral meshing with automatic adaptive refinement. "
                        "COMSOL uses Delaunay triangulation with quality-based smoothing. "
                        "For curved boundaries, isoparametric elements map the reference tetrahedron "
                        "to physical coordinates via the same shape functions used for the solution."
                    ),
                    level="info",
                )
            )

        return blocks
