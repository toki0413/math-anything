"""COMSOL draft engine.

Leverages FEM domain template; adds multiphysics-specific methodology.
"""

from ..schemas import MathSchema
from ..templates import FEMDraftTemplate, FEMParamExtractor
from .base import DraftEngine


class ComsolDraftEngine(DraftEngine):
    """Generate publication methodology for COMSOL Multiphysics simulations."""

    @property
    def engine_name(self) -> str:
        return "comsol"

    def generate(self, schema: MathSchema, fmt: str = "markdown") -> str:
        params = FEMParamExtractor.extract(schema, engine="comsol")
        tpl = FEMDraftTemplate(params)
        tpl.software_name = "COMSOL Multiphysics"
        tpl.element_library_name = "COMSOL Lagrange Elements"
        text = tpl.to_draft_text(fmt=fmt)

        # Add multiphysics-specific section if applicable
        physics = params.get("physics_type", "solid_mechanics")
        if physics != "solid_mechanics":
            if fmt == "markdown":
                mp_section = "\n## Multiphysics Coupling\n\n"
            else:
                mp_section = "\n\\subsection{Multiphysics Coupling}\n"

            if "thermal" in physics and "solid" in physics:
                mp_section += (
                    "A coupled thermomechanical analysis was performed using the monolithic solver. "
                    "Thermal expansion was modeled via the eigenstrain contribution "
                    "$\\boldsymbol{\\varepsilon}_\\mathrm{th} = \\alpha \\Delta T \\mathbf{I}$. "
                    "The weak form couples temperature and displacement fields through the "
                    "thermomechanical stiffness matrix."
                )
            elif "fluid" in physics:
                mp_section += (
                    "A fluid-structure interaction analysis was performed. "
                    "The Navier-Stokes equations were solved for the fluid domain, "
                    "with traction boundary conditions transferring forces to the solid domain."
                )
            else:
                mp_section += (
                    f"A multiphysics simulation was performed coupling {physics.replace('_', '-')} physics. "
                    "The monolithic solver ensures full coupling between all field variables."
                )

            # Insert before caveats
            if fmt == "markdown":
                parts = text.rsplit("## Methodological Notes", 1)
                if len(parts) == 2:
                    text = parts[0] + mp_section + "\n\n## Methodological Notes" + parts[1]
                else:
                    text += mp_section
            else:
                parts = text.rsplit("\\subsection{Methodological Notes", 1)
                if len(parts) == 2:
                    text = parts[0] + mp_section + "\n\n\\subsection{Methodological Notes" + parts[1]
                else:
                    text += mp_section

        return text
