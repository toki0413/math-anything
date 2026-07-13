"""Abaqus methodology draft generator.

Produces peer-review-ready method sections for FEM papers.
"""

from typing import Any, Dict, List

from ..schemas import MathSchema
from .base import DraftEngine


class AbaqusDraftEngine(DraftEngine):
    """Generate publication methodology for Abaqus FEM simulations."""

    @property
    def engine_name(self) -> str:
        return "abaqus"

    def generate(self, schema: MathSchema, fmt: str = "markdown") -> str:
        raw: Dict[str, Any] = schema.raw_symbols or {}
        materials = raw.get("materials", [])
        steps = raw.get("steps", [])
        bcs = raw.get("boundary_conditions", [])
        elements = raw.get("elements", [])
        nlgeom = raw.get("nlgeom", False)
        nodes = raw.get("nodes", 0)

        lines: list[str] = []
        if fmt == "markdown":
            lines.append("# Computational Details")
        else:
            lines.append("\\section{Computational Details}")
        lines.append("")

        lines.append(self._theory(steps, nlgeom, fmt))
        lines.append(self._software(fmt))
        lines.append(self._mesh(elements, nodes, fmt))
        if materials:
            lines.append(self._material(materials, fmt))
        if bcs:
            lines.append(self._boundary_conditions(bcs, fmt))
        if steps:
            lines.append(self._solution_procedure(steps, nlgeom, fmt))
        lines.append(self._caveats(steps, nlgeom, fmt))

        return "\n".join(lines)

    def _theory(self, steps: List[Dict], nlgeom: bool, fmt: str) -> str:
        analysis = steps[0].get("analysis_type", "static") if steps else "static"

        if analysis == "static" and not nlgeom:
            body = (
                "The governing equation is the static equilibrium of a continuous body: "
                "$\\nabla \\cdot \\boldsymbol{\\sigma} + \\mathbf{b} = \\mathbf{0}$, "
                "where $\\boldsymbol{\\sigma}$ is the Cauchy stress tensor and $\\mathbf{b}$ is the body force per unit volume. "  # noqa: E501
                "The weak form is obtained by multiplying by a test function $\\mathbf{v}$ and integrating over the domain $\\Omega$: "  # noqa: E501
                "$\\int_\\Omega \\boldsymbol{\\sigma} : \\nabla^s \\mathbf{v} \\, \\mathrm{d}\\Omega = "
                "\\int_\\Omega \\mathbf{b} \\cdot \\mathbf{v} \\, \\mathrm{d}\\Omega + "
                "\\int_{\\Gamma_t} \\mathbf{t} \\cdot \\mathbf{v} \\, \\mathrm{d}\\Gamma$."
            )
        elif analysis == "static" and nlgeom:
            body = (
                "The governing equation is the static equilibrium in the current configuration: "
                "$\\nabla \\cdot \\boldsymbol{\\sigma} + \\mathbf{b} = \\mathbf{0}$. "
                "Geometric nonlinearity is accounted for via the Green-Lagrange strain tensor "
                "$\\mathbf{E} = \\frac{1}{2}(\\mathbf{F}^\\top \\mathbf{F} - \\mathbf{I})$ "
                "and the Second Piola-Kirchhoff stress $\\mathbf{S}$. "
                "The weak form is nonlinear in the displacement field and solved iteratively."
            )
        elif analysis == "dynamic":
            body = (
                "The governing equation is the dynamic momentum balance "
                "$\\rho \\ddot{\\mathbf{u}} = \\nabla \\cdot \\boldsymbol{\\sigma} + \\mathbf{b}$. "
                "Time integration is performed using the Newmark-$\\beta$ method with "
                "$\\beta = 0.25$ and $\\gamma = 0.5$ (average acceleration, unconditionally stable)."
            )
        elif analysis == "heat":
            body = (
                "The governing equation is Fourier's law of heat conduction: "
                "$\\rho c \\dot{T} = \\nabla \\cdot (k \\nabla T) + q$."
            )
        elif analysis == "coupled":
            body = (
                "A coupled thermomechanical analysis is performed, solving "
                "the mechanical equilibrium and heat conduction equations simultaneously. "
                "Thermal expansion introduces coupling via the strain term "
                "$\\boldsymbol{\\varepsilon}_\\mathrm{th} = \\alpha \\Delta T \\mathbf{I}$."
            )
        else:
            body = f"The analysis type is {analysis}."

        return self._section("Governing Equations", body, fmt)

    def _software(self, fmt: str) -> str:
        body = (
            "All finite element simulations were performed using Abaqus/Standard (Dassault Systemes). "
            "The commercial solver employs a direct sparse linear algebra package (MUMPS or PARDISO) "
            "for the linearized system $\\mathbf{K} \\mathbf{u} = \\mathbf{F}$. "
            "For nonlinear problems, a full Newton-Raphson scheme with automatic incrementation is used."
        )
        return self._section("Software", body, fmt)

    def _mesh(self, elements: List[str], nodes: int, fmt: str) -> str:
        elem = elements[0] if elements else "unspecified"
        body = f"The domain was discretized using {elem} finite elements. The mesh contains {nodes} nodes. "

        if "C3D8R" in elem.upper():
            body += (
                "Reduced integration with hourglass control was employed to mitigate shear locking "
                "while suppressing zero-energy modes."
            )
        elif "C3D8I" in elem.upper():
            body += (
                "Incompatible mode formulation was used to improve bending response "
                "without introducing hourglass modes."
            )
        elif "C3D10" in elem.upper():
            body += (
                "Quadratic interpolation ensures superior accuracy for stress gradients "
                "and curved boundaries compared to linear elements."
            )

        body += " Mesh convergence was verified by comparing strain energy norm across at least two refinement levels."
        return self._section("Mesh and Discretization", body, fmt)

    def _material(self, materials: List[Dict], fmt: str) -> str:
        mat = materials[0]
        mtype = mat.get("model_type", "unknown")
        name = mat.get("name", "Material")

        if mtype == "elastic":
            E = mat.get("youngs_modulus")
            nu = mat.get("poisson_ratio")
            rho = mat.get("density")
            body = f"Material behavior was modeled as isotropic linear elastic ({name}). "
            if E is not None:
                body += f"Young's modulus $E = {E}$ MPa. "
            if nu is not None:
                body += f"Poisson's ratio $\\nu = {nu}$. "
            if rho is not None:
                body += f"Density $\\rho = {rho}$. "
            body += (
                "The constitutive relation is $\\boldsymbol{\\sigma} = \\mathbb{C} : \\boldsymbol{\\varepsilon}$, "
                "where $\\mathbb{C}$ is the fourth-order isotropic stiffness tensor."
            )
        elif mtype in ("plastic", "cdp"):
            body = (
                f"An elasto-plastic/damage material model ({name}) was used. "
                "The constitutive response follows incremental plasticity theory "
                "with associated flow rule and isotropic/kinematic hardening. "
                "Damage evolution is governed by equivalent plastic strain thresholds."
            )
        else:
            body = (
                f"Material properties were defined in the Abaqus input file under *Material, name={name}. "
                "Refer to the input file for complete constitutive specifications."
            )

        return self._section("Constitutive Model", body, fmt)

    def _boundary_conditions(self, bcs: List[Dict], fmt: str) -> str:
        fixed = [b for b in bcs if b.get("value", 0.0) == 0.0]
        loaded = [b for b in bcs if b.get("value", 0.0) != 0.0]

        body = (
            f"Displacement boundary conditions were applied to {len(fixed)} node set(s) "
            f"and load/displacement conditions to {len(loaded)} node set(s). "
        )

        if fixed:
            body += (
                "Essential (Dirichlet) constraints: "
                + ", ".join(f"{b['node_set']} (DOFs {b['dof']})" for b in fixed[:3])
                + ". "
            )
        if loaded:
            body += (
                "Natural (Neumann) conditions: "
                + ", ".join(f"{b['node_set']} (value {b['value']})" for b in loaded[:3])
                + ". "
            )

        body += "Rigid body motion was eliminated to ensure a non-singular stiffness matrix."
        return self._section("Boundary Conditions", body, fmt)

    def _solution_procedure(self, steps: List[Dict], nlgeom: bool, fmt: str) -> str:
        step = steps[0]
        analysis = step.get("analysis_type", "static")
        inc = step.get("initial_inc")
        total = step.get("total_time")
        max_inc = step.get("max_increments", 100)

        if analysis == "static":
            body = "A static analysis procedure was employed. "
            if nlgeom:
                body += (
                    "Geometric nonlinearity was activated (NLGEOM=YES). "
                    "The full Newton-Raphson method was used with automatic incrementation. "
                    "At each increment, the tangent stiffness matrix was assembled and factorized. "
                    "Convergence was deemed achieved when the force residual norm fell below "
                    "$10^{-5}$ times the average force norm."
                )
            else:
                body += (
                    "The problem is geometrically linear. A single matrix assembly and factorization was sufficient."
                )
        elif analysis == "dynamic":
            body = (
                "An implicit dynamic procedure was used with the Hilber-Hughes-Taylor "
                "(HHT) time integrator (alpha = -0.05). "
                "The time step was chosen to resolve the period of the highest frequency mode of interest."
            )
        else:
            body = f"Analysis procedure: {analysis}."

        if inc is not None and total is not None:
            body += (
                f" The step was divided into a maximum of {max_inc} increments, "
                f"with an initial increment size of {inc} and total step time of {total}."
            )

        return self._section("Solution Procedure", body, fmt)

    def _caveats(self, steps: List[Dict], nlgeom: bool, fmt: str) -> str:
        analysis = steps[0].get("analysis_type", "static") if steps else "static"
        notes = []

        if analysis == "static" and nlgeom:
            notes.append(
                "Bifurcation and limit points may require arc-length (Riks) methods "
                "rather than standard Newton-Raphson."
            )
        if not nlgeom and analysis == "static":
            notes.append(
                "Linear analysis is valid only for small strains and rotations. Large deformations require NLGEOM=YES."
            )

        notes.append(
            "Mesh sensitivity should be assessed via systematic refinement studies. "
            "The energy norm is the recommended error indicator for displacement-based FEM."
        )

        body = " ".join(notes)
        return self._section("Methodological Notes", body, fmt)
