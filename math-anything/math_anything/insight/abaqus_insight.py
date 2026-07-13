"""Abaqus-specific mathematical insight generation.

Connects Abaqus FEM input parameters to their underlying mathematical
structures: weak forms, constitutive relations, Newton-Raphson iteration,
and material stability criteria.
"""

from typing import Any, Dict, List, Optional

from ..schemas import MathSchema
from .base import InsightBlock, InsightEngine


class AbaqusInsightEngine(InsightEngine):
    """Generate mathematical insights for Abaqus FEM calculations."""

    @property
    def engine_name(self) -> str:
        return "abaqus"

    def generate(self, schema: MathSchema) -> List[InsightBlock]:
        blocks: List[InsightBlock] = []
        raw: Dict[str, Any] = schema.raw_symbols or {}
        materials = raw.get("materials", [])
        steps = raw.get("steps", [])
        bcs = raw.get("boundary_conditions", [])
        elements = raw.get("elements", [])
        nlgeom = raw.get("nlgeom", False)

        # Problem overview
        blocks.append(self._problem_overview(steps, nlgeom))

        # Material insights
        if materials:
            blocks.extend(self._material_insights(materials))

        # Discretization insight
        if elements:
            blocks.append(self._discretization_insight(elements))

        # Boundary condition insights
        if bcs:
            blocks.append(self._bc_insight(bcs))

        # Solver / convergence insight
        if steps:
            blocks.append(self._solver_insight(steps, nlgeom))

        # Stability warnings
        warning = self._stability_warnings(materials, bcs, steps)
        if warning:
            blocks.append(warning)

        return blocks

    def _problem_overview(self, steps: List[Dict], nlgeom: bool) -> InsightBlock:
        analysis = steps[0].get("analysis_type", "static") if steps else "static"

        descriptions = {
            "static": (
                "The simulation solves the static equilibrium equation div(sigma) + b = 0 "
                "via the finite element method. The weak form is derived by multiplying "
                "the strong form by a test function v and integrating over the domain: "
                "integral_Omega sigma : grad(v) dV = integral_Omega b . v dV + integral_Gamma t . v dS."
            ),
            "dynamic": (
                "The simulation solves the dynamic momentum balance rho * d2u/dt2 = div(sigma) + b "
                "using the Newmark-beta time integration scheme. The weak form introduces "
                "inertial terms: integral_Omega rho * a . v dV + integral_Omega sigma : grad(v) dV = "
                "integral_Omega b . v dV + boundary terms."
            ),
            "heat": (
                "The simulation solves the heat conduction equation div(k grad(T)) + q = rho c dT/dt. "
                "The Galerkin weak form is: integral_Omega k grad(T) . grad(v) dV = "
                "integral_Omega q v dV - integral_Omega rho c dT/dt v dV + boundary flux terms."
            ),
            "coupled": (
                "The simulation solves a coupled thermomechanical problem where "
                "mechanical deformation and thermal diffusion interact through thermal expansion. "
                "The weak form couples displacement and temperature fields via the "
                "thermomechanical stiffness matrix."
            ),
        }

        body = descriptions.get(analysis, descriptions["static"])

        if nlgeom:
            body += (
                "\n\nGeometric nonlinearity (NLGEOM=YES) is enabled, meaning the "
                "strain-displacement relation uses the Green-Lagrange tensor "
                "E = 1/2(F^T F - I) and equilibrium is evaluated in the current configuration. "
                "This requires a Newton-Raphson iteration at each load increment."
            )
        else:
            body += (
                "\n\nGeometric linearity is assumed: strains are infinitesimal "
                "(epsilon = 1/2(grad(u) + grad(u)^T)) and equilibrium is evaluated "
                "in the reference configuration."
            )

        return InsightBlock(
            title="Mathematical Problem Overview",
            content=body,
            level="math",
        )

    def _material_insights(self, materials: List[Dict]) -> List[InsightBlock]:
        blocks = []
        for mat in materials:
            E = mat.get("youngs_modulus")
            nu = mat.get("poisson_ratio")
            rho = mat.get("density")
            mtype = mat.get("model_type", "unknown")

            if mtype == "elastic" and E is not None and nu is not None:
                lam = E * nu / ((1 + nu) * (1 - 2 * nu))
                mu = E / (2 * (1 + nu))
                K = E / (3 * (1 - 2 * nu))
                body = (
                    f"Material '{mat.get('name', 'unknown')}' uses isotropic linear elasticity.\n\n"
                    f"From E = {E} MPa and nu = {nu}:\n"
                    f"  - Shear modulus  mu = E / (2(1+nu)) = {mu:.2f} MPa\n"
                    f"  - First Lame parameter lambda = E*nu/((1+nu)(1-2nu)) = {lam:.2f} MPa\n"
                    f"  - Bulk modulus K = E / (3(1-2nu)) = {K:.2f} MPa\n\n"
                    f"The stiffness tensor is C_ijkl = lambda delta_ij delta_kl + mu (delta_ik delta_jl + delta_il delta_jk). "  # noqa: E501
                    f"Positive definiteness requires E > 0 and -1 < nu < 0.5."
                )
                if rho is not None:
                    body += f"\nDensity = {rho} (units consistent with model)."
                blocks.append(InsightBlock(title=f"Material: {mat.get('name', 'Elastic')}", content=body, level="math"))

            elif mtype in ("plastic", "cdp"):
                body = (
                    f"Material '{mat.get('name', 'unknown')}' uses an elasto-plastic or damage model.\n\n"
                    f"The constitutive relation is sigma = C : (epsilon - epsilon_p), "
                    f"where epsilon_p is the plastic strain tensor. The yield surface "
                    f"f(sigma, q) <= 0 constrains admissible stress states. "
                    f"Return mapping (closest-point projection) enforces consistency."
                )
                blocks.append(InsightBlock(title=f"Material: {mat.get('name', 'Plastic')}", content=body, level="math"))

            elif rho is not None:
                body = (
                    f"Material '{mat.get('name', 'unknown')}' has density = {rho}. "
                    f"Specific constitutive parameters were not fully extracted. "
                    f"Check the .inp file for *Elastic, *Plastic, or other property cards."
                )
                blocks.append(InsightBlock(title=f"Material: {mat.get('name', 'General')}", content=body, level="info"))

        return blocks

    def _discretization_insight(self, elements: List[str]) -> InsightBlock:
        elem = elements[0] if elements else "unknown"

        element_knowledge = {
            "C3D8": (
                "8-node hexahedron (trilinear). Uses tri-linear Lagrange shape functions. "
                "Full integration (2x2x2 Gauss) or reduced integration (1 GP) available. "
                "Prone to shear locking in bending; use incompatible mode (C3D8I) or reduced integration (C3D8R) for bending-dominated problems."  # noqa: E501
            ),
            "C3D8R": (
                "8-node hexahedron with reduced integration (1 Gauss point). "
                "Hourglass stabilization required to suppress zero-energy modes. "
                "Economical for large deformations but check hourglass energy ratio < 5%."
            ),
            "C3D8I": (
                "8-node hexahedron with incompatible modes. "
                "Adds bubble modes to relieve shear locking without hourglass modes. "
                "Excellent for bending-dominated problems."
            ),
            "C3D4": (
                "4-node tetrahedron (linear). Stiff and locks in bending and nearly incompressible deformation. "
                "Use C3D10 (10-node quadratic tet) for accuracy."
            ),
            "C3D10": (
                "10-node quadratic tetrahedron. Accurate for complex geometries and bending. "
                "4 Gauss points. Higher cost but better convergence than linear tets."
            ),
            "CPE4": (
                "4-node plane strain quadrilateral. Assumes epsilon_zz = 0. "
                "Appropriate for thick structures where out-of-plane strain is constrained."
            ),
            "CPS4": (
                "4-node plane stress quadrilateral. Assumes sigma_zz = 0. Appropriate for thin plates and membranes."
            ),
            "S4R": (
                "4-node reduced-integration shell. Uses Mindlin-Reissner shell theory. "
                "Transverse shear deformation included; check shear locking for very thin shells."
            ),
        }

        # Fuzzy match
        desc = None
        for key, value in element_knowledge.items():
            if key in elem.upper():
                desc = value
                break

        if desc is None:
            desc = (
                f"Element type {elem}. Ensure the element formulation is appropriate "
                f"for the expected deformation modes (bending, shear, incompressibility)."
            )

        body = (
            f"Spatial discretization uses {elem} elements.\n\n{desc}\n\n"
            f"Mesh convergence requires that the strain energy error decreases monotonically "
            f"with refinement. The convergence rate for displacement-based FEM is "
            f"O(h^(p+1)) in L2 norm and O(h^p) in energy norm for polynomial degree p."
        )

        return InsightBlock(title="Spatial Discretization", content=body, level="math")

    def _bc_insight(self, bcs: List[Dict]) -> InsightBlock:
        fixed_dofs = set()
        loaded_sets = []
        for bc in bcs:
            dof = bc.get("dof", "")
            val = bc.get("value", 0.0)
            nset = bc.get("node_set", "")
            if val == 0.0:
                fixed_dofs.add(f"{nset}:{dof}")
            else:
                loaded_sets.append(f"{nset} (dof {dof}, value {val})")

        body = f"{len(bcs)} boundary condition(s) detected.\n\nDirichlet (prescribed displacement) constraints:\n"
        if fixed_dofs:
            body += "\n".join(f"  - {fd}" for fd in sorted(fixed_dofs)) + "\n"
        else:
            body += "  (none detected)\n"

        if loaded_sets:
            body += "\nNeumann (applied load/displacement) constraints:\n"
            body += "\n".join(f"  - {ls}" for ls in loaded_sets) + "\n"

        body += (
            "\nIn FEM, Dirichlet conditions modify the stiffness matrix directly "
            "(penalty or Lagrange multiplier method), while Neumann conditions enter "
            "the load vector. Rigid body motion must be fully constrained; "
            "otherwise the stiffness matrix is singular (det(K) = 0)."
        )

        return InsightBlock(title="Boundary Conditions", content=body, level="info")

    def _solver_insight(self, steps: List[Dict], nlgeom: bool) -> InsightBlock:
        step = steps[0] if steps else {}
        analysis = step.get("analysis_type", "static")
        inc = step.get("initial_inc")
        total = step.get("total_time")
        max_inc = step.get("max_increments", 100)

        if analysis == "static" and not nlgeom:
            body = (
                "Linear static analysis: K u = F.\n\n"
                "The global stiffness matrix K is assembled once and factorized. "
                "For N DOFs, direct sparse solvers (e.g., MUMPS, PARDISO) cost O(N^1.5) in 2D "
                "and O(N^2) in 3D. Iterative solvers (e.g., CG with Jacobi preconditioning) "
                "cost O(N) per iteration but may struggle with ill-conditioned contact problems."
            )
        elif analysis == "static" and nlgeom:
            body = (
                "Nonlinear static analysis: Newton-Raphson iteration.\n\n"
                "At each load increment, the tangent stiffness K_t is assembled and factorized. "
                "The displacement correction is delta_u = K_t^-1 R, where R is the residual force vector. "
                "Convergence is checked via ||R|| / ||F_ext|| < tolerance (default ~1e-5). "
                "Line search or arc-length methods improve robustness near limit points."
            )
        elif analysis == "dynamic":
            body = (
                "Dynamic analysis: Newmark-beta time integration.\n\n"
                "The equations M a + C v + K u = F are integrated in time. "
                "Newmark parameters beta=0.25, gamma=0.5 give unconditional stability "
                "(average acceleration method). The time step should resolve the highest "
                "frequency of interest: dt < T_min / 10, where T_min = 2*pi / omega_max."
            )
        else:
            body = f"Analysis type: {analysis}. Solver details depend on the specific procedure."

        if inc is not None and total is not None:
            body += (
                f"\n\nStep control: initial increment = {inc}, total time = {total}, "
                f"max increments = {max_inc}. "
                f"Abaqus automatically adjusts increment size based on convergence. "
                f"If convergence fails, the increment is bisected (cutback)."
            )

        return InsightBlock(title="Solution Procedure", content=body, level="math")

    def _stability_warnings(self, materials: List[Dict], bcs: List[Dict], steps: List[Dict]) -> Optional[InsightBlock]:
        warnings = []

        for mat in materials:
            E = mat.get("youngs_modulus")
            nu = mat.get("poisson_ratio")
            if E is not None and E <= 0:
                warnings.append(f"Material '{mat.get('name')}' has E = {E} <= 0 (unstable).")
            if nu is not None and not (-1 < nu < 0.5):
                warnings.append(
                    f"Material '{mat.get('name')}' has nu = {nu} outside (-1, 0.5) (stiffness not positive definite)."
                )

        # Check rigid body motion constraints (simplified)
        fixed_dofs = set()
        for bc in bcs:
            val = bc.get("value", 0.0)
            if val == 0.0:
                dof = bc.get("dof", "")
                fixed_dofs.add(dof)

        # In 3D, need at least 6 independent constraints (3 translations + 3 rotations)
        # Simplified heuristic: at least 3 fixed displacement components on distinct nodes
        if len(bcs) < 3 and steps and steps[0].get("analysis_type") == "static":
            warnings.append(
                "Fewer than 3 displacement boundary conditions detected. "
                "Rigid body motion may not be fully constrained, causing a singular stiffness matrix."
            )

        if not warnings:
            return None

        return InsightBlock(
            title="Stability & Consistency Warnings",
            content="\n".join(f"  - {w}" for w in warnings),
            level="warning",
        )
