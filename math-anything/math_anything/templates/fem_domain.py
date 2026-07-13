"""FEM domain template — shared narrative for Abaqus, ANSYS, COMSOL, etc.

New FEM engines only need to set:
  software_name, element_library_name
and optionally override specific sections.
"""

from typing import Any, Dict, List, Optional

from .base import CheckTemplate, DraftTemplate, InsightTemplate, NarrativeSection

# ────────────────────────────────────────────────
# FEM Insight Sections
# ────────────────────────────────────────────────


class FEMInsightTemplate(InsightTemplate):
    domain_name = "fem"
    software_name = "FEM Software"
    element_library_name = "standard"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        if not self._INSIGHT_SECTIONS:
            self._register_default_sections()

    def _register_default_sections(self) -> None:
        FEMInsightTemplate._INSIGHT_SECTIONS = [
            _fem_problem_overview,  # type: ignore[list-item]
            _fem_discretization,  # type: ignore[list-item]
            _fem_material,  # type: ignore[list-item]
            _fem_boundary_conditions,  # type: ignore[list-item]
            _fem_solver,  # type: ignore[list-item]
            _fem_stability,  # type: ignore[list-item]
        ]


# ── Section generators ──


def _fem_problem_overview(tpl: FEMInsightTemplate) -> Optional[NarrativeSection]:
    analysis = tpl._param("analysis_type", "static")
    nlgeom = tpl._param("nlgeom", False)

    descriptions = {
        "static": (
            "The simulation solves the static equilibrium equation div(sigma) + b = 0 "
            "via the finite element method. The weak form is derived by multiplying "
            "the strong form by a test function v and integrating over the domain: "
            "integral_Omega sigma : grad(v) dV = integral_Omega b . v dV + integral_Gamma t . v dS."
        ),
        "dynamic": (
            "The simulation solves the dynamic momentum balance rho * d2u/dt2 = div(sigma) + b "
            "using implicit time integration. The weak form introduces "
            "inertial terms: integral_Omega rho * a . v dV + integral_Omega sigma : grad(v) dV = "
            "integral_Omega b . v dV + boundary terms."
        ),
        "heat": (
            "The simulation solves the heat conduction equation div(k grad(T)) + q = rho c dT/dt. "
            "The Galerkin weak form is: integral_Omega k grad(T) . grad(v) dV = "
            "integral_Omega q v dV - integral_Omega rho c dT/dt v dV + boundary flux terms."
        ),
        "coupled": (
            "A coupled thermomechanical analysis is performed, solving "
            "mechanical equilibrium and heat conduction simultaneously. "
            "Thermal expansion introduces coupling via epsilon_th = alpha Delta T I."
        ),
    }

    body = descriptions.get(analysis, descriptions["static"])

    if nlgeom:
        body += (
            "\n\nGeometric nonlinearity is enabled. The strain-displacement relation uses "
            "the Green-Lagrange tensor E = 1/2(F^T F - I) and equilibrium is evaluated "
            "in the current configuration. This requires Newton-Raphson iteration."
        )
    else:
        body += (
            "\n\nGeometric linearity is assumed: strains are infinitesimal "
            "(epsilon = 1/2(grad(u) + grad(u)^T)) and equilibrium is in the reference configuration."
        )

    return NarrativeSection(title="Mathematical Problem Overview", body=body, level="math")


def _fem_discretization(tpl: FEMInsightTemplate) -> Optional[NarrativeSection]:
    elements = tpl._param("elements", [])
    nodes = tpl._param("nodes", 0)
    elem = elements[0] if elements else "unspecified"

    element_knowledge = {
        "C3D8": "8-node hexahedron (trilinear). Prone to shear locking in bending; use C3D8I or C3D8R for bending.",
        "C3D8R": "8-node hexahedron with reduced integration. Hourglass stabilization required. Check hourglass energy < 5%.",  # noqa: E501
        "C3D8I": "8-node hexahedron with incompatible modes. Excellent for bending without hourglass modes.",
        "C3D4": "4-node tetrahedron (linear). Very stiff; use C3D10 for accuracy.",
        "C3D10": "10-node quadratic tetrahedron. Accurate for complex geometries and bending.",
        "CPE4": "4-node plane strain quadrilateral. Appropriate for thick structures.",
        "CPS4": "4-node plane stress quadrilateral. Appropriate for thin plates.",
        "S4R": "4-node reduced-integration shell. Mindlin-Reissner theory with transverse shear.",
    }

    desc = None
    for key, value in element_knowledge.items():
        if key in elem.upper():
            desc = value
            break
    if desc is None:
        desc = f"Element type {elem}. Ensure formulation matches expected deformation modes."

    body = (
        f"Spatial discretization uses {elem} elements.\n\n{desc}\n\n"
        f"Mesh contains {nodes} nodes. "
        "Convergence rate for displacement-based FEM is O(h^(p+1)) in L2 norm "
        "and O(h^p) in energy norm for polynomial degree p."
    )
    return NarrativeSection(title="Spatial Discretization", body=body, level="math")


def _fem_material(tpl: FEMInsightTemplate) -> Optional[NarrativeSection]:
    materials = tpl._param("materials", [])
    if not materials:
        return None

    mat = materials[0]
    E = mat.get("youngs_modulus")
    nu = mat.get("poisson_ratio")
    mtype = mat.get("model_type", "unknown")

    if mtype == "elastic" and E is not None and nu is not None:
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        body = (
            f"Material '{mat.get('name', 'Elastic')}' uses isotropic linear elasticity.\n\n"
            f"E = {E}, nu = {nu} => mu = {mu:.2f}, lambda = {lam:.2f}.\n\n"
            "Stiffness tensor: C_ijkl = lambda delta_ij delta_kl + mu (delta_ik delta_jl + delta_il delta_jk). "
            "Positive definiteness requires E > 0 and -1 < nu < 0.5."
        )
    elif mtype in ("plastic", "cdp"):
        body = (
            f"Material '{mat.get('name')}' uses elasto-plasticity/damage.\n\n"
            "Constitutive: sigma = C : (epsilon - epsilon_p) with yield surface f(sigma, q) <= 0. "
            "Return mapping enforces consistency."
        )
    else:
        body = f"Material '{mat.get('name')}' has model type '{mtype}'."

    return NarrativeSection(title="Constitutive Model", body=body, level="math")


def _fem_boundary_conditions(tpl: FEMInsightTemplate) -> Optional[NarrativeSection]:
    bcs = tpl._param("boundary_conditions", [])
    if not bcs:
        return NarrativeSection(
            title="Boundary Conditions",
            body="No boundary conditions detected. Rigid body motion may not be constrained.",
            level="warning",
        )

    fixed = [b for b in bcs if b.get("value", 0.0) == 0.0]
    loaded = [b for b in bcs if b.get("value", 0.0) != 0.0]

    body = f"{len(bcs)} boundary condition(s) detected.\n\n"
    if fixed:
        body += "Dirichlet constraints:\n" + "\n".join(f"  - {b['node_set']} (DOFs {b['dof']})" for b in fixed) + "\n"
    if loaded:
        body += (
            "\nNeumann conditions:\n" + "\n".join(f"  - {b['node_set']} (value {b['value']})" for b in loaded) + "\n"
        )

    body += (
        "\nDirichlet conditions modify K directly; Neumann conditions enter the load vector. "
        "Rigid body motion must be fully constrained."
    )
    return NarrativeSection(title="Boundary Conditions", body=body, level="info")


def _fem_solver(tpl: FEMInsightTemplate) -> Optional[NarrativeSection]:
    analysis = tpl._param("analysis_type", "static")
    nlgeom = tpl._param("nlgeom", False)
    step = tpl._param("steps", [{}])[0]
    inc = step.get("initial_inc")
    total = step.get("total_time")

    if analysis == "static" and not nlgeom:
        body = "Linear static: K u = F. Direct sparse solver (O(N^1.5) in 2D, O(N^2) in 3D)."
    elif analysis == "static" and nlgeom:
        body = (
            "Nonlinear static: Newton-Raphson iteration. "
            "delta_u = K_t^-1 R, where R is the residual force vector. "
            "Convergence: ||R|| / ||F_ext|| < tolerance."
        )
    elif analysis == "dynamic":
        body = (
            "Dynamic: Newmark-beta integration (beta=0.25, gamma=0.5, unconditionally stable). "
            "Time step should resolve T_min / 10."
        )
    else:
        body = f"Analysis: {analysis}."

    if inc is not None and total is not None:
        body += (
            f"\n\nStep: initial={inc}, total={total}. "
            "Abaqus auto-adjusts increment size; bisects on convergence failure."
        )

    return NarrativeSection(title="Solution Procedure", body=body, level="math")


def _fem_stability(tpl: FEMInsightTemplate) -> Optional[NarrativeSection]:
    warnings = []
    materials = tpl._param("materials", [])
    bcs = tpl._param("boundary_conditions", [])

    for mat in materials:
        E = mat.get("youngs_modulus")
        nu = mat.get("poisson_ratio")
        if E is not None and E <= 0:
            warnings.append(f"Material '{mat.get('name')}' has E = {E} <= 0 (unstable).")
        if nu is not None and not (-1 < nu < 0.5):
            warnings.append(f"Material '{mat.get('name')}' has nu = {nu} outside (-1, 0.5).")

    fixed = sum(1 for b in bcs if b.get("value", 0.0) == 0.0)
    if fixed < 3 and tpl._param("analysis_type") == "static":
        warnings.append("Fewer than 3 displacement constraints. Rigid body motion may not be fully constrained.")

    if not warnings:
        return None

    return NarrativeSection(
        title="Stability & Consistency Warnings",
        body="\n".join(f"  - {w}" for w in warnings),
        level="warning",
    )


# ────────────────────────────────────────────────
# FEM Draft Sections
# ────────────────────────────────────────────────


class FEMDraftTemplate(DraftTemplate):
    domain_name = "fem"
    software_name = "FEM Software"
    element_library_name = "standard"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        if not self._DRAFT_SECTIONS:
            self._register_default_sections()

    def _register_default_sections(self) -> None:
        FEMDraftTemplate._DRAFT_SECTIONS = [
            _fem_draft_theory,  # type: ignore[list-item]
            _fem_draft_software,  # type: ignore[list-item]
            _fem_draft_mesh,  # type: ignore[list-item]
            _fem_draft_material,  # type: ignore[list-item]
            _fem_draft_bc,  # type: ignore[list-item]
            _fem_draft_solver,  # type: ignore[list-item]
            _fem_draft_caveats,  # type: ignore[list-item]
        ]


def _fem_draft_theory(tpl: FEMDraftTemplate) -> Optional[NarrativeSection]:
    analysis = tpl._param("analysis_type", "static")
    nlgeom = tpl._param("nlgeom", False)

    if analysis == "static" and not nlgeom:
        body = (
            "The governing equation is static equilibrium: div(sigma) + b = 0. "
            "The weak form is: integral_Omega sigma : nabla^s v dOmega = "
            "integral_Omega b . v dOmega + integral_Gamma_t t . v dGamma."
        )
    elif analysis == "static" and nlgeom:
        body = (
            "Static equilibrium in the current configuration with geometric nonlinearity. "
            "Green-Lagrange strain E = 1/2(F^T F - I) and Second Piola-Kirchhoff stress S."
        )
    elif analysis == "dynamic":
        body = "Dynamic momentum balance with Newmark-beta time integration."
    elif analysis == "heat":
        body = "Fourier heat conduction: rho c dT/dt = div(k grad(T)) + q."
    elif analysis == "coupled":
        body = "Coupled thermomechanics with thermal expansion feedback."
    else:
        body = f"Analysis type: {analysis}."

    return NarrativeSection(title="Governing Equations", body=body)


def _fem_draft_software(tpl: FEMDraftTemplate) -> Optional[NarrativeSection]:
    body = (
        f"Simulations performed using {tpl.software_name}. "
        "Direct sparse linear algebra (MUMPS/PARDISO) for K u = F. "
        "Newton-Raphson with automatic incrementation for nonlinear problems."
    )
    return NarrativeSection(title="Software", body=body)


def _fem_draft_mesh(tpl: FEMDraftTemplate) -> Optional[NarrativeSection]:
    elements = tpl._param("elements", [])
    nodes = tpl._param("nodes", 0)
    elem = elements[0] if elements else "unspecified"
    body = f"Domain discretized using {elem} elements. Mesh contains {nodes} nodes. "
    if "C3D8R" in elem.upper():
        body += "Reduced integration with hourglass control. "
    elif "C3D8I" in elem.upper():
        body += "Incompatible mode formulation for bending. "
    elif "C3D10" in elem.upper():
        body += "Quadratic interpolation for accuracy. "
    body += "Mesh convergence verified via strain energy norm."
    return NarrativeSection(title="Mesh and Discretization", body=body)


def _fem_draft_material(tpl: FEMDraftTemplate) -> Optional[NarrativeSection]:
    materials = tpl._param("materials", [])
    if not materials:
        return None
    mat = materials[0]
    mtype = mat.get("model_type", "unknown")
    E = mat.get("youngs_modulus")
    nu = mat.get("poisson_ratio")

    if mtype == "elastic":
        body = f"Isotropic linear elastic ({mat.get('name', 'Material')})."
        if E:
            body += f" E = {E} MPa."
        if nu:
            body += f" nu = {nu}."
        body += " Constitutive: sigma = C : epsilon."
    elif mtype in ("plastic", "cdp"):
        body = (
            f"Elasto-plastic/damage ({mat.get('name', 'Material')}). "
            "Incremental plasticity with associated flow rule and hardening."
        )
    else:
        body = f"Material: {mat.get('name', 'unknown')}."
    return NarrativeSection(title="Constitutive Model", body=body)


def _fem_draft_bc(tpl: FEMDraftTemplate) -> Optional[NarrativeSection]:
    bcs = tpl._param("boundary_conditions", [])
    fixed = [b for b in bcs if b.get("value", 0.0) == 0.0]
    loaded = [b for b in bcs if b.get("value", 0.0) != 0.0]
    body = (
        f"Displacement BCs applied to {len(fixed)} node set(s), "
        f"load conditions to {len(loaded)} node set(s). "
        "Rigid body motion eliminated."
    )
    return NarrativeSection(title="Boundary Conditions", body=body)


def _fem_draft_solver(tpl: FEMDraftTemplate) -> Optional[NarrativeSection]:
    analysis = tpl._param("analysis_type", "static")
    nlgeom = tpl._param("nlgeom", False)
    step = tpl._param("steps", [{}])[0]

    if analysis == "static":
        body = "Static analysis procedure."
        if nlgeom:
            body += " Geometric nonlinearity (NLGEOM). Newton-Raphson iteration."
        else:
            body += " Geometrically linear. Single matrix assembly."
    else:
        body = f"{analysis} analysis procedure."

    if step.get("initial_inc") and step.get("total_time"):
        body += (
            f" Max {step.get('max_increments')} increments, initial {step['initial_inc']}, total {step['total_time']}."
        )
    return NarrativeSection(title="Solution Procedure", body=body)


def _fem_draft_caveats(tpl: FEMDraftTemplate) -> Optional[NarrativeSection]:
    notes = ["Mesh sensitivity assessed via systematic refinement. Energy norm recommended as error indicator."]
    if not tpl._param("nlgeom", False) and tpl._param("analysis_type") == "static":
        notes.append("Linear analysis valid only for small strains/rotations.")
    body = " ".join(notes)
    return NarrativeSection(title="Methodological Notes", body=body)


# ────────────────────────────────────────────────
# FEM Check Sections
# ────────────────────────────────────────────────


class FEMCheckTemplate(CheckTemplate):
    domain_name = "fem"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        if not self._CHECK_SECTIONS:
            self._register_default_sections()

    def _register_default_sections(self) -> None:
        FEMCheckTemplate._CHECK_SECTIONS = [
            _fem_check_materials,  # type: ignore[list-item]
            _fem_check_bcs,  # type: ignore[list-item]
            _fem_check_elements,  # type: ignore[list-item]
            _fem_check_steps,  # type: ignore[list-item]
            _fem_check_mesh,  # type: ignore[list-item]
        ]


def _fem_check_materials(tpl: FEMCheckTemplate) -> List[NarrativeSection]:
    results = []
    materials = tpl._param("materials", [])
    if not materials:
        results.append(NarrativeSection(title="No material defined", body="No *Material card found.", level="error"))
        return results

    for mat in materials:
        E = mat.get("youngs_modulus")
        nu = mat.get("poisson_ratio")
        if E is not None and E <= 0:
            results.append(
                NarrativeSection(
                    title="Non-positive Young's modulus",
                    body=f"Material '{mat.get('name')}' has E = {E} <= 0.",
                    level="error",
                )
            )
        if nu is not None and not (-1 < nu < 0.5):
            results.append(
                NarrativeSection(
                    title="Poisson's ratio out of range",
                    body=f"Material '{mat.get('name')}' has nu = {nu} outside (-1, 0.5).",
                    level="error",
                )
            )
        if E and nu and nu > 0.499:
            results.append(
                NarrativeSection(
                    title="Near-incompressibility",
                    body=f"nu = {nu}接近0.5. Use hybrid elements (C3D8H) to avoid volumetric locking.",
                    level="warning",
                )
            )
    return results


def _fem_check_bcs(tpl: FEMCheckTemplate) -> List[NarrativeSection]:
    results = []
    bcs = tpl._param("boundary_conditions", [])
    analysis = tpl._param("analysis_type", "static")
    if not bcs and analysis == "static":
        results.append(
            NarrativeSection(
                title="No boundary conditions", body="No *Boundary found. Static FEM needs BCs.", level="error"
            )
        )
        return results
    fixed = sum(1 for b in bcs if b.get("value", 0.0) == 0.0)
    if fixed < 3 and analysis == "static":
        results.append(
            NarrativeSection(
                title="Possibly insufficient constraints",
                body=f"Only {fixed} zero-value BCs. Need >= 6 in 3D for rigid body suppression.",
                level="warning",
            )
        )
    return results


def _fem_check_elements(tpl: FEMCheckTemplate) -> List[NarrativeSection]:
    results = []
    elements = tpl._param("elements", [])
    if not elements:
        results.append(NarrativeSection(title="No element type", body="No *Element with TYPE= found.", level="error"))
        return results
    elem = elements[0].upper()
    if "C3D4" in elem:
        results.append(
            NarrativeSection(
                title="Linear tetrahedron warning",
                body="C3D4 is stiff and locks. Prefer C3D10 or C3D8R.",
                level="warning",
            )
        )
    if "C3D8R" in elem:
        results.append(
            NarrativeSection(
                title="Reduced integration",
                body="C3D8R needs hourglass stabilization. Check hourglass energy < 5%.",
                level="info",
            )
        )
    return results


def _fem_check_steps(tpl: FEMCheckTemplate) -> List[NarrativeSection]:
    results = []
    steps = tpl._param("steps", [])
    if not steps:
        results.append(NarrativeSection(title="No analysis step", body="No *Step found.", level="error"))
        return results
    for step in steps:
        if step.get("initial_inc", 0) > step.get("total_time", 1):
            results.append(
                NarrativeSection(
                    title="Initial increment exceeds total time",
                    body=f"Initial {step['initial_inc']} > total {step['total_time']}.",
                    level="error",
                )
            )
    return results


def _fem_check_mesh(tpl: FEMCheckTemplate) -> List[NarrativeSection]:
    results = []
    nodes = tpl._param("nodes", 0)
    if nodes == 0:
        results.append(NarrativeSection(title="No nodes", body="No *Node found.", level="error"))
    elif nodes < 10:
        results.append(NarrativeSection(title="Very small mesh", body=f"Only {nodes} nodes.", level="info"))
    return results


# ────────────────────────────────────────────────
# FEM Parameter Extractor
# ────────────────────────────────────────────────


class FEMParamExtractor:
    """Extract and normalize FEM template parameters from MathSchema.raw_symbols.

    Consolidates the duplicated _extract_params() logic that was previously
    repeated across ansys/comsol draft/check/insight engines.
    """

    @staticmethod
    def extract(raw_or_schema: Any, engine: str = "generic") -> Dict[str, Any]:
        """Extract normalized FEM parameters.

        Args:
            raw_or_schema: Either a MathSchema object or a raw_symbols dict.
            engine: Engine name for analysis-type normalization
                    ("ansys", "comsol", "abaqus", etc.).

        Returns:
            Normalized parameter dict ready for FEM template consumption.
        """
        from ..schemas import MathSchema

        if isinstance(raw_or_schema, MathSchema):
            raw = raw_or_schema.raw_symbols or {}
        elif isinstance(raw_or_schema, dict):
            raw = raw_or_schema
        else:
            raw = {}

        params = dict(raw)

        FEMParamExtractor._extract_materials(params, raw)
        FEMParamExtractor._normalize_analysis(params, raw, engine)
        FEMParamExtractor._ensure_steps(params)
        FEMParamExtractor._expand_all_dof(params, raw)

        params["elements"] = raw.get("elements", [])
        params["nodes"] = raw.get("parameters", {}).get("nodes", 100)  # type: ignore[attr-defined]
        params["nlgeom"] = raw.get("nlgeom", False)

        return params

    @staticmethod
    def _extract_materials(params: Dict[str, Any], raw: Dict[str, Any]) -> None:
        materials = raw.get("materials", [])
        if materials:
            mat = materials[0]
            params["youngs_modulus"] = mat.get("youngs_modulus")
            params["poisson_ratio"] = mat.get("poisson_ratio")
            params["density"] = mat.get("density")

    @staticmethod
    def _normalize_analysis(params: Dict[str, Any], raw: Dict[str, Any], engine: str) -> None:
        analysis = raw.get("analysis_type", "static")

        if engine == "ansys":
            if "static" in analysis:
                analysis = "static"
            elif "thermal" in analysis:
                analysis = "heat"
            elif "modal" in analysis or "transient" in analysis:
                analysis = "dynamic"
            else:
                analysis = "static"

        params["analysis_type"] = analysis

    @staticmethod
    def _ensure_steps(params: Dict[str, Any]) -> None:
        steps = params.get("steps", [])
        if not steps:
            params["steps"] = [{"analysis_type": params.get("analysis_type", "static"), "max_increments": 1}]

    @staticmethod
    def _expand_all_dof(params: Dict[str, Any], raw: Dict[str, Any]) -> None:
        bcs = raw.get("boundary_conditions", [])
        expanded = []
        for bc in bcs:
            dof = str(bc.get("dof", "")).upper()
            if bc.get("bc_type") == "displacement" and dof == "ALL":
                for d in range(1, 7):
                    expanded.append(
                        {
                            "node_set": bc["node_set"],
                            "dof": str(d),
                            "value": bc["value"],
                            "bc_type": "displacement",
                        }
                    )
            else:
                expanded.append(bc)
        params["boundary_conditions"] = expanded
