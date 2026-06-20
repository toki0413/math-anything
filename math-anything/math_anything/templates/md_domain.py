"""MD domain template — shared narrative for LAMMPS, GROMACS, etc.

New MD engines only need to set:
  software_name, integrator_name
and optionally override specific sections.
"""

from typing import Any, Dict, List, Optional

from .base import CheckTemplate, DraftTemplate, InsightTemplate, NarrativeSection

# ────────────────────────────────────────────────
# MD Insight Sections
# ────────────────────────────────────────────────


class MDInsightTemplate(InsightTemplate):
    domain_name = "md"
    software_name = "MD Software"
    integrator_name = "Velocity Verlet"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        if not self._INSIGHT_SECTIONS:
            self._register_default_sections()

    def _register_default_sections(self) -> None:
        MDInsightTemplate._INSIGHT_SECTIONS = [
            _md_problem_overview,
            _md_ensemble,
            _md_potential,
            _md_timestep,
            _md_boundary,
            _md_equilibration,
        ]


# ── Section generators ──


def _md_problem_overview(tpl: MDInsightTemplate) -> Optional[NarrativeSection]:
    body = (
        "The simulation solves Hamilton's equations of motion for a system of N interacting particles:\n\n"
        "  dqi/dt = partial H / partial pi\n"
        "  dpi/dt = -partial H / partial qi\n\n"
        "where H = T + V is the classical Hamiltonian, T = sum_i |pi|^2 / (2 mi) is the kinetic energy, "
        "and V(q) is the potential energy surface (PES) approximated by an interatomic potential.\n\n"
        "The equations are integrated numerically using a symplectic integrator (Velocity Verlet), "
        "which preserves phase-space volume and conserves total energy in the microcanonical ensemble."
    )
    return NarrativeSection(title="Mathematical Problem Overview", body=body, level="math")


def _md_ensemble(tpl: MDInsightTemplate) -> Optional[NarrativeSection]:
    ensemble = tpl._param("ensemble", "NVE")
    tpl._param("fixes", [])
    temp = tpl._param("temperature")
    pressure = tpl._param("pressure")

    descriptions = {
        "NVE": (
            "Microcanonical ensemble (NVE): total energy E is conserved exactly (up to numerical drift). "
            "Suitable for studying intrinsic dynamics without thermostat artifacts."
        ),
        "NVT": (
            "Canonical ensemble (NVT): temperature is controlled via a thermostat. "
            "The Nosé-Hoover chain generates the correct Boltzmann distribution by extending "
            "phase space with a fictitious heat bath variable s and its conjugate momentum."
        ),
        "NPT": (
            "Isothermal-isobaric ensemble (NPT): both temperature and pressure are controlled. "
            "The Parrinello-Rahman barostat couples the simulation box to a fictitious piston, "
            "allowing cell shape fluctuations while preserving the correct Gibbs distribution."
        ),
        "Langevin": (
            "Langevin dynamics: stochastic thermostat adds friction and random force terms. "
            "The equation becomes mi dvi/dt = Fi - gamma vi + sqrt(2 gamma kB T / dt) R(t), "
            "where R(t) is delta-correlated white noise. Suitable for implicit solvent or coarse-grained models."
        ),
    }

    body = descriptions.get(ensemble, f"Ensemble: {ensemble}.")

    if temp:
        body += f"\nTarget temperature: {temp} K."
    if pressure:
        body += f"\nTarget pressure: {pressure} (units consistent with model)."

    return NarrativeSection(title="Thermodynamic Ensemble", body=body, level="math")


def _md_potential(tpl: MDInsightTemplate) -> Optional[NarrativeSection]:
    pair_style = tpl._param("pair_style", "")
    pair_args = tpl._param("pair_style_args", [])

    potential_knowledge = {
        "lj": (
            "Lennard-Jones (LJ) 12-6 potential: V(r) = 4 epsilon [(sigma/r)^12 - (sigma/r)^6]. "
            "The r^-12 term models Pauli repulsion; the r^-6 term models dispersion. "
            "No directional bonding; suitable for noble gases and simple fluids."
        ),
        "eam": (
            "Embedded-Atom Method (EAM): E = sum_i F(rho_i) + 1/2 sum_{i!=j} phi(r_ij). "
            "The embedding energy F depends on the local electron density rho_i = sum_j f(r_ij). "
            "Accurate for metallic bonding where conduction electrons mediate interactions."
        ),
        "reax": (
            "ReaxFF: bond order-dependent reactive force field. "
            "E = E_bond + E_over + E_under + E_val + E_pen + E_tors + E_conj + E_vdW + E_Coul. "
            "Bonds break and form dynamically based on interatomic distances. "
            "Suitable for reactive chemistry but ~10x slower than non-reactive potentials."
        ),
        "tersoff": (
            "Tersoff potential: three-body bond-order potential for covalent materials (Si, C, GaN). "
            "V_ij = f_c(r_ij) [f_R(r_ij) + b_ij f_A(r_ij)], where b_ij depends on the local environment. "
            "Captures directional bonding and coordination-dependent elastic properties."
        ),
        "sw": (
            "Stillinger-Weber (SW): three-body potential for silicon and group-IV materials. "
            "Includes two-body stretch and three-body angle terms to stabilize tetrahedral coordination."
        ),
        "buck": (
            "Buckingham potential: V(r) = A exp(-r/rho) - C/r^6. "
            "Exponential repulsion + dispersion. Common for ionic oxides and ceramics."
        ),
    }

    desc = None
    for key, value in potential_knowledge.items():
        if key in pair_style.lower():
            desc = value
            break

    if desc is None:
        desc = (
            f"Potential style: {pair_style}. "
            "Ensure the potential is validated for the chemistry and thermodynamic state of interest."
        )

    body = desc
    if pair_args:
        body += f"\n\nPotential parameters: {pair_args}."

    return NarrativeSection(title="Interatomic Potential", body=body, level="math")


def _md_timestep(tpl: MDInsightTemplate) -> Optional[NarrativeSection]:
    timestep = tpl._param("timestep")
    units = tpl._param("units", "metal")

    if timestep is None:
        return NarrativeSection(
            title="Timestep",
            body="Timestep not specified. Typical values: 1 fs for metals/oxides, 0.5 fs for organics.",
            level="info",
        )

    # Estimate stability limit
    body = (
        f"Integration timestep: {timestep} (units: {units}).\n\n"
        "Symplectic integrators are stable only if the timestep resolves the fastest vibrational mode: "
        "dt < T_fast / (2 pi) ~ 1 / (10 omega_max). For C-H stretches (omega ~ 3000 cm^-1), "
        "T_fast ~ 10 fs, so dt < 1 fs is required."
    )

    if units == "metal":
        if timestep > 0.002:  # 2 fs in metal units = 2e-3 ps
            body += (
                f"\n\nWARNING: dt = {timestep} ps ({timestep * 1000:.1f} fs) may be too large. "
                "For metals with light elements, use dt <= 0.001 ps (1 fs)."
            )
            return NarrativeSection(title="Timestep Stability", body=body, level="warning")
    elif units == "real":
        if timestep > 2.0:
            body += f"\n\nWARNING: dt = {timestep} fs may be too large. For organic systems, use dt <= 1.0 fs."
            return NarrativeSection(title="Timestep Stability", body=body, level="warning")

    body += "\n\nTimestep appears reasonable for the expected vibrational spectrum."
    return NarrativeSection(title="Timestep Stability", body=body, level="info")


def _md_boundary(tpl: MDInsightTemplate) -> Optional[NarrativeSection]:
    boundary = tpl._param("boundary", "p p p")
    dims = boundary.split()

    desc_map = {
        "p": "periodic",
        "f": "fixed (non-periodic)",
        "s": "shrink-wrapped",
        "m": "shrink-wrapped with minimum",
    }
    desc = [desc_map.get(d, d) for d in dims]

    body = f"Boundary conditions: {' / '.join(desc)}.\n\n"

    if all(d == "p" for d in dims):
        body += (
            "Periodic boundaries eliminate surface effects, mimicking bulk behavior. "
            "Reciprocal space interactions are computed via Ewald summation or particle-mesh Ewald (PME)."
        )
    elif all(d == "f" for d in dims):
        body += (
            "Fixed boundaries expose free surfaces. Surface tension and edge effects "
            "will dominate if the system is small."
        )
    else:
        body += (
            "Mixed boundary conditions. Ensure consistency with the intended physics: "
            "periodic in bulk directions, fixed in surface-normal directions for thin films."
        )

    return NarrativeSection(title="Boundary Conditions", body=body, level="info")


def _md_equilibration(tpl: MDInsightTemplate) -> Optional[NarrativeSection]:
    run_steps = tpl._param("run_steps")
    timestep = tpl._param("timestep")
    units = tpl._param("units", "metal")

    if run_steps is None or timestep is None:
        return None

    # Convert to physical time
    if units == "metal":
        total_time = run_steps * timestep  # ps
        time_str = f"{total_time:.2f} ps"
    elif units == "real":
        total_time = run_steps * timestep  # fs
        time_str = f"{total_time:.2f} fs = {total_time / 1000:.2f} ps"
    else:
        time_str = f"{run_steps * timestep} (units: {units})"

    body = (
        f"Simulation length: {run_steps} steps x {timestep} = {time_str}.\n\n"
        "Ensure the run is long enough to sample the relevant configurational space. "
        "For diffusion coefficients, mean-squared displacement should be linear over the run. "
        "For radial distribution functions, at least 10 ps of production is recommended."
    )

    return NarrativeSection(title="Simulation Protocol", body=body, level="info")


# ────────────────────────────────────────────────
# MD Draft Sections
# ────────────────────────────────────────────────


class MDDraftTemplate(DraftTemplate):
    domain_name = "md"
    software_name = "MD Software"
    integrator_name = "Velocity Verlet"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        if not self._DRAFT_SECTIONS:
            self._register_default_sections()

    def _register_default_sections(self) -> None:
        MDDraftTemplate._DRAFT_SECTIONS = [
            _md_draft_theory,
            _md_draft_software,
            _md_draft_potential,
            _md_draft_ensemble,
            _md_draft_protocol,
            _md_draft_boundary,
        ]


def _md_draft_theory(tpl: MDDraftTemplate) -> Optional[NarrativeSection]:
    body = (
        "Classical molecular dynamics integrates Hamilton's equations numerically. "
        "The potential energy surface is approximated by an interatomic potential fitted "
        "to quantum mechanical reference data or experimental observables."
    )
    return NarrativeSection(title="Simulation Method", body=body)


def _md_draft_software(tpl: MDDraftTemplate) -> Optional[NarrativeSection]:
    body = (
        f"Simulations performed using {tpl.software_name}. "
        f"Integration algorithm: {tpl.integrator_name} (symplectic, time-reversible)."
    )
    return NarrativeSection(title="Software", body=body)


def _md_draft_potential(tpl: MDDraftTemplate) -> Optional[NarrativeSection]:
    pair_style = tpl._param("pair_style", "unspecified")
    body = f"Interatomic potential: {pair_style}. "
    if "lj" in pair_style.lower():
        body += "Lennard-Jones 12-6 for van der Waals interactions."
    elif "eam" in pair_style.lower():
        body += "Embedded-Atom Method for metallic bonding."
    elif "reax" in pair_style.lower():
        body += "ReaxFF for reactive chemistry."
    else:
        body += "Refer to potential documentation for functional form and parameterization."
    return NarrativeSection(title="Interatomic Potential", body=body)


def _md_draft_ensemble(tpl: MDDraftTemplate) -> Optional[NarrativeSection]:
    ensemble = tpl._param("ensemble", "NVE")
    temp = tpl._param("temperature")
    pressure = tpl._param("pressure")
    body = f"Thermodynamic ensemble: {ensemble}."
    if temp:
        body += f" Target temperature: {temp} K."
    if pressure:
        body += f" Target pressure: {pressure}."
    return NarrativeSection(title="Thermostat and Barostat", body=body)


def _md_draft_protocol(tpl: MDDraftTemplate) -> Optional[NarrativeSection]:
    timestep = tpl._param("timestep")
    run_steps = tpl._param("run_steps")
    units = tpl._param("units", "metal")
    if timestep and run_steps:
        body = f"Timestep = {timestep} ({units}), total steps = {run_steps}."
    else:
        body = "Integration parameters not fully specified."
    return NarrativeSection(title="Integration Protocol", body=body)


def _md_draft_boundary(tpl: MDDraftTemplate) -> Optional[NarrativeSection]:
    boundary = tpl._param("boundary", "p p p")
    body = f"Boundary conditions: {boundary}."
    return NarrativeSection(title="Boundary Conditions", body=body)


# ────────────────────────────────────────────────
# MD Check Sections
# ────────────────────────────────────────────────


class MDCheckTemplate(CheckTemplate):
    domain_name = "md"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        if not self._CHECK_SECTIONS:
            self._register_default_sections()

    def _register_default_sections(self) -> None:
        MDCheckTemplate._CHECK_SECTIONS = [
            _md_check_timestep,
            _md_check_ensemble,
            _md_check_boundary,
            _md_check_potential,
            _md_check_run_length,
        ]


def _md_check_timestep(tpl: MDCheckTemplate) -> List[NarrativeSection]:
    results = []
    timestep = tpl._param("timestep")
    units = tpl._param("units", "metal")
    if timestep is None:
        results.append(
            NarrativeSection(
                title="Timestep missing",
                body="No timestep specified. Default may be inappropriate.",
                level="warning",
            )
        )
        return results

    if units == "metal" and timestep > 0.002:
        results.append(
            NarrativeSection(
                title="Timestep possibly too large",
                body=f"dt = {timestep} ps ({timestep * 1000:.0f} fs) may exceed vibrational stability limit.",
                level="warning",
            )
        )
    elif units == "real" and timestep > 2.0:
        results.append(
            NarrativeSection(
                title="Timestep possibly too large",
                body=f"dt = {timestep} fs may exceed vibrational stability limit.",
                level="warning",
            )
        )
    else:
        results.append(
            NarrativeSection(
                title="Timestep acceptable",
                body=f"dt = {timestep} appears reasonable.",
                level="info",
            )
        )
    return results


def _md_check_ensemble(tpl: MDCheckTemplate) -> List[NarrativeSection]:
    results = []
    ensemble = tpl._param("ensemble", "NVE")
    tpl._param("fixes", [])
    temp = tpl._param("temperature")

    if ensemble == "NVE" and temp is not None:
        results.append(
            NarrativeSection(
                title="NVE with temperature target",
                body="NVE ensemble does not control temperature. Temperature target will be ignored.",
                level="info",
            )
        )

    if ensemble in ("NVT", "NPT") and temp is None:
        results.append(
            NarrativeSection(
                title="Temperature missing for thermostatted ensemble",
                body=f"{ensemble} requires a target temperature.",
                level="warning",
            )
        )
    return results


def _md_check_boundary(tpl: MDCheckTemplate) -> List[NarrativeSection]:
    results = []
    boundary = tpl._param("boundary", "p p p")
    dims = boundary.split()
    if len(dims) != 3:
        results.append(
            NarrativeSection(
                title="Invalid boundary specification",
                body=f"Boundary '{boundary}' does not have 3 dimensions.",
                level="error",
            )
        )
    return results


def _md_check_potential(tpl: MDCheckTemplate) -> List[NarrativeSection]:
    results = []
    pair_style = tpl._param("pair_style")
    if pair_style is None:
        results.append(
            NarrativeSection(
                title="No pair style",
                body="No pair_style defined. Simulation will crash.",
                level="error",
            )
        )
    return results


def _md_check_run_length(tpl: MDCheckTemplate) -> List[NarrativeSection]:
    results = []
    run_steps = tpl._param("run_steps")
    if run_steps is None:
        results.append(
            NarrativeSection(
                title="No run command",
                body="No run steps defined.",
                level="warning",
            )
        )
    elif run_steps < 100:
        results.append(
            NarrativeSection(
                title="Very short run",
                body=f"Only {run_steps} steps. Insufficient for statistics.",
                level="warning",
            )
        )
    return results
