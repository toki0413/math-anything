"""DFT domain template — shared narrative for VASP, Quantum ESPRESSO, etc.

New DFT engines only need to set:
  software_name, basis_type, pseudopotential_type
and optionally override specific sections.
"""

from typing import Any, Dict, List, Optional

from .base import CheckTemplate, DraftTemplate, InsightTemplate, NarrativeSection

# ────────────────────────────────────────────────
# DFT Insight Sections
# ────────────────────────────────────────────────


class DFTInsightTemplate(InsightTemplate):
    domain_name = "dft"
    software_name = "DFT Software"
    basis_type = "plane-wave"
    pseudopotential_type = "PAW"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        # Register default sections if not already registered
        if not self._INSIGHT_SECTIONS:
            self._register_default_sections()

    def _register_default_sections(self) -> None:
        DFTInsightTemplate._INSIGHT_SECTIONS = [
            _dft_problem_overview,
            _dft_basis_set,
            _dft_kpoint_sampling,
            _dft_scf_convergence,
            _dft_functional,
            _dft_magnetic,
        ]


# ── Section generators ──


def _dft_problem_overview(tpl: DFTInsightTemplate) -> Optional[NarrativeSection]:
    encut = tpl._param("encut")
    ispin = tpl._param("ispin", 1)
    spin_text = "spin-polarized" if ispin == 2 else "spin-restricted"

    body = (
        f"The simulation solves the Kohn-Sham equations within {spin_text} density functional theory. "
        "The Hohenberg-Kohn theorem establishes a one-to-one mapping between the external potential "
        "and the ground-state electron density. The KS ansatz introduces non-interacting orbitals "
        "such that the effective potential includes exchange-correlation effects.\n\n"
        "The weak form is a Galerkin projection onto a basis set: "
        "<phi_i | (-hbar^2/2m nabla^2 + V_eff) | phi_j> = epsilon_j S_ij, "
        "where S_ij is the overlap matrix. For plane-wave bases, S = I (orthonormal)."
    )
    if encut:
        body += (
            f"\n\nThe kinetic energy cutoff ENCUT = {encut} eV truncates the plane-wave expansion, "
            f"corresponding to a maximum wavevector |G|_max = sqrt(2m_e ENCUT)/hbar."
        )
    return NarrativeSection(title="Mathematical Problem Overview", body=body, level="math")


def _dft_basis_set(tpl: DFTInsightTemplate) -> Optional[NarrativeSection]:
    tpl._param("encut")
    prec = tpl._param("prec", "Normal")
    body = (
        f"The {tpl.basis_type} basis expands the Kohn-Sham orbitals as "
        "psi(r) = sum_G c_G exp(i G . r). The cutoff ENCUT controls the cardinality of the basis.\n\n"
        f"Pseudopotentials ({tpl.pseudopotential_type}) replace the strong Coulomb potential near nuclei "
        "with a smoother effective potential, eliminating core electrons from the explicit calculation. "
        "The projector augmented-wave (PAW) method reconstructs the all-electron density from "
        "pseudo-wavefunctions and on-site augmentation charges."
    )
    if prec.lower() in ("low", "medium"):
        body += (
            f"\n\nPREC = {prec} uses a coarser FFT grid than PREC = Accurate, "
            "which can introduce small errors in forces and stresses."
        )
    return NarrativeSection(title="Basis Set and Pseudopotentials", body=body, level="math")


def _dft_kpoint_sampling(tpl: DFTInsightTemplate) -> Optional[NarrativeSection]:
    mesh = tpl._param("kpoint_mesh", [])
    ismear = tpl._param("ismear", 0)
    sigma = tpl._param("sigma", 0.2)

    if not mesh:
        return None

    total_k = mesh[0] * mesh[1] * mesh[2] if len(mesh) == 3 else 0
    body = (
        f"Brillouin zone sampling uses a {mesh[0]}x{mesh[1]}x{mesh[2]} Monkhorst-Pack mesh "
        f"({total_k} k-points in the full zone). "
        "The discrete sum approximates the integral over the BZ: "
        "sum_k w_k f(k) ~ integral_BZ f(k) d^3k / (2pi)^3.\n\n"
    )

    if ismear in (0, -1, 1, 2):
        body += (
            f"Fermi surface smearing: ISMEAR = {ismear}, SIGMA = {sigma} eV. "
            "This replaces the step function occupation with a smooth distribution, "
            "allowing accurate total energies with sparse k-meshes. "
        )
        if ismear == 0:
            body += "Gaussian smearing broadens the Fermi-Dirac distribution."
        elif ismear == -5:
            body += "Tetrahedron method with Blochl corrections (exact for insulators)."
        elif ismear in (1, 2):
            body += "Methfessel-Paxton smearing (preserves Fermi surface topology)."
    elif ismear == -5:
        body += (
            "Tetrahedron method with Blochl corrections is used. "
            "This is exact for semiconductors and insulators but requires dense k-meshes."
        )

    if total_k < 8 and ismear == -5:
        body += (
            "\n\nWARNING: Tetrahedron method with < 8 k-points is unreliable. Use at least a 4x4x4 mesh (64 points)."
        )
        return NarrativeSection(title="k-Point Sampling", body=body, level="warning")

    return NarrativeSection(title="k-Point Sampling", body=body, level="math")


def _dft_scf_convergence(tpl: DFTInsightTemplate) -> Optional[NarrativeSection]:
    ediff = tpl._param("ediff", 1e-6)
    nelm = tpl._param("nelm", 60)
    algo = tpl._param("algo", "Normal")

    body = (
        "Self-consistent field (SCF) iteration solves the fixed-point problem "
        "rho_{n+1} = f[rho_n] via Pulay/Broyden mixing. "
        "The residual ||rho_{n+1} - rho_n|| is driven below a tolerance.\n\n"
        f"Convergence criteria: EDIFF = {ediff} eV (energy difference between SCF steps). "
        f"Maximum SCF cycles: NELM = {nelm}. "
        f"Electronic minimization algorithm: {algo}."
    )

    if algo.upper() in ("FAST", "VERY_FAST"):
        body += (
            "\n\nRMM-DIIS (Fast) extrapolates the density from previous steps, "
            "accelerating convergence but occasionally diverging for difficult systems."
        )
    elif algo.upper() == "DAMPED":
        body += "\n\nDamped algorithm uses linear mixing with a damping factor. Robust but slower than RMM-DIIS."

    return NarrativeSection(title="SCF Convergence", body=body, level="math")


def _dft_functional(tpl: DFTInsightTemplate) -> Optional[NarrativeSection]:
    functional = tpl._param("functional", "PBE")
    lhf = tpl._param("lhfcalc", False)
    hfscreen = tpl._param("hfscreen")

    body = f"Exchange-correlation functional: {functional}."

    if lhf:
        body += (
            "\n\nHybrid functional with exact exchange is enabled. "
            "The total energy mixes DFT exchange with Hartree-Fock exact exchange: "
            "E_xc = alpha E_x^HF + (1-alpha) E_x^DFT + E_c^DFT."
        )
        if hfscreen is not None:
            body += f"\nHSE screening parameter: HFSCREEN = {hfscreen}."

    return NarrativeSection(title="Exchange-Correlation Functional", body=body, level="info")


def _dft_magnetic(tpl: DFTInsightTemplate) -> Optional[NarrativeSection]:
    ispin = tpl._param("ispin", 1)
    if ispin == 1:
        return None

    magmom = tpl._param("magmom")
    body = (
        "Spin-polarized calculation: ISPIN = 2.\n\n"
        "The Kohn-Sham equations are solved separately for spin-up and spin-down channels. "
        "The exchange-correlation potential depends on both densities: "
        "V_xc^sigma = delta E_xc[rho_up, rho_down] / delta rho_sigma."
    )
    if magmom:
        body += f"\nInitial magnetic moments: {magmom}."
    else:
        body += "\nNo initial MAGMOM specified; VASP will use default values."

    return NarrativeSection(title="Magnetic Configuration", body=body, level="info")


# ────────────────────────────────────────────────
# DFT Draft Sections
# ────────────────────────────────────────────────


class DFTDraftTemplate(DraftTemplate):
    domain_name = "dft"
    software_name = "DFT Software"
    basis_type = "plane-wave"
    pseudopotential_type = "PAW"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        if not self._DRAFT_SECTIONS:
            self._register_default_sections()

    def _register_default_sections(self) -> None:
        DFTDraftTemplate._DRAFT_SECTIONS = [
            _dft_draft_theory,
            _dft_draft_software,
            _dft_draft_basis,
            _dft_draft_sampling,
            _dft_draft_convergence,
            _dft_draft_relaxation,
            _dft_draft_caveats,
        ]


def _dft_draft_theory(tpl: DFTDraftTemplate) -> Optional[NarrativeSection]:
    ispin = tpl._param("ispin", 1)
    spin_text = "spin-polarized" if ispin == 2 else "spin-restricted"
    functional = tpl._param("functional", "PBE")

    body = (
        f"The electronic structure calculations were performed within the framework "
        f"of Kohn-Sham density functional theory (DFT) using a {spin_text} formalism. "
        f"Exchange-correlation effects were described by the {functional} functional. "
        "The Kohn-Sham equations were solved self-consistently via iterative diagonalization."
    )
    return NarrativeSection(title="Electronic Structure Theory", body=body)


def _dft_draft_software(tpl: DFTDraftTemplate) -> Optional[NarrativeSection]:
    body = (
        f"All calculations were carried out using {tpl.software_name}. "
        f"The {tpl.basis_type} basis set was employed with {tpl.pseudopotential_type} pseudopotentials. "
        "Projector augmented-wave (PAW) potentials were used to represent the core-valence interaction."
    )
    return NarrativeSection(title="Software and Basis Set", body=body)


def _dft_draft_basis(tpl: DFTDraftTemplate) -> Optional[NarrativeSection]:
    encut = tpl._param("encut")
    body = f"The Kohn-Sham orbitals were expanded in a {tpl.basis_type} basis set. "
    if encut:
        body += f"A kinetic energy cutoff of {encut} eV was used. "
    body += (
        "The FFT grid density was determined automatically to avoid aliasing "
        "of the charge density and augmentation charges."
    )
    return NarrativeSection(title="Basis Set Parameters", body=body)


def _dft_draft_sampling(tpl: DFTDraftTemplate) -> Optional[NarrativeSection]:
    mesh = tpl._param("kpoint_mesh", [])
    if not mesh:
        return None
    body = (
        f"Brillouin zone integration was performed using a {mesh[0]} x {mesh[1]} x {mesh[2]} "
        f"Monkhorst-Pack k-point mesh. "
    )
    return NarrativeSection(title="k-Point Sampling", body=body)


def _dft_draft_convergence(tpl: DFTDraftTemplate) -> Optional[NarrativeSection]:
    ediff = tpl._param("ediff", 1e-6)
    body = (
        f"Electronic self-consistency was deemed achieved when the energy difference "
        f"between successive iterations fell below {ediff} eV. "
        "Geometry optimizations were considered converged when all forces were below 0.01 eV/angstrom."
    )
    return NarrativeSection(title="Convergence Criteria", body=body)


def _dft_draft_relaxation(tpl: DFTDraftTemplate) -> Optional[NarrativeSection]:
    nsw = tpl._param("nsw", 0)
    if nsw <= 0:
        return None
    body = (
        f"Ionic positions were relaxed using the conjugate-gradient algorithm "
        f"with a maximum of {nsw} ionic steps. "
        "The stress tensor was minimized to obtain the equilibrium lattice parameters."
    )
    return NarrativeSection(title="Geometry Optimization", body=body)


def _dft_draft_caveats(tpl: DFTDraftTemplate) -> Optional[NarrativeSection]:
    body = (
        "Standard DFT limitations apply: the exchange-correlation functional is approximate, "
        "and van der Waals interactions are not captured by semi-local functionals. "
        "Zero-point energy and thermal effects were not included unless explicitly stated."
    )
    return NarrativeSection(title="Methodological Notes", body=body)


# ────────────────────────────────────────────────
# DFT Check Sections
# ────────────────────────────────────────────────


class DFTCheckTemplate(CheckTemplate):
    domain_name = "dft"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        if not self._CHECK_SECTIONS:
            self._register_default_sections()

    def _register_default_sections(self) -> None:
        DFTCheckTemplate._CHECK_SECTIONS = [
            _dft_check_encut,
            _dft_check_kmesh,
            _dft_check_smearing,
            _dft_check_scf,
            _dft_check_magnetic,
        ]


def _dft_check_encut(tpl: DFTCheckTemplate) -> List[NarrativeSection]:
    results = []
    encut = tpl._param("encut")
    max_enmax = tpl._param("max_enmax")

    if encut is None:
        results.append(
            NarrativeSection(
                title="ENCUT missing",
                body="ENCUT not explicitly set. VASP will use default ENMAX from POTCAR.",
                level="info",
            )
        )
    elif max_enmax and encut < max_enmax:
        results.append(
            NarrativeSection(
                title="ENCUT below POTCAR ENMAX",
                body=f"ENCUT = {encut} eV < max(ENMAX) = {max_enmax} eV.",
                level="error",
            )
        )
    elif encut < 300:
        results.append(
            NarrativeSection(
                title="ENCUT too low",
                body=f"ENCUT = {encut} eV is below 300 eV.",
                level="warning",
            )
        )
    return results


def _dft_check_kmesh(tpl: DFTCheckTemplate) -> List[NarrativeSection]:
    results = []
    mesh = tpl._param("kpoint_mesh", [])
    ismear = tpl._param("ismear", 0)
    if not mesh:
        return results
    total_k = mesh[0] * mesh[1] * mesh[2] if len(mesh) == 3 else 0
    if ismear == -5 and total_k < 8:
        results.append(
            NarrativeSection(
                title="Sparse k-mesh with tetrahedron",
                body=f"ISMEAR=-5 with only {total_k} k-points is unreliable.",
                level="warning",
            )
        )
    return results


def _dft_check_smearing(tpl: DFTCheckTemplate) -> List[NarrativeSection]:
    results = []
    ismear = tpl._param("ismear", 0)
    sigma = tpl._param("sigma", 0.2)
    if ismear in (0, -1, 1, 2) and sigma > 0.5:
        results.append(
            NarrativeSection(
                title="SIGMA too large",
                body=f"SIGMA = {sigma} eV is large for a metal.",
                level="warning",
            )
        )
    return results


def _dft_check_scf(tpl: DFTCheckTemplate) -> List[NarrativeSection]:
    results = []
    nelm = tpl._param("nelm", 60)
    ediff = tpl._param("ediff", 1e-6)
    if nelm < 10:
        results.append(
            NarrativeSection(
                title="NELM very small",
                body=f"NELM = {nelm} may be insufficient for convergence.",
                level="warning",
            )
        )
    if ediff > 1e-3:
        results.append(
            NarrativeSection(
                title="EDIFF very loose",
                body=f"EDIFF = {ediff} is loose for production calculations.",
                level="warning",
            )
        )
    return results


def _dft_check_magnetic(tpl: DFTCheckTemplate) -> List[NarrativeSection]:
    results = []
    ispin = tpl._param("ispin", 1)
    magmom = tpl._param("magmom")
    if ispin == 1 and magmom is not None:
        results.append(
            NarrativeSection(
                title="MAGMOM ignored",
                body="MAGMOM is set but ISPIN = 1.",
                level="warning",
            )
        )
    return results
