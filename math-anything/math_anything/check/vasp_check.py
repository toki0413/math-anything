"""VASP pre-flight parameter consistency checks.

Rules are derived from mathematical/physical requirements of DFT.
"""

from typing import Any, Dict, List

from ..schemas import MathSchema
from .base import CheckEngine, CheckResult


class VaspCheckEngine(CheckEngine):
    """Validate VASP input parameters before submission."""

    @property
    def engine_name(self) -> str:
        return "vasp"

    def check(self, schema: MathSchema) -> List[CheckResult]:
        raw: Dict[str, Any] = schema.raw_symbols or {}
        incar_raw = raw.get("incar", {})
        incar = self._normalize_incar(incar_raw)
        kpoints = raw.get("kpoints", {})
        mesh = kpoints.get("mesh", {})
        potcar_data = raw.get("potcar", {})

        results: List[CheckResult] = []
        results.extend(self._check_encut(incar, potcar_data))
        results.extend(self._check_smearing(incar, mesh))
        results.extend(self._check_convergence(incar))
        results.extend(self._check_relaxation(incar))
        results.extend(self._check_magnetic(incar))
        results.extend(self._check_ldau(incar))
        results.extend(self._check_hybrid(incar))
        results.extend(self._check_kpoint_density(incar, mesh))
        results.extend(self._check_algo(incar))
        results.extend(self._check_mixing(incar))
        results.extend(self._check_lreal(incar))
        results.extend(self._check_prec(incar))
        results.extend(self._check_output_control(incar))
        return results

    def _normalize_incar(self, incar_raw: Any) -> Dict[str, Any]:
        if hasattr(incar_raw, "get_value"):
            keys = list(incar_raw.parameters.keys()) if hasattr(incar_raw, "parameters") else []
            return {k: incar_raw.get_value(k) for k in keys}
        return incar_raw if isinstance(incar_raw, dict) else {}

    def _check_encut(self, incar: Dict[str, Any], potcar_data: Dict[str, Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        encut = incar.get("ENCUT")
        enmax_list = potcar_data.get("enmax_list", [])
        max_enmax = max(enmax_list) if enmax_list else None

        if encut is None:
            msg = "ENCUT not explicitly set. VASP will use default ENMAX from POTCAR."
            if max_enmax:
                msg += f" Max ENMAX in POTCAR: {max_enmax} eV."
            results.append(
                CheckResult(
                    rule="ENCUT missing",
                    severity="info",
                    message=msg,
                    suggestion="For publishable results, explicitly set ENCUT >= 1.3 * max(ENMAX).",
                )
            )
        elif max_enmax and encut < max_enmax:
            results.append(
                CheckResult(
                    rule="ENCUT below POTCAR ENMAX",
                    severity="error",
                    message=f"ENCUT = {encut} eV < max(ENMAX) = {max_enmax} eV from POTCAR.",
                    suggestion="The plane-wave basis is mathematically incomplete for at least one element. Set ENCUT >= max(ENMAX), preferably 1.3x.",  # noqa: E501
                )
            )
        elif encut < 300:
            results.append(
                CheckResult(
                    rule="ENCUT too low",
                    severity="warning",
                    message=f"ENCUT = {encut} eV is below 300 eV.",
                    suggestion="Most elements need ENCUT >= 400 eV. Low ENCUT causes Pulay stress during cell relaxations.",  # noqa: E501
                )
            )
        elif max_enmax and encut < 1.3 * max_enmax:
            results.append(
                CheckResult(
                    rule="ENCUT may be insufficient",
                    severity="info",
                    message=f"ENCUT = {encut} eV covers ENMAX ({max_enmax} eV) but margin is < 30%.",
                    suggestion="For well-converged results, ENCUT = 1.3 * max(ENMAX) is recommended to suppress Pulay stress.",  # noqa: E501
                )
            )
        return results

    def _check_smearing(self, incar: Dict[str, Any], mesh: Dict[str, Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        ismear = incar.get("ISMEAR", 0)
        sigma = incar.get("SIGMA", 0.2)
        subdiv = mesh.get("subdivisions", [0, 0, 0])
        total_k = subdiv[0] * subdiv[1] * subdiv[2] if subdiv else 0

        if ismear == -5:
            if total_k < 64:
                results.append(
                    CheckResult(
                        rule="Tetrahedron method with sparse k-mesh",
                        severity="error",
                        message=f"ISMEAR=-5 (tetrahedron) requires dense k-mesh, but only {subdiv} ({total_k} k-points) detected.",  # noqa: E501
                        suggestion="For tetrahedron method, use at least 6x6x6 k-points (or denser). For metals, switch to Gaussian (ISMEAR=0) or Methfessel-Paxton (ISMEAR=1).",  # noqa: E501
                    )
                )
            else:
                results.append(
                    CheckResult(
                        rule="Tetrahedron method validation",
                        severity="info",
                        message="ISMEAR=-5 selected. Tetrahedron assumes linear DOS interpolation between k-points.",
                        suggestion="Ensure your k-mesh is dense enough that the DOS varies slowly between adjacent k-points.",  # noqa: E501
                    )
                )

        if ismear in (0, -1, 1, 2):
            if sigma > 0.3:
                results.append(
                    CheckResult(
                        rule="SIGMA too large",
                        severity="warning",
                        message=f"SIGMA = {sigma} eV is large for {self._smear_name(ismear)}.",
                        suggestion="Entropic correction T*S may exceed 1 meV/atom. For accurate total energies, use SIGMA <= 0.2 eV and extrapolate to zero.",  # noqa: E501
                    )
                )
            elif sigma > 0.2:
                results.append(
                    CheckResult(
                        rule="SIGMA moderately large",
                        severity="info",
                        message=f"SIGMA = {sigma} eV is acceptable but not ideal for precision work.",
                        suggestion="For benchmark-quality energies, consider SIGMA <= 0.05-0.1 eV.",
                    )
                )
            elif sigma < 0.01:
                results.append(
                    CheckResult(
                        rule="SIGMA very small",
                        severity="info",
                        message=f"SIGMA = {sigma} eV approaches the zero-temperature limit.",
                        suggestion="Convergence may be slow. Ensure k-mesh density is sufficient.",
                    )
                )

        return results

    def _smear_name(self, ismear: int) -> str:
        names = {
            -5: "tetrahedron",
            -4: "tetrahedron",
            -3: "tetrahedron",
            -2: "partial occupancies",
            -1: "Fermi-Dirac",
            0: "Gaussian",
            1: "Methfessel-Paxton 1st order",
            2: "Methfessel-Paxton 2nd order",
        }
        return names.get(ismear, f"ISMEAR={ismear}")

    def _check_convergence(self, incar: Dict[str, Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        ediff = incar.get("EDIFF", 1e-4)
        nelm = incar.get("NELM", 60)

        if ediff > 1e-3:
            results.append(
                CheckResult(
                    rule="EDIFF too loose",
                    severity="warning",
                    message=f"EDIFF = {ediff} is very loose.",
                    suggestion="For publishable results, use EDIFF <= 1e-5. For accurate forces and stress, EDIFF <= 1e-6.",  # noqa: E501
                )
            )
        elif ediff > 1e-5:
            results.append(
                CheckResult(
                    rule="EDIFF moderate",
                    severity="info",
                    message=f"EDIFF = {ediff} is acceptable for geometry optimization.",
                    suggestion="For final static calculations or DOS, tighten to EDIFF <= 1e-6.",
                )
            )

        if nelm < 40:
            results.append(
                CheckResult(
                    rule="NELM too small",
                    severity="warning",
                    message=f"NELM = {nelm} may be insufficient for SCF convergence.",
                    suggestion="Difficult systems (metals, correlated materials) often need NELM >= 60-100.",
                )
            )

        return results

    def _check_relaxation(self, incar: Dict[str, Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        nsw = incar.get("NSW", 0)
        ibrion = incar.get("IBRION", -1)
        isif = incar.get("ISIF", 2)

        if nsw > 0 and ibrion in (-1, None):
            results.append(
                CheckResult(
                    rule="Relaxation disabled despite NSW > 0",
                    severity="error",
                    message=f"NSW = {nsw} requests ionic steps, but IBRION = {ibrion} disables relaxation.",
                    suggestion="Set IBRION = 2 (conjugate gradient) or 1 (quasi-Newton) to perform geometry optimization.",  # noqa: E501
                )
            )

        if nsw > 0 and ibrion == 5:
            results.append(
                CheckResult(
                    rule="IBRION=5 with finite NSW",
                    severity="warning",
                    message="IBRION=5 (finite differences) is for phonon calculations, not relaxation.",
                    suggestion="For structure relaxation, use IBRION=2. For phonons, set NSW=1 and use appropriate displacement.",  # noqa: E501
                )
            )

        if isif in (3, 4, 5, 6, 7) and nsw == 0:
            results.append(
                CheckResult(
                    rule="Cell relaxation without ionic steps",
                    severity="warning",
                    message=f"ISIF = {isif} requests cell/position relaxation, but NSW = 0.",
                    suggestion="Set NSW >= 20 to allow the optimizer to converge.",
                )
            )

        return results

    def _check_magnetic(self, incar: Dict[str, Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        ispin = incar.get("ISPIN", 1)
        magmom = incar.get("MAGMOM")

        if ispin == 1 and magmom is not None:
            results.append(
                CheckResult(
                    rule="MAGMOM ignored in spin-restricted calculation",
                    severity="warning",
                    message="MAGMOM is set but ISPIN = 1 (spin-restricted).",
                    suggestion="Set ISPIN = 2 to enable spin-polarized calculations and use the provided MAGMOM.",
                )
            )

        if ispin == 2 and magmom is None:
            results.append(
                CheckResult(
                    rule="No initial magnetic moments",
                    severity="info",
                    message="ISPIN = 2 but MAGMOM not explicitly set.",
                    suggestion="VASP will use default moments. For reliable magnetic ground states, explicitly set MAGMOM per atom type.",  # noqa: E501
                )
            )

        return results

    def _check_ldau(self, incar: Dict[str, Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        ldau = incar.get("LDAU", False)
        if not ldau:
            return results

        required = ["LDAUTYPE", "LDAUL", "LDAUU"]
        missing = [r for r in required if incar.get(r) is None]
        if missing:
            results.append(
                CheckResult(
                    rule="Incomplete DFT+U setup",
                    severity="error",
                    message=f"LDAU = .TRUE. but missing parameters: {', '.join(missing)}.",
                    suggestion="DFT+U requires LDAUTYPE, LDAUL, and LDAUU to be fully specified.",
                )
            )
        else:
            results.append(
                CheckResult(
                    rule="DFT+U enabled",
                    severity="info",
                    message="DFT+U parameters are fully specified.",
                    suggestion="Validate U values against experimental band gaps or magnetic moments. U is not transferable across compounds.",  # noqa: E501
                )
            )

        return results

    def _check_hybrid(self, incar: Dict[str, Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        lhf = incar.get("LHFCALC", False)
        if not lhf:
            return results

        encut = incar.get("ENCUT", 0)
        if encut < 400:
            results.append(
                CheckResult(
                    rule="Low ENCUT with hybrid functional",
                    severity="warning",
                    message=f"Hybrid functional (LHFCALC) with ENCUT = {encut} eV may be insufficient.",
                    suggestion="Hybrid functionals are sensitive to basis set completeness. Use ENCUT >= 400 eV, preferably >= 520 eV.",  # noqa: E501
                )
            )

        hfscreen = incar.get("HFSCREEN")
        if hfscreen is not None and hfscreen not in (0.0, 0.2, 0.3):
            results.append(
                CheckResult(
                    rule="Unusual HFSCREEN",
                    severity="info",
                    message=f"HFSCREEN = {hfscreen} is non-standard.",
                    suggestion="Standard values: 0.2 (HSE06), 0.0 (PBE0), 0.3 (HSE03). Ensure this is intentional.",
                )
            )

        return results

    def _check_kpoint_density(self, incar: Dict[str, Any], mesh: Dict[str, Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        subdiv = mesh.get("subdivisions", [0, 0, 0])
        if not subdiv or subdiv == [0, 0, 0]:
            return results

        total_k = subdiv[0] * subdiv[1] * subdiv[2]
        ismear = incar.get("ISMEAR", 0)

        # Metal-like smearing
        if ismear in (0, -1, 1, 2):
            if total_k < 64:
                results.append(
                    CheckResult(
                        rule="Sparse k-mesh for metal",
                        severity="warning",
                        message=f"k-mesh {subdiv} ({total_k} points) may be too sparse for metallic systems.",
                        suggestion="For metals, use at least 6x6x6 (216 points) or denser to resolve Fermi surface.",
                    )
                )

        return results

    def _check_algo(self, incar: Dict[str, Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        algo = str(incar.get("ALGO", "Normal")).upper()
        ismear = incar.get("ISMEAR", 0)
        incar.get("NELM", 60)

        # Fast/RMM is default but can diverge for difficult systems
        if algo in ("FAST", "VERY_FAST") and ismear in (1, 2):
            results.append(
                CheckResult(
                    rule="ALGO=Fast with Methfessel-Paxton",
                    severity="info",
                    message="RMM-DIIS (Fast) with Methfessel-Paxton smearing is efficient but can diverge.",
                    suggestion="If SCF fails to converge, switch to ALGO = Normal or Damped.",
                )
            )

        if algo == "EIGENVAL":
            results.append(
                CheckResult(
                    rule="ALGO=Eigenval is expensive",
                    severity="info",
                    message="Exact diagonalization (Eigenval) scales as O(N^3) and is very slow for large systems.",
                    suggestion="Only use Eigenval for very small systems (< 10 atoms) or when exact eigenvalues are required.",  # noqa: E501
                )
            )

        if algo == "ALL":
            results.append(
                CheckResult(
                    rule="ALGO=All uses all bands",
                    severity="info",
                    message="ALGO=All calculates all bands (not just occupied), increasing cost by ~2x.",
                    suggestion="Use for GW/BSE prep or when unoccupied states are needed. Not needed for standard DFT.",
                )
            )

        if algo == "DAMPED":
            results.append(
                CheckResult(
                    rule="ALGO=Damped selected",
                    severity="info",
                    message="Damped algorithm is robust but slower than RMM-DIIS.",
                    suggestion="Good choice for difficult systems (metals, strongly correlated). If converged quickly, try ALGO=Fast for speedup.",  # noqa: E501
                )
            )

        return results

    def _check_mixing(self, incar: Dict[str, Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        amix = incar.get("AMIX")
        bmix = incar.get("BMIX")
        ismear = incar.get("ISMEAR", 0)

        if amix is not None and amix > 0.6:
            results.append(
                CheckResult(
                    rule="AMIX very large",
                    severity="warning",
                    message=f"AMIX = {amix} is aggressive. Pulay mixing may oscillate.",
                    suggestion="For stable SCF, AMIX <= 0.4 is typical. For metals, AMIX = 0.2 is safer. Reduce if SCF diverges.",  # noqa: E501
                )
            )

        if bmix is not None and bmix < 0.5:
            results.append(
                CheckResult(
                    rule="BMIX very small",
                    severity="warning",
                    message=f"BMIX = {bmix} is very small.",
                    suggestion="BMIX controls Kerker preconditioning cutoff. Typical: 1.0-1.5. Very small BMIX suppresses charge mixing at all wavelengths, slowing convergence.",  # noqa: E501
                )
            )

        if ismear in (0, -1, 1, 2) and amix is None:
            results.append(
                CheckResult(
                    rule="Default AMIX for metal",
                    severity="info",
                    message="Using default AMIX = 0.4 for a metallic system.",
                    suggestion="Default AMIX is usually fine. If SCF oscillates, reduce to AMIX = 0.2 and increase BMIX to 1.0-1.5.",  # noqa: E501
                )
            )

        return results

    def _check_lreal(self, incar: Dict[str, Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        lreal = str(incar.get("LREAL", ".FALSE.")).upper()
        incar.get("NIONS", 0)  # Not available from INCAR alone, but we can estimate from POSCAR if needed

        # Simplified: if LREAL is set to Auto/On without knowing system size
        if lreal in ("AUTO", ".TRUE.", "T", "ON"):
            results.append(
                CheckResult(
                    rule="LREAL approximation enabled",
                    severity="info",
                    message="LREAL = Auto/On uses real-space projection for PAW operators.",
                    suggestion="Speeds up large systems (> 50 atoms) but can introduce small errors. For high-precision results or small systems, use LREAL = .FALSE.",  # noqa: E501
                )
            )
        return results

    def _check_prec(self, incar: Dict[str, Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        prec = str(incar.get("PREC", "Normal")).upper()

        if prec in ("LOW", "MEDIUM"):
            results.append(
                CheckResult(
                    rule="PREC too low",
                    severity="warning",
                    message=f"PREC = {prec} uses coarse FFT grids and approximate charge projection.",
                    suggestion="For publishable results, use PREC = Accurate. Low/Medium is only for quick testing.",
                )
            )
        elif prec == "NORMAL":
            results.append(
                CheckResult(
                    rule="PREC = Normal",
                    severity="info",
                    message="PREC = Normal is acceptable for most geometry optimizations.",
                    suggestion="For final static calculations, DOS, or accurate forces, use PREC = Accurate.",
                )
            )
        elif prec == "SINGLE":
            results.append(
                CheckResult(
                    rule="PREC = Single (reduced precision)",
                    severity="warning",
                    message="PREC = Single uses single-precision FFTs and reduces accuracy.",
                    suggestion="Only use for very large systems where memory is limiting. Results may not be reproducible across machines.",  # noqa: E501
                )
            )

        return results

    def _check_output_control(self, incar: Dict[str, Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        lwave = str(incar.get("LWAVE", ".TRUE.")).upper()
        lcharg = str(incar.get("LCHARG", ".TRUE.")).upper()
        nsw = incar.get("NSW", 0)

        wave_off = lwave in (".FALSE.", "F", "FALSE", "N", "NO")
        charg_off = lcharg in (".FALSE.", "F", "FALSE", "N", "NO")

        if wave_off and charg_off:
            results.append(
                CheckResult(
                    rule="Both WAVECAR and CHGCAR disabled",
                    severity="info",
                    message="LWAVE = .FALSE. and LCHARG = .FALSE.: no wavefunction or charge density files written.",
                    suggestion="If you plan follow-up calculations (DOS, band structure, dielectric), you will need to re-run the SCF. Consider keeping at least LCHARG = .TRUE.",  # noqa: E501
                )
            )

        if nsw > 0 and not charg_off:
            results.append(
                CheckResult(
                    rule="CHGCAR output during relaxation",
                    severity="info",
                    message="LCHARG = .TRUE. during ionic relaxation writes CHGCAR for each step.",
                    suggestion="Only the final CHGCAR is typically useful. For large systems, writing every step creates large files.",  # noqa: E501
                )
            )

        return results
