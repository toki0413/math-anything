"""Quantum ESPRESSO check engine.

Leverages DFT domain template; adds QE-specific checks.
"""

from typing import Any, Dict, List

from ..schemas import MathSchema
from ..templates import DFTCheckTemplate
from .base import CheckEngine, CheckResult


class QuantumEspressoCheckEngine(CheckEngine):
    """Validate Quantum ESPRESSO input parameters."""

    @property
    def engine_name(self) -> str:
        return "quantum_espresso"

    def check(self, schema: MathSchema) -> List[CheckResult]:
        params = self._extract_params(schema)

        # Base DFT checks
        tpl = DFTCheckTemplate(params)
        results = tpl.to_check_results()

        # QE-specific checks
        results.extend(self._check_qe_specific(params))
        return results

    def _check_qe_specific(self, params: Dict[str, Any]) -> List[CheckResult]:
        results = []

        ecutwfc = params.get("ecutwfc")
        ecutrho = params.get("ecutrho")
        ibrav = params.get("ibrav", 0)
        nat = params.get("nat", 0)
        ntyp = params.get("ntyp", 0)
        calculation = params.get("calculation", "scf")
        diagonalization = params.get("diagonalization", "david")
        species = params.get("species", [])

        # ecutwfc check
        if ecutwfc is None:
            results.append(
                CheckResult(
                    rule="ecutwfc missing",
                    severity="error",
                    message="ecutwfc is not defined in &SYSTEM.",
                    suggestion="Set ecutwfc (Ry) based on pseudopotential suggestion. Typical: 30-60 Ry for PBE.",
                )
            )
        elif ecutwfc < 10:
            results.append(
                CheckResult(
                    rule="ecutwfc very low",
                    severity="error",
                    message=f"ecutwfc = {ecutwfc} Ry is extremely low.",
                    suggestion="Most pseudopotentials require ecutwfc >= 30 Ry for converged results.",
                )
            )
        elif ecutwfc < 25:
            results.append(
                CheckResult(
                    rule="ecutwfc low",
                    severity="warning",
                    message=f"ecutwfc = {ecutwfc} Ry may be insufficient.",
                    suggestion="Check convergence with respect to ecutwfc. Typical minimum: 30 Ry.",
                )
            )

        # ecutrho check
        if ecutrho is None and ecutwfc:
            results.append(
                CheckResult(
                    rule="ecutrho missing",
                    severity="info",
                    message="ecutrho not explicitly set; QE will use default (4 * ecutwfc).",
                    suggestion="For PAW or USPP, explicitly set ecutrho = 8-12 * ecutwfc for accurate charge density.",
                )
            )
        elif ecutrho and ecutwfc and ecutrho < 4 * ecutwfc:
            results.append(
                CheckResult(
                    rule="ecutrho too low",
                    severity="warning",
                    message=f"ecutrho = {ecutrho} Ry < 4 * ecutwfc = {4 * ecutwfc} Ry.",
                    suggestion="Increase ecutrho to avoid charge density aliasing. Minimum: 4 * ecutwfc. Recommended: 8-12 * ecutwfc for PAW/USPP.",  # noqa: E501
                )
            )

        # System definition
        if nat == 0:
            results.append(
                CheckResult(
                    rule="No atoms defined",
                    severity="error",
                    message="nat = 0 in &SYSTEM.",
                    suggestion="Set nat to the number of atoms in the unit cell.",
                )
            )
        if ntyp == 0:
            results.append(
                CheckResult(
                    rule="No species defined",
                    severity="error",
                    message="ntyp = 0 in &SYSTEM.",
                    suggestion="Set ntyp to the number of atomic species.",
                )
            )

        if len(species) != ntyp:
            results.append(
                CheckResult(
                    rule="Species mismatch",
                    severity="warning",
                    message=f"ntyp = {ntyp} but {len(species)} species found in ATOMIC_SPECIES.",
                    suggestion="Ensure ntyp matches the number of lines in ATOMIC_SPECIES.",
                )
            )

        # Lattice definition
        if ibrav == 0:
            results.append(
                CheckResult(
                    rule="ibrav = 0",
                    severity="info",
                    message="ibrav = 0 requires explicit CELL_PARAMETERS.",
                    suggestion="Ensure CELL_PARAMETERS is provided in the input file.",
                )
            )

        # Calculation type consistency
        if calculation in ("relax", "vc-relax", "md"):
            nstep = params.get("nstep", 1)
            if nstep <= 1:
                results.append(
                    CheckResult(
                        rule="Short relaxation",
                        severity="warning",
                        message=f"{calculation} with nstep = {nstep}.",
                        suggestion="For structure relaxation, use nstep >= 50 to allow convergence.",
                    )
                )

        # Diagonalization
        if diagonalization == "cg" and nat > 500:
            results.append(
                CheckResult(
                    rule="CG diagonalization for large system",
                    severity="info",
                    message="Conjugate-gradient diagonalization is selected for a large system.",
                    suggestion="CG is robust but slower than Davidson. For large systems, consider 'david' with adequate RAM.",  # noqa: E501
                )
            )

        # K-points
        kpoint_mesh = params.get("kpoint_mesh", [])
        if not kpoint_mesh:
            kpoint_mode = params.get("kpoint_mode", "unknown")
            if kpoint_mode == "gamma":
                results.append(
                    CheckResult(
                        rule="Gamma-only sampling",
                        severity="info",
                        message="Only Gamma point is used for k-point sampling.",
                        suggestion="Gamma-only is fine for large supercells or molecules. For small unit cells, use a denser mesh.",  # noqa: E501
                    )
                )
            elif kpoint_mode == "crystal":
                results.append(
                    CheckResult(
                        rule="Explicit k-points",
                        severity="info",
                        message="Explicit k-point list is used.",
                        suggestion="Ensure the k-point path covers high-symmetry points for band structure calculations.",  # noqa: E501
                    )
                )
            else:
                results.append(
                    CheckResult(
                        rule="No k-point mesh",
                        severity="error",
                        message="No K_POINTS card found or mesh could not be parsed.",
                        suggestion="Add K_POINTS automatic with a Monkhorst-Pack mesh.",
                    )
                )
        else:
            total_k = kpoint_mesh[0] * kpoint_mesh[1] * kpoint_mesh[2] if len(kpoint_mesh) == 3 else 0
            if total_k < 8 and nat <= 10:
                results.append(
                    CheckResult(
                        rule="Sparse k-mesh",
                        severity="warning",
                        message=f"k-mesh {kpoint_mesh} ({total_k} points) may be too sparse.",
                        suggestion="For small unit cells, use at least 4x4x4 (64 points) for converged total energies.",
                    )
                )

        return results

    def _extract_params(self, schema: MathSchema) -> Dict[str, Any]:
        raw = schema.raw_symbols or {}
        params = dict(raw)
        ecutwfc_ry = raw.get("ecutwfc")
        params["encut"] = ecutwfc_ry * 13.6057 if ecutwfc_ry else None  # type: ignore[operator]
        params["ediff"] = raw.get("conv_thr", 1e-6)
        params["nelm"] = raw.get("electron_maxstep", 100)
        params["algo"] = raw.get("mixing_mode", "plain")
        params["ismear"] = self._map_smearing(raw.get("smearing", ""), raw.get("occupations", "fixed"))  # type: ignore[arg-type]
        params["sigma"] = raw.get("degauss", 0.0)
        params["ispin"] = raw.get("nspin", 1)
        params["functional"] = raw.get("functional", "PBE")
        params["nsw"] = raw.get("nstep", 1) if raw.get("calculation") in ("relax", "vc-relax", "md") else 0
        params["kpoint_mesh"] = raw.get("kpoint_mesh", [])
        return params

    def _map_smearing(self, smearing: str, occupations: str) -> int:
        if occupations == "fixed":
            return -5 if not smearing else 0
        smear_lower = smearing.lower()
        if "gauss" in smear_lower or "gaussian" in smear_lower:
            return 0
        elif "m-p" in smear_lower or "mp" in smear_lower or "methfessel" in smear_lower:
            return 1
        elif "marzari" in smear_lower or "cold" in smear_lower or "mv" in smear_lower:
            return -1
        elif "fermi" in smear_lower:
            return -1
        return 0
