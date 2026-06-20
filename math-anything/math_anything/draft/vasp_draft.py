"""VASP methodology draft generator.

Produces peer-review-ready method sections for DFT papers.
"""

from typing import Any, Dict

from ..schemas import MathSchema
from .base import DraftEngine


class VaspDraftEngine(DraftEngine):
    """Generate publication methodology for VASP DFT calculations."""

    @property
    def engine_name(self) -> str:
        return "vasp"

    def generate(self, schema: MathSchema, fmt: str = "markdown") -> str:
        raw: Dict[str, Any] = schema.raw_symbols or {}
        incar_raw = raw.get("incar", {})
        incar = self._normalize_incar(incar_raw)
        kpoints = raw.get("kpoints", {})
        raw.get("structure", {})

        lines: list[str] = []
        if fmt == "markdown":
            lines.append("# Computational Details")
        else:
            lines.append("\\section{Computational Details}")
        lines.append("")

        lines.append(self._theory(incar, fmt))
        lines.append(self._software(incar, fmt))
        lines.append(self._basis(incar, fmt))
        lines.append(self._sampling(kpoints, incar, fmt))
        lines.append(self._convergence(incar, fmt))
        if incar.get("NSW", 0) > 0:
            lines.append(self._relaxation(incar, fmt))
        lines.append(self._caveats(incar, fmt))

        return "\n".join(lines)

    def _normalize_incar(self, incar_raw: Any) -> Dict[str, Any]:
        if hasattr(incar_raw, "get_value"):
            keys = list(incar_raw.parameters.keys()) if hasattr(incar_raw, "parameters") else []
            return {k: incar_raw.get_value(k) for k in keys}
        return incar_raw if isinstance(incar_raw, dict) else {}

    def _theory(self, incar: Dict[str, Any], fmt: str) -> str:
        ispin = incar.get("ISPIN", 1)
        spin_text = "spin-polarized" if ispin == 2 else "spin-restricted"

        xc = self._identify_xc(incar)
        has_ldau = incar.get("LDAU", False)
        is_magnetic = incar.get("ISPIN", 1) == 2

        body = (
            "The electronic structure calculations were performed within the framework "
            "of Kohn-Sham density functional theory (DFT). "
            f"The {spin_text} Kohn-Sham equations were solved self-consistently:\n\n"
        )
        if fmt == "latex":
            body += (
                "\\begin{equation}\n"
                "\\left[-\\frac{\\hbar^2}{2m}\\nabla^2 + V_{\\rm eff}[n](\\mathbf{r})\\right] "
                "\\psi_i(\\mathbf{r}) = \\varepsilon_i \\psi_i(\\mathbf{r})"
                "\\end{equation}\n\n"
            )
        else:
            body += (
                "$$[-\\hbar^2 \\nabla^2 / 2m + V_{\\rm eff}[n](\\mathbf{r})] "
                "\\psi_i(\\mathbf{r}) = \\varepsilon_i \\psi_i(\\mathbf{r})$$\n\n"
            )
        body += (
            "where $V_{\\rm eff}[n]$ is the effective potential depending on the "
            "electron density $n(\\mathbf{r}) = \\sum_i f_i |\\psi_i(\\mathbf{r})|^2$. "
            f"The exchange-correlation energy was described using the {xc} approximation. "
            "Projector augmented-wave (PAW) pseudopotentials were employed to treat "
            "the core-valence electron interactions."
        )
        if has_ldau:
            body += (
                " The DFT+U method was applied to account for strong on-site Coulomb "
                "interactions in localized orbitals, using the Dudarev implementation "
                "where only the effective U_eff = U - J parameter enters the Hamiltonian."
            )
        if is_magnetic:
            body += (
                " Collinear spin-polarized calculations were performed with initial "
                "magnetic moments specified to capture the magnetic ground state."
            )
        return self._section("Electronic Structure Theory", body, fmt)

    def _identify_xc(self, incar: Dict[str, Any]) -> str:
        if incar.get("LHFCALC"):
            hfscreen = incar.get("HFSCREEN")
            if hfscreen == 0.2:
                return "HSE06 hybrid functional (screened exchange, omega=0.2 A^-1)"
            elif hfscreen == 0.0:
                return "PBE0 hybrid functional (unscreened exchange)"
            elif hfscreen is not None:
                return f"screened hybrid functional (HFSCREEN={hfscreen})"
            return "HSE06 hybrid functional"
        if incar.get("GGA"):
            gga = str(incar.get("GGA")).upper()
            mapping = {
                "PE": "PBE",
                "91": "PW91",
                "RP": "rPBE",
                "PS": "PBEsol",
            }
            return mapping.get(gga, gga)
        if incar.get("METAGGA"):
            return f"meta-GGA ({incar.get('METAGGA')})"
        return "LDA (local density approximation)"

    def _software(self, incar: Dict[str, Any], fmt: str) -> str:
        body = "All calculations were carried out using the Vienna Ab initio Simulation Package (VASP)."
        return self._section("Software", body, fmt)

    def _basis(self, incar: Dict[str, Any], fmt: str) -> str:
        encut = incar.get("ENCUT")
        body = "The Kohn-Sham orbitals were expanded in a plane-wave basis set. "
        if encut:
            body += f"A kinetic energy cutoff of {encut} eV was used for the plane-wave expansion. "
        else:
            body += "The default kinetic energy cutoff was employed. "
        body += (
            "This corresponds to a Galerkin projection onto the subspace of plane waves "
            "with $|\\mathbf{G} + \\mathbf{k}|^2 < 2m E_{\\rm cut} / \\hbar^2$."
        )
        return self._section("Basis Set", body, fmt)

    def _sampling(self, kpoints: Dict[str, Any], incar: Dict[str, Any], fmt: str) -> str:
        mesh = kpoints.get("mesh", {})
        subdiv = mesh.get("subdivisions")
        mode = mesh.get("mode", " Monkhorst-Pack")

        body = "Brillouin zone integration was performed using "
        if subdiv:
            body += f"a {subdiv[0]}×{subdiv[1]}×{subdiv[2]} "
            if "gamma" in str(mode).lower():
                body += "Γ-centered "
            body += "Monkhorst-Pack k-point mesh. "
        else:
            body += "an automatic k-point mesh. "

        ismear = incar.get("ISMEAR", 0)
        sigma = incar.get("SIGMA", 0.2)
        if ismear == -5:
            body += (
                "The total energy was computed using the tetrahedron method with "
                "Blochl corrections for accurate Brillouin zone integration."
            )
        elif ismear in (0, -1):
            smear_name = "Gaussian" if ismear == 0 else "Fermi-Dirac"
            body += (
                f"A {smear_name} smearing of width {sigma} eV was applied to the "
                f"occupation numbers near the Fermi level to facilitate convergence."
            )
        elif ismear in (1, 2):
            body += (
                f"Methfessel-Paxton smearing of order {ismear} with width {sigma} eV "
                f"was used for the occupation numbers."
            )

        return self._section("k-Point Sampling", body, fmt)

    def _convergence(self, incar: Dict[str, Any], fmt: str) -> str:
        ediff = incar.get("EDIFF", 1e-4)
        nelm = incar.get("NELM", 60)
        body = (
            f"Self-consistency was achieved when the energy difference between consecutive "
            f"iterations fell below {ediff}. "
            f"The maximum number of self-consistent field iterations was set to {nelm}. "
            "Note that the convergence criterion applies to the charge density residual, "
            "not directly to the total energy eigenvalues."
        )
        return self._section("Convergence Criteria", body, fmt)

    def _relaxation(self, incar: Dict[str, Any], fmt: str) -> str:
        nsw = incar.get("NSW", 0)
        ibrion = incar.get("IBRION", 2)
        isif = incar.get("ISIF", 2)
        ediffg = incar.get("EDIFFG")

        algo_map = {
            1: "quasi-Newton (RMM-DIIS)",
            2: "conjugate gradient",
            3: "damped molecular dynamics",
        }
        algo = algo_map.get(ibrion, f"IBRION={ibrion}")

        relax_type = "atomic positions"
        if isif in (3, 7):
            relax_type = "atomic positions and cell volume"
        elif isif in (4, 5, 6):
            relax_type = "atomic positions, cell shape, and volume"

        body = (
            f"Geometry optimization was performed using the {algo} algorithm. "
            f"A maximum of {nsw} ionic steps were allowed, with relaxation of {relax_type}. "
        )
        if ediffg:
            if ediffg < 0:
                body += f"Optimization was considered converged when the energy change fell below {abs(ediffg)} eV."
            else:
                body += f"Optimization was considered converged when all forces fell below {ediffg} eV/Å."
        return self._section("Geometry Optimization", body, fmt)

    def _caveats(self, incar: Dict[str, Any], fmt: str) -> str:
        notes: list[str] = []
        ismear = incar.get("ISMEAR", 0)
        sigma = incar.get("SIGMA", 0.2)
        if ismear in (0, -1, 1, 2) and sigma > 0.05:
            notes.append(
                f"Finite smearing width ({sigma} eV) introduces an entropic contribution to "
                f"the total energy. For accurate energy comparisons, extrapolation to zero smearing is recommended."
            )
        if incar.get("ISPIN", 1) == 1:
            notes.append(
                "Spin-restricted calculations may not capture magnetic ground states. "
                "For systems with known magnetic ordering, spin-polarized calculations are advised."
            )
        if incar.get("LDAU", False):
            notes.append(
                "DFT+U parameters should be validated against experimental observables "
                "(e.g., band gap, magnetic moments) or benchmarked with hybrid functional calculations. "
                "The choice of U is not transferable across different chemical environments."
            )
        if not notes:
            return ""
        body = " ".join(notes)
        return self._section("Methodological Notes", body, fmt)
