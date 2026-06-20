"""VASP-specific mathematical insight generation.

Connects VASP input parameters to their underlying mathematical
structures: variational principles, Galerkin projections, iterative
solvers, and numerical quadrature.
"""

from typing import Any, Dict, List, Optional

from ..schemas import MathSchema
from .base import InsightBlock, InsightEngine


class VaspInsightEngine(InsightEngine):
    """Generate mathematical insights for VASP DFT calculations."""

    @property
    def engine_name(self) -> str:
        return "vasp"

    def generate(self, schema: MathSchema) -> List[InsightBlock]:
        blocks: List[InsightBlock] = []
        raw: Dict[str, Any] = schema.raw_symbols or {}
        incar_raw = raw.get("incar", {})
        # Normalize IncarResult to plain dict of values
        if hasattr(incar_raw, "get_value"):
            incar = {
                name: incar_raw.get_value(name)
                for name in (incar_raw.parameters.keys() if hasattr(incar_raw, "parameters") else [])
            }
        else:
            incar = incar_raw

        # ── Mathematical problem overview ──
        blocks.append(self._problem_overview(incar))

        # ── Key parameter insights ──
        blocks.extend(self._parameter_insights(incar))

        # ── Convergence analysis ──
        blocks.append(self._convergence_analysis(incar))

        # ── Warnings ──
        warning = self._consistency_warnings(incar)
        if warning:
            blocks.append(warning)

        # ── Sampling quality ──
        blocks.append(self._sampling_insight(raw))

        return blocks

    def _problem_overview(self, incar: Dict[str, Any]) -> InsightBlock:
        ispin = incar.get("ISPIN", 1)
        spin_text = "spin-polarized" if ispin == 2 else "spin-restricted"

        content = (
            "You are solving the Kohn-Sham density functional theory equations:\n"
            "  [-hbar^2 * nabla^2 / 2m + V_eff[n](r)] psi_i(r) = epsilon_i psi_i(r)\n\n"
            "Mathematically, this is a nonlinear eigenvalue problem in a periodic\n"
            "domain (Bloch boundary conditions). The nonlinearity arises because\n"
            "the effective potential V_eff depends on the electron density n(r),\n"
            "which itself is constructed from the eigenfunctions psi_i.\n\n"
            f"Your calculation is {spin_text}. "
        )
        if ispin == 2:
            content += (
                "This doubles the number of Kohn-Sham orbitals (spin-up and spin-down),\n"
                "turning the scalar eigenvalue problem into a coupled 2-component system."
            )
        else:
            content += (
                "This assumes paired electrons and a single scalar potential.\n"
                "For magnetic systems or open-shell molecules, consider ISPIN=2."
            )

        return InsightBlock(
            title="Mathematical Problem",
            content=content,
            level="math",
            params=["ISPIN"],
        )

    def _parameter_insights(self, incar: Dict[str, Any]) -> List[InsightBlock]:
        blocks: List[InsightBlock] = []

        # ENCUT
        encut = incar.get("ENCUT")
        if encut is not None:
            blocks.append(
                InsightBlock(
                    title="ENCUT - Basis Set Truncation",
                    content=(
                        f"ENCUT = {encut} eV\n\n"
                        "Mathematically, this is the truncation energy for the\n"
                        "plane-wave Galerkin projection of the Kohn-Sham orbitals.\n"
                        "The basis set is {G : |k+G|² < 2m·ENCUT/ℏ²}, so ENCUT controls\n"
                        "the resolution of spatial variations in the wavefunctions.\n\n"
                        f"At {encut} eV, the shortest wavelength resolved is\n"
                        f"approximately λ_min ≈ 2π/√(2m·{encut}/ℏ²) ≈ {self._wavelength_estimate(encut)} Å.\n\n"
                        "Consequence: ENCUT must exceed the maximum ENMAX of your\n"
                        "pseudopotentials (check POTCAR), otherwise the basis is\n"
                        "mathematically incomplete for those elements."
                    ),
                    level="info",
                    params=["ENCUT"],
                )
            )

        # ISMEAR + SIGMA
        ismear = incar.get("ISMEAR", 0)
        sigma = incar.get("SIGMA", 0.2)
        smear_names = {
            -5: ("tetrahedron with Blöchl correction", "insulator"),
            -4: ("tetrahedron", "insulator"),
            -3: ("tetrahedron (VASP-specific)", "insulator"),
            -2: ("partial occupancies from file", "special"),
            -1: ("Fermi-Dirac smearing", "metal"),
            0: ("Gaussian smearing", "metal or semiconductor"),
            1: ("Methfessel-Paxton 1st order", "metal"),
            2: ("Methfessel-Paxton 2nd order", "metal"),
        }
        smear_name, typical_use = smear_names.get(ismear, ("unknown", "unknown"))

        blocks.append(
            InsightBlock(
                title="ISMEAR + SIGMA - Brillouin Zone Quadrature",
                content=(
                    f"ISMEAR = {ismear} ({smear_name}), SIGMA = {sigma} eV\n\n"
                    "Mathematically, the total energy requires an integral over\n"
                    "the Brillouin zone: E = Σ_k w_k ∫ f(ε) ε g(ε) dε.\n\n"
                    f"You are using '{smear_name}' to approximate the\n"
                    "Dirac delta at the Fermi level.\n\n"
                    f"SIGMA = {sigma} eV controls the width of the smearing function.\n"
                    "Physical meaning: it replaces the step function occupation\n"
                    "with a smooth crossover of width ~{sigma} eV around E_F.\n\n"
                    "Consequence:\n"
                    f"  • Small SIGMA (< 0.05 eV): closer to true 0 K ground state,\n"
                    f"    but k-point convergence is slower.\n"
                    f"  • Large SIGMA (> 0.2 eV): faster k-convergence, but the\n"
                    f"    total energy is not the true DFT energy - it includes\n"
                    f"    an entropic contribution T·S.\n\n"
                    f"Typical use: {typical_use}."
                ),
                level="info",
                params=["ISMEAR", "SIGMA"],
            )
        )

        return blocks

    def _convergence_analysis(self, incar: Dict[str, Any]) -> InsightBlock:
        ediff = incar.get("EDIFF", 1e-4)
        nelm = incar.get("NELM", 60)

        content = (
            f"EDIFF = {ediff}, NELM = {nelm}\n\n"
            "Mathematically, EDIFF defines the stopping criterion for the\n"
            "self-consistent field (SCF) iteration, which is a fixed-point\n"
            "iteration on the density: n_{j+1} = F[n_j].\n\n"
            f"Your criterion: ||n_{'{j+1}'} - n_{'{j}'}|| < {ediff} (in some metric).\n\n"
            "Important distinction:\n"
            "  • EDIFF controls density (charge) convergence, NOT energy convergence.\n"
            "  • The eigenvalues ε_i may still shift after density is 'converged'.\n"
            "  • For accurate forces, you typically need EDIFF ≤ 1e-6.\n\n"
            f"NELM = {nelm} sets the maximum number of fixed-point iterations.\n"
        )
        if nelm < 60:
            content += (
                f"⚠  Warning: NELM={nelm} is quite low. Difficult systems (metals,\n"
                f"   strongly correlated materials) often need 80-100 SCF steps.\n"
                f"   If SCF does not converge within {nelm} steps, VASP aborts."
            )
        elif nelm > 200:
            content += (
                f"Note: NELM={nelm} is generous. This is fine but wastes walltime\n"
                f"if the iteration is diverging - you will wait {nelm} steps before\n"
                f"VASP gives up."
            )
        else:
            content += f"NELM={nelm} is a standard value for most systems."

        return InsightBlock(
            title="SCF Convergence - Fixed-Point Iteration",
            content=content,
            level="info",
            params=["EDIFF", "NELM"],
        )

    def _consistency_warnings(self, incar: Dict[str, Any]) -> Optional[InsightBlock]:
        warnings: List[str] = []
        ismear = incar.get("ISMEAR", 0)
        sigma = incar.get("SIGMA", 0.2)
        encut = incar.get("ENCUT")
        nsw = incar.get("NSW", 0)
        ibrion = incar.get("IBRION", -1)

        # Metal vs insulator smearing mismatch
        if ismear == -5 and sigma > 0.1:
            warnings.append(
                "ISMEAR=-5 (tetrahedron) with large SIGMA is inconsistent.\n"
                "Tetrahedron method assumes a discrete k-mesh and does not use\n"
                "SIGMA for broadening. SIGMA is ignored, but its presence suggests\n"
                "confusion about the method."
            )

        if ismear == -5:
            warnings.append(
                "ISMEAR=-5 (tetrahedron with Blöchl correction) requires\n"
                "a dense k-mesh (> 4×4×4) to be accurate. For metals or small\n"
                "meshes, use Gaussian (0) or Methfessel-Paxton (1, 2) instead.\n"
                "Mathematical reason: tetrahedron integrates the DOS by linear\n"
                "interpolation between k-points. For metallic DOS (discontinuous\n"
                "at E_F), linear interpolation is only accurate with fine grids."
            )

        if sigma > 0.3 and ismear in (0, -1):
            warnings.append(
                f"SIGMA = {sigma} eV is quite large for Gaussian/Fermi smearing.\n"
                f"The entropic correction T·S becomes significant (> 1 meV/atom).\n"
                f"For total energy comparisons, extrapolate to SIGMA → 0."
            )

        if encut is not None and encut < 300:
            warnings.append(
                f"ENCUT = {encut} eV is low for most elements.\n"
                f"Standard values: 400-520 eV. Low ENCUT causes Pulay stress\n"
                f"(basis incompleteness error) during cell relaxations."
            )

        if nsw > 0 and ibrion in (-1, None):
            warnings.append(
                f"NSW = {nsw} but IBRION = {ibrion}.\n"
                f"NSW requests ionic relaxation, but IBRION=-1 means no relaxation.\n"
                f"VASP will do {nsw} steps of nothing. Set IBRION=2 for CG relaxation."
            )

        if not warnings:
            return None

        return InsightBlock(
            title="Parameter Consistency Warnings",
            content="\n\n".join(f"{i + 1}. {w}" for i, w in enumerate(warnings)),
            level="warning",
        )

    def _sampling_insight(self, raw: Dict[str, Any]) -> InsightBlock:
        kpoints = raw.get("kpoints", {})
        mesh = kpoints.get("mesh")
        if mesh:
            subdiv = mesh.get("subdivisions", [1, 1, 1])
            total = subdiv[0] * subdiv[1] * subdiv[2]
            mode = mesh.get("mode", "unknown")
            return InsightBlock(
                title="k-Point Sampling - Brillouin Zone Integration",
                content=(
                    f"Mesh: {subdiv[0]}×{subdiv[1]}×{subdiv[2]} ({mode})\n"
                    f"Total k-points in irreducible BZ: ~{total} (before symmetry reduction)\n\n"
                    "Mathematically, the k-mesh is a composite Newton-Cotes quadrature\n"
                    "rule for integrals over the Brillouin zone. The error scales as\n"
                    "O(1/N_k) for metals and O(1/N_k²) for insulators.\n\n"
                    "Rule of thumb:\n"
                    "  • Insulators: 4×4×4 is often sufficient\n"
                    "  • Metals: 8×8×8 or denser needed for smooth DOS at E_F\n"
                    "  • 2D materials: use more k-points in-plane (e.g., 12×12×1)"
                ),
                level="tip",
            )
        return InsightBlock(
            title="k-Point Sampling",
            content="No automatic k-mesh detected. Ensure your KPOINTS file is appropriate for your system.",
            level="tip",
        )

    @staticmethod
    def _wavelength_estimate(encut: float) -> str:
        """Rough estimate of shortest wavelength in Å for given ENCUT (eV)."""
        # ℏ²/2m ≈ 3.81 eV·Å², so |G|² = 2m·ENCUT/ℏ² ≈ ENCUT/3.81 Å⁻²
        # λ = 2π/|G| ≈ 2π·√(3.81/ENCUT)
        import math

        lam = 2 * math.pi * math.sqrt(3.81 / max(encut, 1.0))
        return f"{lam:.2f}"
