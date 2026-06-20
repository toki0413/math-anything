"""Quantum ESPRESSO insight engine.

Leverages DFT domain template; only QE-specific overrides needed.
"""

from typing import Any, Dict, List

from ..schemas import MathSchema
from ..templates import DFTInsightTemplate
from .base import InsightBlock, InsightEngine


class QuantumEspressoInsightEngine(InsightEngine):
    """Generate mathematical insights for Quantum ESPRESSO calculations."""

    @property
    def engine_name(self) -> str:
        return "quantum_espresso"

    def generate(self, schema: MathSchema) -> List[InsightBlock]:
        params = self._extract_params(schema)
        tpl = DFTInsightTemplate(params)
        tpl.software_name = "Quantum ESPRESSO"
        tpl.basis_type = "plane-wave"
        tpl.pseudopotential_type = "norm-conserving or PAW (PSLibrary)"

        blocks = tpl.to_insight_blocks()

        # QE-specific: diagonalization algorithm insight
        diag = params.get("diagonalization", "david")
        if diag == "david":
            blocks.append(
                InsightBlock(
                    title="Diagonalization Algorithm",
                    content=(
                        "Quantum ESPRESSO uses the Davidson iterative diagonalization algorithm. "
                        "Davidson constructs a subspace of trial vectors and solves the projected eigenproblem. "
                        "Convergence is accelerated by preconditioning with the diagonal of the kinetic energy. "
                        "For large systems (> 1000 bands), consider 'cg' (conjugate gradient) which has lower memory footprint."
                    ),
                    level="math",
                )
            )
        elif diag == "cg":
            blocks.append(
                InsightBlock(
                    title="Diagonalization Algorithm",
                    content=(
                        "Conjugate-gradient diagonalization is selected. "
                        "CG is robust and has O(N) memory scaling per band, but converges slower than Davidson. "
                        "Recommended for very large systems or when memory is constrained."
                    ),
                    level="math",
                )
            )

        # QE-specific: ultrasoft/PAW vs norm-conserving
        species = params.get("species", [])
        pp_types = set()
        for sp in species:
            pp = sp.get("pseudopotential", "").lower()
            if "us" in pp or "rrkj" in pp:
                pp_types.add("ultrasoft")
            elif "paw" in pp or "kj" in pp:
                pp_types.add("PAW")
            else:
                pp_types.add("norm-conserving")

        if "ultrasoft" in pp_types or "PAW" in pp_types:
            blocks.append(
                InsightBlock(
                    title="Pseudopotential Type",
                    content=(
                        "Ultrasoft or PAW pseudopotentials are used. These allow lower ecutwfc than norm-conserving "
                        "pseudopotentials because the pseudo-wavefunctions are smoother in the core region. "
                        "However, ecutrho must be ~4-8x ecutwfc for accurate charge density representation."
                    ),
                    level="info",
                )
            )

        return blocks

    def _extract_params(self, schema: MathSchema) -> Dict[str, Any]:
        """Map schema.raw_symbols to DFT template params."""
        raw = schema.raw_symbols or {}

        # Map QE-specific keys to template-standard keys
        params = dict(raw)

        # Standardize key names for DFT template
        ecutwfc_ry = raw.get("ecutwfc")
        params["encut"] = ecutwfc_ry * 13.6057 if ecutwfc_ry else None
        params["ediff"] = raw.get("conv_thr", 1e-6)
        params["nelm"] = raw.get("electron_maxstep", 100)
        params["algo"] = raw.get("mixing_mode", "plain")
        params["ismear"] = self._map_smearing(raw.get("smearing", ""), raw.get("occupations", "fixed"))
        params["sigma"] = raw.get("degauss", 0.0)
        params["ispin"] = raw.get("nspin", 1)
        params["functional"] = raw.get("functional", "PBE")
        params["nsw"] = raw.get("nstep", 1) if raw.get("calculation") in ("relax", "vc-relax", "md") else 0
        params["lhfcalc"] = False  # QE uses input_dft for hybrids
        params["kpoint_mesh"] = raw.get("kpoint_mesh", [])

        return params

    def _map_smearing(self, smearing: str, occupations: str) -> int:
        """Map QE smearing to VASP-style ISMEAR equivalent for checks."""
        if occupations == "fixed":
            return -5 if not smearing else 0
        smear_lower = smearing.lower()
        if "gauss" in smear_lower or "gaussian" in smear_lower:
            return 0
        elif "m-p" in smear_lower or "mp" in smear_lower or "methfessel" in smear_lower:
            return 1
        elif "marzari" in smear_lower or "cold" in smear_lower or "mv" in smear_lower:
            return -1  # Marzari-Vanderbilt cold smearing
        elif "fermi" in smear_lower:
            return -1
        return 0
