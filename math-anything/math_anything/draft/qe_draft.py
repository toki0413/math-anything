"""Quantum ESPRESSO draft engine.

Leverages DFT domain template; only QE-specific overrides needed.
"""

from typing import Any, Dict

from ..schemas import MathSchema
from ..templates import DFTDraftTemplate
from .base import DraftEngine


class QuantumEspressoDraftEngine(DraftEngine):
    """Generate publication methodology for Quantum ESPRESSO calculations."""

    @property
    def engine_name(self) -> str:
        return "quantum_espresso"

    def generate(self, schema: MathSchema, fmt: str = "markdown") -> str:
        params = self._extract_params(schema)
        tpl = DFTDraftTemplate(params)
        tpl.software_name = "Quantum ESPRESSO"
        tpl.basis_type = "plane-wave"
        tpl.pseudopotential_type = "norm-conserving or PAW"

        # Generate base draft
        text = tpl.to_draft_text(fmt=fmt)

        # QE-specific additions
        diag = params.get("diagonalization", "david")
        mixing = params.get("mixing_mode", "plain")

        if fmt == "markdown":
            qe_section = "\n## Quantum ESPRESSO Specifics\n\n"
        else:
            qe_section = "\n\\subsection{Quantum ESPRESSO Specifics}\n"

        qe_section += (
            f"The Kohn-Sham orbitals were expanded in a plane-wave basis set with a kinetic energy cutoff "
            f"of {params.get('ecutwfc', 'unspecified')} Ry for wavefunctions "
        )
        if params.get("ecutrho"):
            qe_section += f"and {params['ecutrho']} Ry for charge density. "
        else:
            qe_section += ". "

        qe_section += (
            f"Self-consistency was achieved using {mixing} mixing with beta = {params.get('mixing_beta', 0.7)}. "
            f"Diagonalization was performed via the {diag} algorithm. "
            "Pseudopotentials were taken from the PSLibrary or equivalent repository."
        )

        # Insert before caveats (last section)
        if fmt == "markdown":
            parts = text.rsplit("## Methodological Notes", 1)
            if len(parts) == 2:
                text = parts[0] + qe_section + "\n\n## Methodological Notes" + parts[1]
            else:
                text += qe_section
        else:
            parts = text.rsplit("\\subsection{Methodological Notes", 1)
            if len(parts) == 2:
                text = parts[0] + qe_section + "\n\n\\subsection{Methodological Notes" + parts[1]
            else:
                text += qe_section

        return text

    def _extract_params(self, schema: MathSchema) -> Dict[str, Any]:
        """Map schema.raw_symbols to DFT template params."""
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
