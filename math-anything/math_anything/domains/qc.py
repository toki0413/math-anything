"""QC Domain — Quantum Chemistry as a morphism chain.

Quantum chemistry starts from the many-electron Schrödinger equation and
applies a sequence of approximations:

1. Full Schrödinger → Born-Oppenheimer (lose: nuclear quantum effects)
2. Born-Oppenheimer → Hartree-Fock (lose: electron correlation beyond mean-field)
3. Hartree-Fock → Basis set truncation (lose: completeness)
4. HF → Post-HF (MP2/CCSD/CI): partially recover correlation
5. HF → DFT: replace exact exchange with functional
"""

from __future__ import annotations

from typing import Any

from math_anything.domains import register_domain
from math_anything.domains.base import Domain


@register_domain("qc")
class QCDomain(Domain):
    """QC as instantiation of ManyElectronSchrodinger + HF/Post-HF morphism chain."""

    name = "qc"
    description = "Quantum Chemistry — many-electron Schrödinger with Hartree-Fock and Post-HF methods"
    equation_type = "many_electron_schrodinger"
    default_params = {
        "n_electrons": 10,
        "n_basis_functions": 50,
        "method": "HF",
        "basis_set": "cc-pVDZ",
        "relativistic": False,
    }

    def build_conservation_field(self) -> dict[str, Any]:
        """Build conservation field for many-electron Schrödinger equation."""
        from math_anything.structures.conservation_field import ConservationMatrixField

        field = ConservationMatrixField()
        field.build_from_schrodinger(
            hbar=1.0,
            m=1.0,
        )

        result = {
            "equation_type": "many_electron_schrodinger",
            "conservation_laws": [
                "energy_conservation",
                "particle_number_conservation",
                "total_spin_conservation",
                "angular_momentum_conservation",
            ],
            "symmetries": [
                "time_reversal",
                "spatial_symmetry",
                "permutational_symmetry_antisymmetry",
            ],
        }

        if field.coupling_matrix is not None:
            result["coupling_matrix_shape"] = field.coupling_matrix.shape
        if field.eigenvalues is not None:
            result["eigenvalues"] = [float(e) for e in field.eigenvalues[:6]]  # type: ignore[misc]

        return result

    def build_morphism_chain(self) -> list[dict[str, Any]]:
        """Build the QC morphism chain."""
        chain = []

        # Step 1: Born-Oppenheimer
        chain.append(
            {
                "name": "born_oppenheimer",
                "type": "approximation",
                "description": "Separate electronic and nuclear degrees of freedom",
                "invariants_kept": ["energy_conservation", "particle_number_conservation", "total_spin_conservation"],
                "invariants_lost": ["nuclear_quantum_effects", "vibronic_coupling"],
                "invariants_introduced": ["adiabatic_approximation"],
            }
        )

        # Step 2: Hartree-Fock
        chain.append(
            {
                "name": "hartree_fock",
                "type": "approximation",
                "description": "Replace many-body wavefunction with single Slater determinant",
                "invariants_kept": ["particle_number_conservation", "total_spin_conservation"],
                "invariants_lost": ["electron_correlation", "explicit_many_body_nature"],
                "invariants_introduced": ["mean_field_approximation", "variational_principle"],
            }
        )

        # Step 3: Basis set
        basis = self.params.get("basis_set", "cc-pVDZ")
        n_bf = self.params.get("n_basis_functions", 50)
        chain.append(
            {
                "name": f"basis_set_{basis.lower().replace('-', '_')}",
                "type": "discretization",
                "description": f"Expand orbitals in {basis} basis ({n_bf} functions)",
                "invariants_kept": ["particle_number_conservation"],
                "invariants_lost": ["completeness", "basis_set_superposition_error_free"],
                "invariants_introduced": ["basis_set_convergence", "bsse"],
            }
        )

        # Step 4: Post-HF or DFT
        method = self.params.get("method", "HF")
        if method == "HF":
            chain.append(
                {
                    "name": "hf_self_consistency",
                    "type": "iteration",
                    "description": "Solve HF equations self-consistently",
                    "invariants_kept": ["particle_number_conservation"],
                    "invariants_lost": [],
                    "invariants_introduced": ["scf_convergence_criterion"],
                }
            )
        elif method in ("MP2", "CCSD", "CCSD(T)", "CI"):
            lost = ["full_configuration_interaction"] if method != "CI" else []
            introduced = [f"{method}_correlation_energy"]
            if "CC" in method:
                introduced.append("size_consistency")
            chain.append(
                {
                    "name": f"post_hf_{method.lower().replace('(', '').replace(')', '')}",
                    "type": "approximation",
                    "description": f"Recover electron correlation with {method}",
                    "invariants_kept": ["particle_number_conservation", "total_spin_conservation"],
                    "invariants_lost": lost,
                    "invariants_introduced": introduced,
                }
            )
        elif method == "DFT":
            chain.append(
                {
                    "name": "dft_xc_functional",
                    "type": "approximation",
                    "description": "Replace exact exchange with density functional",
                    "invariants_kept": ["particle_number_conservation"],
                    "invariants_lost": ["exact_exchange", "derivative_discontinuity"],
                    "invariants_introduced": ["xc_functional_approximation", "self_interaction_error"],
                }
            )

        # Step 5: Relativistic effects
        if self.params.get("relativistic", False):
            chain.append(
                {
                    "name": "relativistic_correction",
                    "type": "approximation",
                    "description": "Include scalar relativistic effects (DKH/ZORA)",
                    "invariants_kept": ["energy_conservation"],
                    "invariants_lost": [],
                    "invariants_introduced": ["spin_orbit_coupling"],
                }
            )

        return chain
