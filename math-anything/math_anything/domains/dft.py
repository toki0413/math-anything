"""DFT Domain — Density Functional Theory as a morphism chain.

DFT is not "a software tool" — it's a specific sequence of mathematical
approximations applied to the many-body Schrödinger equation:

1. Many-body Schrödinger → Born-Oppenheimer (lose: nuclear quantum effects)
2. Born-Oppenheimer → Kohn-Sham mapping (lose: explicit electron correlations)
3. Kohn-Sham → Exchange-correlation approximation (lose: exact XC)
4. KS → Pseudopotential (lose: core electron physics)
5. KS → Basis set expansion (lose: completeness)
6. KS → k-point sampling (lose: continuous BZ integration)
7. SCF iteration (emerge: self-consistency constraint)
"""

from __future__ import annotations

from typing import Any

from math_anything.domains import register_domain
from math_anything.domains.base import Domain


@register_domain("dft")
class DFTDomain(Domain):
    """DFT as instantiation of SelfConsistentProblem + KohnSham morphism chain."""

    name = "dft"
    description = "Density Functional Theory — Kohn-Sham mapping of many-body Schrödinger"
    equation_type = "self_consistent"
    default_params = {
        "n_atoms": 1,
        "n_electrons": 1,
        "n_bands": 10,
        "ecutwfc": 50.0,  # Ry
        "k_points": [1, 1, 1],
        "xc_functional": "PBE",
        "pseudopotential": True,
    }

    def build_conservation_field(self) -> dict[str, Any]:
        """Build conservation field for Kohn-Sham equations."""
        from math_anything.structures.conservation_field import ConservationMatrixField

        field = ConservationMatrixField()
        field.build_from_kohn_sham(
            hbar=self.params.get("hbar", 1.0),
            m=self.params.get("m", 1.0),
            V_ext=self.params.get("V_ext", 0.0),
        )

        result = {
            "equation_type": "kohn_sham",
            "conservation_laws": [
                "energy_conservation",
                "particle_number_conservation",
                "norm_conservation",
            ],
            "symmetries": [
                "time_reversal" if not self.params.get("spin_orbit") else None,
                "spatial_symmetry",
            ],
            "eigenvalues": list(field.eigenvalues) if field.eigenvalues is not None else [],
        }
        if field.coupling_matrix is not None:
            result["coupling_matrix_shape"] = field.coupling_matrix.shape
        if field.symplectic_matrix is not None:
            result["symplectic"] = True  # type: ignore[assignment]
        return result

    def build_morphism_chain(self) -> list[dict[str, Any]]:
        """Build the DFT morphism chain."""
        chain = []

        # Step 1: Born-Oppenheimer
        chain.append(
            {
                "name": "born_oppenheimer",
                "type": "approximation",
                "description": "Separate electronic and nuclear degrees of freedom",
                "invariants_kept": ["energy_conservation", "particle_number_conservation"],
                "invariants_lost": ["nuclear_quantum_effects", "vibronic_coupling"],
                "invariants_introduced": ["adiabatic_approximation"],
            }
        )

        # Step 2: Kohn-Sham mapping
        chain.append(
            {
                "name": "kohn_sham_mapping",
                "type": "mapping",
                "description": "Map many-body problem to non-interacting particles in effective potential",
                "invariants_kept": ["energy_conservation", "particle_number_conservation", "norm_conservation"],
                "invariants_lost": ["explicit_electron_correlation", "wavefunction_nature"],
                "invariants_introduced": ["self_consistency"],
            }
        )

        # Step 3: XC approximation
        xc = self.params.get("xc_functional", "PBE")
        chain.append(
            {
                "name": f"xc_approximation_{xc.lower()}",
                "type": "approximation",
                "description": f"Approximate exchange-correlation with {xc}",
                "invariants_kept": ["particle_number_conservation", "norm_conservation"],
                "invariants_lost": ["exact_exchange_correlation"],
                "invariants_introduced": [f"{xc}_xc_functional"],
            }
        )

        # Step 4: Pseudopotential
        if self.params.get("pseudopotential", True):
            chain.append(
                {
                    "name": "pseudopotential",
                    "type": "approximation",
                    "description": "Replace core electrons with effective potential",
                    "invariants_kept": ["particle_number_conservation", "norm_conservation"],
                    "invariants_lost": ["core_electron_physics", "core_level_spectroscopy"],
                    "invariants_introduced": ["pseudopotential_validity_range"],
                }
            )

        # Step 5: Basis set
        ecut = self.params.get("ecutwfc", 50.0)
        chain.append(
            {
                "name": f"plane_wave_basis_ecut{ecut:.0f}",
                "type": "discretization",
                "description": f"Expand wavefunctions in plane waves with Ecut = {ecut} Ry",
                "invariants_kept": ["particle_number_conservation", "norm_conservation"],
                "invariants_lost": ["completeness", "real_space_locality"],
                "invariants_introduced": ["basis_set_convergence"],
            }
        )

        # Step 6: k-point sampling
        k = self.params.get("k_points", [1, 1, 1])
        chain.append(
            {
                "name": f"kpoint_sampling_{'x'.join(str(ki) for ki in k)}",
                "type": "discretization",
                "description": f"Sample BZ with {k} k-point grid",
                "invariants_kept": ["particle_number_conservation"],
                "invariants_lost": ["continuous_bz_integration"],
                "invariants_introduced": ["kpoint_convergence"],
            }
        )

        return chain
