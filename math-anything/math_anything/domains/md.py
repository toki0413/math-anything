"""MD Domain — Molecular Dynamics as a morphism chain.

MD is not "a software tool" — it's a sequence of approximations applied
to Hamiltonian mechanics:

1. Full Hamiltonian → Classical approximation (lose: quantum effects)
2. Classical → Force field (lose: ab initio accuracy)
3. Force field → Cutoff/truncation (lose: long-range interactions)
4. Cutoff → Thermostat/barostat (lose: energy conservation, emerge: ensemble control)
5. Thermostat → Time integration (lose: symplecticity if not symplectic integrator)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from math_anything.domains import register_domain
from math_anything.domains.base import Domain


@register_domain("md")
class MDDomain(Domain):
    """MD as instantiation of HamiltonianSystem + time-stepping morphism chain."""

    name = "md"
    description = "Molecular Dynamics — Hamiltonian mechanics with force field approximation"
    equation_type = "hamiltonian"
    default_params = {
        "n_atoms": 1000,
        "force_field": "Lennard-Jones",
        "cutoff_radius": 12.0,  # Angstrom
        "thermostat": "Nose-Hoover",
        "integrator": "Velocity-Verlet",
        "timestep": 1.0,  # fs
        "temperature": 300.0,  # K
        "ensemble": "NVT",
    }

    def build_conservation_field(self) -> dict[str, Any]:
        """Build conservation field for Hamiltonian system."""
        from math_anything.structures.conservation_field import ConservationMatrixField

        field = ConservationMatrixField()
        n_atoms = self.params.get("n_atoms", 1000)
        dim = n_atoms * 3

        # Build symplectic structure for classical Hamiltonian system
        # J = [[0, I], [-I, 0]] in (q, p) coordinates
        J = np.zeros((2 * dim, 2 * dim))
        J[:dim, dim:] = np.eye(dim)
        J[dim:, :dim] = -np.eye(dim)

        field.symplectic_matrix = J
        field.hamiltonian = "H = Σ p_i²/(2m_i) + V(r_1, ..., r_N)"

        result = {
            "equation_type": "hamiltonian",
            "conservation_laws": [
                "energy_conservation",
                "momentum_conservation",
                "angular_momentum_conservation",
                "phase_space_volume_liouville",
            ],
            "symmetries": [
                "time_reversal",
                "time_translation",
                "spatial_translation",
            ],
        }

        if field.symplectic_matrix is not None:
            result["symplectic"] = True  # type: ignore[assignment]
            result["symplectic_matrix_shape"] = field.symplectic_matrix.shape

        return result

    def build_morphism_chain(self) -> list[dict[str, Any]]:
        """Build the MD morphism chain."""
        chain = []

        # Step 1: Classical approximation
        chain.append(
            {
                "name": "classical_approximation",
                "type": "approximation",
                "description": "Replace quantum dynamics with classical Newtonian mechanics",
                "invariants_kept": ["energy_conservation", "momentum_conservation", "angular_momentum_conservation"],
                "invariants_lost": ["quantum_tunneling", "zero_point_energy", "quantum_statistics"],
                "invariants_introduced": ["classical_limit_validity"],
            }
        )

        # Step 2: Force field
        ff = self.params.get("force_field", "Lennard-Jones")
        chain.append(
            {
                "name": f"force_field_{ff.lower().replace('-', '_')}",
                "type": "approximation",
                "description": f"Approximate interatomic potential with {ff}",
                "invariants_kept": ["momentum_conservation", "angular_momentum_conservation"],
                "invariants_lost": ["ab_initio_accuracy", "electronic_structure"],
                "invariants_introduced": [f"{ff}_parameterization"],
            }
        )

        # Step 3: Cutoff
        rc = self.params.get("cutoff_radius", 12.0)
        chain.append(
            {
                "name": f"cutoff_{rc:.0f}a",
                "type": "approximation",
                "description": f"Truncate interactions at r_c = {rc} Angstrom",
                "invariants_kept": ["momentum_conservation"],
                "invariants_lost": ["long_range_interactions", "energy_conservation"],
                "invariants_introduced": ["cutoff_artifacts"],
            }
        )

        # Step 4: Thermostat/barostat
        thermostat = self.params.get("thermostat", "Nose-Hoover")
        ensemble = self.params.get("ensemble", "NVT")
        chain.append(
            {
                "name": f"thermostat_{thermostat.lower().replace('-', '_')}",
                "type": "approximation",
                "description": f"Control temperature with {thermostat} thermostat ({ensemble} ensemble)",
                "invariants_kept": ["momentum_conservation"],
                "invariants_lost": ["energy_conservation", "phase_space_volume_liouville"],
                "invariants_introduced": [f"{ensemble}_ensemble_distribution", "ergodic_hypothesis"],
            }
        )

        # Step 5: Time integration
        integrator = self.params.get("integrator", "Velocity-Verlet")
        dt = self.params.get("timestep", 1.0)
        is_symplectic = integrator in ("Velocity-Verlet", "Verlet", "Leapfrog")
        chain.append(
            {
                "name": f"integrator_{integrator.lower().replace('-', '_')}",
                "type": "discretization",
                "description": f"Integrate with {integrator}, dt = {dt} fs",
                "invariants_kept": ["momentum_conservation"]
                + (["phase_space_volume_liouville"] if is_symplectic else []),
                "invariants_lost": [] if is_symplectic else ["phase_space_volume_liouville"],
                "invariants_introduced": ["timestep_convergence", "resonance_artifacts"],
            }
        )

        return chain
