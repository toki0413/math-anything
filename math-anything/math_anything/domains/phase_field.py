"""Phase Field Domain — Mesoscale materials modeling as a morphism chain.

Phase field models describe microstructure evolution through continuous
order parameters instead of sharp interfaces:

1. Sharp interface → Diffuse interface (lose: exact interface position, gain: topological freedom)
2. Diffuse interface → Cahn-Hilliard (conserved order parameter, e.g., composition)
3. Or → Allen-Cahn (non-conserved, e.g., grain orientation)
4. Phase field → Anisotropic interface energy (lose: isotropy)
5. Phase field → Coupled mechanics/thermal (lose: decoupled fields)
"""

from __future__ import annotations

from typing import Any

from math_anything.domains import register_domain
from math_anything.domains.base import Domain


@register_domain("phase_field")
class PhaseFieldDomain(Domain):
    """Phase field as instantiation of CahnHilliard/AllenCahn + diffuse interface morphism chain."""

    name = "phase_field"
    description = "Phase Field — mesoscale microstructure evolution via Cahn-Hilliard and Allen-Cahn equations"
    equation_type = "phase_field"
    default_params = {
        "model": "Cahn-Hilliard",
        "n_order_parameters": 1,
        "interface_width": 0.01,
        "mobility": 1.0,
        "gradient_energy": 1.0,
        "anisotropy": False,
        "coupled_mechanics": False,
        "n_grid": 128,
        "domain_size": 1.0,
    }

    def build_conservation_field(self) -> dict[str, Any]:
        """Build conservation field for phase field equations using ConservationMatrixField."""
        from math_anything.structures.conservation_field import ConservationMatrixField

        model = self.params.get("model", "Cahn-Hilliard")
        M = self.params.get("mobility", 1.0)
        kappa = self.params.get("gradient_energy", 1.0)

        field = ConservationMatrixField()

        if model == "Cahn-Hilliard":
            field.build_from_cahn_hilliard(M=M, kappa=kappa)
            conservation_laws = ["mass_conservation", "free_energy_dissipation"]
            conservation_laws_lost = []
        elif model == "Allen-Cahn":
            L = M  # Use mobility as kinetic coefficient
            field.build_from_allen_cahn(L=L, kappa=kappa)
            conservation_laws = ["free_energy_dissipation"]
            conservation_laws_lost = ["mass_conservation"]
        else:
            conservation_laws = ["free_energy_dissipation"]
            conservation_laws_lost = []

        # Build symmetries list
        symmetries = ["translational_invariance"]
        if not self.params.get("anisotropy", False):
            symmetries.append("rotational_invariance")

        # CFL condition
        n = self.params.get("n_grid", 128)
        L_domain = self.params.get("domain_size", 1.0)
        dx = L_domain / n

        if model == "Cahn-Hilliard":
            max_dt = dx**4 / (4 * M * kappa) if M * kappa > 0 else float("inf")
            cfl_type = "4th_order_diffusion"
        else:
            max_dt = dx**2 / (2 * M) if M > 0 else float("inf")
            cfl_type = "2nd_order_diffusion"

        # Merge field data with domain-specific metadata
        field_dict = field.to_dict()
        field_dict.update(
            {
                "equation_type": model,
                "conservation_laws": conservation_laws,
                "conservation_laws_lost": conservation_laws_lost,
                "symmetries": symmetries,
                "cfl_condition": {
                    "type": cfl_type,
                    "dx": dx,
                    "max_dt": max_dt,
                },
            }
        )

        return field_dict

    def build_morphism_chain(self) -> list[dict[str, Any]]:
        """Build the phase field morphism chain."""
        chain = []

        # Step 1: Sharp → Diffuse interface
        W = self.params.get("interface_width", 0.01)
        chain.append(
            {
                "name": f"diffuse_interface_W{W}",
                "type": "approximation",
                "description": f"Replace sharp interface with diffuse interface of width W={W}",
                "invariants_kept": ["free_energy_dissipation"],
                "invariants_lost": ["sharp_interface_position", "exact_surface_area"],
                "invariants_introduced": ["interface_width_parameter", "topological_changes_allowed"],
            }
        )

        # Step 2: Specific model
        model = self.params.get("model", "Cahn-Hilliard")
        if model == "Cahn-Hilliard":
            chain.append(
                {
                    "name": "cahn_hilliard_formulation",
                    "type": "reformulation",
                    "description": "Conserved dynamics: ∂c/∂t = ∇·(M ∇μ), μ = δF/δc",
                    "invariants_kept": ["mass_conservation", "free_energy_dissipation"],
                    "invariants_lost": [],
                    "invariants_introduced": ["chemical_potential", "spinodal_decomposition"],
                }
            )
        elif model == "Allen-Cahn":
            chain.append(
                {
                    "name": "allen_cahn_formulation",
                    "type": "reformulation",
                    "description": "Non-conserved dynamics: ∂η/∂t = -L δF/δη",
                    "invariants_kept": ["free_energy_dissipation"],
                    "invariants_lost": ["mass_conservation"],
                    "invariants_introduced": ["grain_boundary_motion", "curvature_driven_flow"],
                }
            )

        # Step 3: Anisotropy
        if self.params.get("anisotropy", False):
            chain.append(
                {
                    "name": "anisotropic_interface_energy",
                    "type": "approximation",
                    "description": "Include orientation-dependent interface energy",
                    "invariants_kept": ["free_energy_dissipation"],
                    "invariants_lost": ["rotational_invariance"],
                    "invariants_introduced": ["faceted_growth", "dendritic_morphology"],
                }
            )

        # Step 4: Coupled mechanics/thermal
        if self.params.get("coupled_mechanics", False):
            chain.append(
                {
                    "name": "coupled_mechanics",
                    "type": "coupling",
                    "description": "Couple phase field with elasticity/thermal field",
                    "invariants_kept": ["free_energy_dissipation"],
                    "invariants_lost": [],
                    "invariants_introduced": ["eigenstrain", "transformation_strain", "thermal_gradient_driving_force"],
                }
            )

        # Step 5: Spatial discretization
        n = self.params.get("n_grid", 128)
        chain.append(
            {
                "name": f"spatial_discretization_{n}grid",
                "type": "discretization",
                "description": f"Discretize on {n}×{n} uniform grid",
                "invariants_kept": ["free_energy_dissipation"],
                "invariants_lost": ["sub_grid_microstructure", "continuous_field"],
                "invariants_introduced": ["grid_convergence", "numerical_diffusion"],
            }
        )

        return chain
