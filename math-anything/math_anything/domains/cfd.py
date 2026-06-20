"""CFD Domain — Computational Fluid Dynamics as a morphism chain.

CFD is not "a software tool" — it's a sequence of approximations applied
to the Navier-Stokes equations:

1. Full NS → Incompressibility (lose: compressibility, acoustic waves)
2. Incompressible NS → Turbulence model (lose: small-scale dynamics)
3. Turbulence model → Spatial discretization (lose: sub-grid resolution)
4. Discretization → Time integration (lose: temporal continuity)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from math_anything.domains import register_domain
from math_anything.domains.base import Domain


@register_domain("cfd")
class CFDDomain(Domain):
    """CFD as instantiation of NavierStokesProblem + discretization morphism chain."""

    name = "cfd"
    description = "Computational Fluid Dynamics — Navier-Stokes with turbulence modeling"
    equation_type = "navier_stokes"
    default_params = {
        "regime": "incompressible",
        "turbulence_model": "RANS",
        "discretization": "FVM",
        "n_cells": 10000,
        "reynolds_number": 1000.0,
        "mach_number": 0.1,
        "rho": 1.225,  # kg/m^3
        "mu": 1.8e-5,  # Pa·s
        "velocity": 10.0,  # m/s
        "length_scale": 1.0,  # m
    }

    def build_conservation_field(self) -> dict[str, Any]:
        """Build conservation field for Navier-Stokes."""
        from math_anything.structures.conservation_field import ConservationMatrixField

        field = ConservationMatrixField()
        regime = self.params.get("regime", "incompressible")

        if regime == "incompressible":
            field.build_from_navier_stokes(
                gamma=1.4,
                mu=self.params.get("mu", 1.8e-5),
            )
        else:
            field.build_from_euler_equations(gamma=1.4)

        result = {
            "equation_type": regime,
            "conservation_laws": [
                "mass_conservation",
                "momentum_conservation",
                "energy_conservation" if regime != "incompressible" else None,
            ],
            "symmetries": [
                "gallilean_invariance",
                "rotational_symmetry" if not self.params.get("body_forces") else None,
            ],
        }

        if field.coupling_matrix is not None:
            result["coupling_matrix_shape"] = field.coupling_matrix.shape
            result["eigenvalues"] = list(np.real(field.eigenvalues)) if field.eigenvalues is not None else []
        if field.eigenvalues is not None:
            result["characteristic_speeds"] = list(field.eigenvalues)

        # CFL condition
        v = self.params.get("velocity", 10.0)
        L = self.params.get("length_scale", 1.0)
        n = self.params.get("n_cells", 10000)
        dx = L / n ** (1 / 3)
        result["cfl_condition"] = {
            "velocity": v,
            "dx": dx,
            "max_dt": dx / v,
        }

        return result

    def build_morphism_chain(self) -> list[dict[str, Any]]:
        """Build the CFD morphism chain."""
        chain = []
        regime = self.params.get("regime", "incompressible")

        # Step 1: Incompressibility
        if regime == "incompressible":
            chain.append(
                {
                    "name": "incompressibility",
                    "type": "approximation",
                    "description": "Assume divergence-free velocity field (div(u) = 0)",
                    "invariants_kept": ["mass_conservation", "momentum_conservation"],
                    "invariants_lost": ["compressibility", "acoustic_waves", "energy_conservation"],
                    "invariants_introduced": ["pressure_poisson_equation"],
                }
            )

        # Step 2: Turbulence model
        turb = self.params.get("turbulence_model", "RANS")
        if turb != "DNS":
            chain.append(
                {
                    "name": f"turbulence_model_{turb.lower()}",
                    "type": "approximation",
                    "description": f"Model unresolved turbulence with {turb}",
                    "invariants_kept": ["mass_conservation", "momentum_conservation"],
                    "invariants_lost": ["small_scale_dynamics", "turbulent_kinetic_energy_exact"],
                    "invariants_introduced": [f"{turb}_closure_assumption"],
                }
            )

        # Step 3: Spatial discretization
        disc = self.params.get("discretization", "FVM")
        n = self.params.get("n_cells", 10000)
        chain.append(
            {
                "name": f"spatial_discretization_{disc.lower()}",
                "type": "discretization",
                "description": f"Discretize with {disc} on {n} cells",
                "invariants_kept": ["mass_conservation", "momentum_conservation"],
                "invariants_lost": ["sub_grid_resolution", "spatial_continuity"],
                "invariants_introduced": ["numerical_diffusion", "grid_convergence"],
            }
        )

        # Step 4: Time integration
        chain.append(
            {
                "name": "time_integration",
                "type": "discretization",
                "description": "Discretize in time with finite difference",
                "invariants_kept": ["mass_conservation"],
                "invariants_lost": ["temporal_continuity", "time_reversal_symmetry"],
                "invariants_introduced": ["cfl_condition", "temporal_convergence"],
            }
        )

        return chain
