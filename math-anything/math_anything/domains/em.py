"""EM Domain — Electromagnetic field simulation as a morphism chain.

Maxwell's equations are the foundation. The morphism chain shows how
different EM simulation methods approximate the full equations:

1. Full Maxwell → Frequency domain (lose: transient effects)
2. Frequency domain → Quasi-static (lose: displacement current, wave propagation)
3. Full Maxwell → FDTD discretization (lose: sub-cell resolution)
4. FDTD → PML absorbing boundary (lose: open boundary exactness)
5. Full Maxwell → FEM discretization (lose: local conservation)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from math_anything.domains import register_domain
from math_anything.domains.base import Domain


@register_domain("em")
class EMDomain(Domain):
    """EM as instantiation of MaxwellEquations + discretization morphism chain."""

    name = "em"
    description = "Electromagnetic field simulation — Maxwell equations with FDTD/FEM discretization"
    equation_type = "maxwell"
    default_params = {
        "method": "FDTD",
        "frequency_domain": False,
        "quasi_static": False,
        "n_cells": 100,
        "pml_layers": 10,
        "c": 299792458.0,  # m/s
        "epsilon_0": 8.854187817e-12,  # F/m
        "mu_0": 1.2566370614e-6,  # H/m
        "domain_size": 1.0,  # m
    }

    def build_conservation_field(self) -> dict[str, Any]:
        """Build conservation field for Maxwell equations."""
        from math_anything.structures.conservation_field import ConservationMatrixField

        field = ConservationMatrixField()
        c = self.params.get("c", 299792458.0)
        mu_0 = self.params.get("mu_0", 1.2566370614e-6)
        field.build_from_maxwell(
            c=c,
            mu0=mu_0,
        )

        result = {
            "equation_type": "maxwell",
            "conservation_laws": [
                "charge_conservation",
                "energy_conservation_poynting",
                "momentum_conservation_maxwell_stress",
            ],
            "symmetries": [
                "lorentz_invariance",
                "gauge_invariance",
                "duality_electric_magnetic",
            ],
        }

        if field.coupling_matrix is not None:
            result["coupling_matrix_shape"] = field.coupling_matrix.shape
        if field.eigenvalues is not None:
            result["eigenvalues"] = [float(e) for e in field.eigenvalues[:4]]  # type: ignore[misc]

        # CFL for FDTD
        n = self.params.get("n_cells", 100)
        L = self.params.get("domain_size", 1.0)
        dx = L / n
        result["cfl_condition"] = {  # type: ignore[assignment]
            "wave_speed": c,
            "dx": dx,
            "max_dt": dx / (c * np.sqrt(3)),  # 3D Courant condition
        }

        return result

    def build_morphism_chain(self) -> list[dict[str, Any]]:
        """Build the EM morphism chain."""
        chain = []

        # Step 1: Frequency domain
        if self.params.get("frequency_domain", False):
            chain.append(
                {
                    "name": "frequency_domain",
                    "type": "approximation",
                    "description": "Transform to frequency domain via Fourier — assume time-harmonic fields",
                    "invariants_kept": ["energy_conservation_poynting", "charge_conservation"],
                    "invariants_lost": ["transient_effects", "broadband_response"],
                    "invariants_introduced": ["frequency_decoupling", "impedance_concept"],
                }
            )

        # Step 2: Quasi-static
        if self.params.get("quasi_static", False):
            chain.append(
                {
                    "name": "quasi_static",
                    "type": "approximation",
                    "description": "Neglect displacement current ∂D/∂t → elliptic problem",
                    "invariants_kept": ["charge_conservation"],
                    "invariants_lost": ["wave_propagation", "displacement_current", "radiation"],
                    "invariants_introduced": ["electrostatic_approximation", "magnetostatic_approximation"],
                }
            )

        # Step 3: Spatial discretization
        method = self.params.get("method", "FDTD")
        n = self.params.get("n_cells", 100)
        chain.append(
            {
                "name": f"spatial_discretization_{method.lower()}",
                "type": "discretization",
                "description": f"Discretize with {method} on {n} cells",
                "invariants_kept": ["charge_conservation"],
                "invariants_lost": ["sub_cell_resolution", "continuous_field"],
                "invariants_introduced": [
                    "numerical_dispersion",
                    "staircase_approximation" if method == "FDTD" else "mesh_dependency",
                ],
            }
        )

        # Step 4: Absorbing boundary (PML)
        if self.params.get("pml_layers", 0) > 0:
            pml = self.params.get("pml_layers", 10)
            chain.append(
                {
                    "name": f"pml_boundary_{pml}layers",
                    "type": "approximation",
                    "description": f"Perfectly Matched Layer with {pml} cells for absorbing boundaries",
                    "invariants_kept": ["charge_conservation"],
                    "invariants_lost": ["open_boundary_exactness"],
                    "invariants_introduced": ["pml_reflection_coefficient", "pml_stability_condition"],
                }
            )

        return chain
