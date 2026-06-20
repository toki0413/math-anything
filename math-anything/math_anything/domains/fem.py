"""FEM Domain — Finite Element Method as a morphism chain.

FEM is not "a software tool" — it's a sequence of approximations applied
to variational problems:

1. Strong form → Weak form (lose: pointwise satisfaction, gain: variational structure)
2. Weak form → Discrete function space (lose: infinite-dimensional completeness)
3. Discrete space → Quadrature (lose: exact integration)
4. Quadrature → Linear solver (lose: exact solution, gain: computability)
"""

from __future__ import annotations

from typing import Any

from math_anything.domains import register_domain
from math_anything.domains.base import Domain


@register_domain("fem")
class FEMDomain(Domain):
    """FEM as instantiation of VariationalProblem + discretization morphism chain."""

    name = "fem"
    description = "Finite Element Method — variational problems with Galerkin discretization"
    equation_type = "variational"
    default_params = {
        "problem_type": "elliptic",
        "basis_degree": 1,
        "n_elements": 100,
        "quadrature_order": 2,
        "solver": "direct",
        "domain_dim": 1,
    }

    def build_conservation_field(self) -> dict[str, Any]:
        """Build conservation field for variational problem."""
        result = {
            "equation_type": self.params.get("problem_type", "elliptic"),
            "conservation_laws": [
                "variational_consistency",
                "galerkin_orthogonality",
                "coercivity" if self.params.get("problem_type") == "elliptic" else None,
            ],
            "symmetries": [
                "bilinear_form_symmetry" if self.params.get("problem_type") == "elliptic" else None,
            ],
        }
        # Filter out None values from lists
        result["conservation_laws"] = [v for v in result["conservation_laws"] if v is not None]
        result["symmetries"] = [v for v in result["symmetries"] if v is not None]
        return result

    def build_morphism_chain(self) -> list[dict[str, Any]]:
        """Build the FEM morphism chain."""
        chain = []

        # Step 1: Strong → Weak form
        chain.append(
            {
                "name": "weak_formulation",
                "type": "reformulation",
                "description": "Multiply by test function and integrate by parts",
                "invariants_kept": ["variational_consistency"],
                "invariants_lost": ["pointwise_satisfaction", "strong_form_equivalence"],
                "invariants_introduced": ["galerkin_orthogonality", "inf_sup_condition"],
            }
        )

        # Step 2: Discrete function space
        p = self.params.get("basis_degree", 1)
        n = self.params.get("n_elements", 100)
        chain.append(
            {
                "name": f"discrete_space_p{p}_n{n}",
                "type": "discretization",
                "description": f"Restrict to P{p} finite element space with {n} elements",
                "invariants_kept": ["variational_consistency", "galerkin_orthogonality"],
                "invariants_lost": ["infinite_dimensional_completeness", f"approximation_power_beyond_p{p}"],
                "invariants_introduced": ["best_approximation_property", "c0_continuity"],
            }
        )

        # Step 3: Quadrature
        q = self.params.get("quadrature_order", 2)
        chain.append(
            {
                "name": f"quadrature_order_{q}",
                "type": "approximation",
                "description": f"Approximate integrals with order {q} quadrature",
                "invariants_kept": ["variational_consistency"],
                "invariants_lost": ["exact_integration"],
                "invariants_introduced": ["quadrature_convergence"],
            }
        )

        # Step 4: Linear solver
        solver = self.params.get("solver", "direct")
        chain.append(
            {
                "name": f"solver_{solver}",
                "type": "approximation",
                "description": f"Solve linear system with {solver} method",
                "invariants_kept": ["variational_consistency"],
                "invariants_lost": ["exact_solution"],
                "invariants_introduced": ["solver_convergence", "condition_number_dependence"],
            }
        )

        return chain
