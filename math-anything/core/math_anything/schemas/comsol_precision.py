"""COMSOL Mathematical Precision Extractor.

Extracts precise mathematical structures from COMSOL Multiphysics files,
enabling LLMs to understand coupled multiphysics problems.

Design Principles:
- Zero intrusion: Only observe, never modify
- Zero judgment: Describe what is, not what should be
"""

from typing import Any, Dict, List

from .precision import (Approximation, BasePrecisionExtractor,
                        DiscretizationScheme, MathematicalDecoding,
                        MathematicalStructure, PrecisionMetadata,
                        SolutionStrategy, VariableDependency)


class ComsolMathematicalPrecisionExtractor(BasePrecisionExtractor):
    """Extract precise mathematical structures from COMSOL inputs.

    COMSOL handles coupled multiphysics problems.
    """

    def extract_mathematical_structure(
        self, params: Dict[str, Any]
    ) -> MathematicalStructure:
        """Extract the mathematical structure of multiphysics problem."""
        physics_modules = params.get("physics_modules", ["general"])

        if len(physics_modules) == 1:
            return self._single_physics_structure(physics_modules[0], params)
        return self._coupled_physics_structure(physics_modules, params)

    def _single_physics_structure(
        self, physics: str, params: Dict[str, Any]
    ) -> MathematicalStructure:
        """Single physics module structure."""
        physics_map = {
            "solid_mechanics": ("boundary_value_problem", "∇·σ + f = 0"),
            "heat_transfer": ("parabolic_pde", "ρc ∂T/∂t - ∇·(k∇T) = Q"),
            "electromagnetics": ("elliptic_pde", "∇×(μ⁻¹∇×E) - ω²εE = 0"),
            "fluid_flow": ("navier_stokes", "ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + f"),
            "acoustics": ("wave_equation", "∇²p - (1/c²)∂²p/∂t² = 0"),
        }

        problem_type, canonical_form = physics_map.get(
            physics, ("pde_system", "L(u) = f")
        )

        return MathematicalStructure(
            problem_type=problem_type,
            canonical_form=canonical_form,
            properties={
                "formulation": "weak_form",
                "variational": True,
                "multiphysics": False,
            },
            dimension=params.get("dimension", 3),
            function_space="H¹(Ω) or appropriate Sobolev space",
        )

    def _coupled_physics_structure(
        self, modules: List[str], params: Dict[str, Any]
    ) -> MathematicalStructure:
        """Coupled multiphysics structure."""
        return MathematicalStructure(
            problem_type="coupled_pde_system",
            canonical_form="{L₁(u₁, u₂, ...) = f₁, L₂(u₁, u₂, ...) = f₂, ...}",
            properties={
                "formulation": "weak_form",
                "variational": True,
                "multiphysics": True,
                "coupling_type": params.get("coupling_type", "bidirectional"),
                "physics_modules": modules,
            },
            dimension=params.get("dimension", 3),
            function_space="product of Sobolev spaces",
        )

    def extract_variable_dependencies(
        self, params: Dict[str, Any]
    ) -> List[VariableDependency]:
        """Extract variable dependencies including coupling."""
        dependencies = []
        physics_modules = params.get("physics_modules", [])

        if "solid_mechanics" in physics_modules:
            dependencies.append(
                VariableDependency(
                    relation="σ = C : ε(u)",
                    depends_on=["u"],
                    circular=False,
                    mathematical_form="constitutive_relation",
                )
            )

        if "heat_transfer" in physics_modules:
            dependencies.append(
                VariableDependency(
                    relation="q = -k∇T",
                    depends_on=["T"],
                    circular=False,
                    mathematical_form="fourier_law",
                )
            )

        if (
            len(physics_modules) > 1
            and "solid_mechanics" in physics_modules
            and "heat_transfer" in physics_modules
        ):
            dependencies.append(
                VariableDependency(
                    relation="σ(T, u) = C(T) : ε(u)",
                    depends_on=["T", "u"],
                    circular=True,
                    mathematical_form="thermo_mechanical_coupling",
                    physical_interpretation="temperature affects material properties",
                )
            )

        return dependencies

    def extract_discretization_scheme(
        self, params: Dict[str, Any]
    ) -> DiscretizationScheme:
        """Extract discretization scheme."""
        order = params.get("element_order", 2)

        return DiscretizationScheme(
            method="finite_element",
            mathematical_meaning=f"Lagrange elements of order {order}",
            parameters={
                "mesh_type": params.get("mesh_type", "free_tetrahedral"),
                "element_order": order,
            },
            basis_type="lagrange_polynomials",
            completeness=f"C⁰ continuous, order {order}",
        )

    def extract_solution_strategy(self, params: Dict[str, Any]) -> SolutionStrategy:
        """Extract solution strategy."""
        study_type = params.get("study_type", "stationary")

        strategy_map = {
            "stationary": ("newton_raphson", "F(u) = 0, iterate until convergence"),
            "time_dependent": ("time_integration", "∂u/∂t = F(u), BDF/Runge-Kutta"),
            "eigenfrequency": ("eigenvalue_solver", "Kφ = λMφ"),
            "frequency_domain": ("harmonic_solver", "(-ω²M + iωC + K)u = F"),
            "parametric_sweep": (
                "parametric_continuation",
                "solve for parameter range",
            ),
        }

        method, form = strategy_map.get(study_type, ("newton_raphson", "F(u) = 0"))

        return SolutionStrategy(
            method=method,
            mathematical_form=form,
            convergence_criterion=params.get("tolerance", "1e-6"),
            iteration_type=(
                "fully_coupled"
                if params.get("coupling_type") == "bidirectional"
                else "segregated"
            ),
        )

    def extract_approximations(self, params: Dict[str, Any]) -> List[Approximation]:
        """Extract approximations."""
        approximations = [
            Approximation(
                name="continuum_approximation",
                mathematical_form="continuous medium assumption",
                consequence="no discrete microstructure",
                affected_quantities=["all_field_variables"],
            ),
            Approximation(
                name="finite_element_discretization",
                mathematical_form=f"order {params.get('element_order', 2)} Lagrange elements",
                consequence="continuous fields approximated by discrete nodes",
                affected_quantities=["solution_accuracy"],
            ),
        ]

        if params.get("coupling_type") == "unidirectional":
            approximations.append(
                Approximation(
                    name="one_way_coupling",
                    mathematical_form="physics A → physics B (no feedback)",
                    consequence="simplified physics interaction",
                    affected_quantities=["coupling_effects"],
                )
            )

        return approximations

    def extract_mathematical_decoding(
        self, params: Dict[str, Any]
    ) -> MathematicalDecoding:
        """Extract complete decoding."""
        structure = self.extract_mathematical_structure(params)
        physics_modules = params.get("physics_modules", ["general"])

        return MathematicalDecoding(
            core_problem={
                "type": (
                    "multiphysics" if len(physics_modules) > 1 else "single_physics"
                ),
                "equation": structure.canonical_form,
                "physical_meaning": f"coupled {', '.join(physics_modules)}",
            },
            approximations_applied=self.extract_approximations(params),
            solution_method={
                "type": params.get("study_type", "stationary"),
                "solver": (
                    "MUMPS/PARDISO" if params.get("direct_solver") else "iterative"
                ),
            },
            mathematical_hierarchy=[
                {
                    "level": "physical",
                    "description": "conservation laws for each physics",
                },
                {"level": "approximation_1", "description": "continuum hypothesis"},
                {"level": "approximation_2", "description": "constitutive relations"},
                {"level": "coupling", "description": "physics interaction terms"},
                {"level": "discretization", "description": "FE mesh"},
                {"level": "numerical", "description": "nonlinear/linear solver"},
            ],
        )

    def extract_precision_metadata(
        self, params: Dict[str, Any]
    ) -> Dict[str, PrecisionMetadata]:
        """Extract precision metadata."""
        metadata = {}
        for key in ["physics_modules", "study_type", "mesh_type", "element_order"]:
            if key in params:
                metadata[key] = PrecisionMetadata(
                    extraction_confidence=1.0,
                    source="direct_assignment",
                )
        return metadata


def extract_comsol_mathematical_precision(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for COMSOL precision extraction."""
    extractor = ComsolMathematicalPrecisionExtractor()
    return extractor.extract(params).to_dict()
