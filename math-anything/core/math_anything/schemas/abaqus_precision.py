"""Abaqus Mathematical Precision Extractor.

Extracts precise mathematical structures from Abaqus input files,
enabling LLMs to understand the mathematical essence of FEM simulations.

Design Principles:
- Zero intrusion: Only observe, never modify
- Zero judgment: Describe what is, not what should be
"""

from typing import Dict, Any, List
from .precision import (
    MathematicalStructure,
    VariableDependency,
    DiscretizationScheme,
    SolutionStrategy,
    Approximation,
    MathematicalDecoding,
    PrecisionMetadata,
    BasePrecisionExtractor,
)


class AbaqusMathematicalPrecisionExtractor(BasePrecisionExtractor):
    """Extract precise mathematical structures from Abaqus inputs.
    
    Abaqus solves continuum mechanics problems using finite element method.
    """
    
    def extract_mathematical_structure(self, params: Dict[str, Any]) -> MathematicalStructure:
        """Extract the mathematical structure of FEM problem."""
        analysis_type = params.get("analysis_type", "static")
        
        type_map = {
            "static": ("boundary_value_problem", "∇·σ + f = 0"),
            "dynamic": ("initial_boundary_value_problem", "ρü - ∇·σ = f"),
            "thermal": ("parabolic_pde", "ρc ∂T/∂t - ∇·(k∇T) = Q"),
        }
        
        problem_type, canonical_form = type_map.get(analysis_type,
            ("boundary_value_problem", "∇·σ + f = 0"))
        
        return MathematicalStructure(
            problem_type=problem_type,
            canonical_form=canonical_form,
            properties={
                "formulation": "weak_form",
                "variational": True,
                "discretization": "finite_element",
            },
            dimension=3,
            function_space="H¹(Ω)",
        )
    
    def extract_variable_dependencies(self, params: Dict[str, Any]) -> List[VariableDependency]:
        """Extract variable dependencies in FEM."""
        dependencies = [
            VariableDependency(
                relation="σ = C : ε",
                depends_on=["ε"],
                circular=False,
                mathematical_form="constitutive_law",
                physical_interpretation="stress from strain via material law",
            ),
            VariableDependency(
                relation="ε = ½(∇u + (∇u)ᵀ)",
                depends_on=["u"],
                circular=False,
                mathematical_form="strain_displacement",
                physical_interpretation="strain from displacement gradient",
            ),
        ]
        
        if params.get("nonlinear", False):
            dependencies.append(VariableDependency(
                relation="σ = σ(ε, state_variables)",
                depends_on=["ε", "history"],
                circular=True,
                mathematical_form="nonlinear_constitutive",
                physical_interpretation="nonlinear material requires iteration",
            ))
        
        return dependencies
    
    def extract_discretization_scheme(self, params: Dict[str, Any]) -> DiscretizationScheme:
        """Extract the FEM discretization scheme."""
        element_type = params.get("element_type", "C3D8R")
        mesh_size = params.get("mesh_size", 0.01)
        
        return DiscretizationScheme(
            method="finite_element",
            mathematical_meaning="u(x) ≈ Σᵢ Nᵢ(x)uᵢ, where Nᵢ are shape functions",
            parameters={
                "element_type": element_type,
                "mesh_size": f"{mesh_size} m",
            },
            basis_type="polynomial_shape_functions",
            completeness="C⁰ or C¹ continuity",
            convergence_order="O(h^p) where p is polynomial order",
        )
    
    def extract_solution_strategy(self, params: Dict[str, Any]) -> SolutionStrategy:
        """Extract the solution strategy."""
        analysis_type = params.get("analysis_type", "static")
        
        if analysis_type == "static":
            if params.get("nonlinear", False):
                return SolutionStrategy(
                    method="newton_raphson",
                    mathematical_form="K(u)u = F, iterate: u_{n+1} = u_n - K⁻¹R",
                    convergence_criterion=f"|R| < {params.get('tolerance', 1e-6)}",
                    iteration_type="newton_iteration",
                )
            return SolutionStrategy(
                method="direct_solver",
                mathematical_form="Ku = F (linear system)",
                convergence_criterion="N/A (direct solve)",
                iteration_type="none",
            )
        
        return SolutionStrategy(
            method="time_integration",
            mathematical_form="Mü + Cu̇ + Ku = F(t)",
            convergence_criterion=f"Δt = {params.get('dt', 0.001)} s",
            iteration_type="explicit_or_implicit",
        )
    
    def extract_approximations(self, params: Dict[str, Any]) -> List[Approximation]:
        """Extract the approximations in FEM."""
        approximations = [
            Approximation(
                name="continuum_approximation",
                mathematical_form="material is continuous medium",
                consequence="no atomic/molecular structure considered",
                affected_quantities=["stress", "strain", "displacement"],
                theoretical_basis="continuum mechanics",
            ),
            Approximation(
                name="finite_element_discretization",
                mathematical_form=f"{params.get('element_type', 'C3D8R')} elements",
                consequence="continuous domain approximated by discrete elements",
                affected_quantities=["solution_accuracy", "stress_concentration"],
                theoretical_basis="Galerkin method",
            ),
            Approximation(
                name="mesh_resolution",
                mathematical_form=f"element size ≈ {params.get('mesh_size', 0.01)} m",
                consequence="spatial resolution limited by mesh",
                affected_quantities=["gradient_accuracy", "local_phenomena"],
                theoretical_basis="numerical discretization",
            ),
        ]
        
        if params.get("material_model") == "elastic":
            approximations.append(Approximation(
                name="linear_elastic",
                mathematical_form="σ = Eε (Hooke's law)",
                consequence="no plasticity or damage considered",
                affected_quantities=["large_deformation", "failure"],
                theoretical_basis="linear elasticity theory",
            ))
        
        return approximations
    
    def extract_mathematical_decoding(self, params: Dict[str, Any]) -> MathematicalDecoding:
        """Extract complete mathematical decoding."""
        structure = self.extract_mathematical_structure(params)
        
        return MathematicalDecoding(
            core_problem={
                "type": structure.problem_type,
                "equation": structure.canonical_form,
                "physical_meaning": "find equilibrium stress/displacement field",
            },
            approximations_applied=self.extract_approximations(params),
            solution_method={
                "type": "finite_element_method",
                "solver": "Newton-Raphson" if params.get("nonlinear") else "direct",
            },
            mathematical_hierarchy=[
                {"level": "physical", "description": "conservation laws (momentum, energy)"},
                {"level": "approximation_1", "description": "continuum hypothesis"},
                {"level": "approximation_2", "description": "constitutive law"},
                {"level": "discretization", "description": "finite element mesh"},
                {"level": "numerical", "description": "linear/nonlinear solver"},
            ],
        )
    
    def extract_precision_metadata(self, params: Dict[str, Any]) -> Dict[str, PrecisionMetadata]:
        """Extract precision metadata."""
        metadata = {}
        for param_name in ["element_type", "mesh_size", "material_model", "analysis_type"]:
            if param_name in params:
                metadata[param_name] = PrecisionMetadata(
                    extraction_confidence=1.0,
                    source="direct_assignment",
                )
        return metadata


def extract_abaqus_mathematical_precision(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for Abaqus precision extraction."""
    extractor = AbaqusMathematicalPrecisionExtractor()
    return extractor.extract(params).to_dict()
