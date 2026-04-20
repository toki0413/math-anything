"""Ansys Mathematical Precision Extractor.

Extracts precise mathematical structures from Ansys APDL input files,
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


class AnsysMathematicalPrecisionExtractor(BasePrecisionExtractor):
    """Extract precise mathematical structures from Ansys inputs."""
    
    def extract_mathematical_structure(self, params: Dict[str, Any]) -> MathematicalStructure:
        """Extract the mathematical structure."""
        analysis_type = params.get("analysis_type", "static")
        
        type_map = {
            "static": ("boundary_value_problem", "∇·σ + f = 0"),
            "modal": ("eigenvalue_problem", "Kφ = λMφ"),
            "thermal": ("parabolic_pde", "ρc ∂T/∂t - ∇·(k∇T) = Q"),
            "harmonic": ("frequency_domain", "(-ω²M + iωC + K)u = F"),
            "transient": ("initial_boundary_value_problem", "Mü + Cu̇ + Ku = F(t)"),
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
        """Extract variable dependencies."""
        dependencies = [
            VariableDependency(
                relation="σ = D·ε",
                depends_on=["ε"],
                circular=False,
                mathematical_form="stress_strain_relation",
                physical_interpretation="stress from strain via stiffness matrix",
            ),
        ]
        
        if params.get("analysis_type") == "modal":
            dependencies.append(VariableDependency(
                relation="Kφ = λMφ",
                depends_on=["K", "M"],
                circular=False,
                mathematical_form="generalized_eigenvalue",
                physical_interpretation="natural frequencies and mode shapes",
            ))
        
        return dependencies
    
    def extract_discretization_scheme(self, params: Dict[str, Any]) -> DiscretizationScheme:
        """Extract discretization scheme."""
        return DiscretizationScheme(
            method="finite_element",
            mathematical_meaning="u(x) ≈ Σᵢ Nᵢ(x)uᵢ",
            parameters={
                "element_type": params.get("element_type", "SOLID185"),
                "mesh_size": params.get("esize", "auto"),
            },
            basis_type="polynomial_shape_functions",
            completeness="C⁰ continuous",
        )
    
    def extract_solution_strategy(self, params: Dict[str, Any]) -> SolutionStrategy:
        """Extract solution strategy."""
        analysis_type = params.get("analysis_type", "static")
        
        strategy_map = {
            "static": ("direct_solver", "Ku = F"),
            "modal": ("eigenvalue_solver", "Kφ = λMφ (Lanczos/Subspace)"),
            "thermal": ("iterative_solver", "K(T)T = Q"),
            "harmonic": ("complex_solver", "(-ω²M + iωC + K)u = F"),
            "transient": ("time_integration", "Newmark/HTT method"),
        }
        
        method, form = strategy_map.get(analysis_type, ("direct_solver", "Ku = F"))
        
        return SolutionStrategy(
            method=method,
            mathematical_form=form,
            convergence_criterion=params.get("tolerance", "default"),
        )
    
    def extract_approximations(self, params: Dict[str, Any]) -> List[Approximation]:
        """Extract approximations."""
        approximations = [
            Approximation(
                name="continuum_mechanics",
                mathematical_form="material as continuous medium",
                consequence="no microstructure effects",
                affected_quantities=["stress", "strain"],
            ),
            Approximation(
                name="finite_element_discretization",
                mathematical_form=f"{params.get('element_type', 'SOLID185')} elements",
                consequence="discrete approximation of continuous field",
                affected_quantities=["accuracy", "convergence"],
            ),
        ]
        
        if params.get("material_model") == "isotropic_elastic":
            approximations.append(Approximation(
                name="isotropic_elasticity",
                mathematical_form="σ = E/(1+ν) [ε + ν/(1-2ν) tr(ε)I]",
                consequence="simplified material behavior",
                affected_quantities=["anisotropic_response"],
            ))
        
        return approximations
    
    def extract_mathematical_decoding(self, params: Dict[str, Any]) -> MathematicalDecoding:
        """Extract complete decoding."""
        structure = self.extract_mathematical_structure(params)
        
        return MathematicalDecoding(
            core_problem={
                "type": structure.problem_type,
                "equation": structure.canonical_form,
                "physical_meaning": "structural/thermal analysis",
            },
            approximations_applied=self.extract_approximations(params),
            solution_method={
                "type": "finite_element_method",
                "solver": params.get("solver_type", "sparse_direct"),
            },
            mathematical_hierarchy=[
                {"level": "physical", "description": "conservation laws"},
                {"level": "approximation_1", "description": "continuum hypothesis"},
                {"level": "approximation_2", "description": "material model"},
                {"level": "discretization", "description": "FE mesh"},
                {"level": "numerical", "description": "linear algebra solver"},
            ],
        )
    
    def extract_precision_metadata(self, params: Dict[str, Any]) -> Dict[str, PrecisionMetadata]:
        """Extract precision metadata."""
        metadata = {}
        for key in ["EX", "PRXY", "DENS", "element_type", "analysis_type"]:
            if key in params:
                metadata[key] = PrecisionMetadata(
                    extraction_confidence=1.0,
                    source="direct_assignment",
                )
        return metadata


def extract_ansys_mathematical_precision(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for Ansys precision extraction."""
    extractor = AnsysMathematicalPrecisionExtractor()
    return extractor.extract(params).to_dict()
