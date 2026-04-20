"""Multiwfn Mathematical Precision Extractor.

Extracts precise mathematical structures from Multiwfn wavefunction analysis,
enabling LLMs to understand quantum chemical analysis methods.

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


class MultiwfnMathematicalPrecisionExtractor(BasePrecisionExtractor):
    """Extract precise mathematical structures from Multiwfn inputs.
    
    Multiwfn performs wavefunction analysis on quantum chemical results.
    """
    
    def extract_mathematical_structure(self, params: Dict[str, Any]) -> MathematicalStructure:
        """Extract the mathematical structure of analysis."""
        analysis_type = params.get("analysis_type", "electron_density")
        
        analysis_map = {
            "electron_density": ("function_evaluation", "ρ(r) = Σ|ψᵢ(r)|²"),
            "molecular_orbital": ("eigenfunction_display", "ψᵢ(r) = Σcᵢμφμ(r)"),
            "esp": ("electrostatic_potential", "V(r) = ΣZ_A/|r-R_A| - ∫ρ(r')/|r-r'|dr'"),
            "aim": ("topological_analysis", "∇ρ(r) = 0, find critical points"),
            "fukui": ("reactivity_descriptor", "f⁺, f⁻, f⁰ functions"),
            "nbo": ("orbital_analysis", "natural bond orbital decomposition"),
        }
        
        problem_type, canonical_form = analysis_map.get(analysis_type,
            ("function_evaluation", "property calculation"))
        
        return MathematicalStructure(
            problem_type=problem_type,
            canonical_form=canonical_form,
            properties={
                "input_type": "wavefunction_or_density",
                "basis": params.get("basis_type", "unknown"),
            },
            dimension=3,
            function_space="L²(ℝ³)",
        )
    
    def extract_variable_dependencies(self, params: Dict[str, Any]) -> List[VariableDependency]:
        """Extract variable dependencies."""
        dependencies = [
            VariableDependency(
                relation="ρ(r) = Σᵢ nᵢ|ψᵢ(r)|²",
                depends_on=["ψ", "occupations"],
                circular=False,
                mathematical_form="density_construction",
                physical_interpretation="electron density from orbitals",
            ),
        ]
        
        if params.get("analysis_type") == "aim":
            dependencies.append(VariableDependency(
                relation="∇²ρ(r) = 0 at critical points",
                depends_on=["ρ"],
                circular=False,
                mathematical_form="topological_analysis",
                physical_interpretation="find bond critical points",
            ))
        
        return dependencies
    
    def extract_discretization_scheme(self, params: Dict[str, Any]) -> DiscretizationScheme:
        """Extract discretization scheme."""
        return DiscretizationScheme(
            method="numerical_integration_grid",
            mathematical_meaning="∫f(r)dr ≈ Σwᵢf(rᵢ)",
            parameters={
                "grid_type": params.get("grid_type", "becke"),
                "grid_quality": params.get("grid_points", "medium"),
            },
            basis_type="numerical_grid",
            completeness="depends on grid density",
        )
    
    def extract_solution_strategy(self, params: Dict[str, Any]) -> SolutionStrategy:
        """Extract solution strategy."""
        return SolutionStrategy(
            method="post_processing",
            mathematical_form=f"compute {params.get('analysis_type', 'electron_density')} from wavefunction",
            convergence_criterion="N/A (analysis, not iteration)",
            iteration_type="none",
        )
    
    def extract_approximations(self, params: Dict[str, Any]) -> List[Approximation]:
        """Extract approximations."""
        approximations = [
            Approximation(
                name="wavefunction_quality",
                mathematical_form=params.get("method", "unknown"),
                consequence="analysis quality limited by input wavefunction quality",
                affected_quantities=["all_properties"],
                theoretical_basis="depends on quantum chemical method used",
            ),
            Approximation(
                name="numerical_grid",
                mathematical_form=f"{params.get('grid_type', 'becke')} grid",
                consequence="integral accuracy depends on grid density",
                affected_quantities=["integrated_properties"],
                theoretical_basis="numerical quadrature",
            ),
        ]
        
        if params.get("basis_type"):
            approximations.append(Approximation(
                name="basis_set",
                mathematical_form=params.get("basis_type", "unknown"),
                consequence="basis set completeness affects orbital quality",
                affected_quantities=["orbital_shapes", "densities"],
                theoretical_basis="finite basis expansion",
            ))
        
        return approximations
    
    def extract_mathematical_decoding(self, params: Dict[str, Any]) -> MathematicalDecoding:
        """Extract complete decoding."""
        structure = self.extract_mathematical_structure(params)
        
        return MathematicalDecoding(
            core_problem={
                "type": structure.problem_type,
                "equation": structure.canonical_form,
                "physical_meaning": "analyze quantum chemical wavefunction",
            },
            approximations_applied=self.extract_approximations(params),
            solution_method={
                "type": "numerical_evaluation",
                "grid": params.get("grid_type", "becke"),
            },
            mathematical_hierarchy=[
                {"level": "input", "description": "wavefunction from QM calculation"},
                {"level": "basis", "description": "basis set representation"},
                {"level": "analysis", "description": "property computation"},
                {"level": "numerical", "description": "grid integration"},
            ],
        )
    
    def extract_precision_metadata(self, params: Dict[str, Any]) -> Dict[str, PrecisionMetadata]:
        """Extract precision metadata."""
        metadata = {}
        for key in ["analysis_type", "method", "basis_type", "grid_type"]:
            if key in params:
                metadata[key] = PrecisionMetadata(
                    extraction_confidence=1.0,
                    source="direct_assignment",
                )
        return metadata


def extract_multiwfn_mathematical_precision(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for Multiwfn precision extraction."""
    extractor = MultiwfnMathematicalPrecisionExtractor()
    return extractor.extract(params).to_dict()
