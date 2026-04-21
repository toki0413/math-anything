"""VASP Mathematical Precision Extractor.

Extracts precise mathematical structures from VASP input files,
enabling LLMs to understand the mathematical essence of DFT calculations.

Design Principles:
- Zero intrusion: Only observe, never modify
- Zero judgment: Describe what is, not what should be
"""

from typing import Any, Dict, List

from .precision import (
    Approximation,
    BasePrecisionExtractor,
    DiscretizationScheme,
    EnhancedMathSchema,
    MathematicalDecoding,
    MathematicalStructure,
    PrecisionMetadata,
    SolutionStrategy,
    VariableDependency,
)


class VaspMathematicalPrecisionExtractor(BasePrecisionExtractor):
    """Extract precise mathematical structures from VASP inputs.

    Translates VASP parameters into their mathematical significance,
    enabling "decryption" of what the calculation is mathematically doing.
    """

    def extract_mathematical_structure(
        self, params: Dict[str, Any]
    ) -> MathematicalStructure:
        """Extract the mathematical structure of the problem being solved.

        VASP solves the Kohn-Sham equations, which are nonlinear eigenvalue problems.
        """
        return MathematicalStructure(
            problem_type="nonlinear_eigenvalue",
            canonical_form="H[n]ψ = εψ",
            properties={
                "operator_type": "self_adjoint",
                "nonlinearity_source": "density_dependent_potential",
                "variational": True,
            },
            dimension=3,
            function_space="L²(ℝ³)",
        )

    def extract_variable_dependencies(
        self, params: Dict[str, Any]
    ) -> List[VariableDependency]:
        """Extract variable dependencies that require self-consistent iteration.

        The key circular dependency in DFT:
        V_eff depends on n, which depends on ψ, which depends on V_eff
        """
        return [
            VariableDependency(
                relation="V_eff = V_ext + V_H[n] + V_xc[n]",
                depends_on=["n"],
                circular=True,
                mathematical_form="V_eff[n]ψ = εψ",
                physical_interpretation="effective potential depends on electron density",
            ),
            VariableDependency(
                relation="n(r) = Σᵢ fᵢ|ψᵢ(r)|²",
                depends_on=["ψ"],
                circular=True,
                mathematical_form="electron density from occupied orbitals",
                physical_interpretation="density constructed from wavefunctions",
            ),
        ]

    def extract_discretization_scheme(
        self, params: Dict[str, Any]
    ) -> DiscretizationScheme:
        """Extract the discretization scheme from VASP parameters.

        VASP uses plane wave basis with energy cutoff.
        """
        encut = params.get("ENCUT", 520)

        return DiscretizationScheme(
            method="plane_wave_expansion",
            mathematical_meaning="ψ(r) = Σ_G c_G exp(iG·r), where |G|²/2 < E_cut",
            parameters={
                "cutoff_energy": f"{encut} eV",
                "parameter_name": "ENCUT",
            },
            basis_type="plane_wave",
            completeness="controlled_by_cutoff",
            convergence_order="exponential_for_smooth_functions",
        )

    def extract_solution_strategy(self, params: Dict[str, Any]) -> SolutionStrategy:
        """Extract the solution strategy from VASP parameters.

        VASP uses self-consistent field (SCF) iteration.
        """
        ediff = params.get("EDIFF", 1e-4)
        nelm = params.get("NELM", 60)

        return SolutionStrategy(
            method="self_consistent_field",
            mathematical_form="n^{k+1} = F[n^k], iterate until |E^{k+1} - E^k| < ε",
            convergence_criterion=f"|E_n - E_{{n-1}}| < {ediff} eV",
            iteration_type="fixed_point_iteration",
            stability_requirement=f"mixing required for convergence, max {nelm} iterations",
        )

    def extract_approximations(self, params: Dict[str, Any]) -> List[Approximation]:
        """Extract the approximations made in the calculation.

        Lists what approximations are applied, without judging their quality.
        """
        approximations = []

        encut = params.get("ENCUT", 520)
        approximations.append(
            Approximation(
                name="plane_wave_truncation",
                mathematical_form=f"|G|²/2 < {encut} eV",
                consequence="wavefunction expansion limited, affects high-energy states",
                affected_quantities=["wavefunction_accuracy", "total_energy"],
                theoretical_basis="finite basis set approximation",
            )
        )

        ismear = params.get("ISMEAR", 1)
        sigma = params.get("SIGMA", 0.2)
        if ismear >= 0:
            approximations.append(
                Approximation(
                    name="smearing",
                    mathematical_form=f"Gaussian/MP smearing with width {sigma} eV",
                    consequence="occupations smoothed, enables metallic systems",
                    affected_quantities=["fermi_level", "band_energy"],
                    theoretical_basis="finite temperature DFT",
                )
            )
        elif ismear == -5:
            approximations.append(
                Approximation(
                    name="tetrahedron_method",
                    mathematical_form="Blöchl tetrahedron integration",
                    consequence="accurate DOS, not suitable for relaxation",
                    affected_quantities=["density_of_states", "total_energy"],
                    theoretical_basis="tetrahedron integration",
                )
            )

        gga = params.get("GGA", "PE")
        approximations.append(
            Approximation(
                name="exchange_correlation_approximation",
                mathematical_form=f"GGA-{gga} functional",
                consequence="approximate treatment of electron-electron interactions",
                affected_quantities=[
                    "total_energy",
                    "bond_lengths",
                    "reaction_barriers",
                ],
                theoretical_basis="Kohn-Sham DFT with approximate xc functional",
            )
        )

        if params.get("kpoints"):
            grid = params.get("kpoints", {}).get("grid", [1, 1, 1])
            approximations.append(
                Approximation(
                    name="k_point_sampling",
                    mathematical_form=f"Monkhorst-Pack grid {grid}",
                    consequence="Brillouin zone discretized, affects metallic systems",
                    affected_quantities=["fermi_surface", "total_energy"],
                    theoretical_basis="k-point integration approximation",
                )
            )

        return approximations

    def extract_mathematical_decoding(
        self, params: Dict[str, Any]
    ) -> MathematicalDecoding:
        """Extract a complete mathematical decoding of the VASP setup.

        Translates all parameters into their mathematical meaning.
        """
        return MathematicalDecoding(
            core_problem={
                "type": "nonlinear_eigenvalue_problem",
                "equation": "H[n]ψ = εψ",
                "physical_meaning": "solve for ground state electronic structure",
            },
            approximations_applied=self.extract_approximations(params),
            solution_method={
                "type": "self_consistent_iteration",
                "convergence_target": f"|E_n - E_{{n-1}}| < {params.get('EDIFF', 1e-4)} eV",
                "max_iterations": params.get("NELM", 60),
            },
            mathematical_hierarchy=[
                {"level": "physical", "description": "many-body Schrödinger equation"},
                {
                    "level": "approximation_1",
                    "description": "Born-Oppenheimer approximation",
                },
                {"level": "approximation_2", "description": "Kohn-Sham DFT"},
                {
                    "level": "approximation_3",
                    "description": "exchange-correlation functional",
                },
                {
                    "level": "discretization",
                    "description": "plane wave basis truncation",
                },
                {"level": "numerical", "description": "SCF iteration with mixing"},
            ],
        )

    def extract_precision_metadata(
        self, params: Dict[str, Any]
    ) -> Dict[str, PrecisionMetadata]:
        """Extract precision metadata for each parameter.

        Expresses confidence and source of each extracted value.
        """
        metadata = {}

        for param_name in params:
            if param_name in ["ENCUT", "EDIFF", "ISMEAR", "SIGMA", "NELM", "ALGO"]:
                metadata[param_name] = PrecisionMetadata(
                    extraction_confidence=1.0,
                    source="direct_assignment",
                    notes=["explicitly set in INCAR"],
                )
            elif param_name in ["kpoints", "structure"]:
                metadata[param_name] = PrecisionMetadata(
                    extraction_confidence=1.0,
                    source="file_parsing",
                    notes=["extracted from auxiliary file"],
                )
            else:
                metadata[param_name] = PrecisionMetadata(
                    extraction_confidence=0.8,
                    source="default_value",
                    notes=["using VASP default, not explicitly set"],
                )

        return metadata


def extract_vasp_mathematical_precision(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to extract mathematical precision from VASP params.

    Args:
        params: Dictionary of VASP parameters (from INCAR, etc.)

    Returns:
        Dictionary with precise mathematical structure representation
    """
    extractor = VaspMathematicalPrecisionExtractor()
    return extractor.extract(params).to_dict()
