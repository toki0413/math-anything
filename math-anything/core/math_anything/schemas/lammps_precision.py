"""LAMMPS Mathematical Precision Extractor.

Extracts precise mathematical structures from LAMMPS input files,
enabling LLMs to understand the mathematical essence of MD simulations.

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


class LammpsMathematicalPrecisionExtractor(BasePrecisionExtractor):
    """Extract precise mathematical structures from LAMMPS inputs.

    Translates LAMMPS parameters into their mathematical significance,
    enabling "decryption" of what the simulation is mathematically doing.
    """

    def extract_mathematical_structure(
        self, params: Dict[str, Any]
    ) -> MathematicalStructure:
        """Extract the mathematical structure of the MD problem.

        LAMMPS solves Newton's equations of motion for many-body systems.
        Different pair_styles imply different force field functional forms.
        """
        ensemble = params.get("ensemble", "NVE")
        pair_style = params.get("pair_style", "")

        FORCE_FORM = {
            "lj/cut": "F_i = Σ_j 24ε[(2(σ/r_ij)^12 - (σ/r_ij)^6)] r̂_ij / r_ij",
            "lj/cut/coul/long": "F_i = Σ_j [24ε(2(σ/r_ij)^12 - (σ/r_ij)^6)/r_ij + qᵢqⱼ/(4πε₀r_ij²)] r̂_ij",
            "eam": "F_i = -∂[Σⱼ Fᵢ(ρᵢ) + ½Σⱼ φᵢⱼ(r_ij)]/∂r_i",
            "meam": "F_i = -∂[Σⱼ Fᵢ(ρᵢ, tᵢ¹, tᵢ², tᵢ³) + ½Σⱼ φᵢⱼ(r_ij)]/∂r_i",
            "tersoff": "F_i = -∂[½Σⱼ f_c(r_ij)(f_R(r_ij) + b_ij f_A(r_ij))]/∂r_i",
            "sw": "F_i = -∂[Σⱼ φ₂(r_ij) + Σⱼ<k φ₃(r_ij, r_ik, θ_ijk)]/∂r_i",
            "reax/c": "F_i = -∂[E_bond+E_over+E_val+E_pen+E_conj+E_vdW+E_Coul]/∂r_i",
            "buck": "F_i = Σ_j [A_ijexp(-r_ij/ρ_ij)/ρ_ij - 6C_ij/r_ij⁷] r̂_ij",
            "morse": "F_i = Σ_j 2aDₑ[1-exp(-a(r_ij-rₑ))]exp(-a(r_ij-rₑ)) r̂_ij",
            "dpd": "F_i = Σ_j [F_ij^C + F_ij^D + F_ij^R], F_ij^D = -γw_D(r_ij)(v_ij·r̂_ij)r̂_ij",
            "gran/hooke/history": "F_i = Σ_j [k_nδ_ij n̂ - γ_n v_n,ij] (Hertz-Mindlin contact)",
        }

        if ensemble in ["NVT", "NPT"]:
            problem_type = "stochastic_ode"
            canonical_form = f"m_i d²r_i/dt² = F_i + γv_i + R_i(t)"
        else:
            problem_type = "initial_value_ode"
            if pair_style and pair_style in FORCE_FORM:
                canonical_form = FORCE_FORM[pair_style]
            else:
                for key in FORCE_FORM:
                    if key in (pair_style or ""):
                        canonical_form = FORCE_FORM[key]
                        break
                else:
                    canonical_form = "m_i d²r_i/dt² = F_i(r_1, ..., r_N)"

        is_pairwise = any(k in (pair_style or "") for k in ["lj/cut", "buck", "morse", "dpd"])
        is_manybody = any(k in (pair_style or "") for k in ["eam", "meam", "tersoff", "sw"])
        is_reactive = "reax" in (pair_style or "")

        return MathematicalStructure(
            problem_type=problem_type,
            canonical_form=canonical_form,
            properties={
                "hamiltonian": ensemble == "NVE",
                "symplectic": True,
                "reversible": True,
                "pairwise": is_pairwise,
                "manybody": is_manybody,
                "reactive": is_reactive,
            },
            dimension=3 * params.get("n_atoms", 1),
            function_space="ℝ^{3N}",
        )

    def extract_variable_dependencies(
        self, params: Dict[str, Any]
    ) -> List[VariableDependency]:
        """Extract variable dependencies in MD.

        Forces depend on positions, creating the coupled ODE system.
        """
        dependencies = [
            VariableDependency(
                relation="F_i = -∂V/∂r_i",
                depends_on=["r"],
                circular=False,
                mathematical_form="F = -∇V(r)",
                physical_interpretation="forces derived from potential energy",
            ),
            VariableDependency(
                relation="V = Σᵢⱼ V_pair(r_ij) + Σᵢ V_ext(r_i)",
                depends_on=["r"],
                circular=False,
                mathematical_form="total potential energy",
                physical_interpretation="potential from pairwise and external contributions",
            ),
        ]

        if params.get("ensemble") in ["NVT", "NPT"]:
            dependencies.append(
                VariableDependency(
                    relation="T_target = const",
                    depends_on=["thermostat"],
                    circular=False,
                    mathematical_form="thermostat coupling",
                    physical_interpretation="temperature control via thermostat",
                )
            )

        return dependencies

    def extract_discretization_scheme(
        self, params: Dict[str, Any]
    ) -> DiscretizationScheme:
        """Extract the time discretization scheme.

        LAMMPS uses various integrators (Verlet, velocity-Verlet, etc.)
        """
        dt = params.get("dt", 0.001)
        integrator = params.get("integrator", "velocity_verlet")

        return DiscretizationScheme(
            method="finite_difference_time_integration",
            mathematical_meaning="r(t+Δt) = 2r(t) - r(t-Δt) + (F/m)Δt²",
            parameters={
                "timestep": f"{dt} ps",
                "integrator": integrator,
            },
            basis_type="taylor_expansion",
            completeness="energy_conserving_for_symplectic",
            convergence_order="O(Δt²)",
        )

    def extract_solution_strategy(self, params: Dict[str, Any]) -> SolutionStrategy:
        """Extract the solution strategy.

        MD uses time integration, not iterative solution.
        """
        dt = params.get("dt", 0.001)

        return SolutionStrategy(
            method="time_integration",
            mathematical_form="propagate r(t) → r(t+Δt) using velocity-Verlet",
            convergence_criterion="N/A (time evolution, not iterative)",
            iteration_type="explicit_timestep",
            stability_requirement="Δt < τ_characteristic, typically Δt < 0.01 τ_vibration",
        )

    def extract_approximations(self, params: Dict[str, Any]) -> List[Approximation]:
        """Extract the approximations in MD.

        Classical mechanics, force fields, cutoffs, etc.
        """
        approximations = []

        approximations.append(
            Approximation(
                name="classical_mechanics",
                mathematical_form="F = ma (Newton's second law)",
                consequence="no quantum effects, valid when kT >> ℏω",
                affected_quantities=["dynamics", "energy_transfer"],
                theoretical_basis="classical limit of quantum mechanics",
            )
        )

        pair_style = params.get("pair_style", "lj/cut")
        cutoff = params.get("pair_cutoff", 10.0)
        approximations.append(
            Approximation(
                name="force_field_approximation",
                mathematical_form=f"{pair_style} with cutoff {cutoff} Å",
                consequence="interatomic interactions approximated by empirical potential",
                affected_quantities=["forces", "energies", "structures"],
                theoretical_basis="empirical fitting or ab initio derivation",
            )
        )

        if cutoff:
            approximations.append(
                Approximation(
                    name="interaction_cutoff",
                    mathematical_form=f"V(r) = 0 for r > {cutoff} Å",
                    consequence="long-range interactions truncated",
                    affected_quantities=["total_energy", "pressure"],
                    theoretical_basis="computational efficiency",
                )
            )

        dt = params.get("dt", 0.001)
        approximations.append(
            Approximation(
                name="time_discretization",
                mathematical_form=f"Δt = {dt} ps",
                consequence="continuous time approximated by discrete steps",
                affected_quantities=["energy_conservation", "trajectory_accuracy"],
                theoretical_basis="numerical integration",
            )
        )

        if params.get("ensemble") == "NVT":
            approximations.append(
                Approximation(
                    name="thermostat",
                    mathematical_form="Nosé-Hoover or Langevin thermostat",
                    consequence="system coupled to heat bath",
                    affected_quantities=["temperature", "energy_fluctuations"],
                    theoretical_basis="statistical mechanics",
                )
            )

        return approximations

    def extract_mathematical_decoding(
        self, params: Dict[str, Any]
    ) -> MathematicalDecoding:
        """Extract complete mathematical decoding of LAMMPS setup."""
        dt = params.get("dt", 0.001)
        run_steps = params.get("run", 10000)

        return MathematicalDecoding(
            core_problem={
                "type": "initial_value_ode",
                "equation": "m_i d²r_i/dt² = F_i(r)",
                "physical_meaning": "evolve atomic positions under interatomic forces",
            },
            approximations_applied=self.extract_approximations(params),
            solution_method={
                "type": "time_integration",
                "integrator": "velocity-Verlet",
                "timestep": f"{dt} ps",
                "total_time": f"{run_steps * dt} ps",
            },
            mathematical_hierarchy=[
                {"level": "physical", "description": "quantum many-body Schrödinger"},
                {
                    "level": "approximation_1",
                    "description": "Born-Oppenheimer approximation",
                },
                {
                    "level": "approximation_2",
                    "description": "classical mechanics (no quantum nuclei)",
                },
                {"level": "approximation_3", "description": "empirical force field"},
                {"level": "discretization", "description": "finite cutoff radius"},
                {"level": "numerical", "description": "finite timestep integration"},
            ],
        )

    def extract_precision_metadata(
        self, params: Dict[str, Any]
    ) -> Dict[str, PrecisionMetadata]:
        """Extract precision metadata for each parameter."""
        metadata = {}

        for param_name in params:
            if param_name in ["dt", "run", "pair_style", "pair_cutoff"]:
                metadata[param_name] = PrecisionMetadata(
                    extraction_confidence=1.0,
                    source="direct_assignment",
                    notes=["explicitly set in input script"],
                )
            elif param_name in ["ensemble", "integrator"]:
                metadata[param_name] = PrecisionMetadata(
                    extraction_confidence=0.9,
                    source="inferred_from_commands",
                    notes=["inferred from fix and run commands"],
                )
            else:
                metadata[param_name] = PrecisionMetadata(
                    extraction_confidence=0.7,
                    source="default_value",
                    notes=["using LAMMPS default"],
                )

        return metadata


def extract_lammps_mathematical_precision(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to extract mathematical precision from LAMMPS params."""
    extractor = LammpsMathematicalPrecisionExtractor()
    return extractor.extract(params).to_dict()
