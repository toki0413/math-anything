"""GROMACS Mathematical Precision Extractor.

Extracts precise mathematical structures from GROMACS input files,
enabling LLMs to understand the mathematical essence of biomolecular MD.

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


class GromacsMathematicalPrecisionExtractor(BasePrecisionExtractor):
    """Extract precise mathematical structures from GROMACS inputs.
    
    GROMACS specializes in biomolecular molecular dynamics.
    """
    
    def extract_mathematical_structure(self, params: Dict[str, Any]) -> MathematicalStructure:
        """Extract the mathematical structure of MD problem."""
        integrator = params.get("integrator", "md")
        
        if integrator in ["md", "md-vv"]:
            problem_type = "initial_value_ode"
            canonical_form = "m_i d²r_i/dt² = F_i(r)"
        elif integrator == "sd":
            problem_type = "stochastic_ode"
            canonical_form = "m_i d²r_i/dt² = F_i + γv_i + R_i(t)"
        elif integrator == "minimize":
            problem_type = "optimization"
            canonical_form = "min E(r) w.r.t. r"
        else:
            problem_type = "initial_value_ode"
            canonical_form = "m_i d²r_i/dt² = F_i(r)"
        
        return MathematicalStructure(
            problem_type=problem_type,
            canonical_form=canonical_form,
            properties={
                "hamiltonian": integrator in ["md", "md-vv"],
                "symplectic": integrator in ["md", "md-vv"],
                "reversible": True,
            },
            dimension=3 * params.get("n_atoms", 1),
            function_space="ℝ^{3N}",
        )
    
    def extract_variable_dependencies(self, params: Dict[str, Any]) -> List[VariableDependency]:
        """Extract variable dependencies."""
        dependencies = [
            VariableDependency(
                relation="F_i = -∂V/∂r_i",
                depends_on=["r"],
                circular=False,
                mathematical_form="force_from_potential",
                physical_interpretation="forces from empirical force field",
            ),
            VariableDependency(
                relation="V = V_bond + V_angle + V_dihedral + V_LJ + V_coulomb",
                depends_on=["r"],
                circular=False,
                mathematical_form="total_potential_energy",
                physical_interpretation="bonded + nonbonded interactions",
            ),
        ]
        
        if params.get("constraints"):
            dependencies.append(VariableDependency(
                relation="constraint: |r_i - r_j| = d_ij",
                depends_on=["r"],
                circular=True,
                mathematical_form="holonomic_constraint",
                physical_interpretation="SHAKE/LINCS constraint algorithm",
            ))
        
        return dependencies
    
    def extract_discretization_scheme(self, params: Dict[str, Any]) -> DiscretizationScheme:
        """Extract time discretization scheme."""
        dt = params.get("dt", 0.002)
        integrator = params.get("integrator", "md")
        
        integrator_info = {
            "md": ("leap_frog", "r(t+Δt) = r(t) + v(t+Δt/2)Δt"),
            "md-vv": ("velocity_verlet", "symplectic, O(Δt²)"),
            "sd": ("stochastic_dynamics", "Langevin dynamics"),
            "minimize": ("steepest_descent", "gradient descent"),
        }
        
        method, form = integrator_info.get(integrator, ("leap_frog", "standard MD"))
        
        return DiscretizationScheme(
            method=method,
            mathematical_meaning=form,
            parameters={
                "timestep": f"{dt} ps",
                "n_steps": params.get("nsteps", 0),
            },
            basis_type="taylor_expansion",
            convergence_order="O(Δt²)",
        )
    
    def extract_solution_strategy(self, params: Dict[str, Any]) -> SolutionStrategy:
        """Extract solution strategy."""
        dt = params.get("dt", 0.002)
        
        return SolutionStrategy(
            method="time_integration",
            mathematical_form="propagate r(t) → r(t+Δt)",
            convergence_criterion="N/A (time evolution)",
            iteration_type="explicit_timestep",
            stability_requirement=f"Δt < τ_fastest_mode, typically {dt} ps",
        )
    
    def extract_approximations(self, params: Dict[str, Any]) -> List[Approximation]:
        """Extract approximations."""
        approximations = [
            Approximation(
                name="classical_mechanics",
                mathematical_form="F = ma",
                consequence="no quantum nuclear effects",
                affected_quantities=["proton_transfer", "tunneling"],
                theoretical_basis="classical limit",
            ),
            Approximation(
                name="empirical_force_field",
                mathematical_form=params.get("forcefield", "unknown"),
                consequence="interactions approximated by fitted parameters",
                affected_quantities=["energies", "forces", "structures"],
                theoretical_basis="empirical fitting",
            ),
            Approximation(
                name="nonbonded_cutoff",
                mathematical_form=f"V(r) = 0 for r > {params.get('rvdw', 1.0)} nm",
                consequence="long-range interactions truncated or approximated",
                affected_quantities=["electrostatics", "dispersion"],
                theoretical_basis="computational efficiency",
            ),
        ]
        
        if params.get("pbc"):
            approximations.append(Approximation(
                name="periodic_boundary",
                mathematical_form="infinite periodic replication",
                consequence="bulk behavior, no surface effects",
                affected_quantities=["surface_phenomena"],
                theoretical_basis="thermodynamic limit",
            ))
        
        if params.get("constraints"):
            approximations.append(Approximation(
                name="bond_constraints",
                mathematical_form="SHAKE/LINCS algorithm",
                consequence="bonds frozen, allows larger timestep",
                affected_quantities=["fast_vibrations"],
                theoretical_basis="timescale separation",
            ))
        
        return approximations
    
    def extract_mathematical_decoding(self, params: Dict[str, Any]) -> MathematicalDecoding:
        """Extract complete decoding."""
        dt = params.get("dt", 0.002)
        nsteps = params.get("nsteps", 0)
        
        return MathematicalDecoding(
            core_problem={
                "type": "initial_value_ode",
                "equation": "m_i d²r_i/dt² = F_i(r)",
                "physical_meaning": "evolve biomolecular system under force field",
            },
            approximations_applied=self.extract_approximations(params),
            solution_method={
                "type": "time_integration",
                "integrator": params.get("integrator", "md"),
                "timestep": f"{dt} ps",
                "total_time": f"{nsteps * dt} ps",
            },
            mathematical_hierarchy=[
                {"level": "physical", "description": "quantum many-body Schrödinger"},
                {"level": "approximation_1", "description": "Born-Oppenheimer"},
                {"level": "approximation_2", "description": "classical nuclei"},
                {"level": "approximation_3", "description": "empirical force field"},
                {"level": "discretization", "description": "cutoff + PBC"},
                {"level": "numerical", "description": "timestep integration"},
            ],
        )
    
    def extract_precision_metadata(self, params: Dict[str, Any]) -> Dict[str, PrecisionMetadata]:
        """Extract precision metadata."""
        metadata = {}
        for key in ["dt", "nsteps", "integrator", "forcefield", "rvdw"]:
            if key in params:
                metadata[key] = PrecisionMetadata(
                    extraction_confidence=1.0,
                    source="direct_assignment",
                )
        return metadata


def extract_gromacs_mathematical_precision(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for GROMACS precision extraction."""
    extractor = GromacsMathematicalPrecisionExtractor()
    return extractor.extract(params).to_dict()
