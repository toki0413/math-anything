"""MathAdvisor - recommends mathematical analysis tools by simulation engine.

Maps each supported simulation engine (VASP, LAMMPS, Abaqus, etc.) to its
relevant math disciplines, recommended tool pipeline, workflow hint, and
keywords so that downstream agents can auto-configure the right analysis.

Disciplines are tagged as implemented or planned so the frontend can
distinguish actionable recommendations from roadmap items.
"""

from __future__ import annotations

from typing import Any


_GENERIC_ADVISORY: dict[str, Any] = {
    "tools": ["extract", "validate", "verify"],
    "math_disciplines": [
        "Applied Mathematics (domain-specific modeling, numerical methods)",
        "Formal Verification (constraint checking, consistency analysis)",
    ],
    "implemented_disciplines": [
        "Formal Verification (constraint checking, consistency analysis)",
    ],
    "planned_disciplines": [
        "Applied Mathematics (domain-specific modeling, numerical methods)",
    ],
    "workflow_hint": "Extract → Validate constraints → Verify mathematical consistency",
    "keywords": ["simulation", "numerical", "convergence", "consistency"],
}

DISCIPLINE_STATUS: dict[str, bool] = {
    "differential_geometry": True,
    "spectral_analysis": True,
    "group_theory": True,
    "dynamical_systems": True,
    "statistical_mechanics": True,
    "emergence_theory": True,
    "information_theory": True,
    "formal_verification": True,
    "functional_analysis": False,
    "optimization": False,
    "variational_calculus": False,
    "perturbation_theory": False,
    "stochastic_processes": True,
    "topology": True,
    "quantum_chemistry": False,
    "fluid_mechanics": False,
}


def _classify_disciplines(disciplines: list[str]) -> tuple[list[str], list[str]]:
    """Split discipline descriptions into implemented and planned lists."""
    implemented = []
    planned = []
    for disc in disciplines:
        key = ""
        for k in DISCIPLINE_STATUS:
            if k.replace("_", " ") in disc.lower():
                key = k
                break
        if key and DISCIPLINE_STATUS.get(key, False):
            implemented.append(disc)
        else:
            planned.append(disc)
    return implemented, planned


class MathAdvisor:
    ENGINE_ADVISORIES: dict[str, dict[str, Any]] = {
        "vasp": {
            "tools": ["extract", "validate", "verify", "geometry", "proposition"],
            "math_disciplines": [
                "Differential Geometry (Berry curvature, Brillouin zone topology, fiber bundles)",
                "Spectral Analysis (band structure, density of states, gap analysis)",
                "Group Theory (space groups, point groups, symmetry operations)",
                "Formal Verification (convergence constraints, cutoff consistency)",
            ],
            "workflow_hint": (
                "Extract → Validate parameter constraints → "
                "Extract geometry (Berry curvature, metric tensor) → "
                "Verify mathematical statements → "
                "Generate propositions about topological invariants"
            ),
            "keywords": [
                "Berry curvature",
                "Brillouin zone",
                "topological invariant",
                "band gap",
                "k-space",
                "reciprocal lattice",
                "Bloch theorem",
                "symmetry",
            ],
        },
        "lammps": {
            "tools": ["extract", "validate", "emergence", "verify", "proposition"],
            "math_disciplines": [
                "Dynamical Systems (Lyapunov exponents, phase space, attractors)",
                "Statistical Mechanics (ensemble theory, partition functions, fluctuation-dissipation)",
                "Emergence Theory (phase transitions, order parameters, critical exponents)",
                "Information Theory (mutual information, entropy production, multi-scale analysis)",
            ],
            "workflow_hint": (
                "Extract → Validate constraints → "
                "Analyze emergence (phase transitions, order parameters) → "
                "Verify thermodynamic consistency → "
                "Generate propositions about dynamical structure"
            ),
            "keywords": [
                "phase transition",
                "order parameter",
                "Lyapunov",
                "correlation function",
                "diffusion",
                "NVT/NPT",
                "potential energy",
                "radial distribution",
            ],
        },
        "abaqus": {
            "tools": ["extract", "validate", "geometry", "verify", "compare"],
            "math_disciplines": [
                "Differential Geometry (strain-induced curvature, Cauchy-Green tensor, deformation gradients)",
                "Functional Analysis (PDE well-posedness, Sobolev spaces, weak formulations)",
                "Optimization (parameter sensitivity, shape optimization, topology optimization)",
                "Variational Calculus (energy minimization, principle of virtual work)",
            ],
            "workflow_hint": (
                "Extract → Validate material parameters → "
                "Extract geometry (deformation, strain) → "
                "Verify PDE consistency → "
                "Compare with analytical solutions"
            ),
            "keywords": [
                "Cauchy-Green",
                "strain tensor",
                "von Mises",
                "convergence",
                "mesh sensitivity",
                "boundary value problem",
                "weak form",
                "variational",
            ],
        },
        "quantum_espresso": {
            "tools": ["extract", "validate", "verify", "geometry", "proposition"],
            "math_disciplines": [
                "Differential Geometry (Berry phase, polarization, Wannier functions)",
                "Spectral Analysis (phonon dispersion, electronic bands)",
                "Group Theory (crystal symmetries, irreducible representations)",
                "Perturbation Theory (DFPT, linear response)",
            ],
            "workflow_hint": (
                "Extract → Validate convergence → "
                "Extract geometry → "
                "Verify phonon/electronic consistency → "
                "Generate propositions"
            ),
            "keywords": [
                "Berry phase",
                "phonon",
                "DFPT",
                "Wannier",
                "polarization",
                "Fermi surface",
            ],
        },
        "gromacs": {
            "tools": ["extract", "validate", "emergence", "verify", "proposition"],
            "math_disciplines": [
                "Statistical Mechanics (free energy landscapes, PMF, umbrella sampling)",
                "Dynamical Systems (Markov state models, transition paths, metastability)",
                "Information Theory (mutual information between residues, allosteric networks)",
                "Stochastic Processes (Langevin dynamics, Fokker-Planck, Kramers theory)",
            ],
            "workflow_hint": (
                "Extract → Validate force field constraints → "
                "Analyze emergence (folding transitions, binding events) → "
                "Verify free energy consistency → "
                "Generate propositions about conformational landscapes"
            ),
            "keywords": [
                "free energy",
                "Markov state model",
                "PMF",
                "RMSD",
                "folding",
                "binding",
                "conformational",
                "allosteric",
            ],
        },
        "multiwfn": {
            "tools": ["extract", "validate", "verify", "geometry"],
            "math_disciplines": [
                "Quantum Chemistry (electron density, orbital analysis, QTAIM)",
                "Topology (critical points, bond paths, atomic basins)",
                "Information Theory (Shannon entropy, Fisher information, information-theoretic atoms)",
                "Functional Analysis (density functionals, Kohn-Sham equations)",
            ],
            "workflow_hint": (
                "Extract → Validate wavefunction consistency → "
                "Extract geometry (electron density topology) → "
                "Verify topological invariants"
            ),
            "keywords": [
                "electron density",
                "QTAIM",
                "critical point",
                "bond path",
                "orbital",
                "Laplacian",
                "AIM",
            ],
        },
        "ansys": {
            "tools": ["extract", "validate", "geometry", "verify", "compare"],
            "math_disciplines": [
                "Differential Geometry (stress tensors, principal directions, Mohr's circle)",
                "Fluid Dynamics (Navier-Stokes, Reynolds number, turbulence models)",
                "Functional Analysis (PDE well-posedness, finite element spaces)",
                "Optimization (design variables, objective functions, constraints)",
            ],
            "workflow_hint": (
                "Extract → Validate boundary conditions → "
                "Extract geometry (stress/strain fields) → "
                "Verify PDE consistency → "
                "Compare configurations"
            ),
            "keywords": [
                "Navier-Stokes",
                "stress",
                "turbulence",
                "Reynolds",
                "convergence",
                "mesh",
                "boundary condition",
                "CFD",
            ],
        },
        "comsol": {
            "tools": ["extract", "validate", "geometry", "verify", "compare"],
            "math_disciplines": [
                "Functional Analysis (multiphysics PDE coupling, weak formulations)",
                "Variational Calculus (energy minimization across coupled fields)",
                "Differential Geometry (metric structure of coupled domains)",
                "Perturbation Theory (linearization around equilibrium states)",
            ],
            "workflow_hint": (
                "Extract → Validate multiphysics coupling → "
                "Extract geometry (domain structure) → "
                "Verify PDE consistency → "
                "Compare coupled vs decoupled solutions"
            ),
            "keywords": [
                "multiphysics",
                "coupled PDE",
                "weak form",
                "finite element",
                "COMSOL",
                "boundary condition",
                "domain",
                "coupling",
            ],
        },
        "solidworks": {
            "tools": ["extract", "validate", "geometry", "verify"],
            "math_disciplines": [
                "Differential Geometry (CAD surface curvature, geodesics, normal fields)",
                "Optimization (parametric design, sensitivity analysis)",
                "Topology (topological classification of solid models)",
            ],
            "workflow_hint": (
                "Extract → Validate geometric parameters → "
                "Extract geometry (surface structure) → "
                "Verify geometric consistency"
            ),
            "keywords": [
                "CAD",
                "surface",
                "curvature",
                "parametric",
                "solid model",
                "B-rep",
                "NURBS",
                "feature",
            ],
        },
        "voxel": {
            "tools": ["extract", "validate", "emergence", "verify"],
            "math_disciplines": [
                "Topology (persistent homology of voxel structures, Betti numbers)",
                "Information Theory (voxel entropy, mutual information across scales)",
                "Statistical Mechanics (voxel statistics, percolation theory)",
            ],
            "workflow_hint": (
                "Extract → Validate voxel data → "
                "Analyze emergence (topological features) → "
                "Verify structural consistency"
            ),
            "keywords": [
                "voxel",
                "volume",
                "3D grid",
                "topology",
                "percolation",
                "porosity",
                "segmentation",
                "density",
            ],
        },
    }

    def advise(self, engine: str) -> dict[str, Any]:
        key = engine.lower().replace("-", "_").replace(" ", "_")
        advisory = self.ENGINE_ADVISORIES.get(key, _GENERIC_ADVISORY)
        result = dict(advisory)
        disciplines = result.get("math_disciplines", [])
        implemented, planned = _classify_disciplines(disciplines)
        result["implemented_disciplines"] = implemented
        result["planned_disciplines"] = planned
        return result

    def list_engines(self) -> list[str]:
        return list(self.ENGINE_ADVISORIES.keys())

    def get_workflow_hint(self, engine: str) -> str:
        return self.advise(engine)["workflow_hint"]

    def get_recommended_tools(self, engine: str) -> list[str]:
        return self.advise(engine)["tools"]
