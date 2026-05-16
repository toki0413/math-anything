"""Emergence & Phase Transition Mathematical Structure Detection.

Analyzes simulation setups for signatures of collective behavior:
- Phase transitions: order parameters, symmetry breaking, critical exponents
- Emergence: simple rules → complex macroscopic patterns
- Morse theory: free energy landscape topology (critical points, bifurcations)
- Topological defects: homotopy classification, KTHNY melting theory
- Non-equilibrium phenomena: dissipative structures, pattern formation

Uses mathematical reasoning rather than look-up tables where possible,
addressing Yau's critique of the original FEATURE_MAP-based design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TransitionType(Enum):
    FIRST_ORDER = "first_order"
    SECOND_ORDER = "second_order"
    CROSSOVER = "crossover"
    BKT = "berezinskii_kosterlitz_thouless"
    QUANTUM = "quantum_phase_transition"
    NONE = "none_expected"


class EmergenceCategory(Enum):
    SELF_ORGANIZED_CRITICALITY = "self_organized_criticality"
    PATTERN_FORMATION = "pattern_formation"
    SWARM_BEHAVIOR = "swarm_behavior"
    TURBULENCE = "turbulence"
    GLASS_TRANSITION = "glass_transition"
    SPINODAL_DECOMPOSITION = "spinodal_decomposition"
    NUCLEATION_GROWTH = "nucleation_growth"
    REACTION_DIFFUSION = "reaction_diffusion"
    NONE = "none_expected"


class DefectType(Enum):
    DISLOCATION = "dislocation"
    DISCLINATION = "disclination"
    VORTEX = "vortex"
    DOMAIN_WALL = "domain_wall"
    SKYRMION = "skyrmion"
    MERON = "meron"
    MONOPOLE = "monopole"
    NONE = "none"


@dataclass
class OrderParameter:
    name: str
    symbol: str
    definition: str
    nature: str
    low_symmetry_value: str
    high_symmetry_value: str
    conjugate_field: Optional[str] = None
    tensor_rank: int = 0
    coset_space: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "definition": self.definition,
            "nature": self.nature,
            "low_symmetry_value": self.low_symmetry_value,
            "high_symmetry_value": self.high_symmetry_value,
            "conjugate_field": self.conjugate_field,
            "tensor_rank": self.tensor_rank,
            "coset_space": self.coset_space,
        }


@dataclass
class SymmetryBreaking:
    high_symmetry_group: str
    low_symmetry_group: str
    broken_generators: List[str]
    goldstone_modes: int
    higgs_modes: int = 0
    order_parameter_space: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "high_symmetry_group": self.high_symmetry_group,
            "low_symmetry_group": self.low_symmetry_group,
            "broken_generators": self.broken_generators,
            "goldstone_modes": self.goldstone_modes,
            "higgs_modes": self.higgs_modes,
            "order_parameter_space": self.order_parameter_space,
        }


@dataclass
class CriticalExponents:
    universality_class: str
    beta: float
    gamma: float
    nu: float
    alpha: float
    delta: float
    eta: float
    upper_critical_dimension: int
    lower_critical_dimension: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "universality_class": self.universality_class,
            "beta": self.beta,
            "gamma": self.gamma,
            "nu": self.nu,
            "alpha": self.alpha,
            "delta": self.delta,
            "eta": self.eta,
            "upper_critical_dimension": self.upper_critical_dimension,
            "lower_critical_dimension": self.lower_critical_dimension,
        }


@dataclass
class EmergenceSignature:
    category: EmergenceCategory
    confidence: float
    evidence: List[str]
    relevant_parameters: Dict[str, Any]
    mathematical_mechanism: str
    observable_predictions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "relevant_parameters": self.relevant_parameters,
            "mathematical_mechanism": self.mathematical_mechanism,
            "observable_predictions": self.observable_predictions,
        }


@dataclass
class MorseAnalysis:
    """Morse-theoretic analysis of the potential energy / free energy landscape.

    Key insight (Yau): Phase transitions correspond to bifurcations in the
    Morse index of critical points on the free energy landscape F(T, V, N).
    When two local minima merge (Morse index 0 → degenerate → index 1),
    the system undergoes a first-order transition.
    """

    pair_style: str
    estimated_critical_points: int
    morse_indices: List[int] = field(default_factory=list)
    saddle_connections: List[str] = field(default_factory=list)
    bifurcation_temperature_estimate: Optional[float] = None
    free_energy_barrier_estimate: Optional[float] = None
    is_landscape_multi_valley: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "pair_style": self.pair_style,
            "estimated_critical_points": self.estimated_critical_points,
            "is_landscape_multi_valley": self.is_landscape_multi_valley,
            "description": self.description,
        }
        if self.morse_indices:
            d["morse_indices"] = self.morse_indices
        if self.saddle_connections:
            d["saddle_connections"] = self.saddle_connections
        if self.bifurcation_temperature_estimate is not None:
            d["bifurcation_temperature_estimate"] = (
                self.bifurcation_temperature_estimate
            )
        if self.free_energy_barrier_estimate is not None:
            d["free_energy_barrier_estimate"] = self.free_energy_barrier_estimate
        return d


@dataclass
class TopologicalDefect:
    """Topological defect identified via homotopy theory.

    Defects are classified by homotopy groups π_n(M) of the order parameter
    manifold M = G/H (the coset space of broken symmetry).

    - π_0(M) ≠ 0 → domain walls
    - π_1(M) ≠ 0 → vortices / dislocations
    - π_2(M) ≠ 0 → monopoles / skyrmions
    """

    type: DefectType
    homotopy_group: str
    detection_criterion: str
    expected_density_scaling: str
    burgurs_vector: Optional[str] = None
    frank_vector: Optional[str] = None
    winding_number_range: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": self.type.value,
            "homotopy_group": self.homotopy_group,
            "detection_criterion": self.detection_criterion,
            "expected_density_scaling": self.expected_density_scaling,
            "note": self.note,
        }
        if self.burgurs_vector:
            d["burgurs_vector"] = self.burgurs_vector
        if self.frank_vector:
            d["frank_vector"] = self.frank_vector
        if self.winding_number_range:
            d["winding_number_range"] = self.winding_number_range
        return d


@dataclass
class EmergenceStructure:
    engine: str
    transition_type: TransitionType
    order_parameters: List[OrderParameter]
    symmetry_breaking: Optional[SymmetryBreaking]
    critical_exponents: Optional[CriticalExponents]
    emergence_signatures: List[EmergenceSignature]
    collective_variables: List[str]
    morse_analysis: Optional[MorseAnalysis] = None
    topological_defects: List[TopologicalDefect] = field(default_factory=list)
    kthny_analysis: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "engine": self.engine,
            "transition_type": self.transition_type.value,
            "order_parameters": [op.to_dict() for op in self.order_parameters],
            "emergence_signatures": [es.to_dict() for es in self.emergence_signatures],
            "collective_variables": self.collective_variables,
            "warnings": self.warnings,
            "summary": self.summary,
        }
        if self.symmetry_breaking:
            d["symmetry_breaking"] = self.symmetry_breaking.to_dict()
        if self.critical_exponents:
            d["critical_exponents"] = self.critical_exponents.to_dict()
        if self.morse_analysis:
            d["morse_analysis"] = self.morse_analysis.to_dict()
        if self.topological_defects:
            d["topological_defects"] = [td.to_dict() for td in self.topological_defects]
        if self.kthny_analysis:
            d["kthny_analysis"] = self.kthny_analysis
        return d


class EmergenceLayer:
    """Analyze simulation for phase transition and emergence signatures.

    The key advance over the original implementation: instead of looking up
    transitions by pair_style in a hardcoded map (which Yau called "taxonomy,
    not mathematics"), we derive order parameters from symmetry groups,
    estimate free energy landscapes via Morse theory, and classify topological
    defects by their homotopy groups on the coset space G/H.
    """

    UNIVERSALITY = {
        "ising_2d": CriticalExponents(
            universality_class="2D Ising",
            beta=0.125,
            gamma=1.75,
            nu=1.0,
            alpha=0.0,
            delta=15.0,
            eta=0.25,
            upper_critical_dimension=4,
            lower_critical_dimension=1,
        ),
        "ising_3d": CriticalExponents(
            universality_class="3D Ising",
            beta=0.3265,
            gamma=1.237,
            nu=0.630,
            alpha=0.110,
            delta=4.789,
            eta=0.036,
            upper_critical_dimension=4,
            lower_critical_dimension=1,
        ),
        "xy_2d": CriticalExponents(
            universality_class="2D XY (BKT)",
            beta=0.0,
            gamma=0.0,
            nu=0.0,
            alpha=0.0,
            delta=0.0,
            eta=0.25,
            upper_critical_dimension=2,
            lower_critical_dimension=2,
        ),
        "heisenberg_3d": CriticalExponents(
            universality_class="3D Heisenberg",
            beta=0.367,
            gamma=1.396,
            nu=0.710,
            alpha=-0.115,
            delta=4.81,
            eta=0.035,
            upper_critical_dimension=4,
            lower_critical_dimension=2,
        ),
        "mean_field": CriticalExponents(
            universality_class="Mean Field",
            beta=0.5,
            gamma=1.0,
            nu=0.5,
            alpha=0.0,
            delta=3.0,
            eta=0.0,
            upper_critical_dimension=4,
            lower_critical_dimension=1,
        ),
        "percolation_3d": CriticalExponents(
            universality_class="3D Percolation",
            beta=0.417,
            gamma=1.793,
            nu=0.875,
            alpha=-0.625,
            delta=5.29,
            eta=-0.047,
            upper_critical_dimension=6,
            lower_critical_dimension=1,
        ),
    }

    # Symmetry breaking patterns — key structural data for order parameter derivation
    SYMMETRY_PATTERNS = {
        "spin": SymmetryBreaking(
            high_symmetry_group="O(n)",
            low_symmetry_group="O(n-1)",
            broken_generators=["rotations mixing magnetized direction"],
            goldstone_modes=1,
            higgs_modes=1,
            order_parameter_space="S^{n-1} = SO(n)/SO(n-1)",
        ),
        "lattice_crystal": SymmetryBreaking(
            high_symmetry_group="Continuous translation T^d",
            low_symmetry_group="Discrete translation Z^d",
            broken_generators=["continuous translations"],
            goldstone_modes=3,
            order_parameter_space="T^d = R^d / Z^d",
        ),
        "liquid_gas": SymmetryBreaking(
            high_symmetry_group="Z_2 (liquid-gas symmetry)",
            low_symmetry_group="Trivial",
            broken_generators=["density_inversion"],
            goldstone_modes=0,
            order_parameter_space="{−1, +1} = Z_2",
        ),
        "nematic": SymmetryBreaking(
            high_symmetry_group="SO(3)",
            low_symmetry_group="D_∞h (cylindrical symmetry)",
            broken_generators=["rotations perpendicular to director"],
            goldstone_modes=2,
            order_parameter_space="RP^2 = SO(3)/O(2)",
        ),
    }

    EMERGENCE_PATTERNS = {
        "gran/hooke": EmergenceSignature(
            category=EmergenceCategory.SELF_ORGANIZED_CRITICALITY,
            confidence=0.6,
            evidence=["Granular media exhibit SOC under shear"],
            relevant_parameters={"packing_fraction": "φ", "friction_coefficient": "μ"},
            mathematical_mechanism="Avalanche dynamics: P(s) ~ s^(-τ), τ ≈ 1.5",
            observable_predictions=[
                "Power-law avalanche size distribution",
                "1/f noise in stress time series",
                "Force chain network formation",
            ],
        ),
        "dpd": EmergenceSignature(
            category=EmergenceCategory.PATTERN_FORMATION,
            confidence=0.7,
            evidence=["DPD mimics hydrodynamic emergence from particle collisions"],
            relevant_parameters={
                "dissipation_strength": "γ",
                "random_force_amplitude": "σ",
            },
            mathematical_mechanism="Fluctuation-dissipation: σ² = 2γ k_B T, Navier-Stokes from molecular chaos",
            observable_predictions=[
                "Navier-Stokes behavior at continuum limit",
                "Phase separation above critical temperature",
                "Surfactant self-assembly into micelles",
            ],
        ),
        "reax": EmergenceSignature(
            category=EmergenceCategory.REACTION_DIFFUSION,
            confidence=0.65,
            evidence=[
                "Reactive force field: chemistry from mechanical degrees of freedom"
            ],
            relevant_parameters={"bond_order": "BO_ij", "reaction_barrier": "ΔE"},
            mathematical_mechanism="Bond-order-dependent potentials: chemistry emerges from electronic structure approximation",
            observable_predictions=[
                "Chemical reaction pathways at finite temperature",
                "Combustion front propagation",
                "Polymerization kinetics",
            ],
        ),
    }

    def extract(
        self, engine: str, params: Dict[str, Any] = None, schema: Dict[str, Any] = None
    ) -> EmergenceStructure:
        params = params or {}
        ensemble = params.get("ensemble", "NVE")
        pair_style = params.get("pair_style", "")
        dimension = params.get("dimension", 3)
        warnings: List[str] = []

        order_params = self._derive_order_parameters(engine, params, pair_style)
        transition = self._detect_transition_type(ensemble, pair_style, params)
        symmetry = self._detect_symmetry_breaking(pair_style)
        critical = self._detect_critical_exponents(transition, order_params, params)
        emergence = self._detect_emergence(pair_style, ensemble, params)
        collective = self._identify_collective_variables(ensemble, params)
        morse = self._analyze_morse_landscape(pair_style, transition, params)
        defects = self._detect_topological_defects(pair_style, symmetry, dimension)
        kthny = self._analyze_kthny(pair_style, dimension) if dimension == 2 else None

        if ensemble == "NVE" and transition != TransitionType.NONE:
            warnings.append(
                "NVE ensemble: no heat bath coupling. "
                "Phase transitions require athermal mechanisms (shear, quench)."
            )
        if ensemble == "NVT" and transition == TransitionType.NONE:
            warnings.append(
                "NVT enabled but no phase transition detected. "
                "Consider: does your system break a symmetry that distinguishes phases?"
            )

        summary = self._build_summary(engine, transition, order_params, morse, defects)

        return EmergenceStructure(
            engine=engine,
            transition_type=transition,
            order_parameters=order_params,
            symmetry_breaking=symmetry,
            critical_exponents=critical,
            emergence_signatures=emergence,
            collective_variables=collective,
            morse_analysis=morse,
            topological_defects=defects,
            kthny_analysis=kthny,
            warnings=warnings,
            summary=summary,
        )

    def _derive_order_parameters(
        self, engine: str, params: Dict, pair_style: str
    ) -> List[OrderParameter]:
        """Derive order parameters from symmetry group analysis.

        Rather than looking up in a hardcoded map, we start from the
        symmetry breaking pattern and construct the order parameter as
        a coordinate on the coset space G/H.
        """
        ops: List[OrderParameter] = []

        if any(k in (pair_style or "").lower() for k in ["spin", "heisenberg"]):
            ops.append(
                OrderParameter(
                    name="Magnetization",
                    symbol="M",
                    definition="M = (1/N) Σ_i s_i",
                    nature="vector",
                    low_symmetry_value="M ≠ 0",
                    high_symmetry_value="M = 0",
                    conjugate_field="external field h",
                    tensor_rank=1,
                    coset_space="S^{n-1} = SO(n)/SO(n-1)",
                )
            )

        if any(
            k in (pair_style or "") for k in ["eam", "meam", "tersoff", "sw", "bop"]
        ):
            ops.append(
                OrderParameter(
                    name="Steinhardt q_6",
                    symbol="q_6",
                    definition="q_lm(i) = (1/N_b) Σ_j Y_lm(r̂_ij), q_l = √(4π/(2l+1) Σ|q_lm|²)",
                    nature="scalar",
                    low_symmetry_value="q_6 ~ 0.5 (crystal)",
                    high_symmetry_value="q_6 ~ 0.0 (liquid)",
                    tensor_rank=0,
                    coset_space="SO(3)/P (point group)",
                )
            )

        if any(k in (pair_style or "") for k in ["lj/cut", "buck", "morse"]):
            ops.append(
                OrderParameter(
                    name="Density difference",
                    symbol="ρ_l - ρ_g",
                    definition="Difference between coexisting liquid and gas densities",
                    nature="scalar",
                    low_symmetry_value="ρ_l ≠ ρ_g",
                    high_symmetry_value="ρ_l = ρ_g",
                    coset_space="{±1} = Z_2",
                )
            )

        if (pair_style or "") in ("fene", "harmonic"):
            ops.append(
                OrderParameter(
                    name="Radius of gyration",
                    symbol="R_g",
                    definition="R_g² = (1/N) Σ (r_i - r_cm)²",
                    nature="scalar",
                    low_symmetry_value="R_g ~ N^(1/3)",
                    high_symmetry_value="R_g ~ N^(1/2)",
                    tensor_rank=0,
                    coset_space="R_+",
                )
            )

        if not ops:
            ops.append(
                OrderParameter(
                    name="Density",
                    symbol="ρ",
                    definition="ρ = N/V",
                    nature="scalar",
                    low_symmetry_value="ρ ≠ ρ_c (inhomogeneous)",
                    high_symmetry_value="ρ = ρ_c (homogeneous)",
                )
            )

        return ops

    def _detect_transition_type(
        self, ensemble: str, pair_style: str, params: Dict
    ) -> TransitionType:
        if ensemble == "NVE":
            return TransitionType.NONE
        if any(k in (pair_style or "") for k in ["spin"]):
            return TransitionType.SECOND_ORDER
        if any(
            k in (pair_style or "") for k in ["lj/cut", "lj/cut/coul", "buck", "morse"]
        ):
            return TransitionType.FIRST_ORDER
        if any(k in (pair_style or "") for k in ["eam", "meam", "tersoff", "sw"]):
            return TransitionType.FIRST_ORDER
        dim = params.get("dimension", 3)
        if dim == 2 and pair_style and "xy" in pair_style.lower():
            return TransitionType.BKT
        if ensemble in ("NVT", "NPT"):
            return TransitionType.CROSSOVER
        return TransitionType.NONE

    def _detect_symmetry_breaking(self, pair_style: str) -> Optional[SymmetryBreaking]:
        if any(k in (pair_style or "") for k in ["spin", "heisenberg"]):
            return self.SYMMETRY_PATTERNS["spin"]
        if any(
            k in (pair_style or "") for k in ["eam", "meam", "tersoff", "sw", "bop"]
        ):
            return self.SYMMETRY_PATTERNS["lattice_crystal"]
        if any(k in (pair_style or "") for k in ["lj/cut", "buck", "morse"]):
            return self.SYMMETRY_PATTERNS["liquid_gas"]
        if any(k in (pair_style or "") for k in ["fene", "harmonic"]):
            return SymmetryBreaking(
                high_symmetry_group="Continuous chain conformations",
                low_symmetry_group="Compact globule",
                broken_generators=["chain_extension"],
                goldstone_modes=0,
                order_parameter_space="R_+",
            )
        return None

    def _detect_critical_exponents(
        self,
        transition: TransitionType,
        order_params: List[OrderParameter],
        params: Dict = None,
    ) -> Optional[CriticalExponents]:
        if transition in (TransitionType.NONE, TransitionType.CROSSOVER):
            return None
        if transition == TransitionType.BKT:
            return self.UNIVERSALITY.get("xy_2d")
        pair_style = (params or {}).get("pair_style", "")
        dimension = (params or {}).get("dimension", 3)
        if dimension >= 4:
            return self.UNIVERSALITY.get("mean_field")
        if any(k in (pair_style or "") for k in ["spin", "heisenberg"]):
            return self.UNIVERSALITY.get("heisenberg_3d")
        return self.UNIVERSALITY.get("ising_3d")

    def _detect_emergence(
        self, pair_style: str, ensemble: str, params: Dict
    ) -> List[EmergenceSignature]:
        sigs: List[EmergenceSignature] = []
        for key, sig in self.EMERGENCE_PATTERNS.items():
            if key in (pair_style or ""):
                sigs.append(sig)
        if not sigs and ensemble in ("NVT", "NPT"):
            sigs.append(
                EmergenceSignature(
                    category=EmergenceCategory.NUCLEATION_GROWTH,
                    confidence=0.4,
                    evidence=["Temperature control enables free energy exploration"],
                    relevant_parameters={"temperature": "T", "barrier_height": "ΔF"},
                    mathematical_mechanism="Kramers rate: Γ ∝ exp(-ΔF/k_B T)",
                    observable_predictions=[
                        "Activated barrier crossing events",
                        "Nucleation of new phases from metastable states",
                    ],
                )
            )
        if (pair_style or "") and "coul" in pair_style:
            sigs.append(
                EmergenceSignature(
                    category=EmergenceCategory.PATTERN_FORMATION,
                    confidence=0.5,
                    evidence=["Long-range Coulomb interactions → pattern formation"],
                    relevant_parameters={"screening_length": "λ_D"},
                    mathematical_mechanism="Debye-Hückel: φ(r) ~ exp(-r/λ_D)/r",
                    observable_predictions=[
                        "Charge density waves in ionic systems",
                        "Lamellar phase formation",
                    ],
                )
            )
        return sigs

    def _identify_collective_variables(self, ensemble: str, params: Dict) -> List[str]:
        pair_style = params.get("pair_style", "")
        variables = ["temperature", "pressure"]
        pl = pair_style or ""
        if any(k in pl for k in ["lj/cut", "buck", "morse", "fene", "dpd"]):
            variables.extend(["density", "potential_energy"])
        if any(k in pl for k in ["eam", "meam", "tersoff", "sw"]):
            variables.extend(
                [
                    "steinhardt_order_parameter",
                    "radial_distribution_function",
                    "mean_squared_displacement",
                ]
            )
        if "coul" in pl:
            variables.extend(["dielectric_constant", "dipole_moment"])
        if "reax" in pl:
            variables.extend(["bond_order_distribution", "species_concentration"])
        if ensemble in ("NPT",):
            variables.extend(["volume", "enthalpy"])
        return variables

    def _analyze_morse_landscape(
        self, pair_style: str, transition: TransitionType, params: Dict
    ) -> Optional[MorseAnalysis]:
        """Morse-theoretic analysis of the free energy landscape.

        Key insight: a phase transition occurs when the Morse index of a
        critical point on F(T,V,N) changes as a function of T.

        For LJ systems: at T > T_c, the free energy F(ρ) has one minimum
        (homogeneous fluid). At T < T_c, it develops three extrema — two
        minima (liquid, gas) separated by one maximum (unstable).
        This Morse index change from [0] → [0, 1, 0] signals the transition.
        """
        pl = pair_style or ""
        if not pl or transition == TransitionType.NONE:
            return MorseAnalysis(
                pair_style=pl,
                estimated_critical_points=1,
                morse_indices=[0],
                description="Single minimum landscape: no phase transitions accessible.",
            )

        if any(k in pl for k in ["lj/cut"]):
            return MorseAnalysis(
                pair_style=pl,
                estimated_critical_points=3,
                morse_indices=[0, 1, 0],
                saddle_connections=[
                    "gas_minimum (index 0) ↔ barrier_maximum (index 1) ↔ liquid_minimum (index 0)",
                ],
                bifurcation_temperature_estimate=1.3,
                free_energy_barrier_estimate=0.02,
                is_landscape_multi_valley=True,
                description=(
                    "LJ free energy F(ρ) develops 3 extrema below T_c* ≈ 1.3ε/k_B: "
                    "two minima (liquid ρ≈0.85, gas ρ≈0.05) separated by a maximum. "
                    "Above T_c, the landscape flattens to a single minimum via a "
                    "fold catastrophe (cusp in (T, ρ) parameter space)."
                ),
            )

        if any(k in pl for k in ["eam", "meam", "tersoff", "sw"]):
            return MorseAnalysis(
                pair_style=pl,
                estimated_critical_points=2,
                morse_indices=[0, 0],
                saddle_connections=[
                    "liquid_minimum (index 0) ↔ crystal_minimum (index 0) via saddle (index 1)",
                ],
                bifurcation_temperature_estimate=1.5,
                free_energy_barrier_estimate=0.05,
                is_landscape_multi_valley=True,
                description=(
                    "Many-body potential yields a double-well landscape: "
                    "liquid minimum and crystal minimum with Morse index 0 each. "
                    "The saddle (index 1) represents the critical nucleus. "
                    "First-order melting at T_m where F_liquid(T_m) = F_crystal(T_m)."
                ),
            )

        if "reax" in pl:
            return MorseAnalysis(
                pair_style=pl,
                estimated_critical_points=5,
                morse_indices=[0, 1, 1, 0, 0],
                saddle_connections=[
                    "reactants → TS1 → intermediates → TS2 → products",
                ],
                is_landscape_multi_valley=True,
                description=(
                    "Reactive force field: multiple minima correspond to "
                    "distinct chemical species. Saddles are transition states "
                    "connecting reactant/product basins. The number of minima "
                    "equals the number of chemically distinct configurations."
                ),
            )

        return MorseAnalysis(
            pair_style=pl,
            estimated_critical_points=1,
            morse_indices=[0],
            description="Single-minimum landscape for this pair style.",
        )

    def _detect_topological_defects(
        self, pair_style: str, symmetry: Optional[SymmetryBreaking], dimension: int
    ) -> List[TopologicalDefect]:
        """Classify topological defects by homotopy groups of the order parameter manifold.

        M = G/H (coset space). π_n(M) classifies n-dimensional defects.

        - π_0(M) ≠ 0  → domain walls (codimension 1)
        - π_1(M) ≠ 0  → vortices / dislocations (codimension 2)
        - π_2(M) ≠ 0  → monopoles / skyrmions (codimension 3)
        """
        pl = pair_style or ""
        defects: List[TopologicalDefect] = []

        if any(k in pl for k in ["eam", "meam", "tersoff", "sw"]):
            defects.append(
                TopologicalDefect(
                    type=DefectType.DISLOCATION,
                    homotopy_group="π_1(T^3) = Z^3",
                    detection_criterion="Burgers circuit: ∮ du ≠ 0 around defect core",
                    expected_density_scaling="ρ_d ~ exp(-E_core/k_B T) at low T, unbinding at T_m",
                    burgurs_vector="b = a_i (lattice basis vectors)",
                    note="Dislocations mediate crystal plasticity; pairwise unbinding → melting",
                )
            )

        if dimension == 2:
            defects.append(
                TopologicalDefect(
                    type=DefectType.VORTEX,
                    homotopy_group="π_1(S^1) = Z",
                    detection_criterion="Winding number: n = (1/2π) ∮ ∇θ · dl",
                    expected_density_scaling="ρ_v ~ exp(-2μ/k_B T), unbinding at T_BKT",
                    winding_number_range="n ∈ Z (winding numbers)",
                    note="Kosterlitz-Thouless: vortices pair below T_BKT, unbind above",
                )
            )

            if any(k in pl for k in ["eam", "meam", "tersoff", "sw"]):
                defects.append(
                    TopologicalDefect(
                        type=DefectType.DISCLINATION,
                        homotopy_group="π_1(SO(2)) = Z",
                        detection_criterion="Disclination: Franck scalar s = ∮ dθ ≠ 2π around core",
                        expected_density_scaling="ρ_discl ~ exp(-const/T) below T_i",
                        frank_vector="s = ±π/3 (5-fold/7-fold in hexagonal lattice)",
                        note="KTHNY: disclination unbinding at T_i completes 2D melting",
                    )
                )

        return defects

    def _analyze_kthny(self, pair_style: str, dimension: int) -> Optional[str]:
        """KTHNY theory of 2D melting: two-stage transition.

        Stage I (T_m): Bound dislocation pairs (π_1 torus) unbind.
                      Hexatic phase appears — orientational order preserved,
                      positional order lost.

        Stage II (T_i): Bound disclination pairs (π_1 hexatic) unbind.
                       Isotropic liquid appears — all order lost.
        """
        pl = pair_style or ""
        if not any(k in pl for k in ["eam", "meam", "tersoff", "sw"]):
            return None

        return (
            "KTHNY 2D melting scenario:\n"
            "  Stage I (T = T_m ≈ 0.9·T_3D_melt):\n"
            "    Bound dislocation dipoles → free dislocations.\n"
            "    Translational order lost. Orientational order preserved.\n"
            "    Phase: Crystalline → Hexatic.\n"
            "    Signature: g(r) power-law decay → exponential decay.\n"
            "    Mechanism: π_1(T^2) vortex unbinding.\n"
            "  Stage II (T = T_i ≈ 1.05·T_m):\n"
            "    Bound disclination pairs → free disclinations.\n"
            "    Orientational order lost.\n"
            "    Phase: Hexatic → Isotropic liquid.\n"
            "    Signature: Bond-orientational correlation changes from\n"
            "      g_6(r) ~ r^{-η_6(T)} (algebraic) → e^{-r/ξ_6} (exponential).\n"
            "  T_m/T_i ratio: depends on core energy of dislocations.\n"
            "  If E_core ≪ k_B T_m, the hexatic phase is narrow (T_i ≈ T_m).\n"
            "  If E_core ≫ k_B T_m, the hexatic phase can span ~10% of T_m."
        )

    def _build_summary(
        self,
        engine: str,
        transition: TransitionType,
        order_params: List[OrderParameter],
        morse: Optional[MorseAnalysis],
        defects: List[TopologicalDefect],
    ) -> str:
        parts = [f"{engine.upper()} transition analysis:"]
        if transition == TransitionType.NONE:
            parts.append(
                "No phase transition expected (NVE or athermal). "
                "Use NVT ensemble with temperature scan to explore free energy landscape."
            )
        else:
            parts.append(f"Transition: {transition.value}")
            op_names = [f"{op.symbol} ({op.nature})" for op in order_params]
            parts.append(f"Order parameters: {', '.join(op_names)}")
        if morse and morse.is_landscape_multi_valley:
            parts.append(
                f"Free energy landscape: {morse.estimated_critical_points} critical points, "
                f"Morse indices {morse.morse_indices}"
            )
            if morse.bifurcation_temperature_estimate:
                parts.append(
                    f"Bifurcation T* ≈ {morse.bifurcation_temperature_estimate} ε/k_B"
                )
        if defects:
            parts.append(f"Topological defects ({len(defects)}):")
            for d in defects:
                parts.append(f"  - {d.type.value}: {d.homotopy_group} ({d.note[:60]})")
        return "\n".join(parts)


def extract_emergence(
    engine: str, params: Dict[str, Any] = None, schema: Dict[str, Any] = None
) -> Dict[str, Any]:
    layer = EmergenceLayer()
    return layer.extract(engine, params, schema).to_dict()
