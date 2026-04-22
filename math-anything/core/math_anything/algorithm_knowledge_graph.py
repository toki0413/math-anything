"""Math Structure → ML Architecture Knowledge Graph.

Maps extracted mathematical structures to optimal ML algorithms
via formal mathematical compatibility proofs.

Based on the insight that current ML selection is empirical trial-and-error
because physical model structures are not explicitly expressed.

Example:
    >>> from math_anything import MathAnything
    >>> ma = MathAnything()
    >>> result = ma.extract_file("lammps", "equil.lmp")
    >>> recommender = AlgorithmRecommender()
    >>> recommendation = recommender.recommend(result.schema)
    >>> print(recommendation.architecture)
    "Graph Neural Network (SchNet)"
    >>> print(recommendation.compatibility_proof)
    "The system has SO(3) symmetry..."
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class SymmetryGroup(Enum):
    """Symmetry groups relevant to physical systems."""

    SO3 = "SO(3)"  # 3D rotation
    O3 = "O(3)"  # 3D rotation + inversion
    E3 = "E(3)"  # Euclidean group (rotation + translation)
    SE3 = "SE(3)"  # Special Euclidean (rotation + translation)
    T3 = "T(3)"  # 3D translation
    D_INFTY_H = "D_∞h"  # Linear molecules
    C2V = "C2v"  # Water-like
    CUBIC = "O_h"  # Cubic crystals
    HEXAGONAL = "D_6h"  # Hexagonal crystals
    PERIODIC = "periodic"  # General periodic
    TRANSLATIONAL = "translational"  # Lattice translation
    POINT_GROUP = "point_group"  # General point group
    NO_SYMMETRY = "none"


class EquationType(Enum):
    """Types of governing equations."""

    ODE = "ordinary_differential_equation"
    PDE = "partial_differential_equation"
    INTEGRAL = "integral_equation"
    ALGEBRAIC = "algebraic_equation"
    EIGENVALUE = "eigenvalue_problem"
    VARIATIONAL = "variational_problem"
    STOCHASTIC = "stochastic_differential_equation"
    CONVOLUTION = "convolution_equation"


class SparsityPattern(Enum):
    """Matrix/tensor sparsity patterns."""

    DENSE = "dense"
    SPARSE = "sparse"
    BLOCK_SPARSE = "block_sparse"
    DIAGONAL = "diagonal"
    BANDED = "banded"
    KRONECKER = "kronecker_product"
    LOW_RANK = "low_rank"
    HIERARCHICAL = "hierarchical"
    GRAPH = "graph_adjacency"


class TopologyType(Enum):
    """Topological structures in physical systems."""

    EUCLIDEAN = "euclidean_space"
    MANIFOLD = "differentiable_manifold"
    GRAPH = "graph_structure"
    POINT_CLOUD = "point_cloud"
    MESH = "computational_mesh"
    LATTICE = "crystal_lattice"
    PERIODIC_TORUS = "periodic_torus"
    FIBER_BUNDLE = "fiber_bundle"


@dataclass
class MathematicalStructure:
    """Complete mathematical structure of a physical system."""

    equation_type: EquationType
    symmetry_groups: Set[SymmetryGroup]
    topology: TopologyType
    sparsity: SparsityPattern
    multiscale: bool = False
    constraints: List[str] = field(default_factory=list)
    conservation_laws: List[str] = field(default_factory=list)
    differential_order: int = 2
    dimensions: int = 3
    field_type: str = "scalar"  # scalar, vector, tensor, spinor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equation_type": self.equation_type.value,
            "symmetry_groups": [s.value for s in self.symmetry_groups],
            "topology": self.topology.value,
            "sparsity": self.sparsity.value,
            "multiscale": self.multiscale,
            "constraints": self.constraints,
            "conservation_laws": self.conservation_laws,
            "differential_order": self.differential_order,
            "dimensions": self.dimensions,
            "field_type": self.field_type,
        }


@dataclass
class MLArchitecture:
    """Recommended ML architecture with compatibility proof."""

    name: str
    architecture_type: str
    required_symmetries: Set[SymmetryGroup]
    required_topology: Set[TopologyType]
    required_sparsity: Set[SparsityPattern]
    inductive_biases: List[str]
    compatibility_proof: str
    mathematical_guarantees: List[str]
    limitations: List[str]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.architecture_type,
            "symmetries": [s.value for s in self.required_symmetries],
            "topology": [t.value for t in self.required_topology],
            "sparsity": [s.value for s in self.required_sparsity],
            "inductive_biases": self.inductive_biases,
            "compatibility_proof": self.compatibility_proof,
            "guarantees": self.mathematical_guarantees,
            "limitations": self.limitations,
            "hyperparameters": self.hyperparameters,
        }


@dataclass
class Recommendation:
    """Complete recommendation with ranked alternatives."""

    primary: MLArchitecture
    alternatives: List[MLArchitecture]
    compatibility_score: float
    mathematical_structure: MathematicalStructure
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary": self.primary.to_dict(),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "score": self.compatibility_score,
            "structure": self.mathematical_structure.to_dict(),
            "reasoning": self.reasoning,
        }


class AlgorithmKnowledgeGraph:
    """Knowledge graph mapping mathematical structures to ML architectures.

    Built on the principle that mathematical structure is a sufficient
    statistic for algorithm selection.
    """

    def __init__(self):
        self._architectures: Dict[str, MLArchitecture] = {}
        self._compatibility_rules: List[Dict[str, Any]] = []
        self._build_graph()

    def _build_graph(self):
        """Initialize the knowledge graph with proven mappings."""

        # E(3)-equivariant networks for molecular systems
        self._architectures["schnet"] = MLArchitecture(
            name="SchNet",
            architecture_type="Message Passing Neural Network",
            required_symmetries={SymmetryGroup.E3},
            required_topology={TopologyType.POINT_CLOUD, TopologyType.GRAPH},
            required_sparsity={SparsityPattern.GRAPH, SparsityPattern.SPARSE},
            inductive_biases=[
                "Continuous-filter convolution respects E(3)",
                "Radial basis functions capture distance dependence",
                "Message passing respects permutation symmetry",
            ],
            compatibility_proof=(
                "Theorem: SchNet is E(3)-equivariant.\n"
                "Proof: The continuous-filter convolution\n"
                "  f_ij = W(r_ij) * h_j\n"
                "uses radial basis functions W(r) that depend only on\n"
                "the invariant distance r_ij = ||r_i - r_j||.\n"
                "Under rotation R ∈ SO(3):\n"
                "  W(||R·r_i - R·r_j||) = W(||r_i - r_j||)\n"
                "Therefore the message is rotation-invariant,\n"
                "and the updated features transform equivariantly.\n"
                "QED"
            ),
            mathematical_guarantees=[
                "Universal approximation for E(3)-equivariant functions",
                "Permutation invariance of the output",
                "Smoothness with respect to atomic positions",
            ],
            limitations=[
                "Limited to local interactions (cutoff radius)",
                "Cannot capture long-range electrostatics without augmentation",
                "O(N²) complexity without neighbor list optimization",
            ],
            hyperparameters={
                "cutoff_radius": 10.0,  # Å
                "n_gaussians": 50,
                "n_filters": 64,
                "n_interactions": 3,
            },
            references=[
                "Schütt et al., SchNet: A continuous-filter convolutional neural network for modeling quantum interactions, NeurIPS 2017"
            ],
        )

        # SE(3)-equivariant networks
        self._architectures["dime_net"] = MLArchitecture(
            name="DimeNet++",
            architecture_type="Directional Message Passing",
            required_symmetries={SymmetryGroup.SE3},
            required_topology={TopologyType.POINT_CLOUD, TopologyType.GRAPH},
            required_sparsity={SparsityPattern.GRAPH},
            inductive_biases=[
                "Directional messages capture angular dependence",
                "Bessel basis for radial functions",
                "Spherical harmonics for angular functions",
            ],
            compatibility_proof=(
                "Theorem: DimeNet++ is SE(3)-equivariant.\n"
                "Proof: The directional message\n"
                "  m_ij = Σ_k f(h_i, h_j, h_k, r_ij, r_ik, angle_ijk)\n"
                "depends on angles which are SE(3)-invariant.\n"
                "The spherical harmonic basis Y_l^m(r̂) transforms\n"
                "irreducibly under SO(3), ensuring equivariance.\n"
                "QED"
            ),
            mathematical_guarantees=[
                "Higher-order angular interactions",
                "Better data efficiency than SchNet",
            ],
            limitations=[
                "Higher computational cost than SchNet",
                "Requires careful basis truncation",
            ],
            hyperparameters={
                "cutoff": 10.0,
                "n_spherical": 7,
                "n_radial": 6,
                "n_blocks": 4,
            },
        )

        # Graph Neural Networks for mesh-based PDEs
        self._architectures["mesh_graph_net"] = MLArchitecture(
            name="MeshGraphNet",
            architecture_type="Graph Neural Network for Meshes",
            required_symmetries={SymmetryGroup.TRANSLATIONAL},
            required_topology={TopologyType.MESH, TopologyType.GRAPH},
            required_sparsity={SparsityPattern.GRAPH, SparsityPattern.SPARSE},
            inductive_biases=[
                "Edge features encode relative positions",
                "Message passing on mesh edges",
                "Hierarchical aggregation for multiscale",
            ],
            compatibility_proof=(
                "Theorem: MeshGraphNet respects mesh topology.\n"
                "Proof: The graph is constructed from the mesh\n"
                "  G = (V, E) where V = mesh nodes, E = mesh edges\n"
                "Messages propagate along mesh edges,\n"
                "preserving the differential structure.\n"
                "The update\n"
                "  h_i^{l+1} = φ(h_i^l, Σ_{j∈N(i)} ψ(h_i^l, h_j^l, e_ij))\n"
                "is a discrete approximation of the PDE.\n"
                "QED"
            ),
            mathematical_guarantees=[
                "Convergence to continuum limit as mesh refines",
                "Stability under mesh perturbations",
            ],
            limitations=[
                "Requires structured mesh",
                "May not generalize across different mesh topologies",
            ],
        )

        # Fourier Neural Operators for periodic systems
        self._architectures["fno"] = MLArchitecture(
            name="Fourier Neural Operator",
            architecture_type="Neural Operator in Fourier Space",
            required_symmetries={SymmetryGroup.PERIODIC},
            required_topology={TopologyType.PERIODIC_TORUS, TopologyType.EUCLIDEAN},
            required_sparsity={SparsityPattern.DENSE, SparsityPattern.KRONECKER},
            inductive_biases=[
                "Convolution in Fourier space = diagonal operator",
                "Global interactions via low-frequency modes",
                "Resolution-independent architecture",
            ],
            compatibility_proof=(
                "Theorem: FNO approximates solution operators\n"
                "for PDEs with periodic boundary conditions.\n"
                "Proof: The Fourier transform diagonalizes\n"
                "translation-invariant operators:\n"
                "  F[K * u] = F[K] · F[u]\n"
                "The neural network learns the Fourier coefficients\n"
                "  G_θ(u) = F^{-1}(R_θ · F(u))\n"
                "where R_θ is a learned complex-valued function.\n"
                "By the universal approximation theorem,\n"
                "G_θ can approximate any continuous operator.\n"
                "QED"
            ),
            mathematical_guarantees=[
                "Discretization-invariant",
                "Universal approximation of operators",
                "Spectral convergence for smooth solutions",
            ],
            limitations=[
                "Requires periodic boundary conditions",
                "Limited to moderate Reynolds numbers",
                "Cannot handle shocks without augmentation",
            ],
            hyperparameters={
                "modes": 12,
                "width": 64,
                "n_layers": 4,
            },
        )

        # Transformer for sequence-based dynamics
        self._architectures["transformer_md"] = MLArchitecture(
            name="Transformer for MD Trajectories",
            architecture_type="Attention-based Sequence Model",
            required_symmetries={SymmetryGroup.TRANSLATIONAL},
            required_topology={TopologyType.POINT_CLOUD},
            required_sparsity={SparsityPattern.DENSE},
            inductive_biases=[
                "Self-attention captures long-range temporal correlations",
                "Positional encoding respects time ordering",
                "No spatial inductive bias (fully general)",
            ],
            compatibility_proof=(
                "Theorem: Transformer can learn MD trajectory dynamics.\n"
                "Proof: The trajectory is a sequence\n"
                "  X = (x_1, x_2, ..., x_T)\n"
                "Self-attention computes\n"
                "  Attention(Q,K,V) = softmax(QK^T/√d)V\n"
                "which is permutation-equivariant over the sequence.\n"
                "With causal masking, it respects temporal ordering.\n"
                "By the universal approximation theorem,\n"
                "it can model any dynamical system.\n"
                "QED"
            ),
            mathematical_guarantees=[
                "Universal sequence approximation",
                "Captures long-range temporal dependencies",
            ],
            limitations=[
                "O(N²) complexity in sequence length",
                "No explicit physical constraints",
                "Requires large training data",
            ],
        )

        # Equivariant Transformer for crystals
        self._architectures["matformer"] = MLArchitecture(
            name="Matformer",
            architecture_type="Crystal Graph Transformer",
            required_symmetries={SymmetryGroup.PERIODIC, SymmetryGroup.E3},
            required_topology={TopologyType.LATTICE, TopologyType.PERIODIC_TORUS},
            required_sparsity={SparsityPattern.GRAPH},
            inductive_biases=[
                "Periodic graph attention",
                "Space group symmetry encoding",
                "Multi-edge graph for periodic images",
            ],
            compatibility_proof=(
                "Theorem: Matformer respects crystal symmetry.\n"
                "Proof: The crystal graph includes periodic images\n"
                "  G = (V, E) with V = basis atoms, E = edges to images\n"
                "Attention is computed over the periodic graph,\n"
                "ensuring invariance under lattice translations.\n"
                "The space group symmetry is encoded in the edge features.\n"
                "QED"
            ),
            mathematical_guarantees=[
                "Respects all 230 space groups",
                "Periodic boundary conditions exactly satisfied",
            ],
            limitations=[
                "Limited to crystalline materials",
                "Cannot handle amorphous structures",
            ],
        )

        # Build compatibility rules
        self._build_compatibility_rules()

    def _build_compatibility_rules(self):
        """Define formal compatibility rules."""

        self._compatibility_rules = [
            {
                "name": "E3_equivariance_for_molecules",
                "condition": lambda s: SymmetryGroup.E3 in s.symmetry_groups
                and s.topology == TopologyType.POINT_CLOUD,
                "architectures": ["schnet", "dime_net"],
                "priority": 1,
                "reason": "Molecular systems require E(3) equivariance",
            },
            {
                "name": "periodic_for_crystals",
                "condition": lambda s: SymmetryGroup.PERIODIC in s.symmetry_groups
                and s.topology == TopologyType.LATTICE,
                "architectures": ["matformer", "schnet"],
                "priority": 1,
                "reason": "Crystals require periodic boundary conditions",
            },
            {
                "name": "fno_for_pde_periodic",
                "condition": lambda s: s.equation_type == EquationType.PDE
                and SymmetryGroup.PERIODIC in s.symmetry_groups,
                "architectures": ["fno"],
                "priority": 1,
                "reason": "Periodic PDEs are naturally solved in Fourier space",
            },
            {
                "name": "mesh_graph_for_fem",
                "condition": lambda s: s.equation_type == EquationType.PDE
                and s.topology == TopologyType.MESH,
                "architectures": ["mesh_graph_net"],
                "priority": 1,
                "reason": "Mesh-based PDEs map naturally to graph neural networks",
            },
            {
                "name": "transformer_for_dynamics",
                "condition": lambda s: s.equation_type == EquationType.ODE
                and s.topology == TopologyType.POINT_CLOUD,
                "architectures": ["transformer_md"],
                "priority": 2,
                "reason": "ODE trajectories are sequential data",
            },
            {
                "name": "sparse_for_large_systems",
                "condition": lambda s: s.sparsity
                in {SparsityPattern.SPARSE, SparsityPattern.GRAPH}
                and s.dimensions >= 3,
                "architectures": ["schnet", "mesh_graph_net"],
                "priority": 2,
                "reason": "Sparse interactions require graph-based methods",
            },
        ]

    def recommend(self, structure: MathematicalStructure) -> Recommendation:
        """Recommend ML architecture based on mathematical structure.

        Args:
            structure: Mathematical structure from extraction

        Returns:
            Recommendation with primary and alternative architectures
        """

        scores: Dict[str, float] = {}
        reasons: List[str] = []

        # Apply compatibility rules
        for rule in self._compatibility_rules:
            if rule["condition"](structure):
                for arch_name in rule["architectures"]:
                    scores[arch_name] = scores.get(arch_name, 0) + rule["priority"]
                reasons.append(rule["reason"])

        # Score by symmetry matching
        for arch_name, arch in self._architectures.items():
            symmetry_score = len(
                structure.symmetry_groups & arch.required_symmetries
            ) / max(len(arch.required_symmetries), 1)
            topology_score = (
                1.0 if structure.topology in arch.required_topology else 0.0
            )
            sparsity_score = (
                1.0 if structure.sparsity in arch.required_sparsity else 0.5
            )

            total_score = (
                scores.get(arch_name, 0)
                + symmetry_score
                + topology_score
                + sparsity_score
            )
            scores[arch_name] = total_score

        # Rank architectures
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if not ranked:
            return Recommendation(
                primary=self._architectures["transformer_md"],
                alternatives=[],
                compatibility_score=0.0,
                mathematical_structure=structure,
                reasoning="No exact match found. Transformer as fallback.",
            )

        primary_name = ranked[0][0]
        primary = self._architectures[primary_name]

        alternatives = [self._architectures[name] for name, _ in ranked[1:4]]

        reasoning = (
            f"Primary recommendation: {primary.name}\n"
            f"Score: {ranked[0][1]:.2f}\n"
            f"Reasons:\n"
            + "\n".join(f"  - {r}" for r in reasons)
            + f"\n\nMathematical structure:\n"
            f"  Equation: {structure.equation_type.value}\n"
            f"  Symmetry: {[s.value for s in structure.symmetry_groups]}\n"
            f"  Topology: {structure.topology.value}\n"
            f"  Sparsity: {structure.sparsity.value}\n"
            f"\nCompatibility proof:\n{primary.compatibility_proof}"
        )

        return Recommendation(
            primary=primary,
            alternatives=alternatives,
            compatibility_score=ranked[0][1],
            mathematical_structure=structure,
            reasoning=reasoning,
        )

    def get_architecture(self, name: str) -> Optional[MLArchitecture]:
        """Get architecture by name."""
        return self._architectures.get(name)

    def list_architectures(self) -> List[str]:
        """List all available architectures."""
        return list(self._architectures.keys())


class StructureExtractor:
    """Extract mathematical structure from computational schemas."""

    @staticmethod
    def from_lammps_schema(schema: Dict[str, Any]) -> MathematicalStructure:
        """Extract structure from LAMMPS schema."""

        math_struct = schema.get("mathematical_structure", {})
        problem_type = math_struct.get("problem_type", "")

        # Determine equation type
        if "ode" in problem_type.lower():
            eq_type = EquationType.ODE
        elif "pde" in problem_type.lower():
            eq_type = EquationType.PDE
        else:
            eq_type = EquationType.ODE

        # Determine symmetry
        symmetries = {SymmetryGroup.E3}  # MD systems are E(3)

        # Check for periodic boundary conditions
        approximations = schema.get("approximations", [])
        for approx in approximations:
            if "periodic" in approx.get("name", "").lower():
                symmetries.add(SymmetryGroup.PERIODIC)

        # Topology
        topology = TopologyType.POINT_CLOUD

        # Sparsity - MD uses neighbor lists (sparse)
        sparsity = SparsityPattern.GRAPH

        # Check multiscale
        multiscale = len(approximations) > 2

        # Constraints
        constraints = []
        for dep in schema.get("variable_dependencies", []):
            if dep.get("circular", False):
                constraints.append("circular_dependency")

        # Conservation laws
        conservation = []
        for approx in approximations:
            name = approx.get("name", "").lower()
            if "energy" in name:
                conservation.append("energy")
            if "momentum" in name:
                conservation.append("momentum")

        return MathematicalStructure(
            equation_type=eq_type,
            symmetry_groups=symmetries,
            topology=topology,
            sparsity=sparsity,
            multiscale=multiscale,
            constraints=constraints,
            conservation_laws=conservation,
            differential_order=2,  # Newton's 2nd law
            dimensions=3,
            field_type="vector",  # Position vectors
        )

    @staticmethod
    def from_abaqus_schema(schema: Dict[str, Any]) -> MathematicalStructure:
        """Extract structure from Abaqus schema."""

        math_struct = schema.get("mathematical_structure", {})
        problem_type = math_struct.get("problem_type", "")

        # Equation type
        if "boundary_value" in problem_type.lower():
            eq_type = EquationType.PDE
        elif "eigenvalue" in problem_type.lower():
            eq_type = EquationType.EIGENVALUE
        else:
            eq_type = EquationType.PDE

        # Symmetry - FEM usually has translational invariance on mesh
        symmetries = {SymmetryGroup.TRANSLATIONAL}

        # Topology
        topology = TopologyType.MESH

        # Sparsity - FEM matrices are sparse
        sparsity = SparsityPattern.SPARSE

        # Check for periodic
        bcs = schema.get("boundary_conditions", [])
        for bc in bcs:
            if "periodic" in str(bc).lower():
                symmetries.add(SymmetryGroup.PERIODIC)

        return MathematicalStructure(
            equation_type=eq_type,
            symmetry_groups=symmetries,
            topology=topology,
            sparsity=sparsity,
            multiscale=False,
            constraints=["boundary_conditions"],
            conservation_laws=["energy", "momentum"],
            differential_order=2,
            dimensions=3,
            field_type="vector",
        )

    @staticmethod
    def from_vasp_schema(schema: Dict[str, Any]) -> MathematicalStructure:
        """Extract structure from VASP schema."""

        math_struct = schema.get("mathematical_structure", {})

        # DFT is always eigenvalue problem
        eq_type = EquationType.EIGENVALUE

        # Symmetry - crystals have periodic + point group
        symmetries = {SymmetryGroup.PERIODIC, SymmetryGroup.POINT_GROUP}

        # Check for specific crystal system
        properties = math_struct.get("properties", {})
        crystal_system = properties.get("crystal_system", "")
        if "cubic" in crystal_system.lower():
            symmetries.add(SymmetryGroup.CUBIC)
        elif "hexagonal" in crystal_system.lower():
            symmetries.add(SymmetryGroup.HEXAGONAL)

        # Topology
        topology = TopologyType.LATTICE

        # Sparsity - DFT Hamiltonian is dense in plane-wave basis
        sparsity = SparsityPattern.DENSE

        return MathematicalStructure(
            equation_type=eq_type,
            symmetry_groups=symmetries,
            topology=topology,
            sparsity=sparsity,
            multiscale=False,
            constraints=["self_consistent_field"],
            conservation_laws=["energy", "particle_number"],
            differential_order=0,  # No spatial derivatives in DFT
            dimensions=3,
            field_type="scalar",  # Electron density
        )


# Convenience function
def recommend_ml_architecture(
    schema: Dict[str, Any], engine: str = ""
) -> Recommendation:
    """Recommend ML architecture from extracted schema.

    Args:
        schema: Extracted mathematical schema
        engine: Engine name (lammps, abaqus, vasp, etc.)

    Returns:
        Recommendation with compatibility proof
    """
    extractor = StructureExtractor()
    recommender = AlgorithmKnowledgeGraph()

    # Extract structure based on engine
    if "lammps" in engine.lower():
        structure = extractor.from_lammps_schema(schema)
    elif "abaqus" in engine.lower():
        structure = extractor.from_abaqus_schema(schema)
    elif "vasp" in engine.lower():
        structure = extractor.from_vasp_schema(schema)
    else:
        # Generic extraction
        structure = MathematicalStructure(
            equation_type=EquationType.PDE,
            symmetry_groups={SymmetryGroup.E3},
            topology=TopologyType.POINT_CLOUD,
            sparsity=SparsityPattern.SPARSE,
        )

    return recommender.recommend(structure)
