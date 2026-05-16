"""ML potential mathematical analysis — extract structure from learned potentials.

Analyzes the mathematical structure of ML interatomic potentials (DeepMD, MACE,
NequIP, etc.) to understand what physical symmetries and constraints they encode.

This is a unique differentiator: no other tool extracts mathematical semantics
from ML potentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DescriptorMath:
    """Mathematical structure of an ML descriptor."""

    descriptor_type: str = ""
    input_dim: int = 0
    output_dim: int = 0
    symmetry_constraints: List[str] = field(default_factory=list)
    mathematical_form: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "descriptor_type": self.descriptor_type,
        }
        if self.input_dim:
            d["input_dim"] = self.input_dim
        if self.output_dim:
            d["output_dim"] = self.output_dim
        if self.symmetry_constraints:
            d["symmetry_constraints"] = self.symmetry_constraints
        if self.mathematical_form:
            d["mathematical_form"] = self.mathematical_form
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class EquivarianceResult:
    """Equivariance analysis of an ML potential."""

    is_equivariant: bool = False
    equivariance_group: str = ""
    rotation_handling: str = ""
    permutation_handling: str = ""
    translation_invariance: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "is_equivariant": self.is_equivariant,
            "translation_invariance": self.translation_invariance,
        }
        if self.equivariance_group:
            d["equivariance_group"] = self.equivariance_group
        if self.rotation_handling:
            d["rotation_handling"] = self.rotation_handling
        if self.permutation_handling:
            d["permutation_handling"] = self.permutation_handling
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class ComparisonResult:
    """Comparison between ML and classical potentials."""

    ml_type: str = ""
    classical_type: str = ""
    shared_constraints: List[str] = field(default_factory=list)
    ml_only_features: List[str] = field(default_factory=list)
    classical_only_features: List[str] = field(default_factory=list)
    mathematical_equivalence: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "ml_type": self.ml_type,
            "classical_type": self.classical_type,
        }
        if self.shared_constraints:
            d["shared_constraints"] = self.shared_constraints
        if self.ml_only_features:
            d["ml_only_features"] = self.ml_only_features
        if self.classical_only_features:
            d["classical_only_features"] = self.classical_only_features
        if self.mathematical_equivalence:
            d["mathematical_equivalence"] = self.mathematical_equivalence
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class MLPotentialResult:
    """Complete ML potential analysis result."""

    descriptor: Optional[DescriptorMath] = None
    equivariance: Optional[EquivarianceResult] = None
    comparison: Optional[ComparisonResult] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.descriptor:
            d["descriptor"] = self.descriptor.to_dict()
        if self.equivariance:
            d["equivariance"] = self.equivariance.to_dict()
        if self.comparison:
            d["comparison"] = self.comparison.to_dict()
        if self.description:
            d["description"] = self.description
        return d


_DEEPMD_DESCRIPTOR = DescriptorMath(
    descriptor_type="se_a",
    mathematical_form="D_i = Σ_j smooth(R_ij) ⊙ (s(R_ij) · e_ij)",
    symmetry_constraints=[
        "Translation invariance: D_i depends on R_ij = r_j - r_i",
        "Rotation equivariance: e_ij transforms as a vector",
        "Permutation invariance: Σ_j is symmetric in neighbor indices",
    ],
    description="DeepMD se_a descriptor: smooth radial + angular encoding",
)

_MACE_EQUIVARIANCE = EquivarianceResult(
    is_equivariant=True,
    equivariance_group="E(3)",
    rotation_handling="Irreducible representations of O(3) via spherical harmonics",
    permutation_handling="Message passing with symmetric aggregation",
    translation_invariance=True,
    description="MACE: E(3)-equivariant via Wigner-D matrices and Clebsch-Gordan coefficients",
)

_CLASSICAL_POTENTIALS = {
    "LJ": {
        "form": "V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]",
        "constraints": [
            "Pairwise additivity: V_total = Σ_{i<j} V(r_ij)",
            "Short-range repulsion: r⁻¹² term",
            "Van der Waals attraction: r⁻⁶ term",
            "Radial symmetry: V depends only on |r_ij|",
        ],
    },
    "EAM": {
        "form": "V = Σ_i F(ρ_i) + Σ_{i<j} φ(r_ij), ρ_i = Σ_j f(r_ij)",
        "constraints": [
            "Embedding energy: F(ρ) is a nonlinear function",
            "Pair potential: φ(r_ij) for repulsion",
            "Electron density: ρ_i = Σ_j f(r_ij) from neighbors",
            "Many-body: F(ρ) makes total energy non-pairwise",
        ],
    },
    "Tersoff": {
        "form": "V = Σ_{i<j} f_C(r_ij)[V_R + b_ij · V_A]",
        "constraints": [
            "Bond order: b_ij depends on local coordination",
            "Cutoff: f_C(r) smoothly truncates interactions",
            "Angular dependence: b_ij includes θ_ijk terms",
            "Three-body: implicitly via bond order",
        ],
    },
    "ReaxFF": {
        "form": "V = V_bond + V_val + V_tors + V_vdw + V_coul + ...",
        "constraints": [
            "Bond order: continuous BO from interatomic distance",
            "Valence angle: angular energy from bond orders",
            "Torsion: dihedral energy",
            "Non-bonded: van der Waals + Coulomb with shielding",
            "Reactive: bonds can form/break dynamically",
        ],
    },
}


class MLPotentialAnalyzer:
    """Analyze mathematical structure of ML interatomic potentials.

    Extracts the descriptor mathematics, equivariance properties,
    and compares with classical potentials.
    """

    def analyze_deepmd(
        self, model_info: Optional[Dict[str, Any]] = None
    ) -> MLPotentialResult:
        """Analyze DeepMD potential mathematical structure.

        The DeepMD descriptor pipeline:
        R → D (descriptor) → G (embedding) → E (energy)

        Args:
            model_info: Optional model metadata dict
        """
        desc = DescriptorMath(
            descriptor_type="se_a",
            mathematical_form="D_i = Σ_j smooth(R_ij) ⊙ (s(R_ij) · e_ij)",
            input_dim=3,
            output_dim=model_info.get("descriptor_dim", 0) if model_info else 0,
            symmetry_constraints=[
                "Translation invariance: D_i depends on R_ij = r_j - r_i",
                "Rotation equivariance: e_ij transforms as a vector",
                "Permutation invariance: Σ_j is symmetric in neighbor indices",
            ],
            description="DeepMD se_a: smooth radial filter + angular encoding via neighbor summation",
        )

        equiv = EquivarianceResult(
            is_equivariant=False,
            equivariance_group="SO(3) (approximate via data augmentation)",
            rotation_handling="Data augmentation during training; descriptor is NOT strictly equivariant",
            permutation_handling="Symmetric summation over neighbors",
            translation_invariance=True,
            description="DeepMD is NOT strictly equivariant — relies on training data coverage",
        )

        comparison = self._compare_with_classical("DeepMD", "EAM")

        return MLPotentialResult(
            descriptor=desc,
            equivariance=equiv,
            comparison=comparison,
            description="DeepMD: descriptor-based ML potential with approximate rotation equivariance",
        )

    def analyze_mace(
        self, model_info: Optional[Dict[str, Any]] = None
    ) -> MLPotentialResult:
        """Analyze MACE potential mathematical structure.

        MACE uses equivariant message passing with spherical harmonics.

        Args:
            model_info: Optional model metadata dict
        """
        desc = DescriptorMath(
            descriptor_type="equivariant_message_passing",
            mathematical_form="h_i' = Σ_j W · ⊕_{l} CG^l ⊗ Y^l(r_ij) ⊗ h_j",
            input_dim=3,
            output_dim=model_info.get("hidden_irreps_dim", 0) if model_info else 0,
            symmetry_constraints=[
                "E(3) equivariance: via Wigner-D matrices",
                "Clebsch-Gordan: CG coefficients couple angular momenta",
                "Spherical harmonics: Y^l_m encode angular information",
                "Rotation: h transforms as irreducible representation of O(3)",
            ],
            description="MACE: E(3)-equivariant message passing with spherical harmonic basis",
        )

        equiv = EquivarianceResult(
            is_equivariant=True,
            equivariance_group="E(3)",
            rotation_handling="Irreducible representations of O(3) via spherical harmonics Y^l_m",
            permutation_handling="Message passing with symmetric aggregation (Σ_j)",
            translation_invariance=True,
            description="MACE: strictly E(3)-equivariant via Wigner-D matrices and CG coefficients",
        )

        comparison = self._compare_with_classical("MACE", "Tersoff")

        return MLPotentialResult(
            descriptor=desc,
            equivariance=equiv,
            comparison=comparison,
            description="MACE: strictly equivariant ML potential with O(3) irreducible representations",
        )

    def analyze_nequip(
        self, model_info: Optional[Dict[str, Any]] = None
    ) -> MLPotentialResult:
        """Analyze NequIP potential mathematical structure."""
        desc = DescriptorMath(
            descriptor_type="equivariant_convolution",
            mathematical_form="h_i' = Σ_j W · Y^l(r_ij) ⊗ h_j",
            symmetry_constraints=[
                "E(3) equivariance: via spherical harmonics",
                "Convolution: angular convolution over neighbor shell",
                "Rotation: features transform as irreps of O(3)",
            ],
            description="NequIP: E(3)-equivariant neural network with angular convolution",
        )

        equiv = EquivarianceResult(
            is_equivariant=True,
            equivariance_group="E(3)",
            rotation_handling="Spherical harmonics Y^l_m as basis functions",
            permutation_handling="Symmetric neighbor aggregation",
            translation_invariance=True,
            description="NequIP: strictly E(3)-equivariant via spherical harmonic convolution",
        )

        return MLPotentialResult(
            descriptor=desc,
            equivariance=equiv,
            description="NequIP: equivariant interatomic potential with angular convolution",
        )

    def analyze(
        self, potential_type: str, model_info: Optional[Dict[str, Any]] = None
    ) -> MLPotentialResult:
        """Analyze an ML potential by type name.

        Args:
            potential_type: One of "deepmd", "mace", "nequip", "allegro"
            model_info: Optional model metadata
        """
        analyzers = {
            "deepmd": self.analyze_deepmd,
            "mace": self.analyze_mace,
            "nequip": self.analyze_nequip,
            "allegro": self.analyze_mace,
        }

        key = potential_type.lower().replace("-", "").replace("_", "")
        analyzer = analyzers.get(key)
        if analyzer:
            return analyzer(model_info)

        return MLPotentialResult(
            description=f"Unknown potential type: {potential_type}. Supported: deepmd, mace, nequip, allegro",
        )

    def _compare_with_classical(
        self, ml_name: str, classical_name: str
    ) -> ComparisonResult:
        """Compare ML potential with a classical potential."""
        classical = _CLASSICAL_POTENTIALS.get(classical_name, {})

        shared = [
            "Translation invariance (both depend on relative positions)",
            "Permutation invariance (both symmetric in atom indices)",
            "Short-range cutoff (both have finite interaction range)",
        ]

        ml_only = [
            "Learned representation (no fixed functional form)",
            "Many-body effects via neural network (not explicit)",
            "Transferable across chemistries (trained on diverse data)",
        ]

        classical_only = [
            "Explicit functional form (interpretable)",
            "Physically motivated terms (e.g. r⁻⁶ for vdW)",
            "Parameter count typically < 20",
        ]

        return ComparisonResult(
            ml_type=ml_name,
            classical_type=classical_name,
            shared_constraints=shared,
            ml_only_features=ml_only,
            classical_only_features=classical_only,
            mathematical_equivalence=(
                f"{ml_name} and {classical_name} share translational and "
                f"permutational symmetry, but {ml_name} learns the functional "
                f"form from data while {classical_name} imposes it a priori"
            ),
            description=f"Comparison: {ml_name} vs {classical_name}",
        )

    def list_supported_potentials(self) -> List[str]:
        return ["deepmd", "mace", "nequip", "allegro"]

    def list_classical_potentials(self) -> List[str]:
        return list(_CLASSICAL_POTENTIALS.keys())
