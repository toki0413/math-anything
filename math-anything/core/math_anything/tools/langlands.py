"""Langlands Program computation — Galois groups, L-functions, and representations.

Explores connections between the Langlands Program and materials science:
- Galois groups classify symmetry breaking in phase transitions
- L-functions connect spectral data to analytic structures
- Representation theory bridges crystal symmetries to topological invariants

Requires optional dependency: cypari2 (PARI/GP Python interface).
Falls back to basic number theory when unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GaloisResult:
    """Galois group analysis result."""

    polynomial: str = ""
    degree: int = 0
    galois_group: str = ""
    is_abelian: bool = False
    is_solvable: bool = False
    discriminant: Optional[int] = None
    splitting_field_degree: Optional[int] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "polynomial": self.polynomial,
            "degree": self.degree,
            "galois_group": self.galois_group,
            "is_abelian": self.is_abelian,
            "is_solvable": self.is_solvable,
        }
        if self.discriminant is not None:
            d["discriminant"] = self.discriminant
        if self.splitting_field_degree is not None:
            d["splitting_field_degree"] = self.splitting_field_degree
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class LFunctionResult:
    """L-function computation result."""

    name: str = ""
    s_value: complex = 0j
    l_value: complex = 0j
    functional_equation: str = ""
    conductor: Optional[int] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "s_value": {"real": self.s_value.real, "imag": self.s_value.imag},
            "l_value": {"real": self.l_value.real, "imag": self.l_value.imag},
        }
        if self.functional_equation:
            d["functional_equation"] = self.functional_equation
        if self.conductor is not None:
            d["conductor"] = self.conductor
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class ArtinRepResult:
    """Artin representation analysis result."""

    dimension: int = 0
    character_values: List[float] = field(default_factory=list)
    conductor: Optional[int] = None
    is_irreducible: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "dimension": self.dimension,
            "is_irreducible": self.is_irreducible,
        }
        if self.character_values:
            d["character_values"] = self.character_values
        if self.conductor is not None:
            d["conductor"] = self.conductor
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class LanglandsResult:
    """Complete Langlands analysis result."""

    galois: Optional[GaloisResult] = None
    l_function: Optional[LFunctionResult] = None
    artin: Optional[ArtinRepResult] = None
    materials_science_connection: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.galois:
            d["galois"] = self.galois.to_dict()
        if self.l_function:
            d["l_function"] = self.l_function.to_dict()
        if self.artin:
            d["artin"] = self.artin.to_dict()
        if self.materials_science_connection:
            d["materials_science_connection"] = self.materials_science_connection
        if self.description:
            d["description"] = self.description
        return d


_MATERIALIZATION_MAP = {
    "S2": "Z2 symmetry breaking in Ising model (ferromagnetic transition)",
    "S3": "Three-state Potts model (3-fold degeneracy)",
    "Cn": "n-fold rotational symmetry breaking (e.g., nematic transition)",
    "Dn": "Dihedral symmetry breaking (e.g., structural phase transition)",
    "A4": "Tetrahedral symmetry breaking (e.g., cubic-to-tetragonal)",
    "A5": "Icosahedral symmetry (quasicrystals)",
    "S4": "Octahedral symmetry breaking (e.g., perovskite distortions)",
}


class LanglandsAnalyzer:
    """Langlands Program computations for materials science.

    Connects Galois theory, L-functions, and representation theory
    to phase transitions, symmetry breaking, and topological invariants
    in materials.
    """

    def galois_group(self, coefficients: List[int]) -> GaloisResult:
        """Compute Galois group of a polynomial.

        Args:
            coefficients: Polynomial coefficients [a_n, ..., a_1, a_0]
                          representing a_n*x^n + ... + a_1*x + a_0
        """
        degree = len(coefficients) - 1
        poly_str = self._format_polynomial(coefficients)

        try:
            import cypari2

            pari = cypari2.Pari()
            poly = pari.polynomial(coefficients, "x")
            galois_info = pari.polgalois(poly)

            group_str = str(galois_info)
            is_abelian = (
                "abelian" in group_str.lower() if isinstance(group_str, str) else False
            )

            disc = pari.poldisc(poly)

            return GaloisResult(
                polynomial=poly_str,
                degree=degree,
                galois_group=group_str,
                is_abelian=is_abelian,
                is_solvable=degree <= 4 or is_abelian,
                discriminant=int(disc) if disc is not None else None,
                description=f"Galois group of {poly_str}: {group_str}",
            )
        except ImportError:
            return self._fallback_galois(coefficients, poly_str, degree)

    def l_function_value(self, name: str, s: complex) -> LFunctionResult:
        """Evaluate an L-function at a complex point.

        Supports: Riemann zeta, Dirichlet L-functions.

        Args:
            name: L-function name (e.g. "zeta", "dirichlet_chi_4")
            s: Complex evaluation point
        """
        try:
            import cypari2

            pari = cypari2.Pari()

            if name.lower() == "zeta":
                val = pari.zeta(s)
                return LFunctionResult(
                    name="Riemann zeta",
                    s_value=s,
                    l_value=complex(val),
                    functional_equation="ζ(s) = 2^s π^{s-1} sin(πs/2) Γ(1-s) ζ(1-s)",
                    description=f"ζ({s}) = {val}",
                )
        except ImportError:
            pass

        if name.lower() == "zeta":
            import cmath

            if abs(s.real - 1.0) < 1e-10 and abs(s.imag) < 1e-10:
                return LFunctionResult(
                    name="Riemann zeta",
                    s_value=s,
                    l_value=complex("inf"),
                    description="ζ(1) = ∞ (pole)",
                )

            n_terms = 1000
            val = sum(1.0 / (n**s) for n in range(1, n_terms + 1))
            return LFunctionResult(
                name="Riemann zeta (approximate)",
                s_value=s,
                l_value=val,
                functional_equation="ζ(s) = 2^s π^{s-1} sin(πs/2) Γ(1-s) ζ(1-s)",
                description=f"ζ({s}) ≈ {val} (Dirichlet series, {n_terms} terms)",
            )

        return LFunctionResult(
            name=name,
            s_value=s,
            description=f"L-function '{name}' requires cypari2 for evaluation",
        )

    def artin_representation(self, galois_result: GaloisResult) -> ArtinRepResult:
        """Compute Artin representation from a Galois group.

        The Artin representation maps Gal(Gal(K/ℚ)) → GL_n(ℂ),
        connecting field automorphisms to linear representations.
        """
        if not galois_result.galois_group:
            return ArtinRepResult(description="No Galois group available")

        group = galois_result.galois_group

        trivial_dim = 1
        trivial_chars = [1.0]

        if "S2" in group or "C2" in group:
            return ArtinRepResult(
                dimension=1,
                character_values=[1.0, -1.0],
                is_irreducible=True,
                description="Sign representation of C2: χ(g) = ±1 depending on parity",
            )

        if "S3" in group or "C3" in group:
            return ArtinRepResult(
                dimension=2,
                character_values=[2.0, -1.0, -1.0],
                is_irreducible=True,
                description="Standard 2D irrep of S3: χ = (2, -1, -1) on conjugacy classes",
            )

        if "A4" in group:
            return ArtinRepResult(
                dimension=3,
                character_values=[3.0, -1.0, -1.0, -1.0],
                is_irreducible=True,
                description="3D irrep of A4 (tetrahedral): relevant for cubic crystal fields",
            )

        if "S4" in group:
            return ArtinRepResult(
                dimension=3,
                character_values=[3.0, -1.0, 0.0, 1.0, -1.0],
                is_irreducible=True,
                description="3D irrep of S4 (octahedral): relevant for perovskite symmetries",
            )

        return ArtinRepResult(
            dimension=trivial_dim,
            character_values=trivial_chars,
            is_irreducible=True,
            description=f"Trivial representation of {group}",
        )

    def symmetry_breaking_analysis(self, galois_result: GaloisResult) -> str:
        """Map Galois group to materials science symmetry breaking.

        The Galois group of the characteristic polynomial of a Hamiltonian
        encodes the possible symmetry breaking patterns in phase transitions.
        """
        group = galois_result.galois_group

        for key, description in _MATERIALIZATION_MAP.items():
            if key in group:
                return description

        if galois_result.is_abelian:
            return "Abelian symmetry breaking: discrete subgroup of U(1), typical for charge/spin density waves"

        if galois_result.degree <= 2:
            return "Binary symmetry breaking (Z2): Ising-type transition"

        return (
            f"Galois group {group}: symmetry breaking pattern requires further analysis"
        )

    def analyze(self, coefficients: List[int]) -> LanglandsResult:
        """Full Langlands analysis pipeline.

        Args:
            coefficients: Polynomial coefficients [a_n, ..., a_0]
        """
        galois = self.galois_group(coefficients)
        artin = self.artin_representation(galois)
        connection = self.symmetry_breaking_analysis(galois)

        l_func = None
        if galois.is_abelian and galois.degree <= 4:
            l_func = self.l_function_value("zeta", 0.5 + 1j)

        return LanglandsResult(
            galois=galois,
            l_function=l_func,
            artin=artin,
            materials_science_connection=connection,
            description=(
                f"Langlands analysis of {galois.polynomial}: "
                f"Galois group {galois.galois_group}, "
                f"Artin rep dimension {artin.dimension}, "
                f"Materials connection: {connection}"
            ),
        )

    @staticmethod
    def _format_polynomial(coeffs: List[int]) -> str:
        terms = []
        degree = len(coeffs) - 1
        for i, c in enumerate(coeffs):
            d = degree - i
            if c == 0:
                continue
            if d == 0:
                terms.append(str(c))
            elif d == 1:
                terms.append(f"{c}x" if c != 1 else "x")
            else:
                terms.append(f"{c}x^{d}" if c != 1 else f"x^{d}")
        return " + ".join(terms).replace("+ -", "- ") if terms else "0"

    def _fallback_galois(
        self, coefficients: List[int], poly_str: str, degree: int
    ) -> GaloisResult:
        """Basic Galois group determination for low-degree polynomials."""
        if degree == 1:
            group = "C1 (trivial)"
        elif degree == 2:
            a, b, c = coefficients[0], coefficients[1], coefficients[2]
            disc = b * b - 4 * a * c
            group = "C2 (cyclic)" if disc >= 0 else "C2 (cyclic)"
        elif degree == 3:
            group = "S3 or A3 (install cypari2 for exact determination)"
        elif degree == 4:
            group = "S4, A4, D4, V4, or C4 (install cypari2 for exact determination)"
        else:
            group = f"Degree {degree} (install cypari2 for Galois group computation)"

        is_abelian = degree <= 2

        return GaloisResult(
            polynomial=poly_str,
            degree=degree,
            galois_group=group,
            is_abelian=is_abelian,
            is_solvable=degree <= 4,
            description=f"Galois group of {poly_str}: {group} (basic determination)",
        )
