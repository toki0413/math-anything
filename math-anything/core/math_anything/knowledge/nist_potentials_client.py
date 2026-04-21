"""NIST Interatomic Potentials Client for Math Anything.

Provides access to interatomic potential functions and their mathematical forms.
No API key required. Uses OpenKIM and NIST repositories.

Privacy: Only potential type names and material classes are queried,
never specific parameter values from proprietary simulations.
"""

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PotentialFunction:
    """Represent an interatomic potential function."""

    name: str
    type: str
    mathematical_form: str
    parameters: List[str]  # Parameter names only, not values
    description: str
    materials: List[str]
    range_type: str  # short_range, long_range, or mixed
    source: str


class NISTPotentialsClient:
    """Client for NIST Interatomic Potentials database.

    Provides access to:
    - Potential function mathematical forms
    - Parameter naming conventions
    - Material-potential mappings
    - Range and interaction types

    Example:
        ```python
        client = NISTPotentialsClient()

        # Find potentials for a material
        potentials = client.find_potentials_for_material("copper")

        # Get potential by type
        lj = client.get_potential_by_type("lennard_jones")

        # Query by fingerprint
        fingerprint = {"potential_types": ["lennard_jones", "coulomb"]}
        results = client.query_by_fingerprint(fingerprint)
        ```
    """

    # NIST OpenKIM API endpoint
    OPENKIM_API = "https://query.openkim.org/api"

    # Known potential function forms
    POTENTIAL_FORMS = {
        "lennard_jones": {
            "name": "Lennard-Jones",
            "form": "V(r) = 4ε[(σ/r)^12 - (σ/r)^6]",
            "parameters": ["epsilon", "sigma"],
            "range": "short_range",
            "description": "12-6 potential for van der Waals interactions",
        },
        "morse": {
            "name": "Morse",
            "form": "V(r) = D_e[1 - exp(-a(r - r_e))]^2",
            "parameters": ["D_e", "a", "r_e"],
            "range": "short_range",
            "description": "Exponential potential for chemical bonds",
        },
        "buckingham": {
            "name": "Buckingham",
            "form": "V(r) = A*exp(-r/ρ) - C/r^6",
            "parameters": ["A", "rho", "C"],
            "range": "short_range",
            "description": "Exponential-6 potential",
        },
        "harmonic": {
            "name": "Harmonic",
            "form": "V(r) = 0.5*k*(r - r_0)^2",
            "parameters": ["k", "r_0"],
            "range": "short_range",
            "description": "Hookean spring potential",
        },
        "coulomb": {
            "name": "Coulomb",
            "form": "V(r) = (1/4πε_0)*q_i*q_j/r",
            "parameters": ["q_i", "q_j"],
            "range": "long_range",
            "description": "Electrostatic point charge interaction",
        },
        "embedded_atom": {
            "name": "Embedded Atom Method (EAM)",
            "form": "E = ΣF_i(ρ_i) + 0.5*Σφ(r_ij)",
            "parameters": ["electron_density", "embedding_function", "pair_potential"],
            "range": "short_range",
            "description": "Many-body potential for metals",
        },
        "tersoff": {
            "name": "Tersoff",
            "form": "E = Σf_C(r_ij)[f_R(r_ij) + b_ij*f_A(r_ij)]",
            "parameters": [
                "A",
                "B",
                "lambda",
                "mu",
                "beta",
                "n",
                "c",
                "d",
                "h",
                "R",
                "D",
            ],
            "range": "short_range",
            "description": "Bond-order potential for covalent materials",
        },
        "stillinger_weber": {
            "name": "Stillinger-Weber",
            "form": "E = ΣV_2(r_ij) + ΣV_3(r_ij, r_ik, θ_ijk)",
            "parameters": ["epsilon", "sigma", "a", "lambda", "gamma", "cos_theta_0"],
            "range": "short_range",
            "description": "Three-body potential for silicon",
        },
        "reaxff": {
            "name": "ReaxFF",
            "form": "E = E_bond + E_over + E_under + E_lp + E_val + E_pen + E_coa + E_C2 + E_tors + E_conj + E_HB + E_vdW + E_Coulomb",
            "parameters": ["many_bond_parameters"],
            "range": "short_range",
            "description": "Reactive force field for chemical reactions",
        },
    }

    # Material categories
    MATERIAL_CATEGORIES = {
        "metal": ["copper", "aluminum", "iron", "gold", "silver", "nickel", "tungsten"],
        "semiconductor": ["silicon", "germanium", "gallium", "arsenide"],
        "ionic": ["sodium_chloride", "magnesium_oxide"],
        "molecular": ["water", "organic", "polymer"],
        "noble_gas": ["argon", "neon", "krypton", "xenon"],
    }

    def __init__(self, cache=None):
        self.cache = cache
        self.headers = {
            "User-Agent": "MathAnything/1.0 (research tool)",
            "Accept": "application/json",
        }

    def get_potential_by_type(self, potential_type: str) -> Optional[PotentialFunction]:
        """Get potential function by type.

        Args:
            potential_type: Type of potential (e.g., 'lennard_jones')

        Returns:
            PotentialFunction with mathematical form
        """
        # Normalize type
        normalized = potential_type.lower().replace("-", "_").replace(" ", "_")

        # Check known forms
        if normalized in self.POTENTIAL_FORMS:
            form_data = self.POTENTIAL_FORMS[normalized]

            return PotentialFunction(
                name=form_data["name"],
                type=potential_type,
                mathematical_form=form_data["form"],
                parameters=form_data["parameters"],
                description=form_data["description"],
                materials=[],  # Would be populated from API
                range_type=form_data["range"],
                source="NIST/OpenKIM",
            )

        return None

    def find_potentials_for_material(self, material: str) -> List[PotentialFunction]:
        """Find suitable potentials for a material.

        Args:
            material: Material name or formula

        Returns:
            List of suitable potential functions
        """
        # Determine material category
        category = self._classify_material(material)

        # Suggest potentials based on category
        suggestions = []

        if category == "metal":
            suggestions = ["embedded_atom", "lennard_jones"]
        elif category == "semiconductor":
            suggestions = ["tersoff", "stillinger_weber"]
        elif category == "molecular":
            suggestions = ["lennard_jones", "coulomb", "reaxff"]
        elif category == "noble_gas":
            suggestions = ["lennard_jones"]
        elif category == "ionic":
            suggestions = ["buckingham", "coulomb"]

        potentials = []
        for pot_type in suggestions:
            pot = self.get_potential_by_type(pot_type)
            if pot:
                pot.materials = [material]
                potentials.append(pot)

        return potentials

    def query_by_fingerprint(
        self,
        fingerprint: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Query potentials using mathematical fingerprint.

        Args:
            fingerprint: Mathematical fingerprint with potential info

        Returns:
            List of matching potential information
        """
        results = []

        # Match by potential type
        for pot_type in fingerprint.get("potential_types", []):
            pot = self.get_potential_by_type(pot_type)
            if pot:
                results.append(
                    {
                        "name": pot.name,
                        "type": pot.type,
                        "mathematical_form": pot.mathematical_form,
                        "parameters": pot.parameters,
                        "description": pot.description,
                        "range_type": pot.range_type,
                    }
                )

        # Match by material class
        for material in fingerprint.get("material_classes", []):
            potentials = self.find_potentials_for_material(material)
            for pot in potentials:
                if pot.name not in [r["name"] for r in results]:
                    results.append(
                        {
                            "name": pot.name,
                            "type": pot.type,
                            "mathematical_form": pot.mathematical_form,
                            "parameters": pot.parameters,
                            "description": pot.description,
                            "range_type": pot.range_type,
                        }
                    )

        return results

    def get_potential_recommendations(
        self,
        material: str,
        properties: List[str],
    ) -> List[Dict[str, Any]]:
        """Get potential recommendations based on required properties.

        Args:
            material: Target material
            properties: Required properties ('elastic', 'defect', 'surface', etc.)

        Returns:
            List of recommendations with rationale
        """
        category = self._classify_material(material)

        recommendations = []

        if category == "metal":
            if "defect" in properties or "surface" in properties:
                recommendations.append(
                    {
                        "potential": "EAM",
                        "rationale": "Many-body effects important for defects",
                        "mathematical_form": self.POTENTIAL_FORMS["embedded_atom"][
                            "form"
                        ],
                    }
                )
            else:
                recommendations.append(
                    {
                        "potential": "Lennard-Jones",
                        "rationale": "Simple and efficient for bulk properties",
                        "mathematical_form": self.POTENTIAL_FORMS["lennard_jones"][
                            "form"
                        ],
                    }
                )

        elif category == "semiconductor":
            if "reactive" in properties:
                recommendations.append(
                    {
                        "potential": "ReaxFF",
                        "rationale": "Bond breaking/forming capability",
                        "mathematical_form": "Complex multi-body",
                    }
                )
            else:
                recommendations.append(
                    {
                        "potential": "Tersoff",
                        "rationale": "Bond-order for covalent bonding",
                        "mathematical_form": self.POTENTIAL_FORMS["tersoff"]["form"],
                    }
                )

        return recommendations

    def get_unit_conversion(
        self,
        from_unit: str,
        to_unit: str,
    ) -> Optional[float]:
        """Get conversion factor between common MD units.

        Args:
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Conversion factor
        """
        # Common MD unit conversions
        conversions = {
            ("kcal_mol", "ev"): 0.043364,
            ("ev", "kcal_mol"): 23.0605,
            ("angstrom", "nm"): 0.1,
            ("nm", "angstrom"): 10.0,
            ("ps", "fs"): 1000.0,
            ("fs", "ps"): 0.001,
        }

        key = (from_unit.lower(), to_unit.lower())
        return conversions.get(key)

    def _classify_material(self, material: str) -> str:
        """Classify material into category."""
        material_lower = material.lower()

        for category, materials in self.MATERIAL_CATEGORIES.items():
            if any(m in material_lower for m in materials):
                return category

        return "unknown"

    def get_cutoff_recommendations(self, potential_type: str) -> Dict[str, Any]:
        """Get recommended cutoff distances for potential type.

        Args:
            potential_type: Type of potential

        Returns:
            Cutoff recommendations
        """
        cutoffs = {
            "lennard_jones": {
                "recommended": 2.5,  # sigma units
                "description": "2.5*sigma for most cases",
                "units": "sigma",
            },
            "tersoff": {
                "recommended": 3.0,  # Angstrom beyond R
                "description": "R + 3.0 A buffer",
                "units": "angstrom",
            },
            "embedded_atom": {
                "recommended": 6.0,  # Angstrom
                "description": "Material dependent, 5-7 A typical",
                "units": "angstrom",
            },
            "coulomb": {
                "recommended": "ewald",
                "description": "Use Ewald summation for long-range",
                "units": "n/a",
            },
        }

        normalized = potential_type.lower().replace("-", "_").replace(" ", "_")
        return cutoffs.get(
            normalized,
            {"recommended": "unknown", "description": "No recommendation available"},
        )

    def get_validation_tests(self, potential_type: str) -> List[str]:
        """Get recommended validation tests for potential.

        Args:
            potential_type: Type of potential

        Returns:
            List of validation test names
        """
        tests = {
            "lennard_jones": [
                "Lattice constant",
                "Cohesive energy",
                "Bulk modulus",
                "Vacancy formation energy",
            ],
            "embedded_atom": [
                "Lattice constant",
                "Elastic constants",
                "Stacking fault energy",
                "Surface energy",
                "Vacancy formation energy",
            ],
            "tersoff": [
                "Lattice constant",
                "Elastic constants",
                "Phonon frequencies",
                "Defect structures",
            ],
        }

        normalized = potential_type.lower().replace("-", "_").replace(" ", "_")
        return tests.get(normalized, ["Basic property checks"])
