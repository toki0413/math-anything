"""Wikidata Client for Math Anything.

Provides access to mathematical concepts, units, constants, and physical quantities.
No API key required. Uses SPARQL endpoint.

Privacy: Only concept names and unit types are queried,
never specific values or proprietary data.
"""

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class WikidataEntity:
    """Represent a Wikidata entity."""

    id: str
    label: str
    description: str
    properties: Dict[str, Any]
    uri: str


class WikidataClient:
    """Client for Wikidata queries.

    Provides access to:
    - Mathematical concepts and theorems
    - Physical units and constants
    - Physical quantities and dimensions
    - Numerical methods and algorithms

    Example:
        ```python
        client = WikidataClient()

        # Get information about a mathematical concept
        concept = client.get_concept("Laplacian operator")

        # Get unit conversions
        unit = client.get_unit("Pascal")

        # Find related concepts
        related = client.find_related("Navier-Stokes equations")
        ```
    """

    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    API_BASE = "https://www.wikidata.org/w/api.php"

    # Mathematical concept Q-IDs (for direct lookup)
    MATH_CONCEPTS = {
        "partial_differential_equation": "Q2718787",
        "laplacian": "Q173165",
        "navier_stokes_equations": "Q2000011",
        "finite_element_method": "Q236425",
        "taylor_series": "Q131187",
        "fourier_transform": "Q652015",
        "eigenvalue": "Q190524",
        "tensor": "Q173840",
        "boundary_condition": "Q1088667",
        "gradient": "Q173582",
        "divergence": "Q189319",
        "curl": "Q162045",
        "poisson_equation": "Q633674",
        "heat_equation": "Q240775",
        "wave_equation": "Q37105",
        "maxwell_equations": "Q852021",
        "schrodinger_equation": "Q11401",
    }

    # Unit Q-IDs
    UNITS = {
        "meter": "Q11573",
        "kilogram": "Q11570",
        "second": "Q11574",
        "pascal": "Q44395",
        "newton": "Q12438",
        "joule": "Q25269",
        "watt": "Q17009",
        "kelvin": "Q11579",
        "mole": "Q127140",
        "radian": "Q173885",
        "steradian": "Q177389",
        "hertz": "Q39369",
        "poise": "Q215571",
        "pascal_second": "Q207269",
    }

    # Physical constant Q-IDs
    CONSTANTS = {
        "speed_of_light": "Q2111",
        "gravitational_constant": "Q18373",
        "planck_constant": "Q42289",
        "boltzmann_constant": "Q102839",
        "avogadro_constant": "Q18388",
        "gas_constant": "Q18373",
        "electron_charge": "Q2101",
        "vacuum_permittivity": "Q190029",
        "vacuum_permeability": "Q2130478",
    }

    def __init__(self, cache=None):
        self.cache = cache
        self.headers = {
            "User-Agent": "MathAnything/1.0 (research tool)",
            "Accept": "application/json",
        }

    def get_concept(self, concept_name: str) -> Optional[WikidataEntity]:
        """Get mathematical concept by name.

        Args:
            concept_name: Name of the concept

        Returns:
            WikidataEntity with concept information
        """
        # Check cache
        if self.cache:
            cache_key = f"wikidata:concept:{concept_name}"
            cached = self.cache.get(cache_key)
            if cached:
                return WikidataEntity(**cached)

        # Normalize concept name
        normalized = concept_name.lower().replace(" ", "_").replace("-", "_")

        # Check known concepts
        if normalized in self.MATH_CONCEPTS:
            qid = self.MATH_CONCEPTS[normalized]
            return self._get_entity_by_id(qid)

        # Search for concept
        return self._search_entity(concept_name, "Q")

    def get_unit(self, unit_name: str) -> Optional[WikidataEntity]:
        """Get unit information.

        Args:
            unit_name: Name of the unit

        Returns:
            WikidataEntity with unit information
        """
        # Check cache
        if self.cache:
            cache_key = f"wikidata:unit:{unit_name}"
            cached = self.cache.get(cache_key)
            if cached:
                return WikidataEntity(**cached)

        # Normalize
        normalized = unit_name.lower().replace(" ", "_")

        # Check known units
        if normalized in self.UNITS:
            qid = self.UNITS[normalized]
            return self._get_entity_by_id(qid)

        return self._search_entity(unit_name, "Q")

    def get_constant(self, constant_name: str) -> Optional[WikidataEntity]:
        """Get physical constant information.

        Args:
            constant_name: Name of the constant

        Returns:
            WikidataEntity with constant information
        """
        normalized = constant_name.lower().replace(" ", "_")

        if normalized in self.CONSTANTS:
            qid = self.CONSTANTS[normalized]
            return self._get_entity_by_id(qid)

        return self._search_entity(constant_name, "Q")

    def find_related(
        self,
        concept_name: str,
        relation_type: str = "subclass_of",
    ) -> List[WikidataEntity]:
        """Find concepts related to given concept.

        Args:
            concept_name: Starting concept
            relation_type: Type of relation

        Returns:
            List of related entities
        """
        entity = self.get_concept(concept_name)
        if not entity:
            return []

        query = f"""
        SELECT ?item ?itemLabel ?itemDescription WHERE {{
          ?item wdt:P279 wd:{entity.id}.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 10
        """

        return self._execute_sparql(query)

    def get_equation_info(self, equation_name: str) -> Dict[str, Any]:
        """Get information about a mathematical equation.

        Args:
            equation_name: Name of the equation

        Returns:
            Dictionary with equation information
        """
        # Known equation mappings
        equation_map = {
            "navier-stokes": "Q2000011",
            "navier stokes": "Q2000011",
            "heat": "Q240775",
            "wave": "Q37105",
            "poisson": "Q633674",
            "laplace": "Q6752989",
            "schrodinger": "Q11401",
            "maxwell": "Q852021",
            "euler": "Q852062",
            "cauchy-momentum": "Q1985209",
        }

        normalized = equation_name.lower()

        for key, qid in equation_map.items():
            if key in normalized:
                entity = self._get_entity_by_id(qid)
                if entity:
                    return {
                        "name": entity.label,
                        "description": entity.description,
                        "wikidata_id": entity.id,
                        "uri": entity.uri,
                    }

        return {}

    def get_numerical_method_info(self, method_name: str) -> Dict[str, Any]:
        """Get information about a numerical method.

        Args:
            method_name: Name of the numerical method

        Returns:
            Dictionary with method information
        """
        method_map = {
            "finite element": "Q236425",
            "finite difference": "Q2993797",
            "finite volume": "Q2139966",
            "spectral": "Q1423980",
            "monte carlo": "Q232207",
            "molecular dynamics": "Q114151",
            "lattice boltzmann": "Q897811",
            "runge-kutta": "Q725944",
            "newton-raphson": "Q183427",
            "conjugate gradient": "Q1455583",
            "multigrid": "Q1348371",
        }

        normalized = method_name.lower()

        for key, qid in method_map.items():
            if key in normalized:
                entity = self._get_entity_by_id(qid)
                if entity:
                    return {
                        "name": entity.label,
                        "description": entity.description,
                        "wikidata_id": entity.id,
                        "uri": entity.uri,
                    }

        return {}

    def query_by_fingerprint(self, fingerprint: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query Wikidata using mathematical fingerprint.

        Args:
            fingerprint: Mathematical fingerprint

        Returns:
            List of relevant concept information
        """
        results = []

        # Query concepts
        for concept in fingerprint.get("mathematical_concepts", []):
            info = self.get_concept(concept)
            if info:
                results.append(
                    {
                        "type": "concept",
                        "name": info.label,
                        "description": info.description,
                    }
                )

        # Query numerical methods
        for method in fingerprint.get("tensor_operations", []):
            info = self.get_numerical_method_info(method)
            if info:
                results.append({"type": "numerical_method", **info})

        return results

    def _get_entity_by_id(self, qid: str) -> Optional[WikidataEntity]:
        """Get entity by Q-ID."""
        # Check cache
        if self.cache:
            cache_key = f"wikidata:entity:{qid}"
            cached = self.cache.get(cache_key)
            if cached:
                return WikidataEntity(**cached)

        url = f"{self.API_BASE}?action=wbgetentities&ids={qid}&format=json&props=labels|descriptions|claims"

        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            entity_data = data.get("entities", {}).get(qid, {})

            if not entity_data:
                return None

            entity = WikidataEntity(
                id=qid,
                label=entity_data.get("labels", {}).get("en", {}).get("value", ""),
                description=entity_data.get("descriptions", {})
                .get("en", {})
                .get("value", ""),
                properties={},
                uri=f"http://www.wikidata.org/entity/{qid}",
            )

            # Cache result
            if self.cache:
                self.cache.set(
                    cache_key,
                    {
                        "id": entity.id,
                        "label": entity.label,
                        "description": entity.description,
                        "properties": entity.properties,
                        "uri": entity.uri,
                    },
                    ttl=86400,
                )

            return entity

        except Exception as e:
            print(f"Wikidata entity error: {e}")
            return None

    def _search_entity(
        self, search_term: str, type_filter: str = ""
    ) -> Optional[WikidataEntity]:
        """Search for entity by term."""
        url = f"{self.API_BASE}?action=wbsearchentities&search={urllib.parse.quote(search_term)}&format=json&language=en&limit=1"

        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            results = data.get("search", [])
            if results:
                qid = results[0].get("id", "")
                if qid:
                    return self._get_entity_by_id(qid)

            return None

        except Exception as e:
            print(f"Wikidata search error: {e}")
            return None

    def _execute_sparql(self, query: str) -> List[WikidataEntity]:
        """Execute SPARQL query."""
        url = f"{self.SPARQL_ENDPOINT}?query={urllib.parse.quote(query)}&format=json"

        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            results = []
            for binding in data.get("results", {}).get("bindings", []):
                item = binding.get("item", {})
                qid = item.get("value", "").split("/")[-1]

                label = binding.get("itemLabel", {}).get("value", "")
                description = binding.get("itemDescription", {}).get("value", "")

                if qid:
                    results.append(
                        WikidataEntity(
                            id=qid,
                            label=label,
                            description=description,
                            properties={},
                            uri=item.get("value", ""),
                        )
                    )

            return results

        except Exception as e:
            print(f"SPARQL error: {e}")
            return []
