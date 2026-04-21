"""Mathematical Fingerprint Extractor for Privacy-Preserving Knowledge Lookup.

Extracts abstract mathematical signatures without exposing:
- Specific parameter values
- Proprietary model details
- Raw simulation data

The fingerprint contains only the mathematical structure that is needed
for knowledge base queries, similar to how a hash preserves uniqueness
without revealing the original data.
"""

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class FingerprintType(Enum):
    """Types of mathematical fingerprints."""

    EQUATION_STRUCTURE = "equation_structure"
    TENSOR_RANK = "tensor_rank"
    BOUNDARY_CONDITION_TYPE = "bc_type"
    NUMERICAL_METHOD = "numerical_method"
    DISCRETIZATION_SCHEME = "discretization"
    COUPLING_PATTERN = "coupling_pattern"
    POTENTIAL_FORM = "potential_form"
    UNIT_DIMENSION = "unit_dimension"


@dataclass
class MathFingerprint:
    """Privacy-preserving mathematical fingerprint.

    Contains only structural information, no actual values:
    - Equation types and forms (not coefficients)
    - Tensor ranks (not component values)
    - Method names (not parameters)
    - Coupling patterns (not coupling strengths)
    """

    fingerprint_id: str
    fingerprint_type: FingerprintType
    structure_hash: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint_id": self.fingerprint_id,
            "type": self.fingerprint_type.value,
            "structure_hash": self.structure_hash,
            "metadata": self.metadata,
        }


class MathFingerprintExtractor:
    """Extracts mathematical fingerprints for privacy-safe knowledge lookup.

    Example:
        ```python
        schema = harness.extract_math(...)  # Extract from simulation

        extractor = MathFingerprintExtractor()
        fingerprints = extractor.extract_from_schema(schema)

        # Only fingerprints are sent to knowledge base
        # No actual simulation parameters are exposed
        results = arxiv_client.query(fingerprints)
        ```
    """

    def __init__(self):
        self.fingerprints: List[MathFingerprint] = []

    def extract_from_schema(self, schema) -> List[MathFingerprint]:
        """Extract all fingerprints from a MathSchema.

        Args:
            schema: MathSchema to fingerprint

        Returns:
            List of mathematical fingerprints
        """
        self.fingerprints = []

        # Extract equation fingerprints
        self._fingerprint_equations(schema)

        # Extract boundary condition fingerprints
        self._fingerprint_boundary_conditions(schema)

        # Extract numerical method fingerprints
        self._fingerprint_numerical_methods(schema)

        # Extract tensor structure fingerprints
        self._fingerprint_tensors(schema)

        # Extract coupling pattern fingerprints
        self._fingerprint_coupling(schema)

        return self.fingerprints

    def _fingerprint_equations(self, schema):
        """Extract equation structure fingerprints."""
        for eq in getattr(schema, "governing_equations", []):
            # Extract only the mathematical form structure
            # Remove all numeric coefficients
            form_structure = self._abstract_equation_form(eq.mathematical_form)

            fingerprint = MathFingerprint(
                fingerprint_id=f"eq_{eq.id}",
                fingerprint_type=FingerprintType.EQUATION_STRUCTURE,
                structure_hash=self._hash_structure(form_structure),
                metadata={
                    "equation_type": eq.type,
                    "mathematical_form_pattern": form_structure,
                    "variable_types": [v.get("type", "unknown") for v in eq.variables],
                    "tensor_ranks": [
                        v.get("tensor_rank", 0)
                        for v in eq.variables
                        if "tensor_rank" in v
                    ],
                },
            )
            self.fingerprints.append(fingerprint)

    def _fingerprint_boundary_conditions(self, schema):
        """Extract BC type fingerprints (no values)."""
        for bc in getattr(schema, "boundary_conditions", []):
            fingerprint = MathFingerprint(
                fingerprint_id=f"bc_{bc.id}",
                fingerprint_type=FingerprintType.BOUNDARY_CONDITION_TYPE,
                structure_hash=self._hash_structure(bc.type),
                metadata={
                    "bc_type": bc.type,
                    "region_type": self._classify_region(bc.region),
                    "has_tensor": any(v.get("type") == "tensor" for v in bc.variables),
                    "physical_category": self._categorize_physics(bc.physical_meaning),
                },
            )
            self.fingerprints.append(fingerprint)

    def _fingerprint_numerical_methods(self, schema):
        """Extract numerical method fingerprints."""
        for method in getattr(schema, "numerical_methods", []):
            # Extract method type without specific parameters
            disc = method.discretization
            fingerprint = MathFingerprint(
                fingerprint_id=f"method_{method.id}",
                fingerprint_type=FingerprintType.NUMERICAL_METHOD,
                structure_hash=self._hash_structure(method.name),
                metadata={
                    "method_name": method.name,
                    "discretization_type": disc.mesh_type if disc else "unknown",
                    "spatial_order": disc.spatial_order if disc else 0,
                    "temporal_order": disc.temporal_order if disc else 0,
                    "has_adaptive": "adaptive" in method.parameters,
                },
            )
            self.fingerprints.append(fingerprint)

    def _fingerprint_tensors(self, schema):
        """Extract tensor rank fingerprints (no component values)."""
        for obj in getattr(schema, "mathematical_objects", []):
            if hasattr(obj, "tensor_rank") and obj.tensor_rank is not None:
                fingerprint = MathFingerprint(
                    fingerprint_id=f"tensor_{obj.id}",
                    fingerprint_type=FingerprintType.TENSOR_RANK,
                    structure_hash=self._hash_structure(f"rank_{obj.tensor_rank}"),
                    metadata={
                        "tensor_rank": obj.tensor_rank,
                        "object_type": obj.type,
                        "num_components": (
                            len(obj.components)
                            if hasattr(obj, "components") and obj.components
                            else 0
                        ),
                        "has_symmetry": (
                            "symmetric" in str(obj.properties).lower()
                            if hasattr(obj, "properties")
                            else False
                        ),
                    },
                )
                self.fingerprints.append(fingerprint)

    def _fingerprint_coupling(self, schema):
        """Extract coupling pattern fingerprints."""
        for graph in getattr(schema, "computational_graphs", []):
            # Extract coupling topology without edge weights
            coupling_pattern = self._extract_coupling_pattern(graph)

            fingerprint = MathFingerprint(
                fingerprint_id=f"coupling_{graph.id}",
                fingerprint_type=FingerprintType.COUPLING_PATTERN,
                structure_hash=self._hash_structure(
                    json.dumps(coupling_pattern, sort_keys=True)
                ),
                metadata={
                    "num_nodes": len(graph.nodes),
                    "num_edges": len(graph.edges),
                    "coupling_topology": coupling_pattern,
                    "has_feedback_loop": self._detect_feedback(graph),
                },
            )
            self.fingerprints.append(fingerprint)

    def _abstract_equation_form(self, form: str) -> str:
        """Remove numerical coefficients from equation, keep structure.

        Example:
            "3.14*x^2 + 2.5*y = 0" -> "[COEFF]*x^2 + [COEFF]*y = 0"
        """
        import re

        # Replace numbers with placeholder
        # Keep mathematical symbols and operators
        abstract = re.sub(r"\b\d+\.?\d*\b", "[COEFF]", form)

        # Replace specific function parameters
        abstract = re.sub(r"\([^)]*\)", "([ARGS])", abstract)

        return abstract

    def _hash_structure(self, content: str) -> str:
        """Create deterministic hash of structure."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _classify_region(self, region: str) -> str:
        """Classify region type without specifics."""
        region_lower = region.lower()
        if any(x in region_lower for x in ["wall", "boundary", "surface"]):
            return "boundary"
        elif any(x in region_lower for x in ["volume", "domain", "region"]):
            return "domain"
        elif any(x in region_lower for x in ["point", "node"]):
            return "point"
        else:
            return "unspecified"

    def _categorize_physics(self, meaning: str) -> str:
        """Categorize physical meaning."""
        meaning_lower = meaning.lower()
        categories = {
            "force": ["force", "load", "pressure", "traction"],
            "displacement": ["displacement", "velocity", "fix"],
            "thermal": ["temperature", "heat", "thermal"],
            "electromagnetic": ["electric", "magnetic", "field"],
            "flow": ["flow", "inlet", "outlet", "wall"],
        }

        for category, keywords in categories.items():
            if any(kw in meaning_lower for kw in keywords):
                return category
        return "general"

    def _extract_coupling_pattern(self, graph) -> Dict[str, Any]:
        """Extract coupling topology pattern."""
        return {
            "node_types": [node.type for node in graph.nodes.values()],
            "edge_connectivity": [
                {"from": edge.from_node, "to": edge.to_node} for edge in graph.edges
            ],
        }

    def _detect_feedback(self, graph) -> bool:
        """Detect if graph has feedback loops."""
        # Simple cycle detection
        visited = set()
        rec_stack = set()

        def has_cycle(node_id, adjacency):
            visited.add(node_id)
            rec_stack.add(node_id)

            for edge in adjacency.get(node_id, []):
                if edge not in visited:
                    if has_cycle(edge, adjacency):
                        return True
                elif edge in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        # Build adjacency
        adjacency = {}
        for edge in graph.edges:
            if edge.from_node not in adjacency:
                adjacency[edge.from_node] = []
            adjacency[edge.from_node].append(edge.to_node)

        for node_id in graph.nodes:
            if node_id not in visited:
                if has_cycle(node_id, adjacency):
                    return True

        return False

    def extract_for_arxiv(self, schema) -> Dict[str, Any]:
        """Extract fingerprint optimized for arXiv math lookup.

        Focuses on equation types and numerical methods.
        """
        fingerprints = self.extract_from_schema(schema)

        return {
            "equation_patterns": [
                fp.metadata.get("mathematical_form_pattern")
                for fp in fingerprints
                if fp.fingerprint_type == FingerprintType.EQUATION_STRUCTURE
            ],
            "numerical_methods": [
                fp.metadata.get("method_name")
                for fp in fingerprints
                if fp.fingerprint_type == FingerprintType.NUMERICAL_METHOD
            ],
            "tensor_structures": [
                {
                    "rank": fp.metadata.get("tensor_rank"),
                    "type": fp.metadata.get("object_type"),
                }
                for fp in fingerprints
                if fp.fingerprint_type == FingerprintType.TENSOR_RANK
            ],
        }

    def extract_for_wikidata(self, schema) -> Dict[str, Any]:
        """Extract fingerprint optimized for Wikidata lookup.

        Focuses on mathematical concepts and units.
        """
        fingerprints = self.extract_from_schema(schema)

        return {
            "mathematical_concepts": self._extract_concepts(schema),
            "unit_dimensions": self._extract_unit_dimensions(schema),
            "tensor_operations": [
                fp.metadata.get("object_type")
                for fp in fingerprints
                if fp.fingerprint_type == FingerprintType.TENSOR_RANK
            ],
        }

    def extract_for_nist(self, schema) -> Dict[str, Any]:
        """Extract fingerprint optimized for NIST potentials lookup.

        Focuses on potential function forms and materials.
        """
        return {
            "potential_types": self._extract_potential_types(schema),
            "material_classes": self._extract_material_classes(schema),
            "interaction_ranges": self._extract_interaction_ranges(schema),
        }

    def _extract_concepts(self, schema) -> List[str]:
        """Extract mathematical concept names."""
        concepts = set()

        for eq in getattr(schema, "governing_equations", []):
            concepts.add(eq.type)

        for method in getattr(schema, "numerical_methods", []):
            concepts.add(method.name.lower().replace(" ", "_"))

        return list(concepts)

    def _extract_unit_dimensions(self, schema) -> List[str]:
        """Extract unit dimensions from variables."""
        dimensions = set()

        for eq in getattr(schema, "governing_equations", []):
            for var in eq.variables:
                var_type = var.get("type", "")
                if "scalar" in var_type:
                    dimensions.add("dimensionless")
                elif "vector" in var_type:
                    dimensions.add("vector")
                elif "tensor" in var_type:
                    dimensions.add("tensor")

        return list(dimensions)

    def _extract_potential_types(self, schema) -> List[str]:
        """Extract potential function types from force field equations."""
        potentials = []

        for eq in getattr(schema, "governing_equations", []):
            form = eq.mathematical_form.lower()
            if "lennard-jones" in form or "lj" in form:
                potentials.append("lennard_jones")
            elif "coulomb" in form or "electrostatic" in form:
                potentials.append("coulomb")
            elif "harmonic" in form or "spring" in form:
                potentials.append("harmonic")
            elif "morse" in form:
                potentials.append("morse")
            elif "buckingham" in form:
                potentials.append("buckingham")

        return potentials

    def _extract_material_classes(self, schema) -> List[str]:
        """Extract material class hints from schema."""
        # This would typically use heuristics from the schema
        # For now, return empty - would be populated by specific harnesses
        return []

    def _extract_interaction_ranges(self, schema) -> List[str]:
        """Extract interaction range types."""
        ranges = []

        for eq in getattr(schema, "governing_equations", []):
            form = eq.mathematical_form.lower()
            if "1/r" in form or "coulomb" in form:
                ranges.append("long_range")
            elif "exp" in form or "r^" in form:
                ranges.append("short_range")

        return ranges
