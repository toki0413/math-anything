"""Math Anything Diff - Mathematical Semantic Comparison.

Compares two Math Schema objects at the semantic level,
identifying mathematical similarities and differences.

Inspired by git diff, but for mathematical structures rather than text.
"""

import difflib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from math_anything.schemas import BoundaryCondition, GoverningEquation, MathSchema


@dataclass
class DiffResult:
    """Result of mathematical semantic comparison.

    Attributes:
        similarity_score: Overall similarity (0.0 - 1.0)
        common_equations: Equations present in both schemas
        different_equations: Similar but not identical equations
        unique_to_first: Equations only in first schema
        unique_to_second: Equations only in second schema
        constraint_differences: Differences in symbolic constraints
        parameter_mapping: Suggested parameter mappings
        analysis_summary: Human-readable summary
    """

    similarity_score: float = 0.0
    common_equations: List[str] = field(default_factory=list)
    different_equations: List[Dict[str, str]] = field(default_factory=list)
    unique_to_first: List[str] = field(default_factory=list)
    unique_to_second: List[str] = field(default_factory=list)
    constraint_differences: List[Dict[str, Any]] = field(default_factory=list)
    parameter_mapping: List[Dict[str, str]] = field(default_factory=list)
    analysis_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "similarity_score": self.similarity_score,
            "common_equations": self.common_equations,
            "different_equations": self.different_equations,
            "unique_to_first": self.unique_to_first,
            "unique_to_second": self.unique_to_second,
            "constraint_differences": self.constraint_differences,
            "parameter_mapping": self.parameter_mapping,
            "analysis_summary": self.analysis_summary,
        }


class MathDiff:
    """Mathematical semantic diff tool.

    Compares two computational models at the mathematical level,
    not just syntactic differences.

    Example:
        ```python
        from math_anything.repl import MathDiff

        diff = MathDiff.compare(schema1, schema2)

        print(f"Similarity: {diff.similarity_score:.1%}")
        print(f"Common equations: {diff.common_equations}")
        print(f"Analysis: {diff.analysis_summary}")
        ```
    """

    # Equation type similarity matrix
    # Defines which equation types are mathematically related
    EQUATION_SIMILARITY = {
        ("eigenvalue_problem", "eigenvalue_problem"): 1.0,
        ("newtonian_dynamics", "newtonian_dynamics"): 1.0,
        ("equilibrium", "equilibrium"): 1.0,
        ("heat_conduction", "heat_transfer"): 0.9,
        ("navier_stokes", "fluid_flow"): 0.9,
        ("maxwell", "electromagnetic"): 0.9,
        ("newtonian_dynamics", "transient_structural"): 0.7,
        ("static_equilibrium", "equilibrium"): 0.8,
    }

    # Parameter name mappings across engines
    CROSS_ENGINE_MAPPINGS = {
        # VASP <-> Quantum ESPRESSO
        ("ENCUT", "ecutwfc"): "energy_cutoff",
        ("ISMEAR", "smearing"): "smearing_method",
        ("SIGMA", "degauss"): "smearing_width",
        # LAMMPS <-> GROMACS
        ("timestep", "dt"): "time_step",
        ("temp", "ref_t"): "temperature",
        ("pressure", "ref_p"): "pressure",
        # Abaqus <-> Ansys
        ("Young_modulus", "EX"): "elastic_modulus",
        ("Poisson_ratio", "PRXY"): "poissons_ratio",
        ("density", "DENS"): "density",
    }

    @classmethod
    def compare(cls, schema1: MathSchema, schema2: MathSchema) -> DiffResult:
        """Compare two mathematical schemas.

        Args:
            schema1: First MathSchema
            schema2: Second MathSchema

        Returns:
            DiffResult with detailed comparison
        """
        result = DiffResult()

        # Compare governing equations
        eq_comparison = cls._compare_equations(
            (
                schema1.mathematical_model.governing_equations
                if schema1.mathematical_model
                else []
            ),
            (
                schema2.mathematical_model.governing_equations
                if schema2.mathematical_model
                else []
            ),
        )
        result.common_equations = eq_comparison["common"]
        result.different_equations = eq_comparison["different"]
        result.unique_to_first = eq_comparison["unique_first"]
        result.unique_to_second = eq_comparison["unique_second"]

        # Compare boundary conditions
        bc_comparison = cls._compare_boundary_conditions(
            (
                schema1.mathematical_model.boundary_conditions
                if schema1.mathematical_model
                else []
            ),
            (
                schema2.mathematical_model.boundary_conditions
                if schema2.mathematical_model
                else []
            ),
        )

        # Compare constraints
        result.constraint_differences = cls._compare_constraints(schema1, schema2)

        # Find parameter mappings
        result.parameter_mapping = cls._suggest_parameter_mappings(schema1, schema2)

        # Calculate overall similarity
        result.similarity_score = cls._calculate_similarity(
            schema1, schema2, eq_comparison, bc_comparison
        )

        # Generate summary
        result.analysis_summary = cls._generate_summary(result, schema1, schema2)

        return result

    @classmethod
    def _compare_equations(
        cls,
        eqs1: List[GoverningEquation],
        eqs2: List[GoverningEquation],
    ) -> Dict[str, List]:
        """Compare governing equations."""
        common = []
        different = []
        unique_first = []
        unique_second = []

        # Build similarity matrix
        matched_second = set()

        for eq1 in eqs1:
            best_match = None
            best_score = 0.0

            for i, eq2 in enumerate(eqs2):
                if i in matched_second:
                    continue

                score = cls._equation_similarity(eq1, eq2)
                if score > best_score and score > 0.6:  # Threshold
                    best_score = score
                    best_match = (i, eq2)

            if best_match and best_score > 0.9:
                # Very similar - consider common
                common.append(eq1.name or eq1.id)
                matched_second.add(best_match[0])
            elif best_match:
                # Somewhat similar
                different.append(
                    {
                        "first": eq1.name or eq1.id,
                        "second": best_match[1].name or best_match[1].id,
                        "similarity": best_score,
                    }
                )
                matched_second.add(best_match[0])
            else:
                # No match found
                unique_first.append(eq1.name or eq1.id)

        # Find unmatched equations in second list
        for i, eq2 in enumerate(eqs2):
            if i not in matched_second:
                unique_second.append(eq2.name or eq2.id)

        return {
            "common": common,
            "different": different,
            "unique_first": unique_first,
            "unique_second": unique_second,
        }

    @classmethod
    def _equation_similarity(
        cls, eq1: GoverningEquation, eq2: GoverningEquation
    ) -> float:
        """Calculate similarity between two equations (0.0 - 1.0)."""
        scores = []

        # Type similarity
        type_pair = (eq1.type, eq2.type)
        if type_pair in cls.EQUATION_SIMILARITY:
            scores.append(cls.EQUATION_SIMILARITY[type_pair])
        elif (type_pair[1], type_pair[0]) in cls.EQUATION_SIMILARITY:
            scores.append(cls.EQUATION_SIMILARITY[(type_pair[1], type_pair[0])])
        elif eq1.type == eq2.type:
            scores.append(1.0)
        else:
            scores.append(0.0)

        # Mathematical form similarity (text-based)
        if eq1.mathematical_form and eq2.mathematical_form:
            form_sim = difflib.SequenceMatcher(
                None,
                eq1.mathematical_form,
                eq2.mathematical_form,
            ).ratio()
            scores.append(form_sim)

        # Variable overlap
        if eq1.variables and eq2.variables:
            vars1 = set(eq1.variables)
            vars2 = set(eq2.variables)
            if vars1 and vars2:
                overlap = len(vars1 & vars2) / len(vars1 | vars2)
                scores.append(overlap)

        return sum(scores) / len(scores) if scores else 0.0

    @classmethod
    def _compare_boundary_conditions(
        cls,
        bcs1: List[BoundaryCondition],
        bcs2: List[BoundaryCondition],
    ) -> Dict[str, List]:
        """Compare boundary conditions."""
        # Group by type
        bc1_by_type = defaultdict(list)
        bc2_by_type = defaultdict(list)

        for bc in bcs1:
            bc1_by_type[bc.type].append(bc)
        for bc in bcs2:
            bc2_by_type[bc.type].append(bc)

        # Compare within same type
        common_types = set(bc1_by_type.keys()) & set(bc2_by_type.keys())

        return {
            "common_types": list(common_types),
            "unique_first_types": list(
                set(bc1_by_type.keys()) - set(bc2_by_type.keys())
            ),
            "unique_second_types": list(
                set(bc2_by_type.keys()) - set(bc1_by_type.keys())
            ),
        }

    @classmethod
    def _compare_constraints(
        cls,
        schema1: MathSchema,
        schema2: MathSchema,
    ) -> List[Dict[str, Any]]:
        """Compare symbolic constraints."""
        differences = []

        # Get constraints from both schemas
        constraints1 = {}
        constraints2 = {}

        if hasattr(schema1, "symbolic_constraints") and schema1.symbolic_constraints:
            for c in schema1.symbolic_constraints:
                # Extract parameter name from expression
                param = cls._extract_param_from_constraint(c.expression)
                if param:
                    constraints1[param] = c

        if hasattr(schema2, "symbolic_constraints") and schema2.symbolic_constraints:
            for c in schema2.symbolic_constraints:
                param = cls._extract_param_from_constraint(c.expression)
                if param:
                    constraints2[param] = c

        # Find differences
        all_params = set(constraints1.keys()) | set(constraints2.keys())

        for param in all_params:
            c1 = constraints1.get(param)
            c2 = constraints2.get(param)

            if c1 and c2:
                # Both have constraint - compare
                if c1.expression != c2.expression:
                    differences.append(
                        {
                            "parameter": param,
                            "first": c1.expression,
                            "second": c2.expression,
                            "type": "different_constraint",
                        }
                    )
            elif c1:
                differences.append(
                    {
                        "parameter": param,
                        "first": c1.expression,
                        "second": None,
                        "type": "only_in_first",
                    }
                )
            else:
                differences.append(
                    {
                        "parameter": param,
                        "first": None,
                        "second": c2.expression,
                        "type": "only_in_second",
                    }
                )

        return differences

    @classmethod
    def _extract_param_from_constraint(cls, expression: str) -> Optional[str]:
        """Extract parameter name from constraint expression."""
        # Simple heuristic: first word before operator
        import re

        match = re.match(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)", expression)
        return match.group(1) if match else None

    @classmethod
    def _suggest_parameter_mappings(
        cls,
        schema1: MathSchema,
        schema2: MathSchema,
    ) -> List[Dict[str, str]]:
        """Suggest parameter mappings between engines."""
        mappings = []

        # Get raw symbols from both schemas
        symbols1 = schema1.raw_symbols if hasattr(schema1, "raw_symbols") else {}
        symbols2 = schema2.raw_symbols if hasattr(schema2, "raw_symbols") else {}

        params1 = set()
        params2 = set()

        # Extract parameter names
        if isinstance(symbols1, dict):
            for section in symbols1.values():
                if isinstance(section, dict):
                    params1.update(section.keys())

        if isinstance(symbols2, dict):
            for section in symbols2.values():
                if isinstance(section, dict):
                    params2.update(section.keys())

        # Check known cross-engine mappings
        for (p1, p2), meaning in cls.CROSS_ENGINE_MAPPINGS.items():
            if p1 in params1 and p2 in params2:
                mappings.append(
                    {
                        "first": p1,
                        "second": p2,
                        "meaning": meaning,
                        "confidence": "high",
                    }
                )

        # Check for similar names
        for p1 in params1:
            for p2 in params2:
                similarity = difflib.SequenceMatcher(
                    None, p1.lower(), p2.lower()
                ).ratio()
                if similarity > 0.8 and (p1, p2) not in [
                    (m["first"], m["second"]) for m in mappings
                ]:
                    mappings.append(
                        {
                            "first": p1,
                            "second": p2,
                            "meaning": "similar_name",
                            "confidence": "medium",
                            "similarity": round(similarity, 2),
                        }
                    )

        return mappings

    @classmethod
    def _calculate_similarity(
        cls,
        schema1: MathSchema,
        schema2: MathSchema,
        eq_comparison: Dict[str, List],
        bc_comparison: Optional[Dict[str, List]] = None,
    ) -> float:
        """Calculate overall similarity score."""
        scores = []
        weights = []

        # Equation similarity (weight: 0.4)
        total_eqs = (
            len(eq_comparison["common"])
            + len(eq_comparison["different"])
            + len(eq_comparison["unique_first"])
            + len(eq_comparison["unique_second"])
        )
        if total_eqs > 0:
            eq_score = len(eq_comparison["common"]) / total_eqs
            scores.append(eq_score)
            weights.append(0.4)

        # Constraint similarity (weight: 0.3)
        constraints1 = getattr(schema1, "symbolic_constraints", []) or []
        constraints2 = getattr(schema2, "symbolic_constraints", []) or []

        if constraints1 or constraints2:
            all_constraints = set()
            common_constraints = 0

            for c in constraints1:
                param = cls._extract_param_from_constraint(c.expression)
                if param:
                    all_constraints.add(("first", param))

            for c in constraints2:
                param = cls._extract_param_from_constraint(c.expression)
                if param:
                    if ("first", param) in all_constraints:
                        common_constraints += 1
                    all_constraints.add(("second", param))

            if all_constraints:
                constraint_score = common_constraints / len(all_constraints)
                scores.append(constraint_score)
                weights.append(0.3)

        # Boundary condition similarity (weight: 0.1)
        if bc_comparison:
            common_bc_types = len(bc_comparison.get("common_types", []))
            total_bc_types = (
                common_bc_types
                + len(bc_comparison.get("unique_first_types", []))
                + len(bc_comparison.get("unique_second_types", []))
            )
            if total_bc_types > 0:
                bc_score = common_bc_types / total_bc_types
                scores.append(bc_score)
                weights.append(0.1)

        # Numerical method similarity (weight: 0.2)
        if schema1.numerical_method and schema2.numerical_method:
            nm1 = schema1.numerical_method
            nm2 = schema2.numerical_method

            nm_score = 0.0
            if (
                nm1.discretization.space_discretization is not None
                and nm1.discretization.space_discretization
                == nm2.discretization.space_discretization
            ):
                nm_score += 0.5
            if (
                nm1.discretization.time_integrator is not None
                and nm1.discretization.time_integrator
                == nm2.discretization.time_integrator
            ):
                nm_score += 0.5

            scores.append(nm_score)
            weights.append(0.2)

        # Calculate weighted average
        if scores and weights:
            return sum(s * w for s, w in zip(scores, weights)) / sum(weights)

        return 0.0

    @classmethod
    def _generate_summary(
        cls,
        result: DiffResult,
        schema1: MathSchema,
        schema2: MathSchema,
    ) -> str:
        """Generate human-readable summary."""
        parts = []

        similarity_pct = result.similarity_score * 100

        if similarity_pct > 80:
            parts.append(
                f"High similarity ({similarity_pct:.0f}%): These models share most mathematical structures."
            )
        elif similarity_pct > 50:
            parts.append(
                f"Moderate similarity ({similarity_pct:.0f}%): These models have some common elements but differ in important ways."
            )
        else:
            parts.append(
                f"Low similarity ({similarity_pct:.0f}%): These models represent fundamentally different mathematical approaches."
            )

        # Add specifics
        if result.common_equations:
            parts.append(
                f"They share {len(result.common_equations)} governing equations."
            )

        if result.parameter_mapping:
            high_conf = [
                m for m in result.parameter_mapping if m.get("confidence") == "high"
            ]
            if high_conf:
                parts.append(
                    f"Found {len(high_conf)} clear parameter mappings between engines."
                )

        return " ".join(parts)

    @classmethod
    def cross_engine_compare(
        cls,
        schema: MathSchema,
        target_engine: str,
    ) -> Dict[str, Any]:
        """Compare schema against a target engine's typical patterns.

        Useful for migrating settings from one engine to another.

        Args:
            schema: Source MathSchema
            target_engine: Target engine name (e.g., 'quantum_espresso')

        Returns:
            Dictionary with migration suggestions
        """
        suggestions = {
            "target_engine": target_engine,
            "mappable_parameters": [],
            "unmappable_parameters": [],
            "suggested_equations": [],
            "warnings": [],
        }

        # Get parameters from schema
        raw_symbols = getattr(schema, "raw_symbols", {})
        params = {}

        if isinstance(raw_symbols, dict):
            for section_name, section in raw_symbols.items():
                if isinstance(section, dict):
                    params.update(section)

        # Check for known mappings to target engine
        for param_name, param_value in params.items():
            mapped = False
            for (src, tgt), meaning in cls.CROSS_ENGINE_MAPPINGS.items():
                if src == param_name and target_engine in tgt.lower():
                    suggestions["mappable_parameters"].append(
                        {
                            "source": param_name,
                            "target": tgt,
                            "value": param_value,
                            "meaning": meaning,
                        }
                    )
                    mapped = True
                    break

            if not mapped:
                suggestions["unmappable_parameters"].append(param_name)

        return suggestions


# Convenience function
def diff_schemas(schema1: MathSchema, schema2: MathSchema) -> DiffResult:
    """Compare two schemas and return diff result."""
    return MathDiff.compare(schema1, schema2)
