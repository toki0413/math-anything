"""Math Anything Diff - Mathematical structure difference tracking.

This module provides capabilities to compare two Math Schemas and identify
semantic differences in mathematical structures, not just text differences.

Example:
    ```python
    from math_anything.utils import MathDiffer

    differ = MathDiffer()
    report = differ.compare(schema_v1, schema_v2)

    print(f"Boundary conditions changed: {len(report.bc_changes)}")
    for change in report.bc_changes:
        print(f"  {change.type}: {change.path}")
    ```
"""

import json
from dataclasses import dataclass, field
from difflib import unified_diff
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class DiffType(Enum):
    """Types of mathematical differences."""

    # Structural changes
    EQUATION_ADDED = auto()
    EQUATION_REMOVED = auto()
    EQUATION_MODIFIED = auto()

    BC_ADDED = auto()
    BC_REMOVED = auto()
    BC_MODIFIED = auto()
    BC_TYPE_CHANGED = auto()
    BC_TENSOR_RANK_CHANGED = auto()

    # Numerical method changes
    INTEGRATOR_CHANGED = auto()
    TIMESTEP_CHANGED = auto()
    SOLVER_CHANGED = auto()
    ORDER_CHANGED = auto()

    # Conservation changes
    CONSERVATION_GAINED = auto()
    CONSERVATION_LOST = auto()

    # Computational graph changes
    NODE_ADDED = auto()
    NODE_REMOVED = auto()
    EDGE_ADDED = auto()
    EDGE_REMOVED = auto()
    LOOP_TYPE_CHANGED = auto()

    # Parameter changes
    PARAMETER_CHANGED = auto()
    VALUE_CHANGED = auto()

    # Semantic changes
    SYMMETRY_CHANGED = auto()
    CONSTRAINT_CHANGED = auto()

    # Meta changes
    SOURCE_FILE_CHANGED = auto()


@dataclass
class Change:
    """A single detected change."""

    type: DiffType
    path: str  # JSON path-like notation, e.g., "mathematical_model.boundary_conditions[0]"
    old_value: Any = None
    new_value: Any = None
    description: str = ""
    severity: str = "info"  # info, warning, critical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.name,
            "path": self.path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "description": self.description,
            "severity": self.severity,
        }


@dataclass
class DiffReport:
    """Complete difference report between two schemas."""

    old_version: Optional[str] = None
    new_version: Optional[str] = None
    old_source: Optional[str] = None
    new_source: Optional[str] = None

    # Categorized changes
    structural_changes: List[Change] = field(default_factory=list)
    numerical_changes: List[Change] = field(default_factory=list)
    boundary_changes: List[Change] = field(default_factory=list)
    conservation_changes: List[Change] = field(default_factory=list)
    computational_changes: List[Change] = field(default_factory=list)
    parameter_changes: List[Change] = field(default_factory=list)

    @property
    def all_changes(self) -> List[Change]:
        """Get all changes in a single list."""
        return (
            self.structural_changes
            + self.numerical_changes
            + self.boundary_changes
            + self.conservation_changes
            + self.computational_changes
            + self.parameter_changes
        )

    @property
    def critical_changes(self) -> List[Change]:
        """Get critical changes only."""
        return [c for c in self.all_changes if c.severity == "critical"]

    @property
    def has_critical_changes(self) -> bool:
        """Check if there are critical changes."""
        return len(self.critical_changes) > 0

    def get_changes_by_type(self, diff_type: DiffType) -> List[Change]:
        """Get changes filtered by type."""
        return [c for c in self.all_changes if c.type == diff_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "old_source": self.old_source,
            "new_source": self.new_source,
            "summary": {
                "total_changes": len(self.all_changes),
                "critical": len(self.critical_changes),
                "structural": len(self.structural_changes),
                "numerical": len(self.numerical_changes),
                "boundary": len(self.boundary_changes),
                "conservation": len(self.conservation_changes),
                "computational": len(self.computational_changes),
                "parameters": len(self.parameter_changes),
            },
            "changes": [c.to_dict() for c in self.all_changes],
        }

    def to_json(self, indent: int = 2) -> str:
        """Export report as JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def print_summary(self):
        """Print human-readable summary."""
        print("=" * 70)
        print("MATH ANYTHING DIFF REPORT")
        print("=" * 70)

        if self.old_source and self.new_source:
            print(f"\nComparing:")
            print(f"  Old: {self.old_source}")
            print(f"  New: {self.new_source}")

        summary = self.to_dict()["summary"]
        print(f"\nTotal Changes: {summary['total_changes']}")
        print(f"  Critical: {summary['critical']}")
        print(f"  Structural: {summary['structural']}")
        print(f"  Numerical: {summary['numerical']}")
        print(f"  Boundary: {summary['boundary']}")
        print(f"  Conservation: {summary['conservation']}")
        print(f"  Computational: {summary['computational']}")
        print(f"  Parameters: {summary['parameters']}")

        if self.critical_changes:
            print("\n" + "-" * 70)
            print("CRITICAL CHANGES:")
            print("-" * 70)
            for change in self.critical_changes:
                print(f"\n[{change.type.name}] {change.path}")
                print(f"  {change.description}")
                if change.old_value is not None:
                    print(f"  Old: {change.old_value}")
                if change.new_value is not None:
                    print(f"  New: {change.new_value}")


class MathDiffer:
    """Mathematical structure differencer.

    Compares two Math Schemas and identifies semantic differences.

    Example:
        ```python
        differ = MathDiffer()

        # Compare two schemas
        report = differ.compare(schema_v1, schema_v2)

        # Or compare files
        report = differ.compare_files("model_v1.json", "model_v2.json")

        # Check for critical changes
        if report.has_critical_changes:
            print("Warning: Critical mathematical changes detected!")
        ```
    """

    def __init__(self):
        self.report = DiffReport()

    def compare(
        self,
        old_schema: Union[Dict, Any],
        new_schema: Union[Dict, Any],
        old_source: str = "",
        new_source: str = "",
    ) -> DiffReport:
        """Compare two schemas and generate diff report.

        Args:
            old_schema: First schema (dict or MathSchema object)
            new_schema: Second schema (dict or MathSchema object)
            old_source: Optional source identifier for old schema
            new_source: Optional source identifier for new schema

        Returns:
            DiffReport with all detected changes
        """
        # Convert to dict if needed
        old_data = (
            old_schema.to_dict() if hasattr(old_schema, "to_dict") else old_schema
        )
        new_data = (
            new_schema.to_dict() if hasattr(new_schema, "to_dict") else new_schema
        )

        # Initialize report
        self.report = DiffReport(
            old_version=old_data.get("schema_version"),
            new_version=new_data.get("schema_version"),
            old_source=old_source,
            new_source=new_source,
        )

        # Compare all sections
        self._compare_governing_equations(old_data, new_data)
        self._compare_boundary_conditions(old_data, new_data)
        self._compare_numerical_method(old_data, new_data)
        self._compare_conservation_properties(old_data, new_data)
        self._compare_computational_graph(old_data, new_data)
        self._compare_raw_symbols(old_data, new_data)

        return self.report

    def compare_files(self, old_path: str, new_path: str) -> DiffReport:
        """Compare two schema files.

        Args:
            old_path: Path to first JSON file
            new_path: Path to second JSON file

        Returns:
            DiffReport with all detected changes
        """
        with open(old_path, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        with open(new_path, "r", encoding="utf-8") as f:
            new_data = json.load(f)

        return self.compare(old_data, new_data, old_path, new_path)

    def _compare_governing_equations(self, old_data: Dict, new_data: Dict):
        """Compare governing equations."""
        old_model = old_data.get("mathematical_model", {})
        new_model = new_data.get("mathematical_model", {})

        old_eqs = {eq["id"]: eq for eq in old_model.get("governing_equations", [])}
        new_eqs = {eq["id"]: eq for eq in new_model.get("governing_equations", [])}

        old_ids = set(old_eqs.keys())
        new_ids = set(new_eqs.keys())

        # Added equations
        for eq_id in new_ids - old_ids:
            self.report.structural_changes.append(
                Change(
                    type=DiffType.EQUATION_ADDED,
                    path=f"mathematical_model.governing_equations.{eq_id}",
                    new_value=new_eqs[eq_id].get("mathematical_form"),
                    description=f"Governing equation '{eq_id}' added",
                    severity="info",
                )
            )

        # Removed equations
        for eq_id in old_ids - new_ids:
            self.report.structural_changes.append(
                Change(
                    type=DiffType.EQUATION_REMOVED,
                    path=f"mathematical_model.governing_equations.{eq_id}",
                    old_value=old_eqs[eq_id].get("mathematical_form"),
                    description=f"Governing equation '{eq_id}' removed",
                    severity="warning",
                )
            )

        # Modified equations
        for eq_id in old_ids & new_ids:
            old_eq = old_eqs[eq_id]
            new_eq = new_eqs[eq_id]

            if old_eq.get("mathematical_form") != new_eq.get("mathematical_form"):
                self.report.structural_changes.append(
                    Change(
                        type=DiffType.EQUATION_MODIFIED,
                        path=f"mathematical_model.governing_equations.{eq_id}",
                        old_value=old_eq.get("mathematical_form"),
                        new_value=new_eq.get("mathematical_form"),
                        description=f"Governing equation '{eq_id}' modified",
                        severity="critical",
                    )
                )

    def _compare_boundary_conditions(self, old_data: Dict, new_data: Dict):
        """Compare boundary conditions with tensor awareness."""
        old_model = old_data.get("mathematical_model", {})
        new_model = new_data.get("mathematical_model", {})

        old_bcs = {bc["id"]: bc for bc in old_model.get("boundary_conditions", [])}
        new_bcs = {bc["id"]: bc for bc in new_model.get("boundary_conditions", [])}

        old_ids = set(old_bcs.keys())
        new_ids = set(new_bcs.keys())

        # Added BCs
        for bc_id in new_ids - old_ids:
            bc = new_bcs[bc_id]
            mo = bc.get("mathematical_object", {})
            tensor_info = (
                f" (rank-{mo.get('tensor_rank')} tensor)"
                if mo.get("tensor_rank")
                else ""
            )

            self.report.boundary_changes.append(
                Change(
                    type=DiffType.BC_ADDED,
                    path=f"mathematical_model.boundary_conditions.{bc_id}",
                    new_value=bc.get("type"),
                    description=f"Boundary condition '{bc_id}' added{tensor_info}",
                    severity="info",
                )
            )

        # Removed BCs
        for bc_id in old_ids - new_ids:
            self.report.boundary_changes.append(
                Change(
                    type=DiffType.BC_REMOVED,
                    path=f"mathematical_model.boundary_conditions.{bc_id}",
                    old_value=old_bcs[bc_id].get("type"),
                    description=f"Boundary condition '{bc_id}' removed",
                    severity="warning",
                )
            )

        # Modified BCs
        for bc_id in old_ids & new_ids:
            old_bc = old_bcs[bc_id]
            new_bc = new_bcs[bc_id]

            old_mo = old_bc.get("mathematical_object", {})
            new_mo = new_bc.get("mathematical_object", {})

            # Check type change
            if old_bc.get("type") != new_bc.get("type"):
                self.report.boundary_changes.append(
                    Change(
                        type=DiffType.BC_TYPE_CHANGED,
                        path=f"mathematical_model.boundary_conditions.{bc_id}.type",
                        old_value=old_bc.get("type"),
                        new_value=new_bc.get("type"),
                        description=f"Boundary condition '{bc_id}' type changed",
                        severity="critical",
                    )
                )

            # Check tensor rank change
            old_rank = old_mo.get("tensor_rank")
            new_rank = new_mo.get("tensor_rank")
            if old_rank != new_rank:
                self.report.boundary_changes.append(
                    Change(
                        type=DiffType.BC_TENSOR_RANK_CHANGED,
                        path=f"mathematical_model.boundary_conditions.{bc_id}.tensor_rank",
                        old_value=old_rank,
                        new_value=new_rank,
                        description=f"Boundary condition '{bc_id}' tensor rank changed from {old_rank} to {new_rank}",
                        severity="critical",
                    )
                )

            # Check tensor form change
            old_form = old_mo.get("tensor_form")
            new_form = new_mo.get("tensor_form")
            if old_form != new_form:
                self.report.boundary_changes.append(
                    Change(
                        type=DiffType.BC_MODIFIED,
                        path=f"mathematical_model.boundary_conditions.{bc_id}.tensor_form",
                        old_value=old_form,
                        new_value=new_form,
                        description=f"Boundary condition '{bc_id}' tensor form modified",
                        severity="warning",
                    )
                )

            # Check components change
            old_comps = {
                tuple(c.get("index", [])): c.get("value")
                for c in old_mo.get("components", [])
            }
            new_comps = {
                tuple(c.get("index", [])): c.get("value")
                for c in new_mo.get("components", [])
            }

            if old_comps != new_comps:
                changed_indices = set(old_comps.keys()) ^ set(new_comps.keys())
                for idx in changed_indices:
                    self.report.boundary_changes.append(
                        Change(
                            type=DiffType.VALUE_CHANGED,
                            path=f"mathematical_model.boundary_conditions.{bc_id}.components{list(idx)}",
                            old_value=old_comps.get(idx),
                            new_value=new_comps.get(idx),
                            description=f"Tensor component {idx} changed",
                            severity="warning",
                        )
                    )

    def _compare_numerical_method(self, old_data: Dict, new_data: Dict):
        """Compare numerical method settings."""
        old_nm = old_data.get("numerical_method", {})
        new_nm = new_data.get("numerical_method", {})

        old_disc = old_nm.get("discretization", {})
        new_disc = new_nm.get("discretization", {})

        # Check integrator change
        old_integ = old_disc.get("time_integrator")
        new_integ = new_disc.get("time_integrator")
        if old_integ != new_integ:
            self.report.numerical_changes.append(
                Change(
                    type=DiffType.INTEGRATOR_CHANGED,
                    path="numerical_method.discretization.time_integrator",
                    old_value=old_integ,
                    new_value=new_integ,
                    description=f"Time integrator changed from '{old_integ}' to '{new_integ}'",
                    severity="critical",
                )
            )

        # Check timestep change
        old_dt = old_disc.get("time_step")
        new_dt = new_disc.get("time_step")
        if old_dt != new_dt and old_dt is not None and new_dt is not None:
            change_pct = abs(new_dt - old_dt) / old_dt * 100 if old_dt != 0 else 0
            severity = "warning" if change_pct < 50 else "critical"

            self.report.numerical_changes.append(
                Change(
                    type=DiffType.TIMESTEP_CHANGED,
                    path="numerical_method.discretization.time_step",
                    old_value=old_dt,
                    new_value=new_dt,
                    description=f"Time step changed from {old_dt} to {new_dt} ({change_pct:.1f}% change)",
                    severity=severity,
                )
            )

        # Check order change
        old_order = old_disc.get("order")
        new_order = new_disc.get("order")
        if old_order != new_order:
            self.report.numerical_changes.append(
                Change(
                    type=DiffType.ORDER_CHANGED,
                    path="numerical_method.discretization.order",
                    old_value=old_order,
                    new_value=new_order,
                    description=f"Integration order changed from {old_order} to {new_order}",
                    severity="warning",
                )
            )

        # Check solver change
        old_solver = old_nm.get("solver", {}).get("algorithm")
        new_solver = new_nm.get("solver", {}).get("algorithm")
        if old_solver != new_solver:
            self.report.numerical_changes.append(
                Change(
                    type=DiffType.SOLVER_CHANGED,
                    path="numerical_method.solver.algorithm",
                    old_value=old_solver,
                    new_value=new_solver,
                    description=f"Solver algorithm changed from '{old_solver}' to '{new_solver}'",
                    severity="info",
                )
            )

    def _compare_conservation_properties(self, old_data: Dict, new_data: Dict):
        """Compare conservation properties."""
        old_cp = old_data.get("conservation_properties", {})
        new_cp = new_data.get("conservation_properties", {})

        # Check for lost conservation
        for prop_name, old_prop in old_cp.items():
            if prop_name not in new_cp:
                self.report.conservation_changes.append(
                    Change(
                        type=DiffType.CONSERVATION_LOST,
                        path=f"conservation_properties.{prop_name}",
                        old_value=old_prop,
                        description=f"Conservation property '{prop_name}' lost",
                        severity="critical",
                    )
                )

        # Check for gained conservation
        for prop_name, new_prop in new_cp.items():
            if prop_name not in old_cp:
                self.report.conservation_changes.append(
                    Change(
                        type=DiffType.CONSERVATION_GAINED,
                        path=f"conservation_properties.{prop_name}",
                        new_value=new_prop,
                        description=f"Conservation property '{prop_name}' gained",
                        severity="info",
                    )
                )

        # Check for changes in preservation status
        for prop_name in set(old_cp.keys()) & set(new_cp.keys()):
            old_preserved = old_cp[prop_name].get("preserved", False)
            new_preserved = new_cp[prop_name].get("preserved", False)

            if old_preserved and not new_preserved:
                self.report.conservation_changes.append(
                    Change(
                        type=DiffType.CONSERVATION_LOST,
                        path=f"conservation_properties.{prop_name}.preserved",
                        old_value=True,
                        new_value=False,
                        description=f"Conservation of '{prop_name}' no longer guaranteed",
                        severity="critical",
                    )
                )

    def _compare_computational_graph(self, old_data: Dict, new_data: Dict):
        """Compare computational graphs."""
        old_cg = old_data.get("computational_graph", {})
        new_cg = new_data.get("computational_graph", {})

        old_nodes = {n["id"]: n for n in old_cg.get("nodes", [])}
        new_nodes = {n["id"]: n for n in new_cg.get("nodes", [])}

        # Added nodes
        for node_id in set(new_nodes.keys()) - set(old_nodes.keys()):
            self.report.computational_changes.append(
                Change(
                    type=DiffType.NODE_ADDED,
                    path=f"computational_graph.nodes.{node_id}",
                    new_value=new_nodes[node_id].get("type"),
                    description=f"Computational node '{node_id}' added",
                    severity="info",
                )
            )

        # Removed nodes
        for node_id in set(old_nodes.keys()) - set(new_nodes.keys()):
            self.report.computational_changes.append(
                Change(
                    type=DiffType.NODE_REMOVED,
                    path=f"computational_graph.nodes.{node_id}",
                    old_value=old_nodes[node_id].get("type"),
                    description=f"Computational node '{node_id}' removed",
                    severity="warning",
                )
            )

        # Check loop type changes
        for node_id in set(old_nodes.keys()) & set(new_nodes.keys()):
            old_mode = (
                old_nodes[node_id]
                .get("math_semantics", {})
                .get("updates", {})
                .get("mode")
            )
            new_mode = (
                new_nodes[node_id]
                .get("math_semantics", {})
                .get("updates", {})
                .get("mode")
            )

            if old_mode != new_mode:
                self.report.computational_changes.append(
                    Change(
                        type=DiffType.LOOP_TYPE_CHANGED,
                        path=f"computational_graph.nodes.{node_id}.math_semantics.updates.mode",
                        old_value=old_mode,
                        new_value=new_mode,
                        description=f"Update mode changed from '{old_mode}' to '{new_mode}'",
                        severity=(
                            "critical"
                            if "implicit" in str(old_mode)
                            or "implicit" in str(new_mode)
                            else "warning"
                        ),
                    )
                )

    def _compare_raw_symbols(self, old_data: Dict, new_data: Dict):
        """Compare raw symbols for parameter changes."""
        old_rs = old_data.get("raw_symbols", {})
        new_rs = new_data.get("raw_symbols", {})

        # Compare fixes
        old_fixes = old_rs.get("fixes", {})
        new_fixes = new_rs.get("fixes", {})

        for fix_id in set(old_fixes.keys()) & set(new_fixes.keys()):
            old_fix = old_fixes[fix_id]
            new_fix = new_fixes[fix_id]

            if old_fix != new_fix:
                self.report.parameter_changes.append(
                    Change(
                        type=DiffType.PARAMETER_CHANGED,
                        path=f"raw_symbols.fixes.{fix_id}",
                        old_value=old_fix,
                        new_value=new_fix,
                        description=f"Fix '{fix_id}' parameters changed",
                        severity="info",
                    )
                )
