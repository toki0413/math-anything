"""Tests for Math Anything Diff functionality."""

import json
import os
import tempfile
import unittest

from math_anything.schemas import MathematicalModel, MathSchema, MetaInfo
from math_anything.utils import DiffType, MathDiffer


class TestMathDiffer(unittest.TestCase):
    """Test mathematical difference detection."""

    def test_empty_schemas(self):
        """Test comparing two empty schemas."""
        old = MathSchema()
        new = MathSchema()

        differ = MathDiffer()
        report = differ.compare(old, new)

        self.assertEqual(len(report.all_changes), 0)

    def test_equation_added(self):
        """Test detecting added governing equation."""
        from math_anything.schemas import GoverningEquation

        old = MathSchema()
        new = MathSchema()
        new.mathematical_model.governing_equations.append(
            GoverningEquation(
                id="newton",
                type="second_order_ode",
                name="Newton's Law",
                mathematical_form="F = ma",
            )
        )

        differ = MathDiffer()
        report = differ.compare(old, new)

        changes = report.get_changes_by_type(DiffType.EQUATION_ADDED)
        self.assertEqual(len(changes), 1)
        self.assertEqual(
            changes[0].path, "mathematical_model.governing_equations.newton"
        )

    def test_equation_removed(self):
        """Test detecting removed governing equation."""
        from math_anything.schemas import GoverningEquation

        old = MathSchema()
        old.mathematical_model.governing_equations.append(
            GoverningEquation(
                id="old_eq",
                type="ode",
                name="Old Equation",
                mathematical_form="dx/dt = 0",
            )
        )
        new = MathSchema()

        differ = MathDiffer()
        report = differ.compare(old, new)

        changes = report.get_changes_by_type(DiffType.EQUATION_REMOVED)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].severity, "warning")

    def test_integrator_changed(self):
        """Test detecting integrator change."""
        from math_anything.schemas import Discretization, NumericalMethod

        old = MathSchema()
        old.numerical_method = NumericalMethod(
            discretization=Discretization(time_integrator="velocity_verlet")
        )

        new = MathSchema()
        new.numerical_method = NumericalMethod(
            discretization=Discretization(time_integrator="euler")
        )

        differ = MathDiffer()
        report = differ.compare(old, new)

        changes = report.get_changes_by_type(DiffType.INTEGRATOR_CHANGED)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].severity, "critical")
        self.assertEqual(changes[0].old_value, "velocity_verlet")
        self.assertEqual(changes[0].new_value, "euler")

    def test_timestep_changed(self):
        """Test detecting timestep change."""
        from math_anything.schemas import Discretization, NumericalMethod

        old = MathSchema()
        old.numerical_method = NumericalMethod(
            discretization=Discretization(time_step=0.001)
        )

        new = MathSchema()
        new.numerical_method = NumericalMethod(
            discretization=Discretization(time_step=0.002)
        )

        differ = MathDiffer()
        report = differ.compare(old, new)

        changes = report.get_changes_by_type(DiffType.TIMESTEP_CHANGED)
        self.assertEqual(len(changes), 1)
        self.assertIn("100.0%", changes[0].description)  # Change percentage

    def test_bc_tensor_rank_changed(self):
        """Test detecting boundary condition tensor rank change."""
        from math_anything.schemas import BoundaryCondition, MathematicalObject

        old = MathSchema()
        old.mathematical_model.boundary_conditions.append(
            BoundaryCondition(
                id="bc1",
                type="dirichlet",
                domain={},
                mathematical_object=MathematicalObject(
                    field="displacement",
                    tensor_rank=0,
                    tensor_form="scalar",
                ),
                software_implementation={},
            )
        )

        new = MathSchema()
        new.mathematical_model.boundary_conditions.append(
            BoundaryCondition(
                id="bc1",
                type="dirichlet",
                domain={},
                mathematical_object=MathematicalObject(
                    field="displacement",
                    tensor_rank=2,
                    tensor_form="tensor",
                ),
                software_implementation={},
            )
        )

        differ = MathDiffer()
        report = differ.compare(old, new)

        changes = report.get_changes_by_type(DiffType.BC_TENSOR_RANK_CHANGED)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].severity, "critical")
        self.assertEqual(changes[0].old_value, 0)
        self.assertEqual(changes[0].new_value, 2)

    def test_conservation_lost(self):
        """Test detecting lost conservation property."""
        old = MathSchema()
        old.conservation_properties = {"energy": {"preserved": True}}

        new = MathSchema()
        new.conservation_properties = {}

        differ = MathDiffer()
        report = differ.compare(old, new)

        changes = report.get_changes_by_type(DiffType.CONSERVATION_LOST)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].severity, "critical")

    def test_file_comparison(self):
        """Test comparing files."""
        old = MathSchema()
        new = MathSchema()
        new.conservation_properties = {"momentum": {"preserved": True}}

        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = os.path.join(tmpdir, "old.json")
            new_path = os.path.join(tmpdir, "new.json")

            old.save(old_path)
            new.save(new_path)

            differ = MathDiffer()
            report = differ.compare_files(old_path, new_path)

            changes = report.get_changes_by_type(DiffType.CONSERVATION_GAINED)
            self.assertEqual(len(changes), 1)

    def test_report_json_output(self):
        """Test JSON output format."""
        old = MathSchema()
        new = MathSchema()
        new.conservation_properties = {"energy": {"preserved": True}}

        differ = MathDiffer()
        report = differ.compare(old, new)

        json_str = report.to_json()
        data = json.loads(json_str)

        self.assertIn("summary", data)
        self.assertIn("changes", data)
        self.assertIn("total_changes", data["summary"])


class TestDiffReport(unittest.TestCase):
    """Test DiffReport functionality."""

    def test_critical_changes_filter(self):
        """Test filtering critical changes."""
        from math_anything.utils.math_diff import Change, DiffReport, DiffType

        report = DiffReport()
        report.numerical_changes.append(
            Change(
                type=DiffType.INTEGRATOR_CHANGED,
                path="test",
                severity="critical",
            )
        )
        report.parameter_changes.append(
            Change(
                type=DiffType.PARAMETER_CHANGED,
                path="test2",
                severity="info",
            )
        )

        critical = report.critical_changes
        self.assertEqual(len(critical), 1)
        self.assertEqual(critical[0].type, DiffType.INTEGRATOR_CHANGED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
