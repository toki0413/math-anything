"""Tests for Math Schema v1.0 implementation."""

import json
import os
import tempfile
import unittest

from math_anything.schemas import (
    BoundaryCondition,
    ComputationalGraph,
    ComputationalNode,
    Discretization,
    GoverningEquation,
    MathematicalModel,
    MathematicalObject,
    MathSchema,
    MetaInfo,
    NumericalMethod,
    SchemaValidator,
    Solver,
    TensorComponent,
)


class TestMathSchema(unittest.TestCase):
    """Test Math Schema creation and manipulation."""

    def test_create_empty_schema(self):
        """Test creating an empty schema."""
        schema = MathSchema()
        self.assertEqual(schema.schema_version, "1.0.0")
        self.assertIsNotNone(schema.meta)

    def test_schema_to_dict(self):
        """Test converting schema to dictionary."""
        schema = MathSchema()
        data = schema.to_dict()

        self.assertIn("schema_version", data)
        self.assertIn("meta", data)
        self.assertIn("mathematical_model", data)
        self.assertIn("computational_graph", data)

    def test_schema_to_json(self):
        """Test JSON serialization."""
        schema = MathSchema()
        json_str = schema.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        self.assertEqual(data["schema_version"], "1.0.0")

    def test_schema_save_and_load(self):
        """Test saving and loading schema."""
        schema = MathSchema()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save
            schema.save(temp_path)

            # Load
            with open(temp_path, "r") as f:
                loaded_data = json.load(f)

            self.assertEqual(loaded_data["schema_version"], "1.0.0")
        finally:
            os.unlink(temp_path)

    def test_add_governing_equation(self):
        """Test adding governing equation."""
        schema = MathSchema()
        eq = GoverningEquation(
            id="test_eq",
            type="ode",
            name="Test Equation",
            mathematical_form="dx/dt = f(x)",
        )
        schema.mathematical_model.governing_equations.append(eq)

        data = schema.to_dict()
        eqs = data["mathematical_model"]["governing_equations"]
        self.assertEqual(len(eqs), 1)
        self.assertEqual(eqs[0]["id"], "test_eq")


class TestTensorComponents(unittest.TestCase):
    """Test tensor component representation."""

    def test_create_tensor_component(self):
        """Test creating tensor component."""
        comp = TensorComponent(index=[1, 1], value="1.0", unit="m")
        self.assertEqual(comp.index, [1, 1])
        self.assertEqual(comp.value, "1.0")
        self.assertEqual(comp.unit, "m")

    def test_mathematical_object_with_tensor(self):
        """Test mathematical object with tensor components."""
        mo = MathematicalObject(
            field="stress",
            tensor_rank=2,
            tensor_form="σ_{ij}",
            components=[
                TensorComponent(index=[1, 1], value="100.0", unit="MPa"),
                TensorComponent(index=[2, 2], value="50.0", unit="MPa"),
            ],
            symmetry="symmetric",
        )

        data = mo.to_dict()
        self.assertEqual(data["tensor_rank"], 2)
        self.assertEqual(data["symmetry"], "symmetric")
        self.assertEqual(len(data["components"]), 2)


class TestBoundaryCondition(unittest.TestCase):
    """Test boundary condition representation."""

    def test_create_boundary_condition(self):
        """Test creating boundary condition."""
        mo = MathematicalObject(
            field="displacement",
            tensor_rank=2,
            tensor_form="F_{ij}",
        )

        bc = BoundaryCondition(
            id="bc_1",
            type="dirichlet",
            domain={"region": "x-boundary"},
            mathematical_object=mo,
            software_implementation={"command": "fix"},
        )

        data = bc.to_dict()
        self.assertEqual(data["id"], "bc_1")
        self.assertEqual(data["type"], "dirichlet")
        self.assertIn("mathematical_object", data)


class TestComputationalGraph(unittest.TestCase):
    """Test computational graph with explicit/implicit distinction."""

    def test_create_computational_graph(self):
        """Test creating computational graph."""
        cg = ComputationalGraph()

        node = ComputationalNode(
            id="integrator",
            type="time_integration",
            math_semantics={
                "updates": {"mode": "explicit_update"},
            },
        )
        cg.add_node(node)

        data = cg.to_dict()
        self.assertEqual(len(data["nodes"]), 1)
        self.assertEqual(data["nodes"][0]["id"], "integrator")


class TestSchemaValidator(unittest.TestCase):
    """Test schema validation."""

    def test_validate_valid_schema(self):
        """Test validating a valid schema."""
        schema = MathSchema()
        validator = SchemaValidator()

        is_valid = validator.validate(schema.to_dict())
        self.assertTrue(is_valid)

    def test_validate_missing_required_keys(self):
        """Test validation with missing required keys."""
        validator = SchemaValidator()

        # Missing keys
        data = {"schema_version": "1.0.0"}
        is_valid = validator.validate(data)

        self.assertFalse(is_valid)
        self.assertGreater(len(validator.errors), 0)

    def test_validate_version_warning(self):
        """Test version warning."""
        validator = SchemaValidator()

        data = MathSchema().to_dict()
        data["schema_version"] = "2.0.0"

        validator.validate(data)
        self.assertGreater(len(validator.warnings), 0)


class TestNumericalMethod(unittest.TestCase):
    """Test numerical method representation."""

    def test_create_numerical_method(self):
        """Test creating numerical method."""
        nm = NumericalMethod(
            discretization=Discretization(
                time_integrator="velocity_verlet",
                order=2,
                time_step=0.001,
            ),
            solver=Solver(
                algorithm="conjugate_gradient",
                convergence_criterion="force_norm",
            ),
        )

        data = nm.to_dict()
        self.assertEqual(data["discretization"]["time_integrator"], "velocity_verlet")
        self.assertEqual(data["discretization"]["order"], 2)
        self.assertEqual(data["solver"]["algorithm"], "conjugate_gradient")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestMathSchema))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestBoundaryCondition))
    suite.addTests(loader.loadTestsFromTestCase(TestComputationalGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestNumericalMethod))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
