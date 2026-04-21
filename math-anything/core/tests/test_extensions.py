"""Tests for Schema Extension System."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_anything.schemas import (ExtendedMathSchema, ExtensionRegistry,
                                   GraphNeuralNetworkExtension, MathSchema,
                                   MLInteratomicPotentialExtension,
                                   PINNLossExtension, SchemaExtension,
                                   get_available_extensions,
                                   validate_with_extensions)


class TestSchemaExtension(unittest.TestCase):
    """Test schema extension functionality."""

    def test_extension_registration(self):
        """Test that built-in extensions are registered."""
        extensions = get_available_extensions()

        self.assertIn("ml_interatomic_potential", extensions)
        self.assertIn("pinn_loss_function", extensions)
        self.assertIn("graph_neural_network", extensions)

    def test_ml_potential_extension(self):
        """Test ML potential extension."""
        ext = MLInteratomicPotentialExtension()

        # Check schema definition
        schema = ext.get_schema_definition()
        self.assertEqual(schema["type"], "object")
        self.assertIn("architecture", schema["properties"])
        self.assertIn("GAP", schema["properties"]["architecture"]["enum"])

    def test_pinn_loss_validation(self):
        """Test PINN loss validation."""
        ext = PINNLossExtension()

        # Valid data
        valid_data = {
            "residual_weight": 1.0,
            "boundary_weight": 0.5,
            "initial_weight": 0.3,
        }
        self.assertTrue(ext.validate_data(valid_data))

        # Invalid: all zeros
        invalid_data = {
            "residual_weight": 0,
            "boundary_weight": 0,
        }
        self.assertFalse(ext.validate_data(invalid_data))

        # Invalid: negative weight
        negative_data = {
            "residual_weight": -1.0,
        }
        self.assertFalse(ext.validate_data(negative_data))

    def test_gnn_extension_required_fields(self):
        """Test GNN extension required fields."""
        ext = GraphNeuralNetworkExtension()
        schema = ext.get_schema_definition()

        self.assertIn("gnn_type", schema["required"])
        self.assertIn("n_layers", schema["required"])
        self.assertIn("hidden_dim", schema["required"])


class TestExtensionRegistry(unittest.TestCase):
    """Test extension registry."""

    def test_get_extension(self):
        """Test retrieving extension by name."""
        ext_class = ExtensionRegistry.get("ml_interatomic_potential")
        self.assertIsNotNone(ext_class)
        self.assertEqual(ext_class.name, "ml_interatomic_potential")

    def test_create_extension(self):
        """Test creating extension instance."""
        ext = ExtensionRegistry.create("pinn_loss_function")
        self.assertIsNotNone(ext)
        self.assertIsInstance(ext, PINNLossExtension)

    def test_get_all_definitions(self):
        """Test getting all extension definitions."""
        definitions = ExtensionRegistry.get_all_definitions()

        self.assertIn("ml_interatomic_potential", definitions)
        self.assertIn("pinn_loss_function", definitions)

        # Check structure
        for name, defn in definitions.items():
            self.assertIn("metadata", defn)
            self.assertIn("schema", defn)


class TestExtendedMathSchema(unittest.TestCase):
    """Test extended schema with custom objects."""

    def test_add_extension(self):
        """Test adding extension to schema."""
        base = MathSchema()
        extended = ExtendedMathSchema(base)

        ml_data = {
            "architecture": "GAP",
            "cutoff_radius": 5.0,
            "n_descriptors": 100,
        }

        result = extended.add_extension("ml_interatomic_potential", ml_data)
        self.assertTrue(result)

        # Verify stored
        stored = extended.get_extension("ml_interatomic_potential")
        self.assertEqual(stored["architecture"], "GAP")

    def test_add_extension_validation_failure(self):
        """Test that invalid extension data is rejected."""
        base = MathSchema()
        extended = ExtendedMathSchema(base)

        # Invalid PINN loss (all zeros)
        invalid_data = {
            "residual_weight": 0,
            "boundary_weight": 0,
        }

        with self.assertRaises(ValueError):
            extended.add_extension("pinn_loss_function", invalid_data, validate=True)

    def test_add_unknown_extension(self):
        """Test adding unregistered extension."""
        base = MathSchema()
        extended = ExtendedMathSchema(base)

        with self.assertRaises(ValueError):
            extended.add_extension("unknown_extension", {"data": "test"})

    def test_remove_extension(self):
        """Test removing extension."""
        base = MathSchema()
        extended = ExtendedMathSchema(base)

        extended.add_extension(
            "graph_neural_network",
            {
                "gnn_type": "SchNet",
                "n_layers": 3,
                "hidden_dim": 64,
            },
        )

        self.assertIn("graph_neural_network", extended.list_extensions())

        result = extended.remove_extension("graph_neural_network")
        self.assertTrue(result)
        self.assertNotIn("graph_neural_network", extended.list_extensions())

    def test_to_dict_with_extensions(self):
        """Test serialization with extensions."""
        base = MathSchema()
        extended = ExtendedMathSchema(base)

        extended.add_extension(
            "ml_interatomic_potential",
            {
                "architecture": "NeuralNetwork",
                "cutoff_radius": 6.0,
            },
        )

        data = extended.to_dict()

        self.assertIn("extensions", data)
        self.assertIn("ml_interatomic_potential", data["extensions"])
        self.assertIn("data", data["extensions"]["ml_interatomic_potential"])

    def test_save_and_load_with_extensions(self):
        """Test saving and loading extended schema."""
        base = MathSchema()
        extended = ExtendedMathSchema(base)

        extended.add_extension(
            "pinn_loss_function",
            {
                "residual_weight": 1.0,
                "boundary_weight": 0.5,
                "adaptive_weighting": True,
            },
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            extended.save(temp_path)

            # Load and verify
            with open(temp_path, "r") as f:
                loaded = json.load(f)

            self.assertIn("extensions", loaded)
            self.assertIn("pinn_loss_function", loaded["extensions"])
        finally:
            os.unlink(temp_path)


class TestValidateWithExtensions(unittest.TestCase):
    """Test validation of schemas with extensions."""

    def test_valid_extension(self):
        """Test validation with valid extension."""
        data = {
            "schema_version": "1.0.0",
            "meta": {"extracted_by": "test"},
            "extensions": {
                "ml_interatomic_potential": {
                    "data": {
                        "architecture": "GAP",
                        "cutoff_radius": 5.0,
                    }
                }
            },
        }

        report = validate_with_extensions(data)
        self.assertTrue(report["valid"])
        self.assertIn("ml_interatomic_potential", report["extensions"])

    def test_invalid_extension_data(self):
        """Test validation with invalid extension data."""
        data = {
            "schema_version": "1.0.0",
            "extensions": {
                "pinn_loss_function": {
                    "data": {
                        "residual_weight": -1.0,  # Invalid negative
                    }
                }
            },
        }

        report = validate_with_extensions(data)
        self.assertFalse(report["valid"])
        self.assertGreater(len(report["errors"]), 0)

    def test_unknown_extension(self):
        """Test validation with unknown extension."""
        data = {"schema_version": "1.0.0", "extensions": {"unknown_ext": {"data": {}}}}

        report = validate_with_extensions(data)
        # Should warn but not fail
        self.assertTrue(report["valid"])
        self.assertGreater(len(report["warnings"]), 0)

    def test_no_extensions(self):
        """Test validation without extensions."""
        data = {
            "schema_version": "1.0.0",
            "meta": {"extracted_by": "test"},
        }

        report = validate_with_extensions(data)
        self.assertTrue(report["valid"])


class TestCustomExtensionRegistration(unittest.TestCase):
    """Test registering custom extensions."""

    def test_register_custom_extension(self):
        """Test registering a user-defined extension."""

        @ExtensionRegistry.register
        class TestExtension(SchemaExtension):
            name = "test_extension"
            version = "1.0.0"

            def get_schema_definition(self):
                return {"type": "object", "properties": {"value": {"type": "number"}}}

        # Verify registration
        self.assertIn("test_extension", get_available_extensions())

        # Verify it works
        ext = ExtensionRegistry.create("test_extension")
        self.assertIsNotNone(ext)
        self.assertEqual(ext.name, "test_extension")


if __name__ == "__main__":
    unittest.main(verbosity=2)
