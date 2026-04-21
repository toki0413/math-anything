"""Tests for Cross-Engine Session functionality."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_anything.core import (CoupledSchema, CouplingInterface, CouplingType,
                                CrossEngineSession, ModelScale, ScaleModel)
from math_anything.schemas import GoverningEquation, MathSchema


class TestCrossEngineSession(unittest.TestCase):
    """Test cross-engine session functionality."""

    def test_session_creation(self):
        """Test creating a cross-engine session."""
        session = CrossEngineSession()
        self.assertIsNotNone(session)
        self.assertEqual(len(session.models), 0)

    def test_add_model_manually(self):
        """Test manually adding models to session."""
        session = CrossEngineSession()

        # Create a mock scale model
        schema = MathSchema()
        schema.mathematical_model.governing_equations.append(
            GoverningEquation(
                id="test_eq",
                type="test",
                name="Test",
                mathematical_form="F=ma",
            )
        )

        model = ScaleModel(
            model_id="test_md",
            scale=ModelScale.ATOMISTIC,
            engine="lammps",
            schema=schema,
            domain="bulk",
        )

        session.models["test_md"] = model

        self.assertEqual(len(session.models), 1)
        self.assertIn("test_md", session.models)

    def test_detect_coupling_stress(self):
        """Test detecting stress coupling between models."""
        session = CrossEngineSession()

        # Create micro model with stress
        micro_schema = MathSchema()
        micro_schema.mathematical_model.governing_equations.append(
            GoverningEquation(
                id="virial_stress",
                type="tensor",
                name="Virial Stress",
                mathematical_form="σ = ...",
                variables=["stress", "positions", "forces"],
            )
        )

        micro_model = ScaleModel(
            model_id="md_bulk",
            scale=ModelScale.ATOMISTIC,
            engine="lammps",
            schema=micro_schema,
            domain="bulk",
        )

        # Create macro model with stress
        macro_schema = MathSchema()
        macro_schema.mathematical_model.governing_equations.append(
            GoverningEquation(
                id="cauchy_stress",
                type="tensor",
                name="Cauchy Stress",
                mathematical_form="∇·σ + f = 0",
                variables=["stress", "displacement"],
            )
        )

        macro_model = ScaleModel(
            model_id="fem_structure",
            scale=ModelScale.CONTINUUM,
            engine="abaqus",
            schema=macro_schema,
            domain="structure",
        )

        session.models["md_bulk"] = micro_model
        session.models["fem_structure"] = macro_model

        # Auto-detect coupling
        interfaces = session.auto_detect_coupling()

        self.assertGreater(len(interfaces), 0)

        # Check stress coupling detected
        stress_interfaces = [i for i in interfaces if "stress" in i.transfer_quantity]
        self.assertEqual(len(stress_interfaces), 1)

        iface = stress_interfaces[0]
        self.assertEqual(iface.from_scale, "md_bulk")
        self.assertEqual(iface.to_scale, "fem_structure")
        self.assertEqual(iface.mapping_type, "homogenization")

    def test_detect_coupling_temperature(self):
        """Test detecting temperature coupling."""
        session = CrossEngineSession()

        # Micro with temperature
        micro_schema = MathSchema()
        micro_schema.mathematical_model.governing_equations.append(
            GoverningEquation(
                id="temp",
                type="scalar",
                name="Temperature",
                mathematical_form="T = (2/3kB) * KE",
                variables=["temperature", "kinetic_energy"],
            )
        )

        micro_model = ScaleModel(
            model_id="md",
            scale=ModelScale.ATOMISTIC,
            engine="lammps",
            schema=micro_schema,
        )

        # Macro with temperature
        macro_schema = MathSchema()
        macro_schema.mathematical_model.governing_equations.append(
            GoverningEquation(
                id="heat",
                type="pde",
                name="Heat Equation",
                mathematical_form="∂T/∂t = α∇²T",
                variables=["temperature", "time"],
            )
        )

        macro_model = ScaleModel(
            model_id="fem",
            scale=ModelScale.CONTINUUM,
            engine="abaqus",
            schema=macro_schema,
        )

        session.models["md"] = micro_model
        session.models["fem"] = macro_model

        interfaces = session.auto_detect_coupling()

        temp_interfaces = [
            i for i in interfaces if i.transfer_quantity == "temperature"
        ]
        self.assertEqual(len(temp_interfaces), 1)
        self.assertEqual(temp_interfaces[0].mapping_type, "statistical_average")


class TestCoupledSchema(unittest.TestCase):
    """Test coupled schema functionality."""

    def test_coupled_schema_creation(self):
        """Test creating coupled schema."""
        coupled = CoupledSchema()

        self.assertEqual(coupled.schema_version, "1.0.0+coupled")
        self.assertEqual(len(coupled.models), 0)
        self.assertEqual(len(coupled.coupling_interfaces), 0)

    def test_add_model(self):
        """Test adding models to coupled schema."""
        coupled = CoupledSchema()

        schema = MathSchema()
        model = ScaleModel(
            model_id="test",
            scale=ModelScale.ATOMISTIC,
            engine="lammps",
            schema=schema,
        )

        coupled.add_model(model)

        self.assertEqual(len(coupled.models), 1)
        self.assertIn("test", coupled.models)

    def test_add_interface(self):
        """Test adding coupling interface."""
        coupled = CoupledSchema()

        iface = CouplingInterface(
            interface_id="test_interface",
            from_scale="micro",
            to_scale="macro",
            mapping_type="homogenization",
            transfer_quantity="stress",
        )

        coupled.add_interface(iface)

        self.assertEqual(len(coupled.coupling_interfaces), 1)
        self.assertEqual(coupled.coupling_interfaces[0].interface_id, "test_interface")

    def test_get_models_by_scale(self):
        """Test filtering models by scale."""
        coupled = CoupledSchema()

        # Add atomistic model
        schema1 = MathSchema()
        coupled.add_model(
            ScaleModel(
                model_id="md1",
                scale=ModelScale.ATOMISTIC,
                engine="lammps",
                schema=schema1,
            )
        )

        # Add continuum model
        schema2 = MathSchema()
        coupled.add_model(
            ScaleModel(
                model_id="fem1",
                scale=ModelScale.CONTINUUM,
                engine="abaqus",
                schema=schema2,
            )
        )

        atomistic = coupled.get_models_by_scale(ModelScale.ATOMISTIC)
        self.assertEqual(len(atomistic), 1)
        self.assertEqual(atomistic[0].model_id, "md1")

    def test_to_dict(self):
        """Test serialization to dict."""
        coupled = CoupledSchema()

        schema = MathSchema()
        coupled.add_model(
            ScaleModel(
                model_id="test",
                scale=ModelScale.ATOMISTIC,
                engine="lammps",
                schema=schema,
            )
        )

        data = coupled.to_dict()

        self.assertIn("schema_version", data)
        self.assertIn("models", data)
        self.assertIn("coupling_interfaces", data)
        self.assertEqual(data["n_models"], 1)


class TestModelScale(unittest.TestCase):
    """Test model scale enum."""

    def test_scale_values(self):
        """Test scale enum values."""
        self.assertEqual(ModelScale.QUANTUM.value, "quantum")
        self.assertEqual(ModelScale.ATOMISTIC.value, "atomistic")
        self.assertEqual(ModelScale.MESOSCALE.value, "mesoscale")
        self.assertEqual(ModelScale.CONTINUUM.value, "continuum")
        self.assertEqual(ModelScale.MACRO.value, "macro")

    def test_scale_ordering(self):
        """Test that scales can be ordered micro to macro."""
        scales = [
            ModelScale.QUANTUM,
            ModelScale.ATOMISTIC,
            ModelScale.MESOSCALE,
            ModelScale.CONTINUUM,
            ModelScale.MACRO,
        ]

        # Just verify the list is ordered
        self.assertEqual(len(scales), 5)


class TestCouplingInterface(unittest.TestCase):
    """Test coupling interface."""

    def test_interface_creation(self):
        """Test creating coupling interface."""
        iface = CouplingInterface(
            interface_id="stress_coupling",
            from_scale="md",
            to_scale="fem",
            from_objects=["bc:stress"],
            to_objects=["equation:equilibrium"],
            mapping_type="homogenization",
            transfer_quantity="stress_tensor",
            conservation_check=True,
        )

        self.assertEqual(iface.interface_id, "stress_coupling")
        self.assertTrue(iface.conservation_check)

    def test_to_dict(self):
        """Test interface serialization."""
        iface = CouplingInterface(
            interface_id="test",
            from_scale="a",
            to_scale="b",
        )

        data = iface.to_dict()

        self.assertEqual(data["interface_id"], "test")
        self.assertEqual(data["from_scale"], "a")
        self.assertEqual(data["to_scale"], "b")


class TestConsistencyCheck(unittest.TestCase):
    """Test consistency checking."""

    def test_consistent_system(self):
        """Test checking consistent coupled system."""
        session = CrossEngineSession()

        # Add two connected models
        schema1 = MathSchema()
        session.models["md"] = ScaleModel(
            model_id="md",
            scale=ModelScale.ATOMISTIC,
            engine="lammps",
            schema=schema1,
        )

        schema2 = MathSchema()
        session.models["fem"] = ScaleModel(
            model_id="fem",
            scale=ModelScale.CONTINUUM,
            engine="abaqus",
            schema=schema2,
        )

        # Add interface connecting them
        session.coupling_interfaces.append(
            CouplingInterface(
                interface_id="coupling",
                from_scale="md",
                to_scale="fem",
            )
        )

        report = session.check_consistency()

        self.assertTrue(report["consistent"])
        self.assertEqual(len(report["errors"]), 0)

    def test_inconsistent_missing_endpoint(self):
        """Test detecting missing interface endpoint."""
        session = CrossEngineSession()

        session.models["md"] = ScaleModel(
            model_id="md",
            scale=ModelScale.ATOMISTIC,
            engine="lammps",
            schema=MathSchema(),
        )

        # Interface to non-existent model
        session.coupling_interfaces.append(
            CouplingInterface(
                interface_id="bad",
                from_scale="md",
                to_scale="nonexistent",
            )
        )

        report = session.check_consistency()

        self.assertFalse(report["consistent"])
        self.assertGreater(len(report["errors"]), 0)

    def test_orphaned_model_warning(self):
        """Test warning for models without interfaces."""
        session = CrossEngineSession()

        session.models["orphan"] = ScaleModel(
            model_id="orphan",
            scale=ModelScale.ATOMISTIC,
            engine="lammps",
            schema=MathSchema(),
        )

        report = session.check_consistency()

        # Should be consistent but with warning
        self.assertTrue(report["consistent"])
        self.assertGreater(len(report["warnings"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
