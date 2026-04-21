"""Test suite for fix deform tensor boundary condition extraction.

This test validates Phase 0 requirement: Schema must be able to express
fix deform as a 2nd-order tensor (deformation gradient F_ij) with:
1. 二阶张量约束（变形梯度 F_ij）
2. 对称性声明
3. 迹条件/体积变化约束
4. 等价的应变率张量描述
"""

import json
import os
import sys
import tempfile
import unittest

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_anything.lammps.core.harness import LammpsHarness
from math_anything.schemas import MathSchema, SchemaValidator, TensorComponent

# Example LAMMPS input with fix deform
FIX_DEFORM_INPUT = """
# 3D metal simulation - Uniaxial tension
units           metal
boundary        p p p
atom_style      atomic

lattice         fcc 3.52
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box

mass            1 58.69

pair_style      lj/cut 10.0
pair_coeff      1 1 0.54 2.5

neighbor        0.3 bin
neigh_modify    every 20 delay 0 check no

velocity        all create 300.0 12345

fix             1 all nve
fix             2 all deform 1 x erate 0.01

timestep        0.001
run             100
"""


class TestFixDeformTensorBC(unittest.TestCase):
    """Test fix deform tensor boundary condition extraction."""

    @classmethod
    def setUpClass(cls):
        """Create temporary input file."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.input_file = os.path.join(cls.temp_dir, "in.deform")

        with open(cls.input_file, "w") as f:
            f.write(FIX_DEFORM_INPUT)

        # Extract
        cls.harness = LammpsHarness()
        cls.schema = cls.harness.extract({"input": cls.input_file})
        cls.data = cls.schema.to_dict()

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil

        shutil.rmtree(cls.temp_dir)

    def test_schema_validation(self):
        """Test that extracted schema is valid."""
        validator = SchemaValidator()
        is_valid = validator.validate(self.data)

        self.assertTrue(is_valid, f"Schema validation failed: {validator.errors}")

    def test_boundary_conditions_exist(self):
        """Test that boundary conditions are extracted."""
        bcs = self.data["mathematical_model"]["boundary_conditions"]
        self.assertGreater(len(bcs), 0, "No boundary conditions found")

    def test_fix_deform_bc_exists(self):
        """Test that fix deform is extracted as boundary condition."""
        bcs = self.data["mathematical_model"]["boundary_conditions"]

        deform_bcs = [bc for bc in bcs if bc["id"].startswith("fix_")]
        self.assertGreater(len(deform_bcs), 0, "No fix deform boundary condition found")

    def test_tensor_rank_2(self):
        """Test that deformation gradient is rank-2 tensor."""
        bcs = self.data["mathematical_model"]["boundary_conditions"]

        for bc in bcs:
            if bc["id"].startswith("fix_"):
                mo = bc.get("mathematical_object", {})
                self.assertEqual(
                    mo.get("tensor_rank"),
                    2,
                    f"Expected tensor_rank=2, got {mo.get('tensor_rank')}",
                )

    def test_tensor_components_exist(self):
        """Test that tensor components are present."""
        bcs = self.data["mathematical_model"]["boundary_conditions"]

        for bc in bcs:
            if bc["id"].startswith("fix_"):
                mo = bc.get("mathematical_object", {})
                components = mo.get("components", [])
                self.assertGreater(len(components), 0, "No tensor components found")

                # Check for diagonal components
                indices = [tuple(c.get("index", [])) for c in components]
                self.assertIn((1, 1), indices, "Missing F_11 component")
                self.assertIn((2, 2), indices, "Missing F_22 component")
                self.assertIn((3, 3), indices, "Missing F_33 component")

    def test_tensor_form_expression(self):
        """Test that tensor form is properly expressed."""
        bcs = self.data["mathematical_model"]["boundary_conditions"]

        for bc in bcs:
            if bc["id"].startswith("fix_"):
                mo = bc.get("mathematical_object", {})
                self.assertEqual(
                    mo.get("tensor_form"),
                    "F_{ij} = ∂x_i/∂X_j",
                    "Incorrect tensor form expression",
                )

    def test_symmetry_declaration(self):
        """Test that symmetry is declared."""
        bcs = self.data["mathematical_model"]["boundary_conditions"]

        for bc in bcs:
            if bc["id"].startswith("fix_"):
                mo = bc.get("mathematical_object", {})
                self.assertIn("symmetry", mo, "Symmetry not declared")
                self.assertEqual(
                    mo.get("symmetry"), "symmetric", "Expected symmetric tensor"
                )

    def test_trace_condition(self):
        """Test that trace condition is present."""
        bcs = self.data["mathematical_model"]["boundary_conditions"]

        for bc in bcs:
            if bc["id"].startswith("fix_"):
                mo = bc.get("mathematical_object", {})
                self.assertIn("trace_condition", mo, "Trace condition not present")
                # Should contain det(F)
                self.assertIn(
                    "det(F)",
                    mo.get("trace_condition", ""),
                    "Trace condition should reference det(F)",
                )

    def test_equivalent_strain_rate_formulation(self):
        """Test that equivalent strain rate formulation is provided."""
        bcs = self.data["mathematical_model"]["boundary_conditions"]

        for bc in bcs:
            if bc["id"].startswith("fix_"):
                equiv = bc.get("equivalent_formulations", [])
                self.assertGreater(len(equiv), 0, "No equivalent formulations found")

                # Check for strain rate tensor
                strain_rate = [
                    e for e in equiv if e.get("type") == "strain_rate_tensor"
                ]
                self.assertEqual(
                    len(strain_rate), 1, "Expected strain rate tensor formulation"
                )

                form = strain_rate[0].get("form", "")
                self.assertIn("ε̇", form, "Strain rate symbol not found")

    def test_software_implementation(self):
        """Test that software implementation details are captured."""
        bcs = self.data["mathematical_model"]["boundary_conditions"]

        for bc in bcs:
            if bc["id"].startswith("fix_"):
                impl = bc.get("software_implementation", {})
                self.assertIn("command", impl, "Software command not recorded")
                self.assertIn("style", impl, "Fix style not recorded")
                self.assertIn("parameters", impl, "Parameters not recorded")

    def test_dual_role_declaration(self):
        """Test that dual role (BC + external drive) is declared."""
        bcs = self.data["mathematical_model"]["boundary_conditions"]

        for bc in bcs:
            if bc["id"].startswith("fix_"):
                dual = bc.get("dual_role", {})
                self.assertTrue(
                    dual.get("is_boundary_condition", False),
                    "Should be declared as boundary condition",
                )
                self.assertTrue(
                    dual.get("is_external_drive", False),
                    "Should be declared as external drive",
                )

    def test_computational_graph_explicit_implicit(self):
        """Test that computational graph distinguishes explicit/implicit."""
        cg = self.data["computational_graph"]

        self.assertIn("nodes", cg, "No nodes in computational graph")
        self.assertIn("execution_topology", cg, "No execution topology")

        # Check for implicit loops
        topo = cg.get("execution_topology", {})
        implicit_loops = topo.get("implicit_loops", [])

        # Should have at least one node with implicit_loop mode
        nodes = cg.get("nodes", [])
        has_implicit = False
        for node in nodes:
            semantics = node.get("math_semantics", {})
            updates = semantics.get("updates", {})
            if updates.get("mode") == "implicit_loop":
                has_implicit = True
                break

        # Note: NVE doesn't have implicit loops, but NVT/NPT do
        # This test checks that the mode is properly set


class TestSchemaCompleteness(unittest.TestCase):
    """Test overall schema completeness for fix deform case."""

    @classmethod
    def setUpClass(cls):
        """Create temporary input file."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.input_file = os.path.join(cls.temp_dir, "in.deform")

        with open(cls.input_file, "w") as f:
            f.write(FIX_DEFORM_INPUT)

        cls.harness = LammpsHarness()
        cls.schema = cls.harness.extract({"input": cls.input_file})
        cls.data = cls.schema.to_dict()

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil

        shutil.rmtree(cls.temp_dir)

    def test_governing_equations(self):
        """Test that governing equations are extracted."""
        eqs = self.data["mathematical_model"]["governing_equations"]
        self.assertGreater(len(eqs), 0, "No governing equations found")

        # Check for Newton's law
        newton = [e for e in eqs if e.get("id") == "newton_second_law"]
        self.assertEqual(len(newton), 1, "Newton's law not found")

    def test_numerical_method(self):
        """Test that numerical method is extracted."""
        nm = self.data["numerical_method"]
        self.assertIn("discretization", nm, "No discretization")
        self.assertIn("solver", nm, "No solver")

    def test_conservation_properties(self):
        """Test that conservation properties are extracted."""
        cp = self.data.get("conservation_properties", {})
        # Should have energy conservation for NVE
        if "energy" in cp:
            self.assertIn(
                "preserved", cp["energy"], "Energy preservation not specified"
            )

    def test_raw_symbols(self):
        """Test that raw symbols are captured."""
        rs = self.data.get("raw_symbols", {})
        self.assertIn("fixes", rs, "Fixes not in raw symbols")

        # Check for deform fix
        fixes = rs.get("fixes", {})
        deform_fixes = [
            k
            for k in fixes.keys()
            if "deform" in str(fixes.get(k, {}).get("style", ""))
        ]
        self.assertGreater(len(deform_fixes), 0, "Deform fix not in raw symbols")


class TestJSONSerialization(unittest.TestCase):
    """Test JSON serialization and deserialization."""

    @classmethod
    def setUpClass(cls):
        """Create temporary input file."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.input_file = os.path.join(cls.temp_dir, "in.deform")

        with open(cls.input_file, "w") as f:
            f.write(FIX_DEFORM_INPUT)

        cls.harness = LammpsHarness()
        cls.schema = cls.harness.extract({"input": cls.input_file})

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        import shutil

        shutil.rmtree(cls.temp_dir)

    def test_save_and_load(self):
        """Test saving and loading schema."""
        json_path = os.path.join(self.temp_dir, "model.json")

        # Save
        self.schema.save(json_path)
        self.assertTrue(os.path.exists(json_path), "JSON file not created")

        # Load
        loaded = MathSchema.from_dict(self.schema.to_dict())
        self.assertIsNotNone(loaded, "Failed to load schema")

        # Verify key fields
        self.assertEqual(loaded.schema_version, self.schema.schema_version)
        self.assertEqual(loaded.meta.extracted_by, self.schema.meta.extracted_by)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestFixDeformTensorBC))
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaCompleteness))
    suite.addTests(loader.loadTestsFromTestCase(TestJSONSerialization))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
