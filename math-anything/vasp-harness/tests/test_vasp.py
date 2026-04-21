"""Tests for VASP harness."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_anything.schemas import MathSchema
from math_anything.vasp.core.harness import VaspHarness
from math_anything.vasp.core.parser import VaspInputParser, VaspOutputParser

# Example VASP input files
EXAMPLE_INCAR = """
SYSTEM = Si bulk calculation
ENCUT = 300
ISMEAR = 0
SIGMA = 0.1
EDIFF = 1E-6
IBRION = -1
NSW = 0
NELM = 100
ISPIN = 1
ALGO = Normal
"""

EXAMPLE_POSCAR = """
Si
1.0
    2.7150000000    0.0000000000    0.0000000000
    0.0000000000    2.7150000000    0.0000000000
    0.0000000000    0.0000000000    2.7150000000
   Si
    2
Direct
    0.0000000000    0.0000000000    0.0000000000
    0.2500000000    0.2500000000    0.2500000000
"""

EXAMPLE_KPOINTS = """
Automatic mesh
0
Gamma
  8  8  8
  0.  0.  0.
"""

EXAMPLE_OUTCAR = """
 vasp.5.4.4.18Apr17-6-g9f103f2a35 (build Apr 18 2017 09:54:44) complex
 
 POSCAR = Si
 POTCAR:    PAW_PBE Si 05Jan2001
  energy without entropy =      -10.81629359  energy(sigma->0) =      -10.81629359
  E-fermi :  -0.000000
 reached required accuracy - stopping structural energy minimisation
 free  energy   TOTEN  =       -10.81629359 eV
"""


class TestVaspInputParser(unittest.TestCase):
    """Test VASP input file parsing."""

    def test_parse_incar(self):
        """Test parsing INCAR file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".INCAR", delete=False) as f:
            f.write(EXAMPLE_INCAR)
            temp_path = f.name

        try:
            parser = VaspInputParser()
            params = parser.parse_incar(temp_path)

            self.assertEqual(params["ENCUT"], 300)
            self.assertEqual(params["ISMEAR"], 0)
            self.assertEqual(params["SIGMA"], 0.1)
            self.assertEqual(params["ISPIN"], 1)
            self.assertEqual(params["ALGO"], "Normal")
        finally:
            os.unlink(temp_path)

    def test_parse_poscar(self):
        """Test parsing POSCAR file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR", delete=False) as f:
            f.write(EXAMPLE_POSCAR)
            temp_path = f.name

        try:
            parser = VaspInputParser()
            structure = parser.parse_poscar(temp_path)

            self.assertEqual(structure.system_name, "Si")
            self.assertEqual(structure.atom_types, ["Si"])
            self.assertEqual(structure.atom_counts, [2])
            self.assertEqual(structure.coord_type, "Direct")
            self.assertEqual(sum(structure.atom_counts), 2)
        finally:
            os.unlink(temp_path)

    def test_parse_kpoints(self):
        """Test parsing KPOINTS file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".KPOINTS", delete=False
        ) as f:
            f.write(EXAMPLE_KPOINTS)
            temp_path = f.name

        try:
            parser = VaspInputParser()
            kpoints = parser.parse_kpoints(temp_path)

            self.assertEqual(kpoints.mode, "Gamma")
            self.assertEqual(kpoints.grid, [8, 8, 8])
        finally:
            os.unlink(temp_path)


class TestVaspOutputParser(unittest.TestCase):
    """Test VASP output file parsing."""

    def test_parse_outcar(self):
        """Test parsing OUTCAR file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".OUTCAR", delete=False) as f:
            f.write(EXAMPLE_OUTCAR)
            temp_path = f.name

        try:
            parser = VaspOutputParser()
            results = parser.parse_outcar(temp_path)

            self.assertIsNotNone(results.total_energy)
            self.assertAlmostEqual(results.total_energy, -10.81629359)
            self.assertIsNotNone(results.fermi_energy)
            self.assertTrue(results.converged)
        finally:
            os.unlink(temp_path)


class TestVaspHarness(unittest.TestCase):
    """Test VASP harness extraction."""

    def setUp(self):
        """Create temporary VASP files."""
        self.temp_dir = tempfile.mkdtemp()

        self.incar_path = os.path.join(self.temp_dir, "INCAR")
        self.poscar_path = os.path.join(self.temp_dir, "POSCAR")
        self.kpoints_path = os.path.join(self.temp_dir, "KPOINTS")
        self.outcar_path = os.path.join(self.temp_dir, "OUTCAR")

        with open(self.incar_path, "w") as f:
            f.write(EXAMPLE_INCAR)
        with open(self.poscar_path, "w") as f:
            f.write(EXAMPLE_POSCAR)
        with open(self.kpoints_path, "w") as f:
            f.write(EXAMPLE_KPOINTS)
        with open(self.outcar_path, "w") as f:
            f.write(EXAMPLE_OUTCAR)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_extract_kohn_sham_equations(self):
        """Test extracting Kohn-Sham equations."""
        harness = VaspHarness()

        schema = harness.extract(
            {
                "incar": self.incar_path,
                "poscar": self.poscar_path,
                "kpoints": self.kpoints_path,
                "outcar": self.outcar_path,
            }
        )

        # Check governing equations
        equations = schema.mathematical_model.governing_equations
        self.assertGreater(len(equations), 0)

        # Find Kohn-Sham equation
        ks_eqs = [e for e in equations if e.id == "kohn_sham"]
        self.assertEqual(len(ks_eqs), 1)

        ks = ks_eqs[0]
        self.assertEqual(ks.type, "eigenvalue_problem")
        self.assertIn("wavefunction", ks.variables)

    def test_extract_periodic_boundary_conditions(self):
        """Test extracting periodic boundary conditions."""
        harness = VaspHarness()

        schema = harness.extract(
            {
                "incar": self.incar_path,
                "poscar": self.poscar_path,
                "kpoints": self.kpoints_path,
            }
        )

        bcs = schema.mathematical_model.boundary_conditions
        self.assertGreater(len(bcs), 0)

        # Check for periodic BC
        periodic = [bc for bc in bcs if bc.type == "periodic"]
        self.assertEqual(len(periodic), 1)

    def test_extract_bloch_theorem(self):
        """Test extracting Bloch theorem boundary condition."""
        harness = VaspHarness()

        schema = harness.extract(
            {
                "incar": self.incar_path,
                "poscar": self.poscar_path,
                "kpoints": self.kpoints_path,
            }
        )

        bcs = schema.mathematical_model.boundary_conditions
        bloch = [bc for bc in bcs if bc.id == "bloch_theorem"]

        if bloch:
            self.assertEqual(bloch[0].type, "quasi_periodic")

    def test_extract_numerical_method(self):
        """Test extracting numerical method."""
        harness = VaspHarness()

        schema = harness.extract(
            {
                "incar": self.incar_path,
            }
        )

        nm = schema.numerical_method
        self.assertEqual(nm.discretization.space_discretization, "plane_wave_basis")
        self.assertEqual(nm.discretization.time_integrator, "scf_iteration")

    def test_extract_computational_graph(self):
        """Test extracting computational graph with SCF loop."""
        harness = VaspHarness()

        schema = harness.extract(
            {
                "incar": self.incar_path,
            }
        )

        cg = schema.computational_graph
        self.assertGreater(len(cg.nodes), 0)

        # Check for SCF nodes
        scf_nodes = [n for n in cg.nodes if "scf" in n.id]
        self.assertGreater(len(scf_nodes), 0)


class TestVaspSchemaValidation(unittest.TestCase):
    """Test VASP schema validation."""

    def setUp(self):
        """Create temporary VASP files."""
        self.temp_dir = tempfile.mkdtemp()

        self.incar_path = os.path.join(self.temp_dir, "INCAR")
        self.poscar_path = os.path.join(self.temp_dir, "POSCAR")

        with open(self.incar_path, "w") as f:
            f.write(EXAMPLE_INCAR)
        with open(self.poscar_path, "w") as f:
            f.write(EXAMPLE_POSCAR)

    def tearDown(self):
        """Clean up."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_schema_valid(self):
        """Test that extracted schema is valid."""
        from math_anything.schemas import SchemaValidator

        harness = VaspHarness()
        schema = harness.extract(
            {
                "incar": self.incar_path,
                "poscar": self.poscar_path,
            }
        )

        validator = SchemaValidator()
        is_valid = validator.validate(schema.to_dict())

        self.assertTrue(is_valid, f"Validation errors: {validator.errors}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
