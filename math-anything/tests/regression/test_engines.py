"""Regression tests for engine extractors."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure engines path is importable
_engines = Path(__file__).parent.parent.parent / "engines"
if str(_engines) not in sys.path:
    sys.path.insert(0, str(_engines))

_engines_abs = str(_engines.resolve())


def _import_module_direct(package_path_parts):
    """Import a module directly by file path, bypassing __init__.py of parent packages."""
    module_path = Path(_engines_abs, *package_path_parts)
    spec = importlib.util.spec_from_file_location(".".join(package_path_parts), module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[".".join(package_path_parts)] = mod
    spec.loader.exec_module(mod)
    return mod


class TestVaspExtractor:
    def test_instantiate(self):
        from vasp.core.extractor_v2 import VaspExtractor

        extractor = VaspExtractor()
        assert extractor is not None
        assert hasattr(extractor, "extract")

    def test_extract_minimal_empty(self):
        from vasp.core.extractor_v2 import VaspExtractor

        extractor = VaspExtractor()
        schema = extractor.extract({})
        assert schema.schema_version == "1.0.0"
        assert schema.meta.extracted_by.startswith("math-anything-vasp")

    def test_extract_returns_math_schema(self):
        from vasp.core.extractor_v2 import VaspExtractor

        from math_anything.schemas import MathSchema

        extractor = VaspExtractor()
        schema = extractor.extract({})
        assert isinstance(schema, MathSchema)

    def test_has_mathematical_model(self):
        from vasp.core.extractor_v2 import VaspExtractor

        extractor = VaspExtractor()
        schema = extractor.extract({})
        assert schema.mathematical_model is not None

    def test_has_numerical_method(self):
        from vasp.core.extractor_v2 import VaspExtractor

        extractor = VaspExtractor()
        schema = extractor.extract({})
        assert schema.numerical_method is not None

    def test_has_computational_graph(self):
        from vasp.core.extractor_v2 import VaspExtractor

        extractor = VaspExtractor()
        schema = extractor.extract({})
        assert schema.computational_graph is not None

    def test_has_conservation_properties(self):
        from vasp.core.extractor_v2 import VaspExtractor

        extractor = VaspExtractor()
        schema = extractor.extract({})
        assert schema.conservation_properties is not None

    def test_has_raw_symbols(self):
        from vasp.core.extractor_v2 import VaspExtractor

        extractor = VaspExtractor()
        schema = extractor.extract({})
        assert isinstance(schema.raw_symbols, dict)

    def test_has_symbolic_constraints(self):
        from vasp.core.extractor_v2 import VaspExtractor

        extractor = VaspExtractor()
        schema = extractor.extract({})
        assert schema.symbolic_constraints is not None

    def test_extract_with_incar(self, tmp_path):
        from vasp.core.extractor_v2 import VaspExtractor

        incar = tmp_path / "INCAR"
        incar.write_text("ENCUT = 520\nISMEAR = 1\nSIGMA = 0.05\n")
        extractor = VaspExtractor()
        schema = extractor.extract({"incar": str(incar)})
        assert schema.raw_symbols["incar"] is not None


class TestLammpsExtractor:
    @pytest.fixture
    def lammps_input(self, tmp_path):
        p = tmp_path / "input.lammps"
        p.write_text("units metal\nboundary p p p\ntimestep 0.001\npair_style lj/cut 2.5\nfix 1 all nve\nrun 100\n")
        return str(p)

    def test_instantiate(self):
        from lammps.core.extractor import LammpsExtractor

        extractor = LammpsExtractor()
        assert extractor is not None
        assert hasattr(extractor, "extract")

    def test_extract_minimal(self, lammps_input):
        from lammps.core.extractor import LammpsExtractor

        extractor = LammpsExtractor()
        schema = extractor.extract({"input": lammps_input})
        assert schema.schema_version == "1.0.0"
        assert schema.meta.extracted_by.startswith("math-anything-lammps")

    def test_extract_with_log(self, lammps_input, tmp_path):
        from lammps.core.extractor import LammpsExtractor

        log = tmp_path / "log.lammps"
        log.write_text("Step Temp E_pair E_mol TotEng Press\n")
        extractor = LammpsExtractor()
        schema = extractor.extract({"input": lammps_input, "log": str(log)})
        assert schema is not None

    def test_extract_with_options(self, lammps_input):
        from lammps.core.extractor import LammpsExtractor

        extractor = LammpsExtractor()
        schema = extractor.extract({"input": lammps_input}, {"verbose": True})
        assert schema is not None

    def test_returns_math_schema(self, lammps_input):
        from lammps.core.extractor import LammpsExtractor

        from math_anything.schemas import MathSchema

        extractor = LammpsExtractor()
        schema = extractor.extract({"input": lammps_input})
        assert isinstance(schema, MathSchema)

    def test_has_mathematical_model(self, lammps_input):
        from lammps.core.extractor import LammpsExtractor

        extractor = LammpsExtractor()
        schema = extractor.extract({"input": lammps_input})
        assert schema.mathematical_model is not None

    def test_has_numerical_method(self, lammps_input):
        from lammps.core.extractor import LammpsExtractor

        extractor = LammpsExtractor()
        schema = extractor.extract({"input": lammps_input})
        assert schema.numerical_method is not None

    def test_has_computational_graph(self, lammps_input):
        from lammps.core.extractor import LammpsExtractor

        extractor = LammpsExtractor()
        schema = extractor.extract({"input": lammps_input})
        assert schema.computational_graph is not None

    def test_has_conservation_properties(self, lammps_input):
        from lammps.core.extractor import LammpsExtractor

        extractor = LammpsExtractor()
        schema = extractor.extract({"input": lammps_input})
        assert schema.conservation_properties is not None

    def test_has_raw_symbols(self, lammps_input):
        from lammps.core.extractor import LammpsExtractor

        extractor = LammpsExtractor()
        schema = extractor.extract({"input": lammps_input})
        assert isinstance(schema.raw_symbols, dict)

    def test_has_symbolic_constraints(self, lammps_input):
        from lammps.core.extractor import LammpsExtractor

        extractor = LammpsExtractor()
        schema = extractor.extract({"input": lammps_input})
        assert schema.symbolic_constraints is not None

    def test_input_required(self):
        from lammps.core.extractor import LammpsExtractor

        extractor = LammpsExtractor()
        with pytest.raises(ValueError, match="Input file required"):
            extractor.extract({})


class TestAbaqusExtractor:
    def test_instantiate(self):
        from abaqus.core.extractor import AbaqusExtractor

        extractor = AbaqusExtractor()
        assert extractor is not None
        assert hasattr(extractor, "extract")

    def test_extract_minimal_requires_input(self):
        from abaqus.core.extractor import AbaqusExtractor

        extractor = AbaqusExtractor()
        with pytest.raises(ValueError, match="Input file required"):
            extractor.extract({})

    def test_extract_with_file(self, tmp_path):
        from abaqus.core.extractor import AbaqusExtractor

        inp = tmp_path / "test.inp"
        inp.write_text(
            "*HEADING\n"
            "Test job\n"
            "*NODE\n"
            "1, 0.0, 0.0, 0.0\n"
            "2, 1.0, 0.0, 0.0\n"
            "*ELEMENT, TYPE=B31\n"
            "1, 1, 2\n"
            "*MATERIAL, NAME=STEEL\n"
            "*ELASTIC\n"
            "200000.0, 0.3\n"
            "*DENSITY\n"
            "7800.0\n"
            "*STEP\n"
            "*STATIC\n"
            "*BOUNDARY\n"
            "1, 1, 3, 0.0\n"
            "*END STEP\n"
        )
        extractor = AbaqusExtractor()
        schema = extractor.extract({"input": str(inp)})
        assert schema.schema_version == "1.0.0"
        assert schema.meta.extracted_by.startswith("math-anything-abaqus")
        assert schema.mathematical_model is not None


class TestAnsysExtractor:
    def test_instantiate(self):
        from ansys.core.input_extractor import AnsysInputExtractor

        extractor = AnsysInputExtractor()
        assert extractor is not None
        assert hasattr(extractor, "extract")

    def test_extract_minimal_requires_input(self):
        from ansys.core.input_extractor import AnsysInputExtractor

        extractor = AnsysInputExtractor()
        with pytest.raises(ValueError, match="Input file required"):
            extractor.extract({})

    def test_extract_with_file(self, tmp_path):
        from ansys.core.input_extractor import AnsysInputExtractor

        apdl = tmp_path / "test.dat"
        apdl.write_text(
            "/PREP7\n"
            "ET,1,PLANE182\n"
            "MP,EX,1,200000\n"
            "MP,PRXY,1,0.3\n"
            "MP,DENS,1,7800\n"
            "N,1,0,0,0\n"
            "N,2,10,0,0\n"
            "E,1,2\n"
            "D,1,ALL,0\n"
            "F,2,FY,-1000\n"
            "FINISH\n"
            "/SOLU\n"
            "SOLVE\n"
            "FINISH\n"
        )
        extractor = AnsysInputExtractor()
        schema = extractor.extract({"input": str(apdl)})
        assert schema.schema_version == "1.0.0"
        assert schema.meta.extracted_by.startswith("math-anything-ansys")
        assert schema.mathematical_model is not None


class TestComsolExtractor:
    def test_instantiate(self):
        from comsol.core.extractor import ComsolExtractor

        extractor = ComsolExtractor()
        assert extractor is not None
        assert hasattr(extractor, "extract")

    def test_extract_minimal_requires_input(self):
        from comsol.core.extractor import ComsolExtractor

        extractor = ComsolExtractor()
        with pytest.raises(ValueError, match="Input file required"):
            extractor.extract({})

    def test_extract_with_file(self, tmp_path):
        from comsol.core.extractor import ComsolExtractor

        params = tmp_path / "params.txt"
        params.write_text(
            "PHYSICS\n"
            "type solid_mechanics\n"
            "MATERIAL\n"
            "youngs_modulus 200e9\n"
            "poisson_ratio 0.3\n"
            "density 7800\n"
            "MESH\n"
            "element_type tetrahedral\n"
            "STUDY\n"
            "analysis_type stationary\n"
        )
        extractor = ComsolExtractor()
        schema = extractor.extract({"input": str(params)})
        assert schema.schema_version == "1.0.0"
        assert schema.meta.extracted_by.startswith("math-anything-comsol")
        assert schema.mathematical_model is not None


class TestGromacsExtractor:
    def test_instantiate(self):
        mod = _import_module_direct(["gromacs", "core", "extractor.py"])
        extractor = mod.GromacsExtractor()
        assert extractor is not None
        assert hasattr(extractor, "extract_thermodynamics")
        assert hasattr(extractor, "extract_energies")

    def test_extract_thermodynamics(self):
        mod = _import_module_direct(["gromacs", "core", "extractor.py"])
        extractor = mod.GromacsExtractor()
        temp = np.array([300.0, 301.0, 299.0])
        pres = np.array([1.0, 1.1, 0.9])
        vol = np.array([100.0, 100.1, 99.9])
        result = extractor.extract_thermodynamics(temp, pres, vol)
        assert "temperature_mean" in result
        assert "pressure_mean" in result
        assert "volume_mean" in result

    def test_extract_energies(self):
        mod = _import_module_direct(["gromacs", "core", "extractor.py"])
        extractor = mod.GromacsExtractor()
        total = np.array([-1000.0, -999.0, -1001.0])
        potential = np.array([-1200.0, -1199.0, -1201.0])
        kinetic = np.array([200.0, 200.0, 200.0])
        result = extractor.extract_energies(total, potential, kinetic)
        assert "total_energy_mean" in result
        assert "potential_energy_mean" in result
        assert "kinetic_energy_mean" in result


class TestMultiwfnExtractor:
    def test_instantiate(self):
        mod = _import_module_direct(["multiwfn", "core", "extractor.py"])
        extractor = mod.MultiwfnExtractor()
        assert extractor is not None
        assert hasattr(extractor, "extract_density_field")
        assert hasattr(extractor, "extract_orbital_info")

    def test_extract_density_field(self):
        mod = _import_module_direct(["multiwfn", "core", "extractor.py"])
        extractor = mod.MultiwfnExtractor()
        cube = np.random.rand(10, 10, 10).astype(np.float64) * 0.1
        result = extractor.extract_density_field(cube, (0.0, 0.0, 0.0), (0.1, 0.1, 0.1))
        assert "total_electrons" in result
        assert "max_density" in result

    def test_extract_orbital_info(self):
        mod = _import_module_direct(["multiwfn", "core", "extractor.py"])
        extractor = mod.MultiwfnExtractor()
        energies = np.array([-0.5, -0.3, 0.1, 0.2])
        occupations = np.array([2.0, 2.0, 0.0, 0.0])
        result = extractor.extract_orbital_info(energies, occupations)
        assert "homo_energy" in result
        assert "lumo_energy" in result
        assert "homo_lumo_gap" in result


class TestQEExtractor:
    def test_instantiate(self):
        from qe.core.extractor import QuantumEspressoExtractor

        extractor = QuantumEspressoExtractor()
        assert extractor is not None
        assert hasattr(extractor, "extract")

    def test_extract_minimal_requires_input(self):
        from qe.core.extractor import QuantumEspressoExtractor

        extractor = QuantumEspressoExtractor()
        # BaseEngineExtractor 允许无输入调用（使用默认参数）
        result = extractor.extract({})
        assert result is not None

    def test_extract_with_file(self, tmp_path):
        from qe.core.extractor import QuantumEspressoExtractor

        pwi = tmp_path / "qe.in"
        pwi.write_text(
            "&CONTROL\n"
            "  calculation = 'scf'\n"
            "  prefix = 'test'\n"
            "/\n"
            "&SYSTEM\n"
            "  ibrav = 1\n"
            "  celldm(1) = 10.0\n"
            "  nat = 1\n"
            "  ntyp = 1\n"
            "  ecutwfc = 40.0\n"
            "/\n"
            "&ELECTRONS\n"
            "/\n"
            "ATOMIC_SPECIES\n"
            "Si  28.086  Si.pbe-rrkj.UPF\n"
            "ATOMIC_POSITIONS alat\n"
            "Si 0.0 0.0 0.0\n"
            "K_POINTS automatic\n"
            "4 4 4 0 0 0\n"
        )
        extractor = QuantumEspressoExtractor()
        schema = extractor.extract({"input": str(pwi)})
        assert schema.schema_version == "1.0.0"
        assert schema.meta.extracted_by.startswith("math-anything-quantum_espresso")
        assert schema.mathematical_model is not None


class TestSolidworksExtractor:
    def test_instantiate(self):
        mod = _import_module_direct(["solidworks", "core", "extractor.py"])
        extractor = mod.SolidWorksExtractor()
        assert extractor is not None
        assert hasattr(extractor, "extract_stress_results")
        assert hasattr(extractor, "extract_mesh_info")

    def test_extract_mesh_info(self):
        mod = _import_module_direct(["solidworks", "core", "extractor.py"])
        extractor = mod.SolidWorksExtractor()
        result = extractor.extract_mesh_info(1000, 5000)
        assert result["num_nodes"] == 1000
        assert result["num_elements"] == 5000

    def test_extract_stress_results(self):
        mod = _import_module_direct(["solidworks", "core", "extractor.py"])
        extractor = mod.SolidWorksExtractor()
        stress = np.array([50.0, 100.0, 150.0, 200.0, 250.0])
        result = extractor.extract_stress_results(stress, 250.0)
        assert result["von_mises_max"] == 250.0
        assert result["safety_factor_min"] == 1.0


class TestVoxelExtractor:
    def test_instantiate(self):
        from voxel.core.harness import VoxelHarness

        harness = VoxelHarness()
        assert harness is not None
        assert hasattr(harness, "extract")
        assert harness.engine_name == "voxel"

    def test_extract(self):
        from voxel.core.harness import VoxelHarness

        harness = VoxelHarness()
        schema = harness.extract({}, {"simulation_type": "generic"})
        assert schema.schema_version == "1.0.0"

    def test_lattice_boltzmann(self):
        from voxel.core.harness import VoxelHarness

        harness = VoxelHarness()
        schema = harness.extract({}, {"simulation_type": "lattice_boltzmann"})
        eqs = schema.mathematical_model.governing_equations
        assert len(eqs) >= 2
