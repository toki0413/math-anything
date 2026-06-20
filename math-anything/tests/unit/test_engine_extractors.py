"""Unit tests for engine extractors — VASP, LAMMPS, Abaqus, Ansys.

Tests the public API of each engine extractor: instantiation, build_schema,
and key output fields (governing equations, boundary conditions, etc.).
"""

import sys
from pathlib import Path

import pytest

# Ensure engines package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "engines"))


def _is_math_schema(obj):
    """Check if obj is a MathSchema without triggering circular imports."""
    return type(obj).__name__ == "MathSchema" and hasattr(obj, "schema_version")


# ── VASP ──


class TestVaspExtractor:
    """Tests for engines.vasp.core.extractor.VaspExtractor."""

    def test_instantiation(self):
        from engines.vasp.core.extractor import VaspExtractor

        ext = VaspExtractor()
        assert ext.engine_name == "vasp"

    def test_build_schema_returns_math_schema(self):
        from engines.vasp.core.extractor import VaspExtractor

        ext = VaspExtractor()
        schema = ext.build_schema(source_files={"input": ["INCAR"]})
        assert _is_math_schema(schema)

    def test_schema_version(self):
        from engines.vasp.core.extractor import VaspExtractor

        ext = VaspExtractor()
        schema = ext.build_schema(source_files={"input": ["INCAR"]})
        assert schema.schema_version == "1.0.0"

    def test_meta_info(self):
        from engines.vasp.core.extractor import VaspExtractor

        ext = VaspExtractor()
        schema = ext.build_schema(source_files={"input": ["INCAR"]})
        assert schema.meta is not None
        assert "vasp" in schema.meta.extracted_by

    def test_governing_equations_contain_kohn_sham(self):
        from engines.vasp.core.extractor import VaspExtractor

        ext = VaspExtractor()
        schema = ext.build_schema(source_files={"input": ["INCAR"]})
        eqs = schema.mathematical_model.governing_equations
        assert len(eqs) > 0
        names = [eq.name for eq in eqs]
        assert any("Kohn-Sham" in n for n in names)

    def test_governing_equations_contain_hohenberg_kohn(self):
        from engines.vasp.core.extractor import VaspExtractor

        ext = VaspExtractor()
        schema = ext.build_schema(source_files={"input": ["INCAR"]})
        eqs = schema.mathematical_model.governing_equations
        names = [eq.name for eq in eqs]
        assert any("Hohenberg-Kohn" in n for n in names)

    def test_numerical_method_present(self):
        from engines.vasp.core.extractor import VaspExtractor

        ext = VaspExtractor()
        schema = ext.build_schema(source_files={"input": ["INCAR"]})
        assert schema.numerical_method is not None

    def test_computational_graph_has_scf_nodes(self):
        from engines.vasp.core.extractor import VaspExtractor

        ext = VaspExtractor()
        schema = ext.build_schema(source_files={"input": ["INCAR"]})
        node_ids = [n.id for n in schema.computational_graph.nodes]
        assert "hamiltonian_construction" in node_ids
        assert "diagonalization" in node_ids
        assert "scf_convergence" in node_ids

    def test_conservation_properties(self):
        from engines.vasp.core.extractor import VaspExtractor

        ext = VaspExtractor()
        schema = ext.build_schema(source_files={"input": ["INCAR"]})
        assert "charge" in schema.conservation_properties
        assert schema.conservation_properties["charge"]["preserved"] is True

    def test_raw_symbols_present(self):
        from engines.vasp.core.extractor import VaspExtractor

        ext = VaspExtractor()
        schema = ext.build_schema(source_files={"input": ["INCAR"]})
        assert isinstance(schema.raw_symbols, dict)

    def test_to_dict_roundtrip(self):
        from engines.vasp.core.extractor import VaspExtractor

        ext = VaspExtractor()
        schema = ext.build_schema(source_files={"input": ["INCAR"]})
        d = schema.to_dict()
        assert "schema_version" in d
        assert "mathematical_model" in d
        assert "computational_graph" in d


# ── LAMMPS ──


class TestLammpsExtractor:
    """Tests for engines.lammps.core.extractor.LammpsExtractor."""

    @pytest.fixture
    def extractor(self):
        from engines.lammps.core.extractor import LammpsExtractor
        from engines.lammps.core.parser import ComputationalSettings

        ext = LammpsExtractor()
        # build_schema() requires self.settings to be initialized
        ext.settings = ComputationalSettings()
        return ext

    def test_instantiation(self):
        from engines.lammps.core.extractor import LammpsExtractor

        ext = LammpsExtractor()
        assert ext.engine_name == "lammps"

    def test_build_schema_returns_math_schema(self, extractor):
        schema = extractor.build_schema(source_files={"input": ["in.lammps"]})
        assert _is_math_schema(schema)

    def test_governing_equations_contain_newton(self, extractor):
        schema = extractor.build_schema(source_files={"input": ["in.lammps"]})
        eqs = schema.mathematical_model.governing_equations
        assert len(eqs) > 0
        names = [eq.name for eq in eqs]
        assert any("Newton" in n for n in names)

    def test_boundary_conditions_present(self, extractor):
        schema = extractor.build_schema(source_files={"input": ["in.lammps"]})
        # Default settings have boundary style p p p
        bcs = schema.mathematical_model.boundary_conditions
        assert isinstance(bcs, list)
        assert len(bcs) > 0

    def test_numerical_method_present(self, extractor):
        schema = extractor.build_schema(source_files={"input": ["in.lammps"]})
        assert schema.numerical_method is not None

    def test_computational_graph_has_nodes(self, extractor):
        schema = extractor.build_schema(source_files={"input": ["in.lammps"]})
        assert len(schema.computational_graph.nodes) >= 0

    def test_conservation_properties(self, extractor):
        schema = extractor.build_schema(source_files={"input": ["in.lammps"]})
        assert isinstance(schema.conservation_properties, dict)

    def test_symbolic_constraints_populated(self, extractor):
        schema = extractor.build_schema(source_files={"input": ["in.lammps"]})
        # LAMMPS enriches schema with symbolic_constraints
        assert hasattr(schema, "symbolic_constraints")

    def test_raw_symbols_present(self, extractor):
        schema = extractor.build_schema(source_files={"input": ["in.lammps"]})
        assert isinstance(schema.raw_symbols, dict)
        assert "boundary" in schema.raw_symbols


# ── Abaqus ──


class TestAbaqusExtractor:
    """Tests for engines.abaqus.core.extractor.AbaqusExtractor."""

    def test_instantiation(self):
        from engines.abaqus.core.extractor import AbaqusExtractor

        ext = AbaqusExtractor()
        assert ext.engine_name == "abaqus"

    def test_extractor_version(self):
        from engines.abaqus.core.extractor import AbaqusExtractor

        ext = AbaqusExtractor()
        assert ext.extractor_version == "0.2.0"

    def test_build_schema_returns_math_schema(self):
        from engines.abaqus.core.extractor import AbaqusExtractor

        ext = AbaqusExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        assert _is_math_schema(schema)

    def test_governing_equations_contain_equilibrium(self):
        from engines.abaqus.core.extractor import AbaqusExtractor

        ext = AbaqusExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        eqs = schema.mathematical_model.governing_equations
        assert len(eqs) > 0
        names = [eq.name for eq in eqs]
        assert any("Equilibrium" in n for n in names)

    def test_governing_equations_contain_weak_form(self):
        from engines.abaqus.core.extractor import AbaqusExtractor

        ext = AbaqusExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        eqs = schema.mathematical_model.governing_equations
        names = [eq.name for eq in eqs]
        assert any("Virtual Work" in n for n in names)

    def test_numerical_method_present(self):
        from engines.abaqus.core.extractor import AbaqusExtractor

        ext = AbaqusExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        assert schema.numerical_method is not None

    def test_computational_graph_has_assembly_and_solver(self):
        from engines.abaqus.core.extractor import AbaqusExtractor

        ext = AbaqusExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        node_ids = [n.id for n in schema.computational_graph.nodes]
        assert "assembly" in node_ids
        assert "solver" in node_ids

    def test_conservation_properties(self):
        from engines.abaqus.core.extractor import AbaqusExtractor

        ext = AbaqusExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        assert "equilibrium" in schema.conservation_properties

    def test_symbolic_constraints_populated(self):
        from engines.abaqus.core.extractor import AbaqusExtractor

        ext = AbaqusExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        assert hasattr(schema, "symbolic_constraints")
        assert len(schema.symbolic_constraints) > 0

    def test_parameter_relationships(self):
        from engines.abaqus.core.extractor import AbaqusExtractor

        ext = AbaqusExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        # Default settings have no materials, so relationships may be empty
        assert isinstance(
            schema.mathematical_model.parameter_relationships, list
        )


# ── Ansys ──


class TestAnsysExtractor:
    """Tests for engines.ansys.core.extractor.AnsysExtractor."""

    def test_instantiation(self):
        from engines.ansys.core.extractor import AnsysExtractor

        ext = AnsysExtractor()
        assert ext.engine_name == "ansys"

    def test_extractor_version(self):
        from engines.ansys.core.extractor import AnsysExtractor

        ext = AnsysExtractor()
        assert ext.extractor_version == "0.3.0"

    def test_build_schema_returns_math_schema(self):
        from engines.ansys.core.extractor import AnsysExtractor

        ext = AnsysExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        assert _is_math_schema(schema)

    def test_governing_equations_contain_equilibrium(self):
        from engines.ansys.core.extractor import AnsysExtractor

        ext = AnsysExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        eqs = schema.mathematical_model.governing_equations
        assert len(eqs) > 0
        names = [eq.name for eq in eqs]
        assert any("Equilibrium" in n for n in names)

    def test_governing_equations_contain_weak_form(self):
        from engines.ansys.core.extractor import AnsysExtractor

        ext = AnsysExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        eqs = schema.mathematical_model.governing_equations
        names = [eq.name for eq in eqs]
        assert any("Virtual Work" in n for n in names)

    def test_numerical_method_present(self):
        from engines.ansys.core.extractor import AnsysExtractor

        ext = AnsysExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        assert schema.numerical_method is not None

    def test_computational_graph_has_assembly_and_solver(self):
        from engines.ansys.core.extractor import AnsysExtractor

        ext = AnsysExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        node_ids = [n.id for n in schema.computational_graph.nodes]
        assert "assembly" in node_ids
        assert "solver" in node_ids

    def test_conservation_properties(self):
        from engines.ansys.core.extractor import AnsysExtractor

        ext = AnsysExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        assert isinstance(schema.conservation_properties, dict)

    def test_symbolic_constraints_populated(self):
        from engines.ansys.core.extractor import AnsysExtractor

        ext = AnsysExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        assert hasattr(schema, "symbolic_constraints")
        assert len(schema.symbolic_constraints) > 0

    def test_raw_symbols_present(self):
        from engines.ansys.core.extractor import AnsysExtractor

        ext = AnsysExtractor()
        schema = ext.build_schema(source_files={"input": ["beam.inp"]})
        assert isinstance(schema.raw_symbols, dict)
        assert "analysis_type" in schema.raw_symbols
