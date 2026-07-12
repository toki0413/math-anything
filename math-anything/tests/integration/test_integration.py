"""Integration tests for bridge, categories, dimensional, and plugins."""

import math
import sys
from pathlib import Path

import pytest

# Ensure engines path
_engines = Path(__file__).parent.parent.parent / "engines"
if str(_engines) not in sys.path:
    sys.path.insert(0, str(_engines))


class TestPluginRegistry:
    def test_plugin_registry_exists(self):
        from math_anything.plugin import PluginRegistry

        registry = PluginRegistry()
        assert registry is not None

    def test_builtin_engines_registered(self):
        from math_anything.plugin import BUILTIN_ENGINES

        assert "vasp" in BUILTIN_ENGINES
        assert "lammps" in BUILTIN_ENGINES
        assert len(BUILTIN_ENGINES) >= 19


class TestCategoriesGraph:
    def test_add_entity(self, temp_kg):
        eid = temp_kg.add_entity("test_node", "structure", confidence=0.9)
        assert "struct:test_node" == eid

    def test_duplicate_entity(self, temp_kg):
        temp_kg.add_entity("dup", "parameter")
        temp_kg.add_entity("dup", "parameter")
        node = temp_kg.get_entity("dup", "parameter")
        assert node["mentions"] == 2

    def test_add_relation(self, temp_kg):
        a = temp_kg.add_entity("A", "structure")
        b = temp_kg.add_entity("B", "structure")
        temp_kg.add_relation(a, "maps_to", b)
        assert temp_kg.stats()["edges"] == 1

    def test_find_nodes(self, temp_kg):
        temp_kg.add_entity("energy_cutoff", "parameter")
        temp_kg.add_entity("energy", "structure")
        results = temp_kg.find_nodes("energy")
        assert len(results) >= 2

    def test_query(self, temp_kg):
        a = temp_kg.add_entity("ENCUT", "parameter", value="520")
        b = temp_kg.add_entity("basis_truncation", "approximation")
        temp_kg.add_relation(a, "controls", b)
        result = temp_kg.query("ENCUT", depth=1)
        assert len(result["nodes"]) >= 2

    def test_stats(self, temp_kg):
        temp_kg.add_entity("n1", "structure")
        temp_kg.add_entity("n2", "parameter")
        assert temp_kg.stats()["nodes"] == 2


class TestKnowledgeGraphBuilder:
    def test_build_from_structure(self, temp_kg, self_consistent_structure):
        from math_anything.categories.builder import KnowledgeGraphBuilder

        builder = KnowledgeGraphBuilder(temp_kg)
        builder.build_from_structure(self_consistent_structure)
        assert temp_kg.stats()["nodes"] > 3

    def test_build_from_engine(self, temp_kg):
        from math_anything.categories.builder import KnowledgeGraphBuilder

        builder = KnowledgeGraphBuilder(temp_kg)
        builder.build_from_engine("VASP", "NonlinearEigenvalue", {"ENCUT": 520})
        assert temp_kg.has_entity("ENCUT", "parameter")


class TestGraphQueryEngine:
    def test_impact(self, temp_kg):
        from math_anything.categories.query import GraphQueryEngine

        a = temp_kg.add_entity("ENCUT", "parameter")
        b = temp_kg.add_entity("truncation", "approximation")
        temp_kg.add_relation(a, "controls", b)
        qe = GraphQueryEngine(temp_kg)
        result = qe.impact("ENCUT", max_depth=2)
        assert result["impacted_count"] == 1


class TestBridgeIntegration:
    def test_vasp_bridge(self):
        from math_anything.bridge import StructureBridge

        b = StructureBridge()
        r = b.build_from_vasp({"ENCUT": 520, "SIGMA": 0.05, "ISMEAR": 1})
        assert "Self-Consistent" in r["structure"]["name"]

    def test_cfd_bridge(self):
        from math_anything.bridge import StructureBridge

        b = StructureBridge()
        r = b.build_from_cfd({"engine": "OpenFOAM", "regime": "incompressible"})
        assert "Navier" in r["structure"]["name"]
        assert len(r.get("named_pi_groups", {})) >= 2


class TestDimensionalDeep:
    def test_buckingham_suggest(self, buckingham_engine):
        from math_anything.dimensional.scaling_group import BUILTIN_QUANTITIES

        quantities = [
            BUILTIN_QUANTITIES["density"],
            BUILTIN_QUANTITIES["velocity"],
            BUILTIN_QUANTITIES["length"],
            BUILTIN_QUANTITIES["dynamic_viscosity"],
        ]
        pi_groups = buckingham_engine.compute(quantities)
        suggestions = buckingham_engine.suggest_variations(pi_groups, {"rho": 1000})
        assert len(suggestions) > 0

    def test_qm_analyzer(self):
        from math_anything.dimensional.scaling_group import QMDimensionAnalyzer

        qm = QMDimensionAnalyzer()
        pi_groups = qm.analyze_dft({"ENCUT": 520})
        assert len(pi_groups) >= 1


class TestExceptions:
    def test_exception_code(self):
        from math_anything.exceptions import EngineNotFoundError

        e = EngineNotFoundError(detail="test")
        assert len(e.code) > 0

    def test_exception_str(self):
        from math_anything.exceptions import PluginNotFoundError

        e = PluginNotFoundError(detail="engine x not found")
        assert "engine x" in str(e)


class TestRustBridge:
    def test_fallback_available(self):
        from math_anything.rust_bridge import EMLAccelerator, is_rust_available

        acc = EMLAccelerator()
        # Fallback should always work, regardless of Rust availability
        closure = acc.eml_closure([0.0, 1.0, math.e, math.pi], 2, 50)
        assert len(closure) > 0

    def test_buckingham_fallback(self):
        import numpy as np

        from math_anything.rust_bridge import EMLAccelerator

        acc = EMLAccelerator()
        matrix = np.eye(4)[:, :2]
        result = acc.buckingham_pi(matrix)
        assert isinstance(result, list)
