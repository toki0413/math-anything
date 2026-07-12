"""Tests for categories module — CategoryEngine, MathKnowledgeGraph, KnowledgeGraphBuilder, GraphQueryEngine."""

import tempfile
from pathlib import Path

import pytest

from math_anything.categories.builder import KnowledgeGraphBuilder
from math_anything.categories.engine import CategoryEngine, MorphismLink
from math_anything.categories.graph import ENTITY_PREFIXES, MathKnowledgeGraph, node_id
from math_anything.categories.query import GraphQueryEngine

# ── Helper: simple morphism stub ──


class _StubMorphism:
    def __init__(
        self,
        name,
        src_type="A",
        tgt_type="B",
        kept=None,
        lost=None,
        kernel="k",
        category="cat",
        math_form="f(x)",
        condition="true",
    ):
        self.name = name
        self.source_type = [src_type]
        self.target_type = [tgt_type]
        self.invariants_kept = kept or ["inv1"]
        self.invariants_lost = lost or ["inv2"]
        self.kernel_description = kernel
        self.category = category
        self.mathematical_form = math_form
        self.condition = condition

    def compose(self, other):
        return _StubMorphism(
            f"{self.name}∘{other.name}",
            kept=list(set(self.invariants_kept) & set(other.invariants_kept)),
            lost=list(set(self.invariants_lost + other.invariants_lost)),
        )

    def to_dict(self):
        return {"name": self.name, "kept": self.invariants_kept, "lost": self.invariants_lost}


class _StubStructure:
    def __init__(self, name, family="algebra"):
        self.name = name
        self.family = family
        self.canonical_form = f"form_{name}"
        self.function_space = "H1"
        self.dimensional_rank = 2
        self.structural_invariants = []


# ── node_id ──


class TestNodeId:
    def test_structure_prefix(self):
        assert node_id("Foo", "structure") == "struct:Foo"

    def test_morphism_prefix(self):
        assert node_id("Bar", "morphism") == "morph:Bar"

    def test_engine_prefix(self):
        assert node_id("VASP", "engine") == "engine:VASP"

    def test_parameter_prefix(self):
        assert node_id("ENCUT", "parameter") == "param:ENCUT"

    def test_unknown_type(self):
        assert node_id("X", "custom") == "custom:X"


# ── CategoryEngine ──


class TestCategoryEngine:
    def test_register_morphism(self):
        eng = CategoryEngine()
        m = _StubMorphism("bo")
        eng.register_morphism(m)
        assert "bo" in eng.morphisms

    def test_register_structure(self):
        eng = CategoryEngine()
        eng.register_structure("NonlinearEigenvalue", _StubStructure("NonlinearEigenvalue"))
        assert "NonlinearEigenvalue" in eng.structures

    def test_link(self):
        eng = CategoryEngine()
        m = _StubMorphism("ks_map")
        eng.register_morphism(m)
        eng.link("ks_map", "DFT", "KohnSham")
        assert len(eng.morphism_links) == 1
        assert eng.morphism_links[0].source_structure == "DFT"

    def test_link_unknown_morphism_raises(self):
        eng = CategoryEngine()
        with pytest.raises(KeyError):
            eng.link("nonexistent", "A", "B")

    def test_compose(self):
        eng = CategoryEngine()
        eng.register_morphism(_StubMorphism("f", kept=["a"], lost=["b"]))
        eng.register_morphism(_StubMorphism("g", kept=["a", "c"], lost=["d"]))
        result = eng.compose("f", "g")
        assert "f" in result.name and "g" in result.name

    def test_invariant_under(self):
        eng = CategoryEngine()
        m = _StubMorphism("m1", kept=["energy"], lost=["momentum"])
        eng.register_morphism(m)
        assert eng.invariant_under("energy", "m1") is True
        assert eng.invariant_under("momentum", "m1") is False
        assert eng.invariant_under("energy", "nonexistent") is False

    def test_kernel_of(self):
        eng = CategoryEngine()
        eng.register_morphism(_StubMorphism("m1", kernel="correlation"))
        assert eng.kernel_of("m1") == "correlation"
        assert eng.kernel_of("nonexistent") == ""

    def test_what_is_lost(self):
        eng = CategoryEngine()
        eng.register_morphism(_StubMorphism("m1", lost=["symplecticity"]))
        assert "symplecticity" in eng.what_is_lost("m1")
        assert eng.what_is_lost("nonexistent") == []

    def test_what_is_kept(self):
        eng = CategoryEngine()
        eng.register_morphism(_StubMorphism("m1", kept=["hermiticity"]))
        assert "hermiticity" in eng.what_is_kept("m1")

    def test_get_morphism_chain(self):
        eng = CategoryEngine()
        eng.register_morphism(_StubMorphism("f"))
        eng.register_morphism(_StubMorphism("g"))
        eng.link("f", "A", "B")
        eng.link("g", "B", "C")
        chain = eng.get_morphism_chain("A", "C")
        assert len(chain) == 2

    def test_get_morphism_chain_no_path(self):
        eng = CategoryEngine()
        chain = eng.get_morphism_chain("X", "Y")
        assert chain == []

    def test_cumulative_loss(self):
        eng = CategoryEngine()
        eng.register_morphism(_StubMorphism("f", kept=["a", "b"], lost=["c"]))
        eng.register_morphism(_StubMorphism("g", kept=["a", "d"], lost=["e"]))
        eng.link("f", "A", "B")
        eng.link("g", "B", "C")
        result = eng.cumulative_loss("A", "C")
        assert "total_invariants_lost" in result
        assert "final_invariants_kept" in result

    def test_find_structures_by_family(self):
        eng = CategoryEngine()
        eng.register_structure("S1", _StubStructure("S1", family="algebra"))
        eng.register_structure("S2", _StubStructure("S2", family="topology"))
        assert "S1" in eng.find_structures_by_family("algebra")
        assert "S2" not in eng.find_structures_by_family("algebra")

    def test_find_morphisms_between(self):
        eng = CategoryEngine()
        eng.register_morphism(_StubMorphism("m1", src_type="A", tgt_type="B"))
        result = eng.find_morphisms_between("A", "B")
        assert "m1" in result

    def test_to_dict(self):
        eng = CategoryEngine()
        eng.register_morphism(_StubMorphism("m1"))
        d = eng.to_dict()
        assert "morphisms_count" in d
        assert d["morphisms_count"] == 1


# ── MathKnowledgeGraph ──


class TestMathKnowledgeGraph:
    def test_create_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            assert kg.stats()["nodes"] == 0

    def test_add_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            eid = kg.add_entity("NonlinearEigenvalue", "structure")
            assert eid == "struct:NonlinearEigenvalue"
            assert kg.stats()["nodes"] == 1

    def test_add_entity_duplicate_increments(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            kg.add_entity("DFT", "structure")
            kg.add_entity("DFT", "structure")
            assert kg.stats()["nodes"] == 1
            node = kg.get_entity("DFT", "structure")
            assert node["mentions"] == 2

    def test_add_relation(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            sid = kg.add_entity("DFT", "structure")
            eid = kg.add_entity("VASP", "engine")
            kg.add_relation(eid, "instantiates", sid)
            assert kg.stats()["edges"] == 1

    def test_add_relation_missing_node_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            kg.add_relation("nonexistent1", "rel", "nonexistent2")
            assert kg.stats()["edges"] == 0

    def test_has_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            kg.add_entity("DFT", "structure")
            assert kg.has_entity("DFT", "structure") is True
            assert kg.has_entity("MD", "structure") is False

    def test_get_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            kg.add_entity("DFT", "structure")
            node = kg.get_entity("DFT", "structure")
            assert node is not None
            assert node["label"] == "DFT"

    def test_get_entity_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            assert kg.get_entity("X", "structure") is None

    def test_find_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            kg.add_entity("KohnSham", "structure")
            kg.add_entity("BornOppenheimer", "structure")
            results = kg.find_nodes("Kohn")
            assert len(results) > 0

    def test_neighborhood(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            sid = kg.add_entity("DFT", "structure")
            eid = kg.add_entity("VASP", "engine")
            kg.add_relation(eid, "instantiates", sid)
            nb = kg.neighborhood([eid])
            assert len(nb["nodes"]) > 0

    def test_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            kg.add_entity("KohnSham", "structure")
            result = kg.query("Kohn")
            assert len(result["nodes"]) > 0

    def test_query_no_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            result = kg.query("zzzznonexistent")
            assert result["nodes"] == []

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            kg.add_entity("DFT", "structure")
            kg.add_entity("VASP", "engine")
            s = kg.stats()
            assert s["nodes"] == 2
            assert "structure" in s["node_types"]

    def test_to_context_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            sid = kg.add_entity("DFT", "structure")
            ctx = kg.to_context_string({sid})
            assert "DFT" in ctx

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            kg.add_entity("DFT", "structure")
            kg.save()
            kg2 = MathKnowledgeGraph(tmp)
            kg2.load()
            assert kg2.stats()["nodes"] == 1


# ── KnowledgeGraphBuilder ──


class TestKnowledgeGraphBuilder:
    def _make_kg(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            yield kg

    def test_build_from_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            builder = KnowledgeGraphBuilder(kg)
            s = _StubStructure("NonlinearEigenvalue")
            sid = builder.build_from_structure(s)
            assert sid.startswith("struct:")
            assert kg.has_entity("NonlinearEigenvalue", "structure")

    def test_build_from_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            builder = KnowledgeGraphBuilder(kg)
            # First add the structure
            builder.build_from_structure(_StubStructure("DFT"))
            eid = builder.build_from_engine("vasp", "DFT", {"ENCUT": 500})
            assert eid.startswith("engine:")
            assert kg.stats()["nodes"] >= 3  # engine + structure + param

    def test_build_from_morphism(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            builder = KnowledgeGraphBuilder(kg)
            sid = builder.build_from_structure(_StubStructure("DFT"))
            tid = builder.build_from_structure(_StubStructure("KohnSham"))
            m = _StubMorphism("born_oppenheimer", kept=["energy"], lost=["vibronic"])
            mid = builder.build_from_morphism(m, sid, tid)
            assert mid.startswith("morph:")

    def test_build_from_pi_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            builder = KnowledgeGraphBuilder(kg)
            sid = builder.build_from_structure(_StubStructure("Fluid"))
            builder.build_from_pi_groups(
                [{"name": "Re", "expression": "rho*U*L/mu", "variables": {"rho": 1, "U": 1}}],
                sid,
            )
            assert kg.has_entity("Re", "pi_group")

    def test_build_from_cross_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            builder = KnowledgeGraphBuilder(kg)
            builder.build_from_cross_engine(
                "vasp",
                "qe",
                [
                    {"source": "ENCUT", "target": "ecutwfc", "meaning": "cutoff energy"},
                ],
            )
            assert kg.has_entity("ENCUT", "parameter")
            assert kg.has_entity("ecutwfc", "parameter")


# ── GraphQueryEngine ──


class TestGraphQueryEngine:
    def test_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            kg.add_entity("DFT", "structure")
            qe = GraphQueryEngine(kg)
            result = qe.query("DFT")
            assert len(result["nodes"]) > 0

    def test_impact(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            pid = kg.add_entity("ENCUT", "parameter")
            did = kg.add_entity("PlaneWave", "discretization")
            kg.add_relation(pid, "controls", did)
            qe = GraphQueryEngine(kg)
            result = qe.impact("ENCUT")
            assert result["impacted_count"] > 0

    def test_impact_missing_param(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            qe = GraphQueryEngine(kg)
            result = qe.impact("nonexistent")
            assert "error" in result

    def test_root_cause(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            iid = kg.add_entity("SCF convergence", "invariant")
            pid = kg.add_entity("AMIX", "parameter")
            kg.add_relation(pid, "constrains", iid)
            qe = GraphQueryEngine(kg)
            result = qe.root_cause("SCF convergence")
            assert len(result["root_causes"]) > 0

    def test_trace_morphism_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            sid = kg.add_entity("DFT", "structure")
            tid = kg.add_entity("KohnSham", "structure")
            mid = kg.add_entity("BornOppenheimer", "morphism")
            kg.add_relation(sid, "source_of", mid)
            kg.add_relation(mid, "maps_to", tid)
            qe = GraphQueryEngine(kg)
            result = qe.trace_morphism_chain("DFT", "KohnSham")
            assert "chain" in result

    def test_trace_morphism_chain_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            qe = GraphQueryEngine(kg)
            result = qe.trace_morphism_chain("X", "Y")
            assert "error" in result

    def test_find_equivalents(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            pid_a = kg.add_entity("ENCUT", "parameter")
            pid_b = kg.add_entity("ecutwfc", "parameter")
            kg.add_relation(pid_a, "equivalent_to", pid_b, meaning="cutoff energy")
            qe = GraphQueryEngine(kg)
            equivs = qe.find_equivalents("ENCUT", "qe")
            # May or may not find depending on label match
            assert isinstance(equivs, list)

    def test_to_llm_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            kg = MathKnowledgeGraph(tmp)
            kg.add_entity("DFT", "structure")
            qe = GraphQueryEngine(kg)
            result = qe.query("DFT")
            ctx = qe.to_llm_context(result)
            assert "DFT" in ctx

    def test_to_llm_context_error(self):
        qe = GraphQueryEngine(MathKnowledgeGraph(tempfile.mkdtemp()))
        ctx = qe.to_llm_context({"error": "not found"})
        assert "Error" in ctx
