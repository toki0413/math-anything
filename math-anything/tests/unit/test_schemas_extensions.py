"""Tests for schema extensions and precision modules."""

import pytest

from math_anything.schemas.extensions import (
    SchemaExtension,
    ExtensionRegistry,
    ExtensionMetadata,
    ExtendedMathSchema,
    MLInteratomicPotentialExtension,
    PINNLossExtension,
    GraphNeuralNetworkExtension,
    get_available_extensions,
    get_extension_documentation,
    validate_with_extensions,
)
from math_anything.schemas.precision import (
    MathematicalStructure,
    VariableDependency,
    DiscretizationScheme,
    SolutionStrategy,
    Approximation,
    MLContext,
    ModelingGuidance,
    MathematicalDecoding,
    PrecisionMetadata,
    EnhancedMathSchema,
    _filter_none,
)
from math_anything.schemas.math_schema import (
    MathSchema,
    SchemaValidator,
    StreamingSchemaBuilder,
    MathSchemaPool,
    GoverningEquation,
    SymbolicConstraint,
    BoundaryCondition,
    MathematicalObject,
    TensorComponent,
    ComputationalGraph,
    ComputationalNode,
    ComputationalEdge,
    Discretization,
    Solver,
    NumericalMethod,
    ConservationProperty,
    ParameterRelationship,
    MetaInfo,
    UpdateMode,
    TensorRank,
)


# ── ExtensionMetadata ──

class TestExtensionMetadata:

    def test_to_dict(self):
        m = ExtensionMetadata(name="test", version="1.0", description="desc")
        d = m.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "1.0"

    def test_defaults(self):
        m = ExtensionMetadata(name="x", version="0.1")
        assert m.dependencies == []
        assert m.compatible_schema_versions == ["1.0.0"]


# ── SchemaExtension subclasses ──

class TestMLInteratomicPotentialExtension:

    def test_name(self):
        e = MLInteratomicPotentialExtension()
        assert e.name == "ml_interatomic_potential"

    def test_get_schema_definition(self):
        e = MLInteratomicPotentialExtension()
        sd = e.get_schema_definition()
        assert isinstance(sd, dict)
        assert "properties" in sd


class TestPINNLossExtension:

    def test_name(self):
        e = PINNLossExtension()
        assert e.name == "pinn_loss_function"

    def test_get_schema_definition(self):
        e = PINNLossExtension()
        sd = e.get_schema_definition()
        assert isinstance(sd, dict)


class TestGraphNeuralNetworkExtension:

    def test_name(self):
        e = GraphNeuralNetworkExtension()
        assert e.name == "graph_neural_network"

    def test_get_schema_definition(self):
        e = GraphNeuralNetworkExtension()
        sd = e.get_schema_definition()
        assert isinstance(sd, dict)


# ── ExtensionRegistry ──

class TestExtensionRegistry:

    def test_get_available_extensions(self):
        exts = get_available_extensions()
        assert isinstance(exts, list)
        assert len(exts) > 0

    def test_get_extension_documentation(self):
        doc = get_extension_documentation("ml_interatomic_potential")
        assert isinstance(doc, str)

    def test_validate_with_extensions(self):
        result = validate_with_extensions({})
        assert isinstance(result, dict)
        assert "valid" in result


# ── ExtendedMathSchema ──

class TestExtendedMathSchema:

    def test_create(self):
        base = MathSchema()
        es = ExtendedMathSchema(base)
        assert es.base_schema is not None

    def test_add_extension_data(self):
        base = MathSchema()
        es = ExtendedMathSchema(base)
        es.extensions["ml_potential"] = {"architecture": "GAP"}
        assert "ml_potential" in es.extensions

    def test_to_dict(self):
        base = MathSchema()
        es = ExtendedMathSchema(base)
        d = es.to_dict()
        assert isinstance(d, dict)

    def test_add_extension_validated(self):
        base = MathSchema()
        es = ExtendedMathSchema(base)
        result = es.add_extension("ml_interatomic_potential", {"architecture": "GAP"})
        assert result is True

    def test_get_extension(self):
        base = MathSchema()
        es = ExtendedMathSchema(base)
        es.extensions["test"] = {"data": 1}
        assert es.get_extension("test") == {"data": 1}
        assert es.get_extension("missing") is None

    def test_remove_extension(self):
        base = MathSchema()
        es = ExtendedMathSchema(base)
        es.extensions["test"] = {"data": 1}
        assert es.remove_extension("test") is True
        assert es.remove_extension("missing") is False

    def test_list_extensions(self):
        base = MathSchema()
        es = ExtendedMathSchema(base)
        es.extensions["a"] = {}
        es.extensions["b"] = {}
        assert len(es.list_extensions()) == 2


# ── Precision dataclasses ──

class TestMathematicalStructure:

    def test_to_dict(self):
        ms = MathematicalStructure(
            problem_type="eigenvalue",
            canonical_form="H*psi = E*psi",
            dimension=3,
        )
        d = ms.to_dict()
        assert d["problem_type"] == "eigenvalue"
        assert d["dimension"] == 3

    def test_to_dict_filters_none(self):
        ms = MathematicalStructure(problem_type="PDE", canonical_form="nabla^2 u = f")
        d = ms.to_dict()
        assert "dimension" not in d


class TestVariableDependency:

    def test_to_dict(self):
        vd = VariableDependency(
            relation="V_eff depends on n(r)",
            depends_on=["n(r)"],
            circular=True,
        )
        d = vd.to_dict()
        assert d["circular"] is True
        assert "n(r)" in d["depends_on"]


class TestDiscretizationScheme:

    def test_to_dict(self):
        ds = DiscretizationScheme(
            method="plane-wave",
            mathematical_meaning="Galerkin projection onto plane-wave subspace",
            basis_type="plane waves",
        )
        d = ds.to_dict()
        assert d["method"] == "plane-wave"


class TestSolutionStrategy:

    def test_to_dict(self):
        ss = SolutionStrategy(
            method="self-consistent iteration",
            mathematical_form="n_{j+1} = F[n_j]",
        )
        d = ss.to_dict()
        assert d["method"] == "self-consistent iteration"


class TestApproximation:

    def test_to_dict(self):
        a = Approximation(
            name="LDA",
            mathematical_form="E_xc[n] = integral epsilon_xc(n) n(r) dr",
            consequence="underestimates band gaps",
        )
        d = a.to_dict()
        assert d["name"] == "LDA"


class TestMLContext:

    def test_to_dict(self):
        ml = MLContext(
            input_mathematical_roles=[{"name": "structure", "role": "geometry"}],
            output_mathematical_definition="total energy",
        )
        d = ml.to_dict()
        assert d["approximation_type"] == "surrogate_model"


class TestModelingGuidance:

    def test_to_dict(self):
        mg = ModelingGuidance(
            problem_type="DFT",
        )
        d = mg.to_dict()
        assert d["problem_type"] == "DFT"


class TestMathematicalDecoding:

    def test_to_dict(self):
        md = MathematicalDecoding(
            core_problem={"type": "eigenvalue"},
        )
        d = md.to_dict()
        assert d["core_problem"]["type"] == "eigenvalue"


class TestPrecisionMetadata:

    def test_to_dict(self):
        pm = PrecisionMetadata(
            extraction_confidence=0.9,
            source="direct_extraction",
        )
        d = pm.to_dict()
        assert d["extraction_confidence"] == 0.9


class TestEnhancedMathSchema:

    def test_create(self):
        es = EnhancedMathSchema()
        assert es.mathematical_structure is None

    def test_to_dict(self):
        es = EnhancedMathSchema()
        d = es.to_dict()
        assert isinstance(d, dict)

    def test_with_structure(self):
        es = EnhancedMathSchema(
            mathematical_structure=MathematicalStructure(
                problem_type="eigenvalue",
                canonical_form="H*psi = E*psi",
            )
        )
        d = es.to_dict()
        assert "mathematical_structure" in d


# ── _filter_none ──

class TestFilterNone:

    def test_removes_none(self):
        assert _filter_none({"a": 1, "b": None}) == {"a": 1}

    def test_removes_empty_dict(self):
        assert _filter_none({"a": {}}) == {}

    def test_removes_empty_list(self):
        assert _filter_none({"a": []}) == {}

    def test_keeps_zero(self):
        assert _filter_none({"a": 0}) == {"a": 0}

    def test_keeps_false(self):
        assert _filter_none({"a": False}) == {"a": False}


# ── StreamingSchemaBuilder ──

class TestStreamingSchemaBuilder:

    def test_build_empty(self):
        b = StreamingSchemaBuilder()
        schema = b.build()
        assert isinstance(schema, MathSchema)

    def test_add_equations(self):
        b = StreamingSchemaBuilder()
        eq = GoverningEquation(id="eq1", type="PDE", name="Heat", mathematical_form="du/dt = alpha nabla^2 u")
        b.add_equations([eq])
        assert b.field_count == 1
        schema = b.build()
        assert len(schema.mathematical_model.governing_equations) == 1

    def test_add_boundary_conditions(self):
        b = StreamingSchemaBuilder()
        mo = MathematicalObject(field="u", tensor_rank=0, tensor_form="scalar")
        bc = BoundaryCondition(
            id="bc1", type="dirichlet", domain={},
            mathematical_object=mo, software_implementation={},
        )
        b.add_boundary_conditions([bc])
        assert b.field_count == 1

    def test_add_conservation(self):
        b = StreamingSchemaBuilder()
        cp = ConservationProperty(quantity="energy", preserved=True)
        b.add_conservation({"energy": cp})
        schema = b.build()
        assert "energy" in schema.conservation_properties


# ── MathSchemaPool ──

class TestMathSchemaPool:

    def test_acquire_and_release(self):
        # Clear pool first
        MathSchemaPool._pool.clear()
        schema = MathSchemaPool.acquire()
        assert isinstance(schema, MathSchema)
        MathSchemaPool.release(schema)
        assert len(MathSchemaPool._pool) == 1

    def test_acquire_from_pool(self):
        MathSchemaPool._pool.clear()
        s1 = MathSchemaPool.acquire()
        MathSchemaPool.release(s1)
        s2 = MathSchemaPool.acquire()
        assert s2 is s1

    def test_release_clears_data(self):
        MathSchemaPool._pool.clear()
        s = MathSchemaPool.acquire()
        s.mathematical_model.governing_equations.append(
            GoverningEquation(id="eq1", type="PDE", name="Test", mathematical_form="x=0")
        )
        MathSchemaPool.release(s)
        s2 = MathSchemaPool.acquire()
        assert len(s2.mathematical_model.governing_equations) == 0


# ── SchemaValidator ──

class TestSchemaValidator:

    def test_valid_schema_dict(self):
        v = SchemaValidator()
        data = MathSchema().to_dict()
        assert v.validate(data) is True

    def test_missing_top_level_key(self):
        v = SchemaValidator()
        data = {"schema_version": "1.0.0"}
        assert v.validate(data) is False
        assert len(v.errors) > 0

    def test_missing_meta_keys(self):
        v = SchemaValidator()
        data = {"schema_version": "1.0.0", "meta": {}, "mathematical_model": {}, "numerical_method": {}, "computational_graph": {}}
        assert v.validate(data) is False

    def test_unsupported_version_warning(self):
        v = SchemaValidator()
        data = MathSchema().to_dict()
        data["schema_version"] = "2.0.0"
        v.validate(data)
        assert len(v.warnings) > 0

    def test_validate_file_not_found(self):
        v = SchemaValidator()
        assert v.validate_file("/nonexistent/path.json") is False


# ── MathSchema serialization ──

class TestMathSchemaSerialization:

    def test_to_json(self):
        s = MathSchema()
        j = s.to_json()
        assert "schema_version" in j

    def test_to_dict_roundtrip(self):
        s = MathSchema()
        s.add_governing_equation(
            GoverningEquation(id="eq1", type="PDE", name="Test", mathematical_form="x=0")
        )
        d = s.to_dict()
        s2 = MathSchema.from_dict(d)
        assert len(s2.mathematical_model.governing_equations) == 1

    def test_add_symbolic_constraint(self):
        s = MathSchema()
        s.add_symbolic_constraint(SymbolicConstraint(expression="dt < dx^2/2D"))
        assert len(s.symbolic_constraints) == 1

    def test_add_parameter_relationship(self):
        s = MathSchema()
        s.add_parameter_relationship(
            ParameterRelationship(name="CFL", expression="dt < dx^2/2D", variables=["dt", "dx", "D"])
        )
        assert len(s.mathematical_model.parameter_relationships) == 1

    def test_add_boundary_condition(self):
        s = MathSchema()
        mo = MathematicalObject(field="u", tensor_rank=0, tensor_form="scalar")
        bc = BoundaryCondition(id="bc1", type="dirichlet", domain={}, mathematical_object=mo, software_implementation={})
        s.add_boundary_condition(bc)
        assert len(s.mathematical_model.boundary_conditions) == 1

    def test_add_numerical_method(self):
        s = MathSchema()
        nm = NumericalMethod(
            discretization=Discretization(time_integrator="Verlet"),
            solver=Solver(algorithm="CG"),
        )
        s.add_numerical_method(nm)
        assert s.numerical_method.solver.algorithm == "CG"

    def test_computational_graph(self):
        g = ComputationalGraph()
        g.add_node(ComputationalNode(id="n1", type="scf", math_semantics={}))
        g.add_edge(ComputationalEdge(from_node="n1", to_node="n2", data_type="density", dependency="sequential"))
        d = g.to_dict()
        assert len(d["nodes"]) == 1
        assert len(d["edges"]) == 1


# ── Enums ──

class TestEnums:

    def test_update_mode(self):
        assert UpdateMode.EXPLICIT_UPDATE.value == "explicit_update"
        assert UpdateMode.IMPLICIT_LOOP.value == "implicit_loop"

    def test_tensor_rank(self):
        assert TensorRank.SCALAR.value == 0
        assert TensorRank.VECTOR.value == 1
        assert TensorRank.MATRIX.value == 2
