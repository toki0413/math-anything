"""Unit tests for the exception hierarchy.

Tests that each exception has code/detail/suggestion attributes,
the inheritance hierarchy is correct, and string representation works.
"""

import pytest

from math_anything.exceptions import (
    ConfigurationError,
    ConservationLawViolationError,
    ConstraintViolationError,
    ContradictoryInvariantsError,
    DimensionalAnalysisError,
    DimensionalInconsistencyError,
    EngineNotFoundError,
    EntityNotFoundError,
    ExtractionError,
    ExtractionFileNotFoundError,
    GraphQueryError,
    InvalidStructurePropertyError,
    KnowledgeGraphError,
    MathAnythingError,
    MissingAPIKeyError,
    MissingInvariantError,
    MorphismCompositionError,
    MorphismError,
    MorphismNotApplicableError,
    MorphismNotFoundError,
    ParseError,
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
    StructureError,
    UnknownUnitError,
    UnsupportedEngineError,
    ValidationError,
)

# ── Base class ──


class TestMathAnythingError:
    def test_has_code_attribute(self):
        err = MathAnythingError()
        assert hasattr(err, "code")
        assert err.code == "UNKNOWN_ERROR"

    def test_has_detail_attribute(self):
        err = MathAnythingError(detail="something broke")
        assert err.detail == "something broke"

    def test_default_detail(self):
        err = MathAnythingError()
        assert err.detail == "An unexpected error occurred."

    def test_has_suggestion_attribute(self):
        err = MathAnythingError(suggestion="try again")
        assert err.suggestion == "try again"

    def test_default_suggestion_empty(self):
        err = MathAnythingError()
        assert err.suggestion == ""

    def test_str_without_suggestion(self):
        err = MathAnythingError(detail="oops")
        s = str(err)
        assert "[UNKNOWN_ERROR]" in s
        assert "oops" in s
        assert "Suggestion" not in s

    def test_str_with_suggestion(self):
        err = MathAnythingError(detail="oops", suggestion="fix it")
        s = str(err)
        assert "[UNKNOWN_ERROR]" in s
        assert "oops" in s
        assert "Suggestion: fix it" in s

    def test_is_exception_subclass(self):
        assert issubclass(MathAnythingError, Exception)

    def test_context_kwargs(self):
        err = MathAnythingError(detail="x", engine="vasp")
        assert err.context == {"engine": "vasp"}


# ── Structure errors ──


class TestStructureErrors:
    def test_structure_error_code(self):
        err = StructureError()
        assert err.code == "STRUCTURE_ERROR"

    def test_structure_error_inherits(self):
        assert issubclass(StructureError, MathAnythingError)

    def test_invalid_structure_property_code(self):
        err = InvalidStructurePropertyError()
        assert err.code == "INVALID_STRUCTURE_PROPERTY"

    def test_invalid_structure_property_inherits(self):
        assert issubclass(InvalidStructurePropertyError, StructureError)

    def test_missing_invariant_code(self):
        err = MissingInvariantError()
        assert err.code == "MISSING_INVARIANT"

    def test_missing_invariant_inherits(self):
        assert issubclass(MissingInvariantError, StructureError)

    def test_contradictory_invariants_code(self):
        err = ContradictoryInvariantsError()
        assert err.code == "CONTRADICTORY_INVARIANTS"

    def test_contradictory_invariants_inherits(self):
        assert issubclass(ContradictoryInvariantsError, StructureError)


# ── Morphism errors ──


class TestMorphismErrors:
    def test_morphism_error_code(self):
        err = MorphismError()
        assert err.code == "MORPHISM_ERROR"

    def test_morphism_error_inherits(self):
        assert issubclass(MorphismError, MathAnythingError)

    def test_morphism_not_applicable_code(self):
        err = MorphismNotApplicableError()
        assert err.code == "MORPHISM_NOT_APPLICABLE"

    def test_morphism_not_applicable_inherits(self):
        assert issubclass(MorphismNotApplicableError, MorphismError)

    def test_morphism_composition_code(self):
        err = MorphismCompositionError()
        assert err.code == "MORPHISM_COMPOSITION_ERROR"

    def test_morphism_composition_inherits(self):
        assert issubclass(MorphismCompositionError, MorphismError)

    def test_morphism_not_found_code(self):
        err = MorphismNotFoundError()
        assert err.code == "MORPHISM_NOT_FOUND"

    def test_morphism_not_found_inherits(self):
        assert issubclass(MorphismNotFoundError, MorphismError)


# ── Extraction errors ──


class TestExtractionErrors:
    def test_extraction_error_code(self):
        err = ExtractionError()
        assert err.code == "EXTRACTION_ERROR"

    def test_extraction_error_inherits(self):
        assert issubclass(ExtractionError, MathAnythingError)

    def test_engine_not_found_code(self):
        err = EngineNotFoundError()
        assert err.code == "ENGINE_NOT_FOUND"

    def test_engine_not_found_inherits(self):
        assert issubclass(EngineNotFoundError, ExtractionError)

    def test_parse_error_code(self):
        err = ParseError()
        assert err.code == "PARSE_ERROR"

    def test_parse_error_inherits(self):
        assert issubclass(ParseError, ExtractionError)

    def test_unsupported_engine_code(self):
        err = UnsupportedEngineError()
        assert err.code == "UNSUPPORTED_ENGINE"

    def test_unsupported_engine_inherits(self):
        assert issubclass(UnsupportedEngineError, ExtractionError)

    def test_file_not_found_code(self):
        err = ExtractionFileNotFoundError()
        assert err.code == "EXTRACTION_FILE_NOT_FOUND"

    def test_file_not_found_inherits(self):
        assert issubclass(ExtractionFileNotFoundError, ExtractionError)

    def test_file_not_found_is_extraction_error(self):
        """ExtractionFileNotFoundError should be catchable as ExtractionError."""
        with pytest.raises(ExtractionError):
            raise ExtractionFileNotFoundError("missing file")


# ── Validation errors ──


class TestValidationErrors:
    def test_validation_error_code(self):
        err = ValidationError()
        assert err.code == "VALIDATION_ERROR"

    def test_validation_error_inherits(self):
        assert issubclass(ValidationError, MathAnythingError)

    def test_dimensional_inconsistency_code(self):
        err = DimensionalInconsistencyError()
        assert err.code == "DIMENSIONAL_INCONSISTENCY"

    def test_dimensional_inconsistency_inherits(self):
        assert issubclass(DimensionalInconsistencyError, ValidationError)

    def test_constraint_violation_code(self):
        err = ConstraintViolationError()
        assert err.code == "CONSTRAINT_VIOLATION"

    def test_constraint_violation_inherits(self):
        assert issubclass(ConstraintViolationError, ValidationError)

    def test_conservation_law_violation_code(self):
        err = ConservationLawViolationError()
        assert err.code == "CONSERVATION_LAW_VIOLATED"

    def test_conservation_law_violation_inherits(self):
        assert issubclass(ConservationLawViolationError, ValidationError)


# ── Knowledge graph errors ──


class TestKnowledgeGraphErrors:
    def test_kg_error_code(self):
        err = KnowledgeGraphError()
        assert err.code == "KG_ERROR"

    def test_kg_error_inherits(self):
        assert issubclass(KnowledgeGraphError, MathAnythingError)

    def test_entity_not_found_code(self):
        err = EntityNotFoundError()
        assert err.code == "ENTITY_NOT_FOUND"

    def test_entity_not_found_inherits(self):
        assert issubclass(EntityNotFoundError, KnowledgeGraphError)

    def test_graph_query_error_code(self):
        err = GraphQueryError()
        assert err.code == "GRAPH_QUERY_ERROR"

    def test_graph_query_error_inherits(self):
        assert issubclass(GraphQueryError, KnowledgeGraphError)


# ── Dimensional analysis errors ──


class TestDimensionalAnalysisErrors:
    def test_dimensional_analysis_error_code(self):
        err = DimensionalAnalysisError()
        assert err.code == "DIMENSIONAL_ANALYSIS_ERROR"

    def test_dimensional_analysis_error_inherits(self):
        assert issubclass(DimensionalAnalysisError, MathAnythingError)

    def test_unknown_unit_code(self):
        err = UnknownUnitError()
        assert err.code == "UNKNOWN_UNIT"

    def test_unknown_unit_inherits(self):
        assert issubclass(UnknownUnitError, DimensionalAnalysisError)


# ── Configuration errors ──


class TestConfigurationErrors:
    def test_configuration_error_code(self):
        err = ConfigurationError()
        assert err.code == "CONFIG_ERROR"

    def test_configuration_error_inherits(self):
        assert issubclass(ConfigurationError, MathAnythingError)

    def test_missing_api_key_code(self):
        err = MissingAPIKeyError()
        assert err.code == "MISSING_API_KEY"

    def test_missing_api_key_inherits(self):
        assert issubclass(MissingAPIKeyError, ConfigurationError)


# ── Plugin errors ──


class TestPluginErrors:
    def test_plugin_error_code(self):
        err = PluginError()
        assert err.code == "PLUGIN_ERROR"

    def test_plugin_error_inherits(self):
        assert issubclass(PluginError, MathAnythingError)

    def test_plugin_not_found_code(self):
        err = PluginNotFoundError()
        assert err.code == "PLUGIN_NOT_FOUND"

    def test_plugin_not_found_inherits(self):
        assert issubclass(PluginNotFoundError, PluginError)

    def test_plugin_load_error_code(self):
        err = PluginLoadError()
        assert err.code == "PLUGIN_LOAD_ERROR"

    def test_plugin_load_error_inherits(self):
        assert issubclass(PluginLoadError, PluginError)


# ── Cross-hierarchy catch ──


class TestCrossHierarchyCatch:
    def test_all_catchable_as_base(self):
        """Every custom exception should be catchable as MathAnythingError."""
        subclasses = [
            StructureError,
            InvalidStructurePropertyError,
            MissingInvariantError,
            ContradictoryInvariantsError,
            MorphismError,
            MorphismNotApplicableError,
            MorphismCompositionError,
            MorphismNotFoundError,
            ExtractionError,
            EngineNotFoundError,
            ParseError,
            UnsupportedEngineError,
            ExtractionFileNotFoundError,
            ValidationError,
            DimensionalInconsistencyError,
            ConstraintViolationError,
            ConservationLawViolationError,
            KnowledgeGraphError,
            EntityNotFoundError,
            GraphQueryError,
            DimensionalAnalysisError,
            UnknownUnitError,
            ConfigurationError,
            MissingAPIKeyError,
            PluginError,
            PluginNotFoundError,
            PluginLoadError,
        ]
        for cls in subclasses:
            assert issubclass(cls, MathAnythingError), f"{cls.__name__} is not a subclass of MathAnythingError"

    def test_catch_extraction_subtree(self):
        """All extraction-related errors should be catchable as ExtractionError."""
        extraction_subclasses = [
            EngineNotFoundError,
            ParseError,
            UnsupportedEngineError,
            ExtractionFileNotFoundError,
        ]
        for cls in extraction_subclasses:
            assert issubclass(cls, ExtractionError), f"{cls.__name__} is not a subclass of ExtractionError"

    def test_catch_validation_subtree(self):
        """All validation-related errors should be catchable as ValidationError."""
        validation_subclasses = [
            DimensionalInconsistencyError,
            ConstraintViolationError,
            ConservationLawViolationError,
        ]
        for cls in validation_subclasses:
            assert issubclass(cls, ValidationError), f"{cls.__name__} is not a subclass of ValidationError"
