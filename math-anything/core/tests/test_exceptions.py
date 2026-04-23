"""Tests for exception handling."""

import pytest
from math_anything.exceptions import (
    ConfigurationError,
    FileAccessError,
    FileNotFoundError,
    MathAnythingError,
    ParseError,
    SecurityError,
    TierAnalysisError,
    UnsupportedEngineError,
    ValidationError,
)


class TestMathAnythingError:
    """Test base exception class."""

    def test_basic_exception(self):
        """Test basic exception creation."""
        err = MathAnythingError("Test message")
        assert err.message == "Test message"
        assert err.error_code == "UNKNOWN_ERROR"
        assert str(err) == "Test message"

    def test_exception_with_code(self):
        """Test exception with error code."""
        err = MathAnythingError("Test", error_code="TEST_001")
        assert err.error_code == "TEST_001"

    def test_exception_with_details(self):
        """Test exception with details."""
        err = MathAnythingError("Test", details={"key": "value"})
        assert err.details == {"key": "value"}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        err = MathAnythingError(
            "Test message", error_code="TEST_001", details={"key": "value"}
        )
        d = err.to_dict()
        assert d["message"] == "Test message"
        assert d["error_code"] == "TEST_001"
        assert d["details"] == {"key": "value"}
        assert d["type"] == "MathAnythingError"


class TestUnsupportedEngineError:
    """Test unsupported engine exception."""

    def test_creation(self):
        """Test exception creation."""
        err = UnsupportedEngineError(
            "vasp6", available_engines=["vasp", "lammps", "quantum_espresso"]
        )
        assert err.engine == "vasp6"
        assert "vasp6" in err.message
        assert "vasp" in err.message
        assert err.error_code == "UNSUPPORTED_ENGINE"


class TestFileNotFoundError:
    """Test file not found exception."""

    def test_creation(self):
        """Test exception creation."""
        err = FileNotFoundError("/path/to/file.lmp")
        assert err.filepath == "/path/to/file.lmp"
        assert "/path/to/file.lmp" in err.message
        assert err.error_code == "FILE_NOT_FOUND"


class TestFileAccessError:
    """Test file access error."""

    def test_creation(self):
        """Test exception creation."""
        err = FileAccessError("/etc/passwd", "Security restriction")
        assert err.filepath == "/etc/passwd"
        assert err.reason == "Security restriction"
        assert "Security restriction" in err.message
        assert err.error_code == "FILE_ACCESS_DENIED"


class TestParseError:
    """Test parse error exception."""

    def test_creation(self):
        """Test basic creation."""
        err = ParseError("test.lmp", "LammpsParser")
        assert err.filepath == "test.lmp"
        assert err.parser == "LammpsParser"
        assert err.error_code == "PARSE_ERROR"

    def test_with_line_info(self):
        """Test with line number and content."""
        err = ParseError(
            "test.lmp",
            "LammpsParser",
            line_number=42,
            line_content="fix 1 all nvt temp 300.0 300.0 100.0",
        )
        assert err.line_number == 42
        assert err.line_content == "fix 1 all nvt temp 300.0 300.0 100.0"
        assert "line 42" in err.message

    def test_with_original_error(self):
        """Test with original exception."""
        original = ValueError("Invalid syntax")
        err = ParseError("test.lmp", "LammpsParser", original_error=original)
        assert "Invalid syntax" in err.details["original_error"]


class TestValidationError:
    """Test validation error."""

    def test_creation(self):
        """Test exception creation."""
        err = ValidationError("ENCUT", -100, "must be positive")
        assert err.parameter == "ENCUT"
        assert err.value == -100
        assert err.constraint == "must be positive"
        assert "ENCUT" in err.message
        assert err.error_code == "VALIDATION_ERROR"


class TestTierAnalysisError:
    """Test tier analysis error."""

    def test_creation(self):
        """Test exception creation."""
        err = TierAnalysisError("ADVANCED", "Insufficient data")
        assert err.tier == "ADVANCED"
        assert err.reason == "Insufficient data"
        assert err.error_code == "TIER_ANALYSIS_ERROR"


class TestSecurityError:
    """Test security error."""

    def test_creation(self):
        """Test exception creation."""
        err = SecurityError("PATH_TRAVERSAL", "Detected ../ in path")
        assert err.violation_type == "PATH_TRAVERSAL"
        assert "PATH_TRAVERSAL" in err.message
        assert err.error_code == "SECURITY_VIOLATION"


class TestConfigurationError:
    """Test configuration error."""

    def test_creation(self):
        """Test exception creation."""
        err = ConfigurationError("precision", "ultra", ["low", "normal", "high"])
        assert err.config_key == "precision"
        assert "ultra" in err.message
        assert "low, normal, high" in err.message
        assert err.error_code == "CONFIGURATION_ERROR"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
