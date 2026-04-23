"""Math Anything exception hierarchy.

Provides structured exception handling for different error scenarios.
"""

from typing import Any, Dict, List, Optional


class MathAnythingError(Exception):
    """Base exception for all Math Anything errors."""

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "type": self.__class__.__name__,
        }


class UnsupportedEngineError(MathAnythingError):
    """Raised when an unsupported engine is specified."""

    def __init__(self, engine: str, available_engines: List[str], **kwargs):
        message = (
            f"Engine '{engine}' is not supported. "
            f"Available engines: {', '.join(available_engines)}"
        )
        super().__init__(
            message,
            error_code="UNSUPPORTED_ENGINE",
            details={"engine": engine, "available_engines": available_engines},
            **kwargs,
        )
        self.engine = engine
        self.available_engines = available_engines


class FileNotFoundError(MathAnythingError):
    """Raised when input file is not found or inaccessible."""

    def __init__(self, filepath: str, **kwargs):
        message = f"File not found or inaccessible: {filepath}"
        super().__init__(
            message,
            error_code="FILE_NOT_FOUND",
            details={"filepath": filepath},
            **kwargs,
        )
        self.filepath = filepath


class FileAccessError(MathAnythingError):
    """Raised when file access is denied (security restriction)."""

    def __init__(self, filepath: str, reason: str, **kwargs):
        message = f"File access denied: {filepath}. Reason: {reason}"
        super().__init__(
            message,
            error_code="FILE_ACCESS_DENIED",
            details={"filepath": filepath, "reason": reason},
            **kwargs,
        )
        self.filepath = filepath
        self.reason = reason


class ParseError(MathAnythingError):
    """Raised when file parsing fails."""

    def __init__(
        self,
        filepath: str,
        parser: str,
        line_number: Optional[int] = None,
        line_content: Optional[str] = None,
        original_error: Optional[Exception] = None,
        **kwargs,
    ):
        message = f"Failed to parse {filepath} with {parser}"
        if line_number:
            message += f" at line {line_number}"

        details = {
            "filepath": filepath,
            "parser": parser,
        }
        if line_number:
            details["line_number"] = line_number
        if line_content:
            details["line_content"] = line_content
        if original_error:
            details["original_error"] = str(original_error)

        super().__init__(message, error_code="PARSE_ERROR", details=details, **kwargs)
        self.filepath = filepath
        self.parser = parser
        self.line_number = line_number
        self.line_content = line_content
        self.original_error = original_error


class ValidationError(MathAnythingError):
    """Raised when parameter validation fails."""

    def __init__(
        self,
        parameter: str,
        value: Any,
        constraint: str,
        **kwargs,
    ):
        message = f"Validation failed for '{parameter}': {value} violates {constraint}"
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            details={
                "parameter": parameter,
                "value": str(value),
                "constraint": constraint,
            },
            **kwargs,
        )
        self.parameter = parameter
        self.value = value
        self.constraint = constraint


class TierAnalysisError(MathAnythingError):
    """Raised when tiered analysis fails."""

    def __init__(
        self,
        tier: Any,
        reason: str,
        file_analysis: Optional[Dict] = None,
        **kwargs,
    ):
        message = f"Tier analysis failed at tier {tier}: {reason}"
        super().__init__(
            message,
            error_code="TIER_ANALYSIS_ERROR",
            details={
                "tier": str(tier),
                "reason": reason,
                "file_analysis": file_analysis,
            },
            **kwargs,
        )
        self.tier = tier
        self.reason = reason


class SecurityError(MathAnythingError):
    """Raised when a security violation is detected."""

    def __init__(self, violation_type: str, details: str, **kwargs):
        message = f"Security violation ({violation_type}): {details}"
        super().__init__(
            message,
            error_code="SECURITY_VIOLATION",
            details={"violation_type": violation_type, "details": details},
            **kwargs,
        )
        self.violation_type = violation_type


class ConfigurationError(MathAnythingError):
    """Raised when configuration is invalid."""

    def __init__(
        self, config_key: str, invalid_value: Any, valid_options: List[str], **kwargs
    ):
        message = (
            f"Invalid configuration for '{config_key}': {invalid_value}. "
            f"Valid options: {', '.join(valid_options)}"
        )
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            details={
                "config_key": config_key,
                "invalid_value": str(invalid_value),
                "valid_options": valid_options,
            },
            **kwargs,
        )
        self.config_key = config_key
