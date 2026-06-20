"""OpenAPI 3.0 schema for the Math Anything API.

Auto-generated specification documenting all HTTP endpoints,
request/response models, and error schemas.
"""

from __future__ import annotations

from typing import Any


def get_openapi_schema() -> dict[str, Any]:
    """Return OpenAPI 3.0 specification for the Math Anything API."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Math Anything API",
            "description": (
                "Extract, verify, and compare mathematical structures from computational materials science engines."
            ),
            "version": "3.0.0",
            "license": {"name": "MIT"},
        },
        "servers": [
            {"url": "http://localhost:8000", "description": "Local development"},
        ],
        "paths": {
            "/extract": {
                "post": {
                    "summary": "Extract mathematical structures from simulation parameters",
                    "operationId": "extract",
                    "tags": ["extraction"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ExtractRequest"},
                                "example": {
                                    "engine": "vasp",
                                    "params": {
                                        "ENCUT": 520,
                                        "SIGMA": 0.05,
                                        "EDIFF": 1e-6,
                                        "kpoints": [4, 4, 4],
                                    },
                                    "validate": True,
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Extraction result",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ExtractionResult"},
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid request body",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                                }
                            },
                        },
                        "422": {
                            "description": "Unsupported engine or invalid params",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                                }
                            },
                        },
                    },
                }
            },
            "/verify": {
                "post": {
                    "summary": "Verify extracted mathematical structures",
                    "operationId": "verify",
                    "tags": ["verification"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/VerifyRequest"},
                                "example": {
                                    "engine": "vasp",
                                    "params": {"ENCUT": 520, "EDIFF": 1e-6},
                                    "schema": {
                                        "mathematical_structure": {
                                            "canonical_form": "H[n]ψ = εψ",
                                        }
                                    },
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Verification result",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/VerifyResult"},
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid request body",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                                }
                            },
                        },
                    },
                }
            },
            "/engines": {
                "get": {
                    "summary": "List available engines",
                    "operationId": "listEngines",
                    "tags": ["discovery"],
                    "responses": {
                        "200": {
                            "description": "List of supported engine names",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/EnginesResponse"},
                                    "example": {
                                        "engines": [
                                            "vasp",
                                            "lammps",
                                            "abaqus",
                                            "ansys",
                                            "comsol",
                                            "gromacs",
                                            "multiwfn",
                                        ],
                                    },
                                }
                            },
                        },
                    },
                }
            },
            "/health": {
                "get": {
                    "summary": "Health check endpoint",
                    "operationId": "healthCheck",
                    "tags": ["operations"],
                    "responses": {
                        "200": {
                            "description": "System health status",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthResponse"},
                                    "example": {
                                        "status": "healthy",
                                        "version": "3.0.0",
                                        "rust_acceleration": False,
                                        "engines_available": [
                                            "vasp",
                                            "lammps",
                                            "abaqus",
                                            "ansys",
                                            "comsol",
                                            "gromacs",
                                            "multiwfn",
                                        ],
                                        "uptime_seconds": 3600.0,
                                        "python_version": "3.11.5",
                                    },
                                }
                            },
                        },
                        "503": {
                            "description": "Service unhealthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthResponse"},
                                }
                            },
                        },
                    },
                }
            },
            "/metrics": {
                "get": {
                    "summary": "Prometheus-style metrics",
                    "operationId": "getMetrics",
                    "tags": ["operations"],
                    "responses": {
                        "200": {
                            "description": "Operational metrics",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/MetricsResponse"},
                                    "example": {
                                        "total_extractions": 42,
                                        "total_verifications": 17,
                                        "rust_acceleration_available": False,
                                        "engines_count": 7,
                                        "average_extraction_time_ms": 12.5,
                                    },
                                }
                            },
                        },
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "ExtractRequest": {
                    "type": "object",
                    "required": ["engine", "params"],
                    "properties": {
                        "engine": {
                            "type": "string",
                            "description": "Engine name",
                            "enum": [
                                "vasp",
                                "lammps",
                                "abaqus",
                                "ansys",
                                "comsol",
                                "gromacs",
                                "multiwfn",
                            ],
                        },
                        "params": {
                            "type": "object",
                            "description": "Engine-specific parameters as key-value pairs",
                            "additionalProperties": True,
                        },
                        "validate": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to validate constraints after extraction",
                        },
                    },
                },
                "ExtractionResult": {
                    "type": "object",
                    "required": ["engine", "success", "schema"],
                    "properties": {
                        "engine": {
                            "type": "string",
                            "description": "Engine name used for extraction",
                        },
                        "files": {
                            "type": "object",
                            "description": "Parsed input file contents",
                            "additionalProperties": True,
                        },
                        "schema": {
                            "type": "object",
                            "description": "Enhanced mathematical schema",
                            "additionalProperties": True,
                        },
                        "success": {
                            "type": "boolean",
                            "description": "Whether extraction succeeded",
                        },
                        "errors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Error messages if extraction failed",
                        },
                        "warnings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Warning messages",
                        },
                    },
                },
                "VerifyRequest": {
                    "type": "object",
                    "required": ["engine", "params", "schema"],
                    "properties": {
                        "engine": {
                            "type": "string",
                            "description": "Engine name",
                        },
                        "params": {
                            "type": "object",
                            "description": "Engine parameters used during extraction",
                            "additionalProperties": True,
                        },
                        "schema": {
                            "type": "object",
                            "description": "Previously extracted schema to verify",
                            "additionalProperties": True,
                        },
                    },
                },
                "VerifyResult": {
                    "type": "object",
                    "required": ["valid", "violations"],
                    "properties": {
                        "valid": {
                            "type": "boolean",
                            "description": "Whether all constraints are satisfied",
                        },
                        "violations": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/ConstraintViolation"},
                            "description": "List of constraint violations",
                        },
                        "warnings": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "ConstraintViolation": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "expression": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["theorem", "consistency", "conservation", "heuristic"],
                        },
                        "message": {"type": "string"},
                    },
                },
                "EnginesResponse": {
                    "type": "object",
                    "required": ["engines"],
                    "properties": {
                        "engines": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of supported engine names",
                        },
                    },
                },
                "HealthResponse": {
                    "type": "object",
                    "required": ["status", "version", "engines_available"],
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["healthy", "degraded", "unhealthy"],
                            "description": "Overall system health",
                        },
                        "version": {
                            "type": "string",
                            "description": "Package version",
                        },
                        "rust_acceleration": {
                            "type": "boolean",
                            "description": "Whether Rust acceleration is available",
                        },
                        "engines_available": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of available engine names",
                        },
                        "uptime_seconds": {
                            "type": "number",
                            "description": "Seconds since module import",
                        },
                        "python_version": {
                            "type": "string",
                            "description": "Python runtime version",
                        },
                    },
                },
                "MetricsResponse": {
                    "type": "object",
                    "required": [
                        "total_extractions",
                        "total_verifications",
                        "rust_acceleration_available",
                        "engines_count",
                        "average_extraction_time_ms",
                    ],
                    "properties": {
                        "total_extractions": {
                            "type": "integer",
                            "description": "Cumulative extraction call count",
                        },
                        "total_verifications": {
                            "type": "integer",
                            "description": "Cumulative verification call count",
                        },
                        "rust_acceleration_available": {
                            "type": "boolean",
                            "description": "Whether Rust backend is active",
                        },
                        "engines_count": {
                            "type": "integer",
                            "description": "Number of registered engines",
                        },
                        "average_extraction_time_ms": {
                            "type": "number",
                            "description": "Mean extraction wall-clock time in milliseconds",
                        },
                    },
                },
                "ErrorResponse": {
                    "type": "object",
                    "required": ["code", "detail"],
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Machine-readable error code (UPPER_SNAKE_CASE)",
                        },
                        "detail": {
                            "type": "string",
                            "description": "Human-readable error description",
                        },
                        "suggestion": {
                            "type": "string",
                            "description": "Actionable fix suggestion",
                        },
                    },
                    "example": {
                        "code": "UNSUPPORTED_ENGINE",
                        "detail": "Engine 'foobar' not supported.",
                        "suggestion": "Available: vasp, lammps, abaqus, ansys, comsol, gromacs, multiwfn",
                    },
                },
            },
        },
    }
