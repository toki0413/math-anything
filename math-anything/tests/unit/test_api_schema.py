"""Unit tests for api_schema.py — OpenAPI 3.0 specification."""

import pytest

from math_anything.api_schema import get_openapi_schema


# ── get_openapi_schema ──

class TestGetOpenAPISchema:
    def test_returns_dict(self):
        schema = get_openapi_schema()
        assert isinstance(schema, dict)

    def test_openapi_version(self):
        schema = get_openapi_schema()
        assert schema["openapi"] == "3.0.3"

    def test_info_block(self):
        schema = get_openapi_schema()
        info = schema["info"]
        assert info["title"] == "Math Anything API"
        assert "version" in info
        assert "description" in info
        assert info["license"]["name"] == "MIT"

    def test_servers_block(self):
        schema = get_openapi_schema()
        servers = schema["servers"]
        assert isinstance(servers, list)
        assert len(servers) >= 1
        assert "url" in servers[0]

    def test_paths_block(self):
        schema = get_openapi_schema()
        paths = schema["paths"]
        assert isinstance(paths, dict)
        assert "/extract" in paths
        assert "/verify" in paths
        assert "/engines" in paths
        assert "/health" in paths
        assert "/metrics" in paths


class TestExtractEndpoint:
    def test_extract_is_post(self):
        schema = get_openapi_schema()
        assert "post" in schema["paths"]["/extract"]

    def test_extract_has_request_body(self):
        schema = get_openapi_schema()
        post = schema["paths"]["/extract"]["post"]
        assert "requestBody" in post
        assert post["requestBody"]["required"] is True

    def test_extract_has_responses(self):
        schema = get_openapi_schema()
        post = schema["paths"]["/extract"]["post"]
        responses = post["responses"]
        assert "200" in responses
        assert "400" in responses
        assert "422" in responses

    def test_extract_has_tags(self):
        schema = get_openapi_schema()
        post = schema["paths"]["/extract"]["post"]
        assert "extraction" in post["tags"]

    def test_extract_example(self):
        schema = get_openapi_schema()
        post = schema["paths"]["/extract"]["post"]
        example = post["requestBody"]["content"]["application/json"]["example"]
        assert example["engine"] == "vasp"
        assert "params" in example


class TestVerifyEndpoint:
    def test_verify_is_post(self):
        schema = get_openapi_schema()
        assert "post" in schema["paths"]["/verify"]

    def test_verify_has_request_body(self):
        schema = get_openapi_schema()
        post = schema["paths"]["/verify"]["post"]
        assert "requestBody" in post

    def test_verify_has_responses(self):
        schema = get_openapi_schema()
        post = schema["paths"]["/verify"]["post"]
        assert "200" in post["responses"]
        assert "400" in post["responses"]

    def test_verify_has_tags(self):
        schema = get_openapi_schema()
        post = schema["paths"]["/verify"]["post"]
        assert "verification" in post["tags"]


class TestEnginesEndpoint:
    def test_engines_is_get(self):
        schema = get_openapi_schema()
        assert "get" in schema["paths"]["/engines"]

    def test_engines_has_response(self):
        schema = get_openapi_schema()
        get = schema["paths"]["/engines"]["get"]
        assert "200" in get["responses"]

    def test_engines_example(self):
        schema = get_openapi_schema()
        get = schema["paths"]["/engines"]["get"]
        example = get["responses"]["200"]["content"]["application/json"]["example"]
        assert "engines" in example
        assert "vasp" in example["engines"]


class TestHealthEndpoint:
    def test_health_is_get(self):
        schema = get_openapi_schema()
        assert "get" in schema["paths"]["/health"]

    def test_health_has_200_and_503(self):
        schema = get_openapi_schema()
        get = schema["paths"]["/health"]["get"]
        assert "200" in get["responses"]
        assert "503" in get["responses"]

    def test_health_example(self):
        schema = get_openapi_schema()
        get = schema["paths"]["/health"]["get"]
        example = get["responses"]["200"]["content"]["application/json"]["example"]
        assert example["status"] == "healthy"
        assert "version" in example
        assert "engines_available" in example


class TestMetricsEndpoint:
    def test_metrics_is_get(self):
        schema = get_openapi_schema()
        assert "get" in schema["paths"]["/metrics"]

    def test_metrics_has_response(self):
        schema = get_openapi_schema()
        get = schema["paths"]["/metrics"]["get"]
        assert "200" in get["responses"]

    def test_metrics_example(self):
        schema = get_openapi_schema()
        get = schema["paths"]["/metrics"]["get"]
        example = get["responses"]["200"]["content"]["application/json"]["example"]
        assert "total_extractions" in example
        assert "total_verifications" in example


# ── Components / schemas ──

class TestComponentsSchemas:
    def test_has_components(self):
        schema = get_openapi_schema()
        assert "components" in schema

    def test_has_schemas(self):
        schema = get_openapi_schema()
        assert "schemas" in schema["components"]

    def test_has_extract_request_schema(self):
        schema = get_openapi_schema()
        schemas = schema["components"]["schemas"]
        assert "ExtractRequest" in schemas
        assert "engine" in schemas["ExtractRequest"]["properties"]
        assert "params" in schemas["ExtractRequest"]["properties"]

    def test_has_extraction_result_schema(self):
        schema = get_openapi_schema()
        schemas = schema["components"]["schemas"]
        assert "ExtractionResult" in schemas

    def test_has_verify_request_schema(self):
        schema = get_openapi_schema()
        schemas = schema["components"]["schemas"]
        assert "VerifyRequest" in schemas

    def test_has_verify_result_schema(self):
        schema = get_openapi_schema()
        schemas = schema["components"]["schemas"]
        assert "VerifyResult" in schemas

    def test_has_constraint_violation_schema(self):
        schema = get_openapi_schema()
        schemas = schema["components"]["schemas"]
        assert "ConstraintViolation" in schemas

    def test_has_engines_response_schema(self):
        schema = get_openapi_schema()
        schemas = schema["components"]["schemas"]
        assert "EnginesResponse" in schemas

    def test_has_health_response_schema(self):
        schema = get_openapi_schema()
        schemas = schema["components"]["schemas"]
        assert "HealthResponse" in schemas

    def test_has_metrics_response_schema(self):
        schema = get_openapi_schema()
        schemas = schema["components"]["schemas"]
        assert "MetricsResponse" in schemas

    def test_has_error_response_schema(self):
        schema = get_openapi_schema()
        schemas = schema["components"]["schemas"]
        assert "ErrorResponse" in schemas
        assert "code" in schemas["ErrorResponse"]["properties"]
        assert "detail" in schemas["ErrorResponse"]["properties"]


class TestSchemaReferences:
    def test_extract_request_ref(self):
        schema = get_openapi_schema()
        ref = schema["paths"]["/extract"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        assert ref == "#/components/schemas/ExtractRequest"

    def test_extract_response_ref(self):
        schema = get_openapi_schema()
        ref = schema["paths"]["/extract"]["post"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        assert ref == "#/components/schemas/ExtractionResult"

    def test_verify_request_ref(self):
        schema = get_openapi_schema()
        ref = schema["paths"]["/verify"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        assert ref == "#/components/schemas/VerifyRequest"


class TestSchemaRequiredFields:
    def test_extract_request_required(self):
        schema = get_openapi_schema()
        req = schema["components"]["schemas"]["ExtractRequest"]["required"]
        assert "engine" in req
        assert "params" in req

    def test_extraction_result_required(self):
        schema = get_openapi_schema()
        req = schema["components"]["schemas"]["ExtractionResult"]["required"]
        assert "engine" in req
        assert "success" in req
        assert "schema" in req

    def test_verify_result_required(self):
        schema = get_openapi_schema()
        req = schema["components"]["schemas"]["VerifyResult"]["required"]
        assert "valid" in req
        assert "violations" in req

    def test_engines_response_required(self):
        schema = get_openapi_schema()
        req = schema["components"]["schemas"]["EnginesResponse"]["required"]
        assert "engines" in req

    def test_error_response_required(self):
        schema = get_openapi_schema()
        req = schema["components"]["schemas"]["ErrorResponse"]["required"]
        assert "code" in req
        assert "detail" in req


class TestSchemaEnumValues:
    def test_extract_request_engine_enum(self):
        schema = get_openapi_schema()
        engine_prop = schema["components"]["schemas"]["ExtractRequest"]["properties"]["engine"]
        assert "enum" in engine_prop
        assert "vasp" in engine_prop["enum"]
        assert "lammps" in engine_prop["enum"]

    def test_health_status_enum(self):
        schema = get_openapi_schema()
        status_prop = schema["components"]["schemas"]["HealthResponse"]["properties"]["status"]
        assert "enum" in status_prop
        assert "healthy" in status_prop["enum"]
        assert "degraded" in status_prop["enum"]
        assert "unhealthy" in status_prop["enum"]

    def test_constraint_violation_severity_enum(self):
        schema = get_openapi_schema()
        sev = schema["components"]["schemas"]["ConstraintViolation"]["properties"]["severity"]
        assert "enum" in sev
        assert "theorem" in sev["enum"]
        assert "consistency" in sev["enum"]
