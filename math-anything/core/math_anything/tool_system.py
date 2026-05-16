"""MathTool type system - type-safe tool registration framework.

Inspired by Claude Code's buildTool pattern. Each tool is a self-contained,
self-describing unit with input validation, permission checks, concurrency
markers, and execution logic.

Design principles:
- Conservative safety defaults (read-only, concurrency-safe by default)
- Pydantic input schemas for automatic JSON Schema generation
- Async-first execution model
- LLM function-calling compatible tool definitions
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O")


class PermissionResult(BaseModel):
    behavior: Literal["allow", "deny", "ask"] = "allow"
    reason: str = ""


class ToolContext(BaseModel):
    session_id: str = ""
    schema_context: dict[str, Any] | None = None
    llm_config: dict[str, Any] | None = None
    engine: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel, Generic[O]):
    success: bool = True
    data: O | None = None
    error: str = ""
    display: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MathTool(Generic[I, O]):
    """A self-contained, type-safe tool definition.

    Attributes:
        name: Unique tool identifier (e.g. 'extract', 'verify')
        description: Human-readable description for LLM tool selection
        input_schema: Pydantic model class for input validation
        is_concurrency_safe: Whether this tool can run concurrently with others
        is_read_only: Whether this tool only reads data (no side effects)
        check_permissions: Permission check before execution
        call: The actual async execution function
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: type[I],
        call: Callable[[I, ToolContext], Awaitable[ToolResult[O]]],
        is_concurrency_safe: Callable[[I], bool] | None = None,
        is_read_only: Callable[[I], bool] | None = None,
        check_permissions: Callable[[I, ToolContext], PermissionResult] | None = None,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.call = call
        self.is_concurrency_safe = is_concurrency_safe or (lambda _: True)
        self.is_read_only = is_read_only or (lambda _: True)
        self.check_permissions = check_permissions or (
            lambda _, __: PermissionResult(behavior="allow")
        )

    def get_json_schema(self) -> dict[str, Any]:
        return self.input_schema.model_json_schema()

    def get_tool_definition_for_llm(self) -> dict[str, Any]:
        schema = self.get_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    async def safe_call(
        self, raw_input: dict[str, Any], ctx: ToolContext
    ) -> ToolResult[O]:
        perm = (
            self.check_permissions.__func__(raw_input, ctx)
            if hasattr(self.check_permissions, "__func__")
            else self.check_permissions(raw_input, ctx)
        )
        if isinstance(perm, PermissionResult) and perm.behavior == "deny":
            return ToolResult(success=False, error=f"Permission denied: {perm.reason}")

        try:
            validated = self.input_schema.model_validate(raw_input)
        except Exception as e:
            return ToolResult(success=False, error=f"Input validation failed: {e}")

        try:
            return await self.call(validated, ctx)
        except Exception as e:
            return ToolResult(success=False, error=f"Tool execution failed: {e}")


def build_math_tool(
    name: str,
    description: str,
    input_schema: type[I],
    call: Callable[[I, ToolContext], Awaitable[ToolResult[O]]],
    is_concurrency_safe: Callable[[I], bool] | None = None,
    is_read_only: Callable[[I], bool] | None = None,
    check_permissions: Callable[[I, ToolContext], PermissionResult] | None = None,
) -> MathTool[I, O]:
    """Build a MathTool with conservative safety defaults.

    Defaults:
    - is_concurrency_safe: True (math tools are generally stateless)
    - is_read_only: True (math analysis doesn't modify files)
    - check_permissions: allow (no dangerous operations in math tools)
    """
    return MathTool(
        name=name,
        description=description,
        input_schema=input_schema,
        call=call,
        is_concurrency_safe=is_concurrency_safe,
        is_read_only=is_read_only,
        check_permissions=check_permissions,
    )
