"""MathAgentLoop - LLM-driven iterative agent loop.

Inspired by Claude Code's queryLoop pattern. The loop:
1. Builds messages from user input + context
2. Calls LLM with tool definitions
3. If LLM requests tool use -> execute tool -> inject result -> continue
4. If LLM produces text only -> stream to client -> done

Supports:
- OpenAI and Anthropic API providers
- Streaming text output (text_delta events)
- Tool call execution with progress reporting
- Abort via cancel messages
- Schema context injection
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

from .tool_registry import ToolRegistry
from .tool_system import ToolContext, ToolResult


@dataclass
class AgentEvent(ABC):
    event_type: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.event_type}


@dataclass
class TextDeltaEvent(AgentEvent):
    event_type: str = "text_delta"
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.event_type, "text": self.text}


@dataclass
class ToolCallStartEvent(AgentEvent):
    event_type: str = "tool_call_start"
    tool_name: str = ""
    tool_use_id: str = ""
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.event_type,
            "tool_name": self.tool_name,
            "tool_use_id": self.tool_use_id,
            "args": self.args,
        }


@dataclass
class ToolProgressEvent(AgentEvent):
    event_type: str = "tool_progress"
    tool_use_id: str = ""
    progress: float = 0.0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.event_type,
            "tool_use_id": self.tool_use_id,
            "progress": self.progress,
            "message": self.message,
        }


@dataclass
class ToolResultEvent(AgentEvent):
    event_type: str = "tool_result"
    tool_use_id: str = ""
    tool_name: str = ""
    success: bool = True
    display: str = ""
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.event_type,
            "tool_use_id": self.tool_use_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "display": self.display,
        }
        if self.data is not None:
            if isinstance(self.data, dict):
                payload["data"] = self.data
            elif hasattr(self.data, "model_dump"):
                payload["data"] = self.data.model_dump()
            elif hasattr(self.data, "to_dict"):
                payload["data"] = self.data.to_dict()
            else:
                payload["data"] = str(self.data)
        return payload


@dataclass
class ErrorEvent(AgentEvent):
    event_type: str = "error"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.event_type, "error": self.error}


@dataclass
class DoneEvent(AgentEvent):
    event_type: str = "done"
    reason: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.event_type, "reason": self.reason}


def build_system_prompt(context: dict[str, Any]) -> str:
    parts = []

    parts.append(
        "You are Math-Anything, an AI assistant that helps computational materials "
        "science researchers discover deeper mathematical structures in their "
        "simulation data. Many researchers only perform basic analysis on their "
        "data — you help them unlock deeper insights using rigorous mathematical tools."
    )

    parts.append(
        "\nYou have access to the following tools:\n"
        "- extract: Extract mathematical structures (governing equations, boundary "
        "conditions, constraints) from VASP, LAMMPS, Abaqus, etc.\n"
        "- validate: Validate mathematical consistency of extracted schemas\n"
        "- compare: Compare two mathematical schemas across engines or models\n"
        "- verify: Perform multi-layer formal verification (symbolic, type system, "
        "logic, LLM semantic, Lean4)\n"
        "- proposition: Generate mathematical propositions and proof tasks from schemas\n"
        "- cross_validate: Build a cross-validation matrix (methods × conclusions)\n"
        "- dual_perspective: Analyze from geometric and analytic perspectives\n"
        "- emergence: Analyze phase transitions and emergent phenomena\n"
        "- geometry: Extract differential geometry structures (manifolds, curvature, "
        "fiber bundles)"
    )

    engine = context.get("engine")
    if engine:
        try:
            from .advisor import MathAdvisor

            advisor = MathAdvisor()
            advisory = advisor.advise(engine)
            if advisory:
                parts.append(
                    f"\nEngine-specific analysis guidance ({engine}):\n{advisory}"
                )
        except ImportError:
            pass

    schema_ctx = context.get("schema_context")
    if schema_ctx:
        schema_str = json.dumps(schema_ctx, ensure_ascii=False, indent=2)
        parts.append(f"\nCurrent schema context:\n```json\n{schema_str}\n```")

    parts.append(
        "\nWhen a user asks about a computational science file or mathematical structure:\n"
        "1. First use the extract tool to get the schema\n"
        "2. Then use validate/verify/proposition as appropriate\n"
        "3. Explain results in clear mathematical language\n"
        "4. Use LaTeX notation for equations\n"
        "\nIf the user provides an engine name and parameters, use the extract tool directly.\n"
        "If the user asks about validation or verification, use the appropriate tool after extraction."
    )

    return "\n".join(parts)


class MathAgentLoop:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm_config: dict[str, Any] | None = None,
        firewall: Any = None,
    ):
        self.tools = tool_registry
        self.llm_config = llm_config or {}
        self._firewall = firewall
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    async def run(
        self,
        user_input: str,
        context: dict[str, Any] | None = None,
        on_event: Callable[[AgentEvent], Awaitable[None]] | None = None,
    ) -> None:
        self._cancelled = False
        context = context or {}

        messages: list[dict[str, Any]] = self._build_messages(user_input, context)
        tool_defs = self.tools.get_tool_definitions_for_llm()

        max_iterations = 20
        for _ in range(max_iterations):
            if self._cancelled:
                if on_event:
                    await on_event(DoneEvent(reason="cancelled"))
                return

            try:
                response_chunks = self._call_llm(messages, tool_defs)
            except Exception as e:
                if on_event:
                    await on_event(ErrorEvent(error=str(e)))
                return

            tool_calls: list[dict[str, Any]] = []
            assistant_content = ""

            async for chunk in response_chunks:
                if self._cancelled:
                    break

                if chunk.get("type") == "text_delta":
                    assistant_content += chunk["text"]
                    if on_event:
                        await on_event(TextDeltaEvent(text=chunk["text"]))
                elif chunk.get("type") == "tool_call":
                    tool_calls.append(chunk)

            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            elif tool_calls:
                tool_call_contents = []
                for tc in tool_calls:
                    tool_call_contents.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["args"],
                        }
                    )
                messages.append({"role": "assistant", "content": tool_call_contents})

            if not tool_calls:
                break

            tool_results = await self._execute_tool_calls(tool_calls, context, on_event)
            for tr in tool_results:
                messages.append(tr)

        if on_event:
            await on_event(DoneEvent(reason="completed"))

    def _build_messages(
        self, user_input: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        system_prompt = build_system_prompt(context)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        messages.append({"role": "user", "content": user_input})
        return messages

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            from .firewall import DataFirewall

            if self._firewall and self._firewall.is_enabled:
                messages = self._firewall.sanitize_messages(messages)
        except ImportError:
            pass

        provider = self.llm_config.get("provider", "openai")
        api_key = self.llm_config.get("api_key", "")
        base_url = self.llm_config.get("base_url", "")
        model = self.llm_config.get("model", "")

        if not api_key:
            yield {
                "type": "text_delta",
                "text": "LLM API key not configured. Please set it in Settings.\n\nI can still help you with direct tool calls. What would you like to analyze?",
            }
            return

        if provider == "anthropic":
            async for chunk in self._call_anthropic(
                messages, tool_defs, api_key, model
            ):
                yield chunk
        else:
            async for chunk in self._call_openai(
                messages, tool_defs, api_key, base_url, model
            ):
                yield chunk

    async def _call_openai(
        self,
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
        api_key: str,
        base_url: str,
        model: str,
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            import httpx
        except ImportError:
            yield {
                "type": "text_delta",
                "text": "httpx not installed. Run: pip install httpx",
            }
            return

        url = (
            base_url.rstrip("/") + "/chat/completions"
            if base_url
            else "https://api.openai.com/v1/chat/completions"
        )
        model = model or "gpt-4o"

        payload = {
            "model": model,
            "messages": messages,
            "tools": tool_defs if tool_defs else None,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield {
                        "type": "text_delta",
                        "text": f"LLM API error ({resp.status_code}): {body.decode()[:200]}",
                    }
                    return

                current_tool_calls: dict[int, dict[str, Any]] = {}

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    if delta.get("content"):
                        yield {"type": "text_delta", "text": delta["content"]}

                    tc_deltas = delta.get("tool_calls", [])
                    for tc_delta in tc_deltas:
                        idx = tc_delta.get("index", 0)
                        if idx not in current_tool_calls:
                            current_tool_calls[idx] = {
                                "id": tc_delta.get(
                                    "id", f"call_{uuid.uuid4().hex[:8]}"
                                ),
                                "name": "",
                                "args_str": "",
                            }
                        if tc_delta.get("id"):
                            current_tool_calls[idx]["id"] = tc_delta["id"]
                        if tc_delta.get("function", {}).get("name"):
                            current_tool_calls[idx]["name"] = tc_delta["function"][
                                "name"
                            ]
                        if tc_delta.get("function", {}).get("arguments"):
                            current_tool_calls[idx]["args_str"] += tc_delta["function"][
                                "arguments"
                            ]

                for idx in sorted(current_tool_calls.keys()):
                    tc = current_tool_calls[idx]
                    try:
                        args = json.loads(tc["args_str"]) if tc["args_str"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    yield {
                        "type": "tool_call",
                        "id": tc["id"],
                        "name": tc["name"],
                        "args": args,
                    }

    async def _call_anthropic(
        self,
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
        api_key: str,
        model: str,
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            import httpx
        except ImportError:
            yield {
                "type": "text_delta",
                "text": "httpx not installed. Run: pip install httpx",
            }
            return

        url = "https://api.anthropic.com/v1/messages"
        model = model or "claude-sonnet-4-20250514"

        anthropic_tools = []
        for td in tool_defs:
            func = td.get("function", td)
            anthropic_tools.append(
                {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )

        openai_msgs = [m for m in messages if m["role"] != "system"]
        system_msg = "\n".join(m["content"] for m in messages if m["role"] == "system")

        payload = {
            "model": model,
            "max_tokens": 4096,
            "system": system_msg,
            "messages": openai_msgs,
            "tools": [
                {
                    "type": "custom",
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["input_schema"],
                }
                for t in anthropic_tools
            ],
            "stream": True,
        }
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield {
                        "type": "text_delta",
                        "text": f"Anthropic API error ({resp.status_code}): {body.decode()[:200]}",
                    }
                    return

                current_tool: dict[str, Any] = {}

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield {"type": "text_delta", "text": delta.get("text", "")}
                        elif delta.get("type") == "input_json_delta":
                            current_tool["args_str"] = current_tool.get(
                                "args_str", ""
                            ) + delta.get("partial_json", "")

                    elif event_type == "content_block_start":
                        block = event.get("content_block", {})
                        if block.get("type") == "tool_use":
                            current_tool = {
                                "id": block.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                                "name": block.get("name", ""),
                                "args_str": "",
                            }

                    elif event_type == "content_block_stop":
                        if current_tool.get("name"):
                            try:
                                args = json.loads(current_tool.get("args_str", "{}"))
                            except json.JSONDecodeError:
                                args = {}
                            yield {
                                "type": "tool_call",
                                "id": current_tool["id"],
                                "name": current_tool["name"],
                                "args": args,
                            }
                            current_tool = {}

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        context: dict[str, Any],
        on_event: Callable[[AgentEvent], Awaitable[None]] | None = None,
    ) -> list[dict[str, Any]]:
        results = []
        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            tool_use_id = tc.get("id", "")

            tool = self.tools.get(tool_name)
            if not tool:
                if on_event:
                    await on_event(ErrorEvent(error=f"Unknown tool: {tool_name}"))
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                    }
                )
                continue

            if on_event:
                await on_event(
                    ToolCallStartEvent(
                        tool_name=tool_name,
                        tool_use_id=tool_use_id,
                        args=tool_args,
                    )
                )

            tool_ctx = ToolContext(
                session_id=context.get("session_id", ""),
                schema_context=context.get("schema_context"),
                llm_config=self.llm_config or None,
                engine=tool_args.get("engine", ""),
                params=tool_args.get("params", {}),
            )

            result: ToolResult = await tool.safe_call(tool_args, tool_ctx)

            if on_event:
                await on_event(
                    ToolResultEvent(
                        tool_use_id=tool_use_id,
                        tool_name=tool_name,
                        success=result.success,
                        display=result.display,
                        data=result.data,
                    )
                )

            result_content = result.display or result.error
            if result.data is not None:
                if hasattr(result.data, "model_dump"):
                    result_content = json.dumps(
                        result.data.model_dump(), ensure_ascii=False
                    )
                elif hasattr(result.data, "to_dict"):
                    result_content = json.dumps(
                        result.data.to_dict(), ensure_ascii=False
                    )
                elif isinstance(result.data, dict):
                    result_content = json.dumps(result.data, ensure_ascii=False)

            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": result_content,
                }
            )

        return results
