"""Extract Agent - Extract mathematical structures from simulation inputs."""

from typing import Any, Dict

from .base import AgentResult, AgentStatus, BaseAgent


class ExtractAgent(BaseAgent):
    """Extract mathematical structures from simulation engine inputs.

    Wraps the existing MathAnything extraction API into the agent framework.

    Input context:
        - 'engine': Engine name (required)
        - 'params': Engine parameters dict (optional)
        - 'filepath': Path to input file (optional)

    Output data:
        - 'schema': Extracted mathematical schema
        - 'constraints': List of symbolic constraints
        - 'approximations': List of approximations
    """

    @property
    def name(self) -> str:
        return "extract"

    @property
    def description(self) -> str:
        return "Extract mathematical structures from simulation inputs"

    def validate_inputs(self, context: Dict[str, Any]) -> list:
        missing = []
        if "engine" not in context:
            missing.append("engine")
        if "params" not in context and "filepath" not in context:
            missing.append("params or filepath")
        return missing

    def run(self, context: Dict[str, Any]) -> AgentResult:
        from math_anything import MathAnything

        engine = context["engine"]
        ma = MathAnything()

        if "filepath" in context:
            result = ma.extract_file(engine, context["filepath"])
        else:
            result = ma.extract(engine, context.get("params", {}))

        schema = result.schema if hasattr(result, "schema") else {}
        if isinstance(schema, dict):
            pass
        elif hasattr(schema, "to_dict"):
            schema = schema.to_dict()

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={
                "schema": schema,
                "constraints": (
                    schema.get("constraints", []) if isinstance(schema, dict) else []
                ),
                "approximations": (
                    schema.get("approximations", []) if isinstance(schema, dict) else []
                ),
            },
            metadata={"engine": engine},
        )
