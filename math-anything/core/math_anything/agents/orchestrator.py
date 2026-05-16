"""Agent Orchestrator - Coordinate agents into workflows."""

from typing import Any, Callable, Dict, List, Optional

from .base import AgentResult, AgentStatus, BaseAgent
from .compare_agent import CompareAgent
from .extract_agent import ExtractAgent
from .proposition_agent import PropositionAgent
from .validate_agent import ValidateAgent
from .verify_agent import VerifyAgent


class PipelineStep:
    """A single step in a pipeline."""

    def __init__(
        self, agent: BaseAgent, input_mapping: Optional[Dict[str, str]] = None
    ):
        self.agent = agent
        self.input_mapping = input_mapping or {}

    def prepare_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Map context keys for this step."""
        if not self.input_mapping:
            return context
        mapped = dict(context)
        for target_key, source_key in self.input_mapping.items():
            if source_key in context:
                mapped[target_key] = context[source_key]
        return mapped


class AgentOrchestrator:
    """Orchestrate multiple agents into analysis workflows.

    Pre-built pipelines:
    - 'full': extract -> validate -> proposition
    - 'verify': proposition -> verify
    - 'compare': extract_a -> extract_b -> compare
    - 'extract_only': extract -> validate

    Custom pipelines can be built by adding steps manually.

    Example:
        >>> orch = AgentOrchestrator()
        >>> result = orch.run_pipeline("full", engine="vasp", params={"ENCUT": 520})
        >>> print(result["validate"]["data"]["valid"])
    """

    def __init__(self):
        self.agents = {
            "extract": ExtractAgent(),
            "validate": ValidateAgent(),
            "compare": CompareAgent(),
            "proposition": PropositionAgent(),
            "verify": VerifyAgent(),
        }

    def run_pipeline(self, pipeline_name: str, **kwargs) -> Dict[str, Any]:
        """Run a named pipeline.

        Args:
            pipeline_name: Name of the pipeline to run
            **kwargs: Input parameters passed to the pipeline

        Returns:
            Dict mapping agent names to their AgentResult dicts
        """
        context = dict(kwargs)
        steps = self._get_pipeline_steps(pipeline_name)
        return self._execute_steps(steps, context)

    def run_custom(self, steps: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Run a custom pipeline.

        Args:
            steps: List of step dicts, each with 'agent' and optional 'input_mapping'
            **kwargs: Input parameters

        Returns:
            Dict mapping agent names to their AgentResult dicts
        """
        context = dict(kwargs)
        pipeline_steps = []
        for step_def in steps:
            agent_name = step_def.get("agent", "")
            if agent_name not in self.agents:
                raise ValueError(
                    f"Unknown agent: {agent_name}. Available: {list(self.agents.keys())}"
                )
            pipeline_steps.append(
                PipelineStep(
                    agent=self.agents[agent_name],
                    input_mapping=step_def.get("input_mapping", {}),
                )
            )
        return self._execute_steps(pipeline_steps, context)

    def _get_pipeline_steps(self, name: str) -> List[PipelineStep]:
        """Get pipeline step definitions by name."""
        if name == "full":
            return [
                PipelineStep(self.agents["extract"]),
                PipelineStep(self.agents["validate"], {"schema": "schema"}),
                PipelineStep(self.agents["proposition"], {"schema": "schema"}),
            ]
        elif name == "verify":
            return [
                PipelineStep(self.agents["proposition"], {"schema": "schema"}),
                PipelineStep(self.agents["verify"]),
            ]
        elif name == "compare":
            return [
                PipelineStep(
                    self.agents["extract"],
                    input_mapping={
                        "engine": "engine_a",
                        "params": "params_a",
                        "filepath": "filepath_a",
                    },
                ),
                PipelineStep(
                    self.agents["extract"],
                    input_mapping={
                        "engine": "engine_b",
                        "params": "params_b",
                        "filepath": "filepath_b",
                    },
                ),
                PipelineStep(self.agents["compare"]),
            ]
        elif name == "extract_only":
            return [
                PipelineStep(self.agents["extract"]),
                PipelineStep(self.agents["validate"], {"schema": "schema"}),
            ]
        else:
            raise ValueError(
                f"Unknown pipeline: {name}. "
                f"Available: full, verify, compare, extract_only"
            )

    def _execute_steps(
        self, steps: List[PipelineStep], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute pipeline steps sequentially, passing results forward."""
        results: Dict[str, Any] = {}

        for i, step in enumerate(steps):
            step_context = step.prepare_context(context)

            if i > 0:
                for prev_name, prev_result in results.items():
                    if isinstance(prev_result, dict) and "data" in prev_result:
                        for key, value in prev_result["data"].items():
                            if key not in step_context:
                                step_context[key] = value

            result = step.agent.safe_run(step_context)

            results[step.agent.name] = result.to_dict()

            if (
                result.status == AgentStatus.SUCCESS
                or result.status == AgentStatus.PARTIAL
            ):
                context[step.agent.name + "_result"] = result
                for key, value in result.data.items():
                    if key not in context:
                        context[key] = value

        results["_pipeline_context"] = {
            k: v for k, v in context.items() if not k.endswith("_result")
        }

        return results

    def list_pipelines(self) -> List[str]:
        """List available pipeline names."""
        return ["full", "verify", "compare", "extract_only"]

    def list_agents(self) -> List[str]:
        """List available agent names."""
        return list(self.agents.keys())
