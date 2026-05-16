"""Proposition Agent - Generate mathematical propositions and proof tasks."""

from typing import Any, Dict

from .base import AgentResult, AgentStatus, BaseAgent


class PropositionAgent(BaseAgent):
    """Generate mathematical propositions from extracted schemas.

    Wraps the existing PropositionGenerator into the agent framework.

    Input context:
        - 'schema': Extracted mathematical schema (required)

    Output data:
        - 'propositions': MathematicalPropositions data
        - 'proof_tasks': List of proof tasks
        - 'validation_tasks': List of validation tasks
        - 'consistency_checks': List of consistency check tasks
    """

    @property
    def name(self) -> str:
        return "proposition"

    @property
    def description(self) -> str:
        return "Generate mathematical propositions and proof tasks"

    def validate_inputs(self, context: Dict[str, Any]) -> list:
        if "schema" not in context:
            return ["schema"]
        return []

    def run(self, context: Dict[str, Any]) -> AgentResult:
        from math_anything.proposition import PropositionGenerator

        schema = context["schema"]
        if not isinstance(schema, dict):
            schema = schema.to_dict() if hasattr(schema, "to_dict") else {}

        gen = PropositionGenerator()
        propositions = gen.translate(schema)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={
                "propositions": propositions.to_dict(),
                "proof_tasks": [t.to_dict() for t in propositions.proof_tasks],
                "validation_tasks": [
                    t.to_dict() for t in propositions.validation_tasks
                ],
                "consistency_checks": [
                    t.to_dict() for t in propositions.consistency_checks
                ],
                "comparison_tasks": [
                    t.to_dict() for t in propositions.comparison_tasks
                ],
                "error_analysis": [t.to_dict() for t in propositions.error_analysis],
                "total_tasks": len(propositions.all_tasks()),
            },
        )
