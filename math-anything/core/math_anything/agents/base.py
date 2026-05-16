"""Base Agent - Abstract base class for all Math Anything agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentResult:
    """Standard result from any agent."""

    agent_name: str
    status: AgentStatus
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status in (AgentStatus.SUCCESS, AgentStatus.PARTIAL)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "status": self.status.value,
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Each agent handles one specific aspect of the mathematical analysis pipeline.
    Agents are designed to be composable - the Orchestrator chains them together.

    Subclasses must implement:
        - name: Agent identifier
        - run(): Main execution logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent name."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        return ""

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> AgentResult:
        """Execute the agent's task.

        Args:
            context: Shared context dict containing inputs and previous agent results.
                     Common keys:
                     - 'engine': Engine name (e.g., 'vasp', 'lammps')
                     - 'params': Engine parameters dict
                     - 'schema': Extracted mathematical schema
                     - 'propositions': Generated propositions
                     - 'proof_text': LLM-generated proof

        Returns:
            AgentResult with status, data, and any errors/warnings
        """
        ...

    def validate_inputs(self, context: Dict[str, Any]) -> List[str]:
        """Validate required inputs are present in context.

        Returns:
            List of missing input keys (empty if all present)
        """
        return []

    def safe_run(self, context: Dict[str, Any]) -> AgentResult:
        """Run with error handling wrapper.

        Catches exceptions and returns them as failed AgentResult
        instead of propagating.
        """
        missing = self.validate_inputs(context)
        if missing:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SKIPPED,
                errors=[f"Missing required inputs: {', '.join(missing)}"],
            )

        try:
            return self.run(context)
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=[f"{type(e).__name__}: {str(e)}"],
            )
