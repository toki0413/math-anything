"""Agent Architecture for Math Anything.

Modular agent system that orchestrates mathematical analysis workflows.
Each agent handles a specific aspect of the extraction-verification pipeline.

Agents:
    ExtractAgent    - Extract mathematical structures from simulation inputs
    ValidateAgent   - Validate parameter consistency and constraints
    CompareAgent    - Cross-engine and cross-model comparison
    PropositionAgent - Generate mathematical propositions and proof tasks
    VerifyAgent     - Verify proofs against mathematical tasks
    Orchestator     - Coordinate agents into workflows

Example:
    >>> from math_anything.agents import AgentOrchestrator
    >>> orch = AgentOrchestrator()
    >>> result = orch.run_pipeline("vasp", {"ENCUT": 520, "SIGMA": 0.05})
"""

from math_anything.agents.base import AgentResult, BaseAgent
from math_anything.agents.compare_agent import CompareAgent
from math_anything.agents.extract_agent import ExtractAgent
from math_anything.agents.orchestrator import AgentOrchestrator
from math_anything.agents.proposition_agent import PropositionAgent
from math_anything.agents.validate_agent import ValidateAgent
from math_anything.agents.verify_agent import VerifyAgent

__all__ = [
    "BaseAgent",
    "AgentResult",
    "ExtractAgent",
    "ValidateAgent",
    "CompareAgent",
    "PropositionAgent",
    "VerifyAgent",
    "AgentOrchestrator",
]
