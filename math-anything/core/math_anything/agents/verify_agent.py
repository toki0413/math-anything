"""Verify Agent - Verify proofs against mathematical tasks."""

from typing import Any, Dict

from .base import AgentResult, AgentStatus, BaseAgent


class VerifyAgent(BaseAgent):
    """Verify LLM-generated proofs using the ProofVerifier.

    Implements the closed-loop verification:
    1. Verify proof against task requirements
    2. If failed, generate revision prompt
    3. Optionally re-verify revised proof

    Input context:
        - 'task': MathematicalTask to verify against (required)
        - 'proof_text': LLM-generated proof text (required)
        - 'max_revisions': Max revision rounds (optional, default 0)

    Output data:
        - 'verification': VerificationResult data
        - 'status': Verification status
        - 'confidence': Confidence score
        - 'revised_prompt': Prompt for LLM revision (if failed)
    """

    @property
    def name(self) -> str:
        return "verify"

    @property
    def description(self) -> str:
        return "Verify proofs against mathematical tasks"

    def validate_inputs(self, context: Dict[str, Any]) -> list:
        missing = []
        if "task" not in context:
            missing.append("task")
        if "proof_text" not in context:
            missing.append("proof_text")
        return missing

    def run(self, context: Dict[str, Any]) -> AgentResult:
        from math_anything.proof_verifier import ProofReviser, ProofVerifier

        task = context["task"]
        proof_text = context["proof_text"]
        max_revisions = context.get("max_revisions", 0)

        verifier = ProofVerifier()
        result = verifier.verify(task, proof_text)

        data = {
            "verification": result.to_dict(),
            "status": result.status.value,
            "confidence": result.confidence,
            "issues_count": len(result.issues),
        }

        if not result.passed:
            reviser = ProofReviser()
            revised_prompt = reviser.revise(task, proof_text, result)
            data["revised_prompt"] = revised_prompt

            if max_revisions > 0 and "llm_generate_fn" in context:
                from math_anything.proof_verifier import VerificationPipeline

                pipeline = VerificationPipeline()
                pipeline_result = pipeline.run(
                    task,
                    proof_text,
                    max_revisions=max_revisions,
                    llm_generate_fn=context["llm_generate_fn"],
                )
                data["pipeline_result"] = pipeline_result

        status = AgentStatus.SUCCESS if result.passed else AgentStatus.PARTIAL

        return AgentResult(
            agent_name=self.name,
            status=status,
            data=data,
        )
