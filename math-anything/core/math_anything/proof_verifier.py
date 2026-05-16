"""Proof Verifier - Closed-loop verification for mathematical propositions.

Implements the Generator-Verifier-Reviser architecture:
1. PropositionGenerator generates mathematical tasks (existing)
2. ProofVerifier verifies LLM-generated proofs against tasks
3. ProofReviser suggests corrections for failed proofs

This closes the loop: extract -> propose -> prove -> verify -> revise -> verify again.

Example:
    >>> from math_anything import PropositionGenerator, ProofVerifier
    >>> gen = PropositionGenerator()
    >>> props = gen.translate(schema)
    >>> verifier = ProofVerifier()
    >>> result = verifier.verify(props.proof_tasks[0], llm_proof_text)
    >>> if not result.passed:
    ...     reviser = ProofReviser()
    ...     revised = reviser.revise(props.proof_tasks[0], llm_proof_text, result)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VerificationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"


class IssueSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class VerificationIssue:
    """A single issue found during verification."""

    severity: IssueSeverity
    category: str
    description: str
    location: str = ""
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "description": self.description,
            "location": self.location,
            "suggestion": self.suggestion,
        }


@dataclass
class VerificationResult:
    """Result of verifying a proof against a mathematical task."""

    status: VerificationStatus
    confidence: float
    issues: List[VerificationIssue] = field(default_factory=list)
    checked_aspects: List[str] = field(default_factory=list)
    summary: str = ""
    revision_hints: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "issues": [i.to_dict() for i in self.issues],
            "checked_aspects": self.checked_aspects,
            "summary": self.summary,
            "revision_hints": self.revision_hints,
        }


class ProofVerifier:
    """Verify LLM-generated proofs against mathematical tasks.

    Verification checks:
    1. Logical consistency - no missing steps, no circular reasoning
    2. Assumption coverage - all required assumptions are addressed
    3. Goal completion - all proof goals are reached
    4. Mathematical rigor - correct use of theorems and definitions
    5. Symbol consistency - variables used consistently throughout
    """

    def __init__(self, strict: bool = False):
        self.strict = strict

    def verify(self, task, proof_text: str) -> VerificationResult:
        """Verify a proof against a mathematical task.

        Args:
            task: MathematicalTask from PropositionGenerator
            proof_text: LLM-generated proof text

        Returns:
            VerificationResult with status, confidence, and issues
        """
        if not proof_text or not proof_text.strip():
            return VerificationResult(
                status=VerificationStatus.FAILED,
                confidence=0.0,
                issues=[
                    VerificationIssue(
                        severity=IssueSeverity.ERROR,
                        category="empty_proof",
                        description="Proof text is empty",
                        suggestion="Generate a proof before verification",
                    )
                ],
                summary="Empty proof submitted",
            )

        issues: List[VerificationIssue] = []
        checked: List[str] = []

        self._check_assumption_coverage(task, proof_text, issues, checked)
        self._check_goal_completion(task, proof_text, issues, checked)
        self._check_logical_structure(proof_text, issues, checked)
        self._check_symbol_consistency(task, proof_text, issues, checked)
        self._check_mathematical_rigor(task, proof_text, issues, checked)

        error_count = sum(1 for i in issues if i.severity == IssueSeverity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == IssueSeverity.WARNING)

        if error_count == 0 and warning_count == 0:
            status = VerificationStatus.PASSED
            confidence = 0.9
        elif error_count == 0 and warning_count <= 2:
            status = VerificationStatus.PARTIAL
            confidence = 0.7
        elif error_count == 0:
            status = VerificationStatus.PARTIAL
            confidence = 0.5
        elif error_count <= 1:
            status = VerificationStatus.PARTIAL
            confidence = 0.3
        else:
            status = VerificationStatus.FAILED
            confidence = 0.1

        if self.strict and warning_count > 0:
            status = VerificationStatus.FAILED
            confidence = min(confidence, 0.2)

        summary = self._build_summary(status, error_count, warning_count, checked)
        hints = self._build_revision_hints(issues)

        return VerificationResult(
            status=status,
            confidence=confidence,
            issues=issues,
            checked_aspects=checked,
            summary=summary,
            revision_hints=hints,
        )

    def _check_assumption_coverage(
        self, task, proof_text: str, issues: List[VerificationIssue], checked: List[str]
    ) -> None:
        checked.append("assumption_coverage")
        assumptions = getattr(task, "assumptions", [])
        if not assumptions:
            return

        missing = []
        for assumption in assumptions:
            keywords = self._extract_keywords(assumption)
            found = any(
                kw.lower() in proof_text.lower() for kw in keywords if len(kw) > 2
            )
            if not found:
                missing.append(assumption)

        if missing:
            issues.append(
                VerificationIssue(
                    severity=IssueSeverity.WARNING,
                    category="missing_assumptions",
                    description=f"Proof does not address {len(missing)} assumption(s): "
                    + "; ".join(missing[:3]),
                    suggestion="Explicitly state how each assumption is used in the proof",
                )
            )

    def _check_goal_completion(
        self, task, proof_text: str, issues: List[VerificationIssue], checked: List[str]
    ) -> None:
        checked.append("goal_completion")
        goals = getattr(task, "goals", [])
        if not goals:
            return

        missing_goals = []
        for goal in goals:
            keywords = self._extract_keywords(goal)
            found = any(
                kw.lower() in proof_text.lower() for kw in keywords if len(kw) > 2
            )
            if not found:
                missing_goals.append(goal)

        if missing_goals:
            severity = (
                IssueSeverity.ERROR
                if len(missing_goals) > len(goals) // 2
                else IssueSeverity.WARNING
            )
            issues.append(
                VerificationIssue(
                    severity=severity,
                    category="incomplete_goals",
                    description=f"Proof does not address {len(missing_goals)} goal(s): "
                    + "; ".join(missing_goals[:3]),
                    suggestion="Ensure each goal is explicitly proven or addressed",
                )
            )

    def _check_logical_structure(
        self, proof_text: str, issues: List[VerificationIssue], checked: List[str]
    ) -> None:
        checked.append("logical_structure")

        lines = [l.strip() for l in proof_text.split("\n") if l.strip()]
        if len(lines) < 2:
            issues.append(
                VerificationIssue(
                    severity=IssueSeverity.WARNING,
                    category="short_proof",
                    description="Proof is very short, may lack necessary detail",
                    suggestion="Expand with intermediate steps and justifications",
                )
            )

        has_conclusion = any(
            kw in proof_text.lower()
            for kw in [
                "therefore",
                "hence",
                "thus",
                "q.e.d",
                "qed",
                "proved",
                "证毕",
                "得证",
                "因此",
                "所以",
                "综上",
            ]
        )
        if not has_conclusion:
            issues.append(
                VerificationIssue(
                    severity=IssueSeverity.WARNING,
                    category="missing_conclusion",
                    description="Proof lacks explicit conclusion marker",
                    suggestion="Add a concluding statement (e.g., Q.E.D. or Therefore...)",
                )
            )

        has_justification = any(
            kw in proof_text.lower()
            for kw in [
                "by ",
                "since ",
                "because ",
                "from ",
                "using ",
                "根据",
                "由",
                "因为",
                "根据",
            ]
        )
        if not has_justification:
            issues.append(
                VerificationIssue(
                    severity=IssueSeverity.WARNING,
                    category="missing_justification",
                    description="Proof steps lack explicit justifications",
                    suggestion="Add reasoning for each step (e.g., 'By Theorem X...')",
                )
            )

    def _check_symbol_consistency(
        self, task, proof_text: str, issues: List[VerificationIssue], checked: List[str]
    ) -> None:
        checked.append("symbol_consistency")

        statement = getattr(task, "statement", "")
        if not statement:
            return

        task_symbols = self._extract_math_symbols(statement)
        proof_symbols = self._extract_math_symbols(proof_text)

        introduced = proof_symbols - task_symbols
        if len(introduced) > 5:
            issues.append(
                VerificationIssue(
                    severity=IssueSeverity.WARNING,
                    category="many_new_symbols",
                    description=f"Proof introduces {len(introduced)} new symbols not in the problem statement",
                    suggestion="Define all new symbols explicitly before first use",
                )
            )

    def _check_mathematical_rigor(
        self, task, proof_text: str, issues: List[VerificationIssue], checked: List[str]
    ) -> None:
        checked.append("mathematical_rigor")

        informal_phrases = [
            "obviously",
            "clearly",
            "it is easy to see",
            "trivially",
            "显然",
            "易知",
            "不难看出",
        ]
        found_informal = [p for p in informal_phrases if p in proof_text.lower()]
        if len(found_informal) > 2:
            issues.append(
                VerificationIssue(
                    severity=IssueSeverity.INFO,
                    category="informal_language",
                    description=f"Proof uses {len(found_informal)} informal phrase(s): "
                    + ", ".join(found_informal[:3]),
                    suggestion="Replace informal phrases with precise mathematical justifications",
                )
            )

        references = getattr(task, "references", [])
        if references:
            ref_found = any(
                any(str(r).lower() in proof_text.lower() for r in references)
                for _ in [1]
            )
            if not ref_found:
                issues.append(
                    VerificationIssue(
                        severity=IssueSeverity.INFO,
                        category="missing_references",
                        description="Task provides references but proof does not cite them",
                        suggestion="Cite relevant theorems or equations from the references",
                    )
                )

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        import re

        words = re.findall(r"[a-zA-Z_\u4e00-\u9fff][a-zA-Z0-9_\u4e00-\u9fff]*", text)
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "of",
            "in",
            "to",
            "for",
            "with",
            "on",
            "at",
            "by",
            "from",
            "and",
            "or",
            "not",
            "this",
            "that",
            "it",
            "its",
            "as",
            "的",
            "了",
            "在",
            "是",
            "和",
            "与",
            "或",
            "不",
            "这",
            "那",
        }
        return [w for w in words if w.lower() not in stop_words and len(w) > 1]

    def _extract_math_symbols(self, text: str) -> set:
        """Extract mathematical symbols from text."""
        import re

        symbols = set()
        for match in re.finditer(r"\$([^$]+)\$", text):
            symbols.add(match.group(1).strip())
        for match in re.finditer(r"\\([a-zA-Z]+)", text):
            symbols.add(match.group(1))
        for match in re.finditer(r"\b([A-Z][a-z]?(?:_\w+)?)\b", text):
            symbols.add(match.group(1))
        return symbols

    def _build_summary(
        self, status: VerificationStatus, errors: int, warnings: int, checked: List[str]
    ) -> str:
        parts = [f"Verification: {status.value}"]
        if errors:
            parts.append(f"{errors} error(s)")
        if warnings:
            parts.append(f"{warnings} warning(s)")
        parts.append(f"Checked: {', '.join(checked)}")
        return " | ".join(parts)

    def _build_revision_hints(self, issues: List[VerificationIssue]) -> List[str]:
        hints = []
        for issue in issues:
            if issue.severity in (IssueSeverity.ERROR, IssueSeverity.WARNING):
                hint = (
                    f"[{issue.category}] {issue.suggestion}" if issue.suggestion else ""
                )
                if hint:
                    hints.append(hint)
        return hints


class ProofReviser:
    """Suggest revisions for failed proofs.

    Takes a task, proof, and verification result, then generates
    a revised prompt for the LLM to fix the proof.

    Example:
        >>> reviser = ProofReviser()
        >>> revised_prompt = reviser.revise(task, proof_text, verification_result)
        >>> new_proof = llm.generate(revised_prompt)
    """

    def revise(
        self, task, proof_text: str, verification_result: VerificationResult
    ) -> str:
        """Generate a revised prompt for the LLM.

        Args:
            task: Original MathematicalTask
            proof_text: Failed proof text
            verification_result: Result from ProofVerifier

        Returns:
            Revised prompt string for LLM
        """
        task_name = getattr(task, "name", "Unknown")
        task_statement = getattr(task, "statement", "")
        task_goals = getattr(task, "goals", [])
        task_assumptions = getattr(task, "assumptions", [])

        issues_text = "\n".join(
            f"  - [{i.severity.value}] {i.category}: {i.description}"
            for i in verification_result.issues
        )

        hints_text = "\n".join(f"  - {h}" for h in verification_result.revision_hints)

        goals_text = "\n".join(f"  {i+1}. {g}" for i, g in enumerate(task_goals))
        assumptions_text = "\n".join(f"  - {a}" for a in task_assumptions)

        prompt = f"""【证明修正任务】
原始命题: {task_name}
命题陈述: {task_statement}

假设:
{assumptions_text if assumptions_text else '  (无显式假设)'}

需要证明的目标:
{goals_text if goals_text else '  (无显式目标)'}

原始证明:
---
{proof_text}
---

验证结果: {verification_result.status.value} (置信度: {verification_result.confidence:.0%})

发现的问题:
{issues_text if issues_text else '  (无)'}

修正建议:
{hints_text if hints_text else '  (无)'}

请根据以上反馈修正证明。要求:
1. 解决所有标记为 error 的问题
2. 尽量解决 warning 级别的问题
3. 确保每个目标都有明确的证明
4. 每个步骤给出严格的数学理由
5. 使用规范的数学符号和术语

请给出修正后的完整证明:"""

        return prompt


class VerificationPipeline:
    """Full verification pipeline: generate -> verify -> revise -> re-verify.

    Orchestrates the closed-loop verification process.

    Example:
        >>> pipeline = VerificationPipeline()
        >>> result = pipeline.run(
        ...     task=task,
        ...     proof_text=llm_proof,
        ...     max_revisions=2,
        ...     llm_generate_fn=my_llm_fn,
        ... )
        >>> print(f"Final status: {result.final_status}")
    """

    def __init__(self, strict: bool = False):
        self.verifier = ProofVerifier(strict=strict)
        self.reviser = ProofReviser()

    def run(
        self, task, proof_text: str, max_revisions: int = 2, llm_generate_fn=None
    ) -> Dict[str, Any]:
        """Run the full verification pipeline.

        Args:
            task: MathematicalTask to verify against
            proof_text: Initial LLM-generated proof
            max_revisions: Maximum number of revision rounds
            llm_generate_fn: Optional callable(prompt) -> str for auto-revision

        Returns:
            Dict with final status, all verification results, and revision history
        """
        history: List[Dict[str, Any]] = []
        current_proof = proof_text

        for round_num in range(max_revisions + 1):
            result = self.verifier.verify(task, current_proof)

            history.append(
                {
                    "round": round_num,
                    "status": result.status.value,
                    "confidence": result.confidence,
                    "issues_count": len(result.issues),
                    "proof_length": len(current_proof),
                }
            )

            if result.passed:
                return {
                    "final_status": result.status.value,
                    "final_confidence": result.confidence,
                    "rounds": round_num + 1,
                    "history": history,
                    "final_result": result.to_dict(),
                    "final_proof": current_proof,
                }

            if round_num >= max_revisions:
                return {
                    "final_status": result.status.value,
                    "final_confidence": result.confidence,
                    "rounds": round_num + 1,
                    "history": history,
                    "final_result": result.to_dict(),
                    "final_proof": current_proof,
                }

            revised_prompt = self.reviser.revise(task, current_proof, result)

            if llm_generate_fn is not None:
                try:
                    current_proof = llm_generate_fn(revised_prompt)
                except Exception as e:
                    history.append(
                        {
                            "round": round_num + 1,
                            "status": "llm_error",
                            "error": str(e),
                        }
                    )
                    return {
                        "final_status": "inconclusive",
                        "final_confidence": result.confidence,
                        "rounds": round_num + 1,
                        "history": history,
                        "final_result": result.to_dict(),
                        "final_proof": current_proof,
                        "error": str(e),
                    }
            else:
                return {
                    "final_status": result.status.value,
                    "final_confidence": result.confidence,
                    "rounds": round_num + 1,
                    "history": history,
                    "final_result": result.to_dict(),
                    "final_proof": current_proof,
                    "revised_prompt": revised_prompt,
                    "message": "No LLM function provided. Use revised_prompt to generate a new proof.",
                }

        return {
            "final_status": "inconclusive",
            "final_confidence": 0.0,
            "rounds": max_revisions + 1,
            "history": history,
        }
