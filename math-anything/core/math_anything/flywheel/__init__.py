"""Data Flywheel - Feedback loop for continuous improvement.

The flywheel captures verification results and uses them to improve:
1. Extraction patterns (which patterns lead to correct extractions)
2. Verification rules (which checks are most predictive)
3. Proposition quality (which propositions are provable)
4. Common failure modes (systematic errors to avoid)

This addresses Liang Wenfeng's critique: "You have no training flywheel."

Example:
    >>> from math_anything.flywheel import DataFlywheel
    >>> fw = DataFlywheel()
    >>> fw.record("extract", {"engine": "vasp", "success": True, "schema": {...}})
    >>> fw.record("verify", {"task_id": "t1", "status": "failed", "issues": [...]})
    >>> stats = fw.get_stats()
    >>> patterns = fw.get_failure_patterns()
"""

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class RecordType(Enum):
    EXTRACT = "extract"
    VALIDATE = "validate"
    PROPOSITION = "proposition"
    VERIFY = "verify"
    COMPARE = "compare"


@dataclass
class FlywheelRecord:
    """A single record in the flywheel."""

    record_type: RecordType
    timestamp: float
    success: bool
    data: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.record_type.value,
            "timestamp": self.timestamp,
            "success": self.success,
            "data": self.data,
            "tags": self.tags,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlywheelRecord":
        return cls(
            record_type=RecordType(d["type"]),
            timestamp=d["timestamp"],
            success=d["success"],
            data=d.get("data", {}),
            tags=d.get("tags", []),
            session_id=d.get("session_id", ""),
        )


@dataclass
class FailurePattern:
    """A recurring failure pattern identified by the flywheel."""

    pattern_id: str
    description: str
    frequency: int
    affected_engines: List[str]
    affected_checks: List[str]
    suggested_fix: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "frequency": self.frequency,
            "affected_engines": self.affected_engines,
            "affected_checks": self.affected_checks,
            "suggested_fix": self.suggested_fix,
            "confidence": self.confidence,
        }


@dataclass
class FlywheelStats:
    """Aggregate statistics from the flywheel."""

    total_records: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    success_rate_by_type: Dict[str, float] = field(default_factory=dict)
    success_rate_trend: Dict[str, List[Tuple[float, float]]] = field(
        default_factory=dict
    )
    common_issues: List[Tuple[str, int]] = field(default_factory=list)
    engine_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "by_type": self.by_type,
            "success_rate_by_type": self.success_rate_by_type,
            "common_issues": self.common_issues[:10],
            "engine_performance": self.engine_performance,
        }


class EngineDegradation:
    """Tracks per-engine degradation state based on recent failure history.

    When an engine accumulates enough consecutive failures, it enters
    a degraded mode that downstream consumers can check before running
    expensive operations.
    """

    def __init__(self, failure_threshold: int = 3, window: int = 10):
        self.failure_threshold = failure_threshold
        self.window = window
        self._degraded: Dict[str, bool] = {}
        self._callbacks: List[Any] = []

    def check(self, engine: str, records: List[FlywheelRecord]) -> bool:
        recent = [
            r
            for r in records[-self.window :]
            if r.data.get("engine") == engine and not r.success
        ]
        was_degraded = self._degraded.get(engine, False)
        is_degraded = len(recent) >= self.failure_threshold
        self._degraded[engine] = is_degraded

        if is_degraded and not was_degraded:
            for cb in self._callbacks:
                cb(engine, recent)

        return is_degraded

    def is_degraded(self, engine: str) -> bool:
        return self._degraded.get(engine, False)

    def on_degradation(self, callback) -> None:
        self._callbacks.append(callback)

    def reset(self, engine: str = "") -> None:
        if engine:
            self._degraded.pop(engine, None)
        else:
            self._degraded.clear()


class DataFlywheel:
    """Data flywheel for continuous improvement of math-anything.

    The flywheel stores every extraction, validation, proposition, and
    verification result. It then analyzes patterns to:

    1. Identify systematic failure modes
    2. Track improvement over time
    3. Suggest targeted fixes
    4. Build a training dataset for local LLM fine-tuning
    5. Auto-degrade engines with repeated failures

    Storage: JSONL file (one record per line) for simplicity and portability.
    """

    def __init__(self, storage_path: Optional[str] = None, failure_threshold: int = 3):
        if storage_path is None:
            storage_dir = Path.home() / ".math_anything" / "flywheel"
            storage_dir.mkdir(parents=True, exist_ok=True)
            storage_path = str(storage_dir / "records.jsonl")
        self.storage_path = storage_path
        self._records: List[FlywheelRecord] = []
        self._loaded = False
        self.degradation = EngineDegradation(failure_threshold=failure_threshold)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._records.append(
                                FlywheelRecord.from_dict(json.loads(line))
                            )
                        except (json.JSONDecodeError, KeyError, ValueError):
                            pass

    def record(
        self,
        record_type: str,
        data: Dict[str, Any],
        success: bool = True,
        tags: Optional[List[str]] = None,
        session_id: str = "",
    ) -> None:
        """Record a new data point in the flywheel.

        Args:
            record_type: One of 'extract', 'validate', 'proposition', 'verify', 'compare'
            data: Result data (schema, issues, etc.)
            success: Whether the operation succeeded
            tags: Optional tags for categorization
            session_id: Optional session identifier
        """
        self._ensure_loaded()
        rec = FlywheelRecord(
            record_type=RecordType(record_type),
            timestamp=time.time(),
            success=success,
            data=data,
            tags=tags or [],
            session_id=session_id,
        )
        self._records.append(rec)
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

        engine = data.get("engine", "")
        if engine and not success:
            self.degradation.check(engine, self._records)

    def get_stats(self) -> FlywheelStats:
        """Compute aggregate statistics from all records."""
        self._ensure_loaded()
        stats = FlywheelStats()
        stats.total_records = len(self._records)
        if not self._records:
            return stats

        by_type: Counter = Counter()
        success_by_type: Dict[str, List[bool]] = defaultdict(list)
        issue_counter: Counter = Counter()
        engine_stats: Dict[str, Dict[str, List[bool]]] = defaultdict(
            lambda: {"extract": [], "validate": [], "verify": []}
        )

        for rec in self._records:
            t = rec.record_type.value
            by_type[t] += 1
            success_by_type[t].append(rec.success)

            if not rec.success and "issues" in rec.data:
                for issue in rec.data["issues"]:
                    if isinstance(issue, dict):
                        key = issue.get("category", "unknown")
                    else:
                        key = str(issue)
                    issue_counter[key] += 1

            engine = rec.data.get("engine", "unknown")
            if rec.record_type.value in engine_stats.get(engine, {}):
                engine_stats[engine][rec.record_type.value].append(rec.success)

        stats.by_type = dict(by_type)
        stats.success_rate_by_type = {
            t: sum(v) / len(v) for t, v in success_by_type.items() if v
        }
        stats.common_issues = issue_counter.most_common(10)

        for engine, type_stats in engine_stats.items():
            stats.engine_performance[engine] = {
                t: sum(v) / len(v) for t, v in type_stats.items() if v
            }

        return stats

    def get_failure_patterns(self, min_frequency: int = 2) -> List[FailurePattern]:
        """Identify recurring failure patterns.

        Args:
            min_frequency: Minimum number of occurrences to be considered a pattern

        Returns:
            List of FailurePattern sorted by frequency (descending)
        """
        self._ensure_loaded()
        failure_records = [r for r in self._records if not r.success]
        if not failure_records:
            return []

        issue_by_category: Dict[str, List[FlywheelRecord]] = defaultdict(list)
        for rec in failure_records:
            for issue in rec.data.get("issues", []):
                if isinstance(issue, dict):
                    cat = issue.get("category", "unknown")
                else:
                    cat = str(issue)
                issue_by_category[cat].append(rec)

        patterns = []
        for category, records in issue_by_category.items():
            if len(records) < min_frequency:
                continue

            engines = list(set(r.data.get("engine", "unknown") for r in records))
            descriptions = []
            for r in records:
                for issue in r.data.get("issues", []):
                    if isinstance(issue, dict) and issue.get("category") == category:
                        descriptions.append(issue.get("description", ""))

            fix = self._suggest_fix(category, records)

            patterns.append(
                FailurePattern(
                    pattern_id=f"FP-{category[:20]}-{len(records)}",
                    description=f"Recurring {category} failure ({len(records)} times)",
                    frequency=len(records),
                    affected_engines=engines,
                    affected_checks=[category],
                    suggested_fix=fix,
                    confidence=min(1.0, len(records) / 10),
                )
            )

        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns

    def get_training_data(
        self, record_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Export records as training data for local LLM fine-tuning.

        Args:
            record_type: Filter by type, or None for all

        Returns:
            List of training examples with input/output pairs
        """
        self._ensure_loaded()
        training = []

        for rec in self._records:
            if record_type and rec.record_type.value != record_type:
                continue

            if rec.record_type == RecordType.VERIFY:
                proof_text = rec.data.get("proof_text", "")
                issues = rec.data.get("issues", [])
                if proof_text and issues:
                    training.append(
                        {
                            "input": proof_text,
                            "output": json.dumps(issues, ensure_ascii=False),
                            "label": "valid" if rec.success else "invalid",
                            "type": "proof_verification",
                        }
                    )

            elif rec.record_type == RecordType.EXTRACT:
                engine = rec.data.get("engine", "")
                schema = rec.data.get("schema", {})
                if engine and schema:
                    training.append(
                        {
                            "input": json.dumps(
                                {
                                    "engine": engine,
                                    "params": rec.data.get("params", {}),
                                },
                                ensure_ascii=False,
                            ),
                            "output": json.dumps(schema, ensure_ascii=False),
                            "label": "correct" if rec.success else "incorrect",
                            "type": "schema_extraction",
                        }
                    )

        return training

    def export_training_jsonl(
        self, output_path: str, record_type: Optional[str] = None
    ) -> int:
        """Export training data as JSONL file for LLM fine-tuning.

        Returns:
            Number of training examples exported
        """
        data = self.get_training_data(record_type)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return len(data)

    def _suggest_fix(self, category: str, records: List[FlywheelRecord]) -> str:
        """Suggest a fix based on failure pattern."""
        fix_map = {
            "missing_assumptions": "Add explicit assumption coverage check in extraction",
            "incomplete_goals": "Ensure all proof goals are enumerated before generation",
            "missing_conclusion": "Add conclusion marker requirement to proof prompt",
            "missing_justification": "Require explicit justifications for each proof step",
            "missing_references": "Include reference citations in proposition generation",
            "informal_language": "Add formality constraints to proof generation prompt",
            "many_new_symbols": "Add symbol definition requirement to proof prompt",
            "short_proof": "Require minimum proof length or step count",
            "circular_dependency": "Check for circular variable dependencies in schema",
            "below_minimum": "Validate parameter ranges before extraction",
            "above_maximum": "Validate parameter ranges before extraction",
        }
        return fix_map.get(category, f"Review and fix {category} issues systematically")

    def clear(self) -> None:
        """Clear all flywheel data."""
        self._records = []
        self._loaded = True
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
