"""Data structures for tiered analysis system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AnalysisTier(Enum):
    """Analysis complexity levels."""

    BASIC = 1
    ENHANCED = 2
    PROFESSIONAL = 3
    ADVANCED = 4
    COMPLETE = 5


@dataclass
class ComplexityScore:
    """Complexity score for tier recommendation."""

    total: float = 0.0
    system_size: float = 0.0
    time_scale: float = 0.0
    constraints: float = 0.0
    data_availability: float = 0.0

    def __str__(self) -> str:
        return f"ComplexityScore(total={self.total:.1f}, size={self.system_size:.1f}, time={self.time_scale:.1f})"


@dataclass
class ResourceRequirements:
    """Resource requirements for analysis."""

    cpu_time_seconds: float = 1.0
    memory_gb: float = 1.0
    gpu_required: bool = False
    gpu_memory_gb: float = 0.0
    additional_packages: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        gpu_str = f", GPU={self.gpu_memory_gb}GB" if self.gpu_required else ""
        return (
            f"Resources: CPU={self.cpu_time_seconds}s, RAM={self.memory_gb}GB{gpu_str}"
        )


@dataclass
class FileAnalysis:
    """Analysis of input file properties."""

    file_path: str = ""
    file_type: str = ""
    engine: str = ""
    num_atoms: int = 0
    simulation_time: float = 0.0
    has_constraints: bool = False
    constraint_types: List[str] = field(default_factory=list)
    has_training_data: bool = False
    training_data_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "engine": self.engine,
            "num_atoms": self.num_atoms,
            "simulation_time": self.simulation_time,
            "has_constraints": self.has_constraints,
            "constraint_types": self.constraint_types,
            "has_training_data": self.has_training_data,
            "training_data_size": self.training_data_size,
        }


@dataclass
class TierRecommendation:
    """Tier recommendation result."""

    recommended_tier: AnalysisTier
    suitable_tiers: List[AnalysisTier]
    reasons: List[str]
    complexity_score: ComplexityScore
    estimated_time: str
    required_resources: ResourceRequirements
    upgrade_triggers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_tier": self.recommended_tier.value,
            "suitable_tiers": [t.value for t in self.suitable_tiers],
            "reasons": self.reasons,
            "complexity_score": self.complexity_score.total,
            "estimated_time": self.estimated_time,
            "required_resources": str(self.required_resources),
            "upgrade_triggers": self.upgrade_triggers,
        }


@dataclass
class TieredAnalysisResult:
    """Result of tiered analysis."""

    tier: AnalysisTier
    file_analysis: FileAnalysis
    math_schema: Optional[Any] = None
    detailed_params: Optional[Dict[str, Any]] = None
    topology_info: Optional[Dict[str, Any]] = None
    manifold_info: Optional[Dict[str, Any]] = None
    morse_info: Optional[Dict[str, Any]] = None
    latent_info: Optional[Dict[str, Any]] = None
    ml_recommendations: Optional[List[Dict[str, Any]]] = None
    validation: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    upgrade_suggested: bool = False
    upgrade_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "tier": self.tier.value,
            "file_analysis": self.file_analysis.to_dict(),
            "warnings": self.warnings,
            "errors": self.errors,
            "upgrade_suggested": self.upgrade_suggested,
            "upgrade_reason": self.upgrade_reason,
        }

        if self.math_schema is not None:
            result["math_schema"] = str(self.math_schema)
        if self.detailed_params:
            result["detailed_params"] = self.detailed_params
        if self.topology_info:
            result["topology_info"] = self.topology_info
        if self.manifold_info:
            result["manifold_info"] = self.manifold_info
        if self.morse_info:
            result["morse_info"] = self.morse_info
        if self.latent_info:
            result["latent_info"] = self.latent_info
        if self.ml_recommendations:
            result["ml_recommendations"] = self.ml_recommendations
        if self.validation:
            result["validation"] = self.validation

        return result
