"""Pydantic models for Math Anything.

Provides type-safe data models with validation for all API inputs and outputs.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class AnalysisTierEnum(str, Enum):
    """Analysis complexity levels."""

    BASIC = "basic"
    ENHANCED = "enhanced"
    PROFESSIONAL = "professional"
    ADVANCED = "advanced"
    COMPLETE = "complete"


class EngineType(str, Enum):
    """Supported computational engines."""

    VASP = "vasp"
    LAMMPS = "lammps"
    ABAQUS = "abaqus"
    ANSYS = "ansys"
    COMSOL = "comsol"
    GROMACS = "gromacs"
    MULTIWFN = "multiwfn"


class ProblemType(str, Enum):
    """Mathematical problem types."""

    NONLINEAR_EIGENVALUE = "nonlinear_eigenvalue"
    INITIAL_VALUE_ODE = "initial_value_ode"
    BOUNDARY_VALUE_PDE = "boundary_value_pde"
    VARIATIONAL = "variational"
    EIGENVALUE = "eigenvalue"
    STOCHASTIC_ODE = "stochastic_ode"


class ExtractionRequest(BaseModel):
    """Request model for mathematical structure extraction.

    Attributes:
        engine: Computational engine name.
        params: Dictionary of engine parameters.
        validate_params: Whether to validate constraints.
    """

    engine: EngineType = Field(..., description="Computational engine name")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Engine parameters"
    )
    validate_params: bool = Field(default=True, description="Validate constraints")

    @field_validator("engine", mode="before")
    @classmethod
    def normalize_engine(cls, v):
        """Normalize engine name to lowercase."""
        return v.lower() if isinstance(v, str) else v


class FileExtractionRequest(BaseModel):
    """Request model for file-based extraction.

    Attributes:
        engine: Computational engine name.
        filepath: Path to input file or dict of file paths.
        validate_params: Whether to validate constraints.
    """

    engine: EngineType = Field(..., description="Computational engine name")
    filepath: Union[str, Dict[str, str]] = Field(..., description="Input file path(s)")
    validate_params: bool = Field(default=True, description="Validate constraints")


class MathematicalStructure(BaseModel):
    """Mathematical structure extracted from simulation.

    Attributes:
        problem_type: Type of mathematical problem.
        canonical_form: Canonical mathematical representation.
        variable_dependencies: Variable dependency graph.
    """

    problem_type: Optional[ProblemType] = Field(
        None, description="Type of mathematical problem"
    )
    canonical_form: Optional[str] = Field(
        None, description="Canonical mathematical representation"
    )
    variable_dependencies: Dict[str, List[str]] = Field(
        default_factory=dict, description="Variable dependency graph"
    )


class Constraint(BaseModel):
    """Mathematical constraint.

    Attributes:
        expression: Constraint expression.
        satisfied: Whether constraint is satisfied.
        description: Human-readable description.
    """

    expression: str = Field(..., description="Constraint expression")
    satisfied: bool = Field(..., description="Whether constraint is satisfied")
    description: Optional[str] = Field(None, description="Description")


class Approximation(BaseModel):
    """Approximation applied in the simulation.

    Attributes:
        name: Approximation name.
        level: Hierarchy level.
        description: Description of the approximation.
    """

    name: str = Field(..., description="Approximation name")
    level: int = Field(..., description="Hierarchy level (0=exact)")
    description: Optional[str] = Field(None, description="Description")


class ExtractionResultModel(BaseModel):
    """Result model for extraction operations.

    Attributes:
        engine: Engine name.
        success: Whether extraction succeeded.
        mathematical_structure: Extracted mathematical structure.
        constraints: List of constraints.
        approximations: List of approximations.
        errors: List of error messages.
        warnings: List of warning messages.
    """

    engine: str = Field(..., description="Engine name")
    success: bool = Field(..., description="Whether extraction succeeded")
    mathematical_structure: MathematicalStructure = Field(
        default_factory=MathematicalStructure
    )
    constraints: List[Constraint] = Field(default_factory=list)
    approximations: List[Approximation] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class TieredAnalysisRequest(BaseModel):
    """Request for tiered analysis.

    Attributes:
        filepath: Path to simulation file.
        tier: Specific tier to use (optional).
        auto_tier: Automatically determine tier.
        min_tier: Minimum tier level.
        max_tier: Maximum tier level.
    """

    filepath: str = Field(..., description="Path to simulation file")
    tier: Optional[AnalysisTierEnum] = Field(None, description="Analysis tier")
    auto_tier: bool = Field(default=True, description="Auto-detect tier")
    min_tier: int = Field(default=1, ge=1, le=5, description="Minimum tier")
    max_tier: int = Field(default=5, ge=1, le=5, description="Maximum tier")

    @field_validator("max_tier")
    @classmethod
    def max_tier_gte_min_tier(cls, v, info):
        """Ensure max_tier >= min_tier."""
        min_tier = info.data.get("min_tier")
        if min_tier is not None and v < min_tier:
            raise ValueError("max_tier must be >= min_tier")
        return v


class ComplexityScore(BaseModel):
    """Complexity score for tier recommendation.

    Attributes:
        total: Total complexity score (0-100).
        system_size: System size component.
        time_scale: Time scale component.
        constraints: Constraint complexity component.
        data_availability: Training data availability component.
    """

    total: float = Field(default=0.0, ge=0.0, le=100.0)
    system_size: float = Field(default=0.0, ge=0.0, le=50.0)
    time_scale: float = Field(default=0.0, ge=0.0, le=30.0)
    constraints: float = Field(default=0.0, ge=0.0, le=20.0)
    data_availability: float = Field(default=0.0, ge=0.0, le=10.0)


class ResourceRequirements(BaseModel):
    """Resource requirements for analysis.

    Attributes:
        cpu_time_seconds: Estimated CPU time.
        memory_gb: Estimated memory usage.
        gpu_required: Whether GPU is required.
        gpu_memory_gb: Required GPU memory.
    """

    cpu_time_seconds: float = Field(default=1.0, gt=0)
    memory_gb: float = Field(default=1.0, gt=0)
    gpu_required: bool = Field(default=False)
    gpu_memory_gb: float = Field(default=0.0, ge=0.0)


class TierRecommendation(BaseModel):
    """Tier recommendation result.

    Attributes:
        recommended_tier: Recommended analysis tier.
        suitable_tiers: List of suitable tiers.
        reasons: Recommendation reasons.
        complexity_score: Complexity score.
        estimated_time: Estimated analysis time.
        required_resources: Resource requirements.
    """

    recommended_tier: AnalysisTierEnum
    suitable_tiers: List[AnalysisTierEnum]
    reasons: List[str] = Field(default_factory=list)
    complexity_score: ComplexityScore = Field(default_factory=ComplexityScore)
    estimated_time: str = Field(default="< 1 second")
    required_resources: ResourceRequirements = Field(
        default_factory=ResourceRequirements
    )


class TopologyInfo(BaseModel):
    """Topology analysis results.

    Attributes:
        betti_numbers: Betti numbers [b0, b1, b2].
        connected_components: Number of connected components.
        loops: Number of independent loops.
        voids: Number of voids.
    """

    betti_numbers: List[int] = Field(default_factory=lambda: [1, 0, 0])
    connected_components: int = Field(default=1)
    loops: int = Field(default=0)
    voids: int = Field(default=0)


class ManifoldInfo(BaseModel):
    """Manifold analysis results.

    Attributes:
        dimension: Manifold dimension.
        metric_type: Metric type.
        has_symplectic_structure: Whether symplectic structure exists.
    """

    dimension: int = Field(default=0)
    metric_type: str = Field(default="euclidean")
    has_symplectic_structure: bool = Field(default=True)


class LatentInfo(BaseModel):
    """Latent space analysis results.

    Attributes:
        recommended_dimension: Recommended latent dimension.
        encoder_type: Recommended encoder type.
        equivariance: List of equivariance groups.
        estimated_speedup: Estimated speedup.
    """

    recommended_dimension: int = Field(default=16, ge=1)
    encoder_type: str = Field(default="VAE")
    equivariance: List[str] = Field(default_factory=list)
    estimated_speedup: str = Field(default="1x")


class TieredAnalysisResult(BaseModel):
    """Result of tiered analysis.

    Attributes:
        tier: Analysis tier used.
        success: Whether analysis succeeded.
        topology_info: Topology analysis results.
        manifold_info: Manifold analysis results.
        latent_info: Latent space info.
        errors: List of errors.
        warnings: List of warnings.
    """

    tier: AnalysisTierEnum
    success: bool = Field(default=True)
    topology_info: Optional[TopologyInfo] = None
    manifold_info: Optional[ManifoldInfo] = None
    latent_info: Optional[LatentInfo] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ComparisonRequest(BaseModel):
    """Request for comparing two extraction results.

    Attributes:
        result1: First extraction result.
        result2: Second extraction result.
        critical_only: Only show critical changes.
    """

    result1: ExtractionResultModel = Field(..., description="First result")
    result2: ExtractionResultModel = Field(..., description="Second result")
    critical_only: bool = Field(default=False)


class ComparisonResult(BaseModel):
    """Result of comparing two mathematical structures.

    Attributes:
        equations_changed: Whether equations changed.
        boundary_conditions_changed: Whether BCs changed.
        approximations_added: New approximations.
        approximations_removed: Removed approximations.
        critical_changes: Critical changes detected.
    """

    equations_changed: bool = Field(default=False)
    boundary_conditions_changed: bool = Field(default=False)
    approximations_added: List[str] = Field(default_factory=list)
    approximations_removed: List[str] = Field(default_factory=list)
    critical_changes: List[str] = Field(default_factory=list)


class VisualizationRequest(BaseModel):
    """Request for generating visualization.

    Attributes:
        result: Extraction result to visualize.
        output_format: Output format.
        output_path: Optional file path to save.
    """

    result: ExtractionResultModel
    output_format: str = Field(default="mermaid", pattern="^(mermaid|graphviz|html)$")
    output_path: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response model.

    Attributes:
        error_code: Error code.
        message: Error message.
        details: Additional error details.
        type: Exception type name.
    """

    error_code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    type: str = Field(default="MathAnythingError")
