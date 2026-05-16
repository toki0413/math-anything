"""Math Anything - Mathematical structure extraction for computational materials.

Math Anything extracts universal mathematical structures (governing equations,
boundary conditions, numerical methods, computational graphs) from computational
software (VASP, LAMMPS, Abaqus, etc.) and outputs them as LLM-native structured data.

Quick Start:
    ```python
    from math_anything import MathAnything

    # Simple API
    ma = MathAnything()
    result = ma.extract("vasp", {"ENCUT": 520, "SIGMA": 0.05})
    print(result.schema["mathematical_structure"]["canonical_form"])

    # With file parsing
    result = ma.extract_file("vasp", "INCAR")
    print(result.to_mermaid())  # Visualize as diagram
    ```
"""

__version__ = "1.0.0"

# New simplified API
from .api import (
    ExtractionResult,
    InputFileNotFoundError,
    MathAnything,
    MathAnythingError,
    ParseError,
    UnsupportedEngineError,
    extract,
    extract_file,
)
from .core.extractor import ExtractorEngine

# Legacy API (for backward compatibility)
from .core.harness import HarnessRegistry, MathAnythingHarness
from .core.session import ExtractionSession

# EML Symbolic Regression
from .eml_v2 import (
    ExprBuilder,
    ImprovedSymbolicRegression,
    Node,
    NodeType,
    SymbolicRegression,
    discover_equation,
    eml,
)

# Exception Handling
from .exceptions import (
    ConfigurationError,
    FileAccessError,
    MathAnythingError,
    ParseError,
    SecurityError,
    TierAnalysisError,
    UnsupportedEngineError,
    ValidationError,
)

# Multi-variable Discovery
from .multivar import MultiVariableDiscovery, analyze_interactions, discover_multivar

# Math Proposition Generation
from .proposition import (
    MathematicalPropositions,
    MathematicalTask,
    PropositionGenerator,
    TaskType,
)

# Proof Verification (Closed Loop)
from .proof_verifier import (
    ProofReviser,
    ProofVerifier,
    VerificationPipeline,
    VerificationResult,
    VerificationStatus,
)

# Agent Architecture
from .agents import (
    AgentOrchestrator,
    AgentResult,
    BaseAgent,
    CompareAgent,
    ExtractAgent,
    PropositionAgent,
    ValidateAgent,
    VerifyAgent,
)
from .schemas import (
    BoundaryCondition,
    ComputationalGraph,
    GoverningEquation,
    MathematicalModel,
    MathSchema,
    NumericalMethod,
    SchemaValidator,
)

# Security Utilities
from .security import (
    FileSizeValidator,
    PathSecurityValidator,
    default_path_validator,
    default_size_validator,
    validate_file_size,
    validate_filepath,
)

# Expression Simplification
from .simplifier import ExpressionSimplifier, simplify

# Tiered Analysis System
from .tiered import (
    AnalysisTier,
    ComplexityScore,
    FileAnalysis,
    ResourceRequirements,
    TieredAnalysisResult,
    TieredAnalyzer,
    TierRecommendation,
    TierRecommender,
)
from .tiered.tier_recommender import analyze_file_properties
from .tiered.tiered_analyzer import analyze as tiered_analyze

# Visualization
from .visualization import Visualizer, save_html, to_graphviz, to_mermaid

# Differential Geometry Layer
from .geometry import (
    CurvatureInfo,
    CurvatureType,
    DifferentialGeometryLayer,
    FiberBundle,
    GeometricStructure,
    Manifold,
    ManifoldType,
    MetricTensor,
    SymmetryGroup,
    SymmetryType,
    compute_christoffel,
)

# Math Advisor
from .advisor import MathAdvisor, DISCIPLINE_STATUS

# Analysis Tools
from .tools.symmetry import SymmetryAnalyzer, SymmetryAnalysisResult
from .tools.tda import TDAAnalyzer, TopologyResult
from .tools.spectral import SpectralAnalyzer, SpectralAnalysisResult
from .tools.dynamics import DynamicsAnalyzer, DynamicsAnalysisResult
from .tools.viz import InteractiveVisualizer
from .tools.ml_potential import MLPotentialAnalyzer, MLPotentialResult
from .tools.langlands import LanglandsAnalyzer, LanglandsResult
from .tools.sindy import SINDyDiscoverer, SINDyResult
from .sandbox import SandboxExecutor, SandboxResult, SandboxConfig

# Provenance & References
from .provenance import Provenance, ProvenanceChain, ProvenanceTracker
from .references import ConstraintReference, ReferenceTracker

# Data Flywheel
from .flywheel import (
    DataFlywheel,
    EngineDegradation,
    FailurePattern,
    FlywheelRecord,
    FlywheelStats,
    RecordType,
)

# Formal Verification (Multi-layer)
from .formal_verifier import (
    Dimension,
    FormalStatus,
    FormalVerificationResult,
    FormalVerifier,
    LayerResult,
    LogicVerifier,
    LLMSemanticVerifier,
    MathType,
    SymbolicVerifier,
    TypeSystemVerifier,
    TypedSymbol,
    VerificationLayer,
)

# Lean4 + Mathlib Bridge (Optional Layer 5)
from .lean4_bridge import (
    Lean4Bridge,
    LeanVerificationResult,
    LeanVerificationStatus,
    get_lean4_bridge,
    schema_to_lean,
    verify_with_lean4,
)

# Validation Toolkit (Cross-validation, Falsifiable Predictions, Dual Perspective)
from .validation_toolkit import (
    CrossValidationMatrix,
    DualPerspectiveAnalyzer,
    DualPerspectiveResult,
    FalsifiablePrediction,
    FalsifiablePredictionTable,
    MethodConclusionCell,
    PredictionStatus,
    ValidationStatus,
    create_cross_validation_from_schema,
    create_prediction_table_from_schema,
)

# Tool System + Agent Loop
from .tool_system import (
    MathTool,
    PermissionResult,
    ToolContext,
    ToolResult,
    build_math_tool,
)
from .tool_registry import (
    ToolRegistry,
    build_default_registry,
)
from .agent_loop import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    MathAgentLoop,
    TextDeltaEvent,
    ToolCallStartEvent,
    ToolProgressEvent,
    ToolResultEvent,
)

FormalResult = FormalVerificationResult


def load_harness(engine_name: str) -> MathAnythingHarness:
    """Load a harness by engine name.

    Args:
        engine_name: Name of the engine (e.g., 'lammps', 'vasp').

    Returns:
        Harness instance.

    Raises:
        ValueError: If engine not found.

    Example:
        ```python
        harness = ma.load_harness("lammps")
        schema = harness.extract({"input": "in.file"})
        ```
    """
    harness = HarnessRegistry.create(engine_name)
    if harness is None:
        available = HarnessRegistry.list_engines()
        raise ValueError(
            f"Engine '{engine_name}' not found. " f"Available engines: {available}"
        )
    return harness


def list_engines() -> list:
    """List all available engines.

    Returns:
        List of engine names.
    """
    try:
        from .api import ENGINE_EXTRACTORS
        return list(ENGINE_EXTRACTORS.keys())
    except Exception:
        return HarnessRegistry.list_engines()


__all__ = [
    # Version
    "__version__",
    # New API
    "MathAnything",
    "ExtractionResult",
    "extract",
    "extract_file",
    # Exceptions
    "MathAnythingError",
    "UnsupportedEngineError",
    "InputFileNotFoundError",
    "FileAccessError",
    "ParseError",
    "ValidationError",
    "SecurityError",
    "TierAnalysisError",
    "ConfigurationError",
    # Security
    "PathSecurityValidator",
    "FileSizeValidator",
    "validate_filepath",
    "validate_file_size",
    "default_path_validator",
    "default_size_validator",
    # Visualization
    "Visualizer",
    "to_mermaid",
    "to_graphviz",
    "save_html",
    # EML Symbolic Regression
    "NodeType",
    "Node",
    "ExprBuilder",
    "ImprovedSymbolicRegression",
    "SymbolicRegression",
    "discover_equation",
    "eml",
    # Expression Simplification
    "ExpressionSimplifier",
    "simplify",
    # Multi-variable Discovery
    "MultiVariableDiscovery",
    "discover_multivar",
    "analyze_interactions",
    # Math Proposition Generation
    "PropositionGenerator",
    "MathematicalPropositions",
    "MathematicalTask",
    "TaskType",
    # Proof Verification
    "ProofVerifier",
    "ProofReviser",
    "VerificationPipeline",
    "VerificationResult",
    "VerificationStatus",
    # Agent Architecture
    "AgentOrchestrator",
    "AgentResult",
    "BaseAgent",
    "ExtractAgent",
    "ValidateAgent",
    "CompareAgent",
    "PropositionAgent",
    "VerifyAgent",
    # Legacy API
    "load_harness",
    "list_engines",
    "MathAnythingHarness",
    "HarnessRegistry",
    "ExtractorEngine",
    "ExtractionSession",
    "MathSchema",
    "SchemaValidator",
    "MathematicalModel",
    "NumericalMethod",
    "ComputationalGraph",
    "BoundaryCondition",
    "GoverningEquation",
    # Tiered Analysis
    "TieredAnalyzer",
    "AnalysisTier",
    "TierRecommender",
    "TierRecommendation",
    "TieredAnalysisResult",
    "FileAnalysis",
    "ComplexityScore",
    "ResourceRequirements",
    "tiered_analyze",
    "analyze_file_properties",
    # Differential Geometry
    "DifferentialGeometryLayer",
    "Manifold",
    "ManifoldType",
    "MetricTensor",
    "CurvatureInfo",
    "CurvatureType",
    "SymmetryGroup",
    "SymmetryType",
    "FiberBundle",
    "GeometricStructure",
    "compute_christoffel",
    # Math Advisor
    "MathAdvisor",
    "DISCIPLINE_STATUS",
    # Analysis Tools
    "SymmetryAnalyzer",
    "SymmetryAnalysisResult",
    "TDAAnalyzer",
    "TopologyResult",
    "SpectralAnalyzer",
    "SpectralAnalysisResult",
    "DynamicsAnalyzer",
    "DynamicsAnalysisResult",
    "InteractiveVisualizer",
    "MLPotentialAnalyzer",
    "MLPotentialResult",
    "LanglandsAnalyzer",
    "LanglandsResult",
    "SINDyDiscoverer",
    "SINDyResult",
    "SandboxExecutor",
    "SandboxResult",
    "SandboxConfig",
    # Provenance & References
    "Provenance",
    "ProvenanceChain",
    "ProvenanceTracker",
    "ConstraintReference",
    "ReferenceTracker",
    # Data Flywheel
    "DataFlywheel",
    "EngineDegradation",
    "FlywheelRecord",
    "FlywheelStats",
    "FailurePattern",
    "RecordType",
    # Formal Verification
    "FormalVerifier",
    "FormalVerificationResult",
    "FormalResult",
    "FormalStatus",
    "LayerResult",
    "VerificationLayer",
    "SymbolicVerifier",
    "TypeSystemVerifier",
    "LogicVerifier",
    "LLMSemanticVerifier",
    "TypedSymbol",
    "MathType",
    "Dimension",
    # Lean4 Bridge
    "Lean4Bridge",
    "LeanVerificationResult",
    "LeanVerificationStatus",
    "get_lean4_bridge",
    "schema_to_lean",
    "verify_with_lean4",
    # Validation Toolkit
    "CrossValidationMatrix",
    "DualPerspectiveAnalyzer",
    "DualPerspectiveResult",
    "FalsifiablePrediction",
    "FalsifiablePredictionTable",
    "MethodConclusionCell",
    "PredictionStatus",
    "ValidationStatus",
    "create_cross_validation_from_schema",
    "create_prediction_table_from_schema",
    # Tool System + Agent Loop
    "MathTool",
    "PermissionResult",
    "ToolContext",
    "ToolResult",
    "build_math_tool",
    "ToolRegistry",
    "build_default_registry",
    "AgentEvent",
    "DoneEvent",
    "ErrorEvent",
    "MathAgentLoop",
    "TextDeltaEvent",
    "ToolCallStartEvent",
    "ToolProgressEvent",
    "ToolResultEvent",
]
