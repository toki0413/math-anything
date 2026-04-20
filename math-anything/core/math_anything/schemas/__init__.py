"""Math Schema v1.0 - Core schema definitions for Math Anything."""

from .math_schema import (
    MathSchema,
    SchemaValidator,
    MetaInfo,
    MathematicalModel,
    NumericalMethod,
    ComputationalGraph,
    ComputationalNode,
    ComputationalEdge,
    BoundaryCondition,
    GoverningEquation,
    MathematicalObject,
    TensorComponent,
    Discretization,
    Solver,
    ConservationProperty,
    UpdateMode,
    TensorRank,
    SymbolicConstraint,
    ParameterRelationship,
)

from .extensions import (
    SchemaExtension,
    ExtensionRegistry,
    ExtendedMathSchema,
    ExtensionMetadata,
    MLInteratomicPotentialExtension,
    PINNLossExtension,
    GraphNeuralNetworkExtension,
    get_available_extensions,
    get_extension_documentation,
    validate_with_extensions,
)

from .precision import (
    MathematicalStructure,
    VariableDependency,
    DiscretizationScheme,
    SolutionStrategy,
    Approximation,
    MLContext,
    ModelingGuidance,
    MathematicalDecoding,
    PrecisionMetadata,
    EnhancedMathSchema,
    BasePrecisionExtractor,
)

from .vasp_precision import (
    VaspMathematicalPrecisionExtractor,
    extract_vasp_mathematical_precision,
)

from .lammps_precision import (
    LammpsMathematicalPrecisionExtractor,
    extract_lammps_mathematical_precision,
)

from .abaqus_precision import (
    AbaqusMathematicalPrecisionExtractor,
    extract_abaqus_mathematical_precision,
)

from .ansys_precision import (
    AnsysMathematicalPrecisionExtractor,
    extract_ansys_mathematical_precision,
)

from .comsol_precision import (
    ComsolMathematicalPrecisionExtractor,
    extract_comsol_mathematical_precision,
)

from .gromacs_precision import (
    GromacsMathematicalPrecisionExtractor,
    extract_gromacs_mathematical_precision,
)

from .multiwfn_precision import (
    MultiwfnMathematicalPrecisionExtractor,
    extract_multiwfn_mathematical_precision,
)

__all__ = [
    # Core schema
    "MathSchema",
    "SchemaValidator",
    "MetaInfo",
    "MathematicalModel",
    "NumericalMethod",
    "ComputationalGraph",
    "ComputationalNode",
    "ComputationalEdge",
    "BoundaryCondition",
    "GoverningEquation",
    "MathematicalObject",
    "TensorComponent",
    "Discretization",
    "Solver",
    "ConservationProperty",
    "UpdateMode",
    "TensorRank",
    "SymbolicConstraint",
    "ParameterRelationship",
    # Extensions
    "SchemaExtension",
    "ExtensionRegistry",
    "ExtendedMathSchema",
    "ExtensionMetadata",
    "MLInteratomicPotentialExtension",
    "PINNLossExtension",
    "GraphNeuralNetworkExtension",
    "get_available_extensions",
    "get_extension_documentation",
    "validate_with_extensions",
    # Mathematical Precision
    "MathematicalStructure",
    "VariableDependency",
    "DiscretizationScheme",
    "SolutionStrategy",
    "Approximation",
    "MLContext",
    "ModelingGuidance",
    "MathematicalDecoding",
    "PrecisionMetadata",
    "EnhancedMathSchema",
    "BasePrecisionExtractor",
    # Engine-specific precision extractors
    "VaspMathematicalPrecisionExtractor",
    "extract_vasp_mathematical_precision",
    "LammpsMathematicalPrecisionExtractor",
    "extract_lammps_mathematical_precision",
    "AbaqusMathematicalPrecisionExtractor",
    "extract_abaqus_mathematical_precision",
    "AnsysMathematicalPrecisionExtractor",
    "extract_ansys_mathematical_precision",
    "ComsolMathematicalPrecisionExtractor",
    "extract_comsol_mathematical_precision",
    "GromacsMathematicalPrecisionExtractor",
    "extract_gromacs_mathematical_precision",
    "MultiwfnMathematicalPrecisionExtractor",
    "extract_multiwfn_mathematical_precision",
]