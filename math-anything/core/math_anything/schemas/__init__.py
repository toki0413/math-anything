"""Math Schema v1.0 - Core schema definitions for Math Anything."""

from .abaqus_precision import (AbaqusMathematicalPrecisionExtractor,
                               extract_abaqus_mathematical_precision)
from .ansys_precision import (AnsysMathematicalPrecisionExtractor,
                              extract_ansys_mathematical_precision)
from .comsol_precision import (ComsolMathematicalPrecisionExtractor,
                               extract_comsol_mathematical_precision)
from .extensions import (ExtendedMathSchema, ExtensionMetadata,
                         ExtensionRegistry, GraphNeuralNetworkExtension,
                         MLInteratomicPotentialExtension, PINNLossExtension,
                         SchemaExtension, get_available_extensions,
                         get_extension_documentation, validate_with_extensions)
from .gromacs_precision import (GromacsMathematicalPrecisionExtractor,
                                extract_gromacs_mathematical_precision)
from .lammps_precision import (LammpsMathematicalPrecisionExtractor,
                               extract_lammps_mathematical_precision)
from .math_schema import (BoundaryCondition, ComputationalEdge,
                          ComputationalGraph, ComputationalNode,
                          ConservationProperty, Discretization,
                          GoverningEquation, MathematicalModel,
                          MathematicalObject, MathSchema, MetaInfo,
                          NumericalMethod, ParameterRelationship,
                          SchemaValidator, Solver, SymbolicConstraint,
                          TensorComponent, TensorRank, UpdateMode)
from .multiwfn_precision import (MultiwfnMathematicalPrecisionExtractor,
                                 extract_multiwfn_mathematical_precision)
from .precision import (Approximation, BasePrecisionExtractor,
                        DiscretizationScheme, EnhancedMathSchema,
                        MathematicalDecoding, MathematicalStructure, MLContext,
                        ModelingGuidance, PrecisionMetadata, SolutionStrategy,
                        VariableDependency)
from .vasp_precision import (VaspMathematicalPrecisionExtractor,
                             extract_vasp_mathematical_precision)

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
