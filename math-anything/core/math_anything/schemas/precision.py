"""Mathematical Precision Extensions for Math Schema v1.0.

This module adds precise mathematical structure representations to Math Schema,
enabling LLMs to understand the mathematical essence of computational models.

Design Principles:
- Zero intrusion: Only observe and report, never modify user input
- Zero judgment: Describe what is, not what should be
- Mathematical precision: Express structures in canonical mathematical forms
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _filter_none(data: dict) -> dict:
    """Remove None values and empty containers from dict."""
    return {
        k: v
        for k, v in data.items()
        if v is not None and not (isinstance(v, (dict, list)) and len(v) == 0)
    }


@dataclass
class MathematicalStructure:
    """Formal representation of the mathematical structure.

    Expresses the problem in canonical mathematical form,
    enabling LLMs to understand what kind of problem is being solved.
    """

    problem_type: str
    canonical_form: str
    properties: Dict[str, Any] = field(default_factory=dict)
    dimension: Optional[int] = None
    function_space: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _filter_none(asdict(self))


@dataclass
class VariableDependency:
    """Mathematical dependency between variables.

    Expresses how variables depend on each other mathematically,
    revealing the structure that requires iterative solution.
    """

    relation: str
    depends_on: List[str]
    circular: bool = False
    mathematical_form: Optional[str] = None
    physical_interpretation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _filter_none(asdict(self))


@dataclass
class DiscretizationScheme:
    """Precise description of discretization method.

    Expresses how continuous mathematics is approximated numerically,
    enabling LLMs to understand the approximation hierarchy.
    """

    method: str
    mathematical_meaning: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    basis_type: Optional[str] = None
    completeness: Optional[str] = None
    convergence_order: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _filter_none(asdict(self))


@dataclass
class SolutionStrategy:
    """Mathematical expression of solution strategy.

    Describes how the mathematical problem is solved,
    not whether it's correct, but what it mathematically means.
    """

    method: str
    mathematical_form: str
    convergence_criterion: Optional[str] = None
    iteration_type: Optional[str] = None
    stability_requirement: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _filter_none(asdict(self))


@dataclass
class Approximation:
    """Expression of an approximation in the mathematical model.

    Describes what approximation is made and its mathematical consequence.
    Does NOT judge whether the approximation is good or bad.
    """

    name: str
    mathematical_form: str
    consequence: str
    affected_quantities: List[str] = field(default_factory=list)
    theoretical_basis: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _filter_none(asdict(self))


@dataclass
class MLContext:
    """Mathematical context for machine learning models.

    Expresses what mathematical relationship the ML model is approximating,
    enabling LLMs to understand ML as a mathematical surrogate.
    """

    input_mathematical_roles: List[Dict[str, str]] = field(default_factory=list)
    output_mathematical_definition: str = ""
    approximation_type: str = "surrogate_model"
    target_physics: Optional[str] = None
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _filter_none(asdict(self))


@dataclass
class ModelingGuidance:
    """Guidance for mathematical modeling.

    Provides information about what mathematical components are needed
    for a certain type of problem, without prescribing specific values.
    """

    problem_type: str
    required_components: List[Dict[str, Any]] = field(default_factory=list)
    optional_components: List[Dict[str, Any]] = field(default_factory=list)
    typical_choices: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _filter_none(asdict(self))


@dataclass
class MathematicalDecoding:
    """Decoding of a computational setup into mathematical meaning.

    Translates software-specific parameters into their mathematical significance,
    enabling "decryption" of black-box computational setups.
    """

    core_problem: Dict[str, Any] = field(default_factory=dict)
    approximations_applied: List[Approximation] = field(default_factory=list)
    solution_method: Dict[str, Any] = field(default_factory=dict)
    mathematical_hierarchy: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.core_problem:
            result["core_problem"] = self.core_problem
        if self.approximations_applied:
            result["approximations_applied"] = [
                a.to_dict() for a in self.approximations_applied
            ]
        if self.solution_method:
            result["solution_method"] = self.solution_method
        if self.mathematical_hierarchy:
            result["mathematical_hierarchy"] = self.mathematical_hierarchy
        return result


@dataclass
class PrecisionMetadata:
    """Metadata about the precision of extracted information.

    Expresses confidence levels and sources of information,
    enabling LLMs to understand what is certain vs inferred.
    """

    extraction_confidence: float = 1.0
    source: str = "direct_extraction"
    alternative_interpretations: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _filter_none(asdict(self))


@dataclass
class EnhancedMathSchema:
    """Enhanced Math Schema with mathematical precision.

    Extends the base Math Schema with precise mathematical structure
    representations while maintaining zero intrusion and zero judgment.

    Usage:
        This is an optional extension layer. The base Math Schema remains
        unchanged. This layer adds mathematical precision when available.
    """

    mathematical_structure: Optional[MathematicalStructure] = None
    variable_dependencies: List[VariableDependency] = field(default_factory=list)
    discretization_scheme: Optional[DiscretizationScheme] = None
    solution_strategy: Optional[SolutionStrategy] = None
    approximations: List[Approximation] = field(default_factory=list)
    ml_context: Optional[MLContext] = None
    modeling_guidance: Optional[ModelingGuidance] = None
    mathematical_decoding: Optional[MathematicalDecoding] = None
    precision_metadata: Dict[str, PrecisionMetadata] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.mathematical_structure:
            result["mathematical_structure"] = self.mathematical_structure.to_dict()
        if self.variable_dependencies:
            result["variable_dependencies"] = [
                v.to_dict() for v in self.variable_dependencies
            ]
        if self.discretization_scheme:
            result["discretization_scheme"] = self.discretization_scheme.to_dict()
        if self.solution_strategy:
            result["solution_strategy"] = self.solution_strategy.to_dict()
        if self.approximations:
            result["approximations"] = [a.to_dict() for a in self.approximations]
        if self.ml_context:
            result["ml_context"] = self.ml_context.to_dict()
        if self.modeling_guidance:
            result["modeling_guidance"] = self.modeling_guidance.to_dict()
        if self.mathematical_decoding:
            result["mathematical_decoding"] = self.mathematical_decoding.to_dict()
        if self.precision_metadata:
            result["precision_metadata"] = {
                k: v.to_dict() for k, v in self.precision_metadata.items()
            }
        return result


class BasePrecisionExtractor(ABC):
    """Base class for engine-specific precision extractors.

    Provides common interface and shared utilities.
    Subclasses implement engine-specific extraction logic.
    """

    @abstractmethod
    def extract_mathematical_structure(
        self, params: Dict[str, Any]
    ) -> MathematicalStructure:
        """Extract the mathematical structure of the problem."""
        pass

    @abstractmethod
    def extract_variable_dependencies(
        self, params: Dict[str, Any]
    ) -> List[VariableDependency]:
        """Extract variable dependencies."""
        pass

    @abstractmethod
    def extract_discretization_scheme(
        self, params: Dict[str, Any]
    ) -> DiscretizationScheme:
        """Extract discretization scheme."""
        pass

    @abstractmethod
    def extract_solution_strategy(self, params: Dict[str, Any]) -> SolutionStrategy:
        """Extract solution strategy."""
        pass

    @abstractmethod
    def extract_approximations(self, params: Dict[str, Any]) -> List[Approximation]:
        """Extract approximations made."""
        pass

    @abstractmethod
    def extract_mathematical_decoding(
        self, params: Dict[str, Any]
    ) -> MathematicalDecoding:
        """Extract complete mathematical decoding."""
        pass

    def extract_precision_metadata(
        self, params: Dict[str, Any]
    ) -> Dict[str, PrecisionMetadata]:
        """Extract precision metadata. Default implementation."""
        return {}

    def extract(self, params: Dict[str, Any]) -> EnhancedMathSchema:
        """Extract complete enhanced schema. Template method."""
        return EnhancedMathSchema(
            mathematical_structure=self.extract_mathematical_structure(params),
            variable_dependencies=self.extract_variable_dependencies(params),
            discretization_scheme=self.extract_discretization_scheme(params),
            solution_strategy=self.extract_solution_strategy(params),
            approximations=self.extract_approximations(params),
            mathematical_decoding=self.extract_mathematical_decoding(params),
            precision_metadata=self.extract_precision_metadata(params),
        )
