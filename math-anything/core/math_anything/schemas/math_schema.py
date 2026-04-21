"""Math Schema v1.0 - Structured mathematical representation for computational materials.

This module defines the core schema for extracting mathematical structures from
computational software (VASP, LAMMPS, Abaqus, etc.) into LLM-native structured data.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class UpdateMode(Enum):
    """Computational update mode - explicit vs implicit."""

    EXPLICIT_UPDATE = "explicit_update"
    IMPLICIT_LOOP = "implicit_loop"
    SYMPLECTIC_INTEGRATOR = "symplectic_integrator"


class TensorRank(Enum):
    """Tensor rank for mathematical objects."""

    SCALAR = 0
    VECTOR = 1
    MATRIX = 2
    TENSOR_3 = 3
    TENSOR_4 = 4


@dataclass
class TensorComponent:
    """A single component of a tensor with index and value."""

    index: List[int]
    value: str
    unit: str = "dimensionless"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "value": self.value,
            "unit": self.unit,
        }


@dataclass
class MathematicalObject:
    """Base class for mathematical objects with tensor support."""

    field: str
    tensor_rank: int
    tensor_form: str
    components: List[TensorComponent] = field(default_factory=list)
    symmetry: Optional[str] = None
    trace_condition: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "field": self.field,
            "tensor_rank": self.tensor_rank,
            "tensor_form": self.tensor_form,
            "components": [c.to_dict() for c in self.components],
        }
        if self.symmetry:
            result["symmetry"] = self.symmetry
        if self.trace_condition:
            result["trace_condition"] = self.trace_condition
        return result


@dataclass
class SymbolicConstraint:
    """Symbolic mathematical constraint between parameters.

    Captures parameter relationships as mathematical inequalities
    for LLM symbolic reasoning, not just numerical comparison.

    Example:
        expression: "tau > 0.5"
        variables: ["tau"]
        inferred_from: "source code line 123"
    """

    expression: str
    description: str = ""
    variables: List[str] = field(default_factory=list)
    inferred_from: Optional[str] = None
    confidence: float = 0.5  # Auto-generation confidence

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "expression": self.expression,
            "description": self.description,
            "variables": self.variables,
            "confidence": self.confidence,
        }
        if self.inferred_from:
            result["inferred_from"] = self.inferred_from
        return result


@dataclass
class ParameterRelationship:
    """Relationship between parameters in mathematical form.

    Captures symbolic relationships like:
    - "dt < dx^2 / (2*D)" (CFL condition)
    - "mu = E / (2*(1+nu))" (Elastic modulus relationship)
    """

    name: str
    expression: str
    variables: List[str]
    relation_type: str = "inequality"  # equality, inequality, bound
    description: str = ""
    physical_meaning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "expression": self.expression,
            "variables": self.variables,
            "relation_type": self.relation_type,
            "description": self.description,
        }
        if self.physical_meaning:
            result["physical_meaning"] = self.physical_meaning
        return result


@dataclass
class GoverningEquation:
    """A governing equation in the mathematical model."""

    id: str
    type: str
    name: str
    mathematical_form: str
    variables: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    symbolic_constraints: List[SymbolicConstraint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "mathematical_form": self.mathematical_form,
            "variables": self.variables,
            "parameters": self.parameters,
            "description": self.description,
        }
        if self.symbolic_constraints:
            result["symbolic_constraints"] = [
                c.to_dict() for c in self.symbolic_constraints
            ]
        return result

    def add_constraint(self, constraint: SymbolicConstraint):
        """Add a symbolic constraint to this equation."""
        self.symbolic_constraints.append(constraint)


@dataclass
class BoundaryCondition:
    """Boundary condition with tensor-complete mathematical expression."""

    id: str
    type: str
    domain: Dict[str, Any]
    mathematical_object: MathematicalObject
    software_implementation: Dict[str, Any]
    dual_role: Optional[Dict[str, Any]] = None
    equivalent_formulations: List[Dict[str, str]] = field(default_factory=list)
    symbolic_constraints: List[SymbolicConstraint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "type": self.type,
            "domain": self.domain,
            "mathematical_object": self.mathematical_object.to_dict(),
            "software_implementation": self.software_implementation,
        }
        if self.dual_role:
            result["dual_role"] = self.dual_role
        if self.equivalent_formulations:
            result["equivalent_formulations"] = self.equivalent_formulations
        if self.symbolic_constraints:
            result["symbolic_constraints"] = [
                c.to_dict() for c in self.symbolic_constraints
            ]
        return result

    def add_constraint(self, constraint: SymbolicConstraint):
        """Add a symbolic constraint to this BC."""
        self.symbolic_constraints.append(constraint)


@dataclass
class ComputationalNode:
    """A node in the computational graph."""

    id: str
    type: str
    math_semantics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "math_semantics": self.math_semantics,
        }


@dataclass
class ComputationalEdge:
    """An edge in the computational graph representing data flow."""

    from_node: str
    to_node: str
    data_type: str
    dependency: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.from_node,
            "to": self.to_node,
            "data_type": self.data_type,
            "dependency": self.dependency,
        }


@dataclass
class ComputationalGraph:
    """Computational graph with explicit/implicit loop distinction."""

    version: str = "1.0"
    nodes: List[ComputationalNode] = field(default_factory=list)
    edges: List[ComputationalEdge] = field(default_factory=list)
    execution_topology: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "execution_topology": self.execution_topology,
        }

    def add_node(self, node: ComputationalNode):
        self.nodes.append(node)

    def add_edge(self, edge: ComputationalEdge):
        self.edges.append(edge)


@dataclass
class Discretization:
    """Discretization method for numerical solution."""

    time_integrator: Optional[str] = None
    space_discretization: Optional[str] = None
    time_step: Optional[float] = None
    order: Optional[int] = None
    stability_condition: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.time_integrator is not None:
            result["time_integrator"] = self.time_integrator
        if self.space_discretization is not None:
            result["space_discretization"] = self.space_discretization
        if self.time_step is not None:
            result["time_step"] = self.time_step
        if self.order is not None:
            result["order"] = self.order
        if self.stability_condition is not None:
            result["stability_condition"] = self.stability_condition
        return result


@dataclass
class Solver:
    """Algebraic solver configuration."""

    algorithm: Optional[str] = None
    convergence_criterion: Optional[str] = None
    tolerance: Optional[float] = None
    max_iterations: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.algorithm is not None:
            result["algorithm"] = self.algorithm
        if self.convergence_criterion is not None:
            result["convergence_criterion"] = self.convergence_criterion
        if self.tolerance is not None:
            result["tolerance"] = self.tolerance
        if self.max_iterations is not None:
            result["max_iterations"] = self.max_iterations
        return result


@dataclass
class NumericalMethod:
    """Numerical method configuration."""

    discretization: Discretization = field(default_factory=Discretization)
    solver: Solver = field(default_factory=Solver)
    parallelization: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discretization": self.discretization.to_dict(),
            "solver": self.solver.to_dict(),
            "parallelization": self.parallelization,
        }


@dataclass
class ConservationProperty:
    """Conservation property of the numerical scheme."""

    quantity: str
    preserved: bool
    mechanism: Optional[str] = None
    error_bound: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "quantity": self.quantity,
            "preserved": self.preserved,
        }
        if self.mechanism:
            result["mechanism"] = self.mechanism
        if self.error_bound:
            result["error_bound"] = self.error_bound
        return result


@dataclass
class MathematicalModel:
    """Complete mathematical model description."""

    governing_equations: List[GoverningEquation] = field(default_factory=list)
    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)
    initial_conditions: List[Dict[str, Any]] = field(default_factory=list)
    constitutive_relations: List[Dict[str, Any]] = field(default_factory=list)
    coupling_conditions: List[Dict[str, Any]] = field(default_factory=list)
    parameter_relationships: List[ParameterRelationship] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "governing_equations": [e.to_dict() for e in self.governing_equations],
            "boundary_conditions": [bc.to_dict() for bc in self.boundary_conditions],
            "initial_conditions": self.initial_conditions,
            "constitutive_relations": self.constitutive_relations,
            "coupling_conditions": self.coupling_conditions,
            "parameter_relationships": [
                pr.to_dict() for pr in self.parameter_relationships
            ],
        }

    def add_parameter_relationship(self, relationship: ParameterRelationship):
        """Add a parameter relationship to the model."""
        self.parameter_relationships.append(relationship)


@dataclass
class MetaInfo:
    """Metadata for the extracted mathematical model."""

    extracted_by: str
    extractor_version: str
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_files: Dict[str, List[str]] = field(default_factory=dict)
    explicit_material_context: Optional[str] = None
    material_context_declaration: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.material_context_declaration:
            self.material_context_declaration = {
                "contains_material_specific_parameters": False,
                "contains_material_specific_geometry": False,
                "contains_material_specific_initial_state": False,
                "extraction_scope": "mathematical_structure_only",
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "extracted_by": self.extracted_by,
            "extractor_version": self.extractor_version,
            "extracted_at": self.extracted_at,
            "source_files": self.source_files,
            "explicit_material_context": self.explicit_material_context,
            "material_context_declaration": self.material_context_declaration,
        }


@dataclass
class MathSchema:
    """Complete Math Schema v1.0 representation."""

    schema_version: str = "1.0.0"
    meta: MetaInfo = field(default=None)
    mathematical_model: MathematicalModel = field(default_factory=MathematicalModel)
    numerical_method: NumericalMethod = field(default_factory=NumericalMethod)
    conservation_properties: Dict[str, Any] = field(default_factory=dict)
    computational_graph: ComputationalGraph = field(default_factory=ComputationalGraph)
    raw_symbols: Dict[str, Any] = field(default_factory=dict)
    symbolic_constraints: List[SymbolicConstraint] = field(default_factory=list)

    def __post_init__(self):
        if self.meta is None:
            self.meta = MetaInfo(
                extracted_by="math-anything-core",
                extractor_version="0.1.0",
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "schema_version": self.schema_version,
            "meta": self.meta.to_dict(),
            "mathematical_model": self.mathematical_model.to_dict(),
            "numerical_method": self.numerical_method.to_dict(),
            "conservation_properties": self.conservation_properties,
            "computational_graph": self.computational_graph.to_dict(),
            "raw_symbols": self.raw_symbols,
            "symbolic_constraints": [c.to_dict() for c in self.symbolic_constraints],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str):
        """Save to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def add_symbolic_constraint(self, constraint: SymbolicConstraint):
        """Add a symbolic constraint to the schema."""
        self.symbolic_constraints.append(constraint)

    def add_parameter_relationship(self, relationship: ParameterRelationship):
        """Add a parameter relationship to the model."""
        self.mathematical_model.add_parameter_relationship(relationship)

    def add_governing_equation(self, equation: GoverningEquation):
        """Add a governing equation."""
        self.mathematical_model.governing_equations.append(equation)

    def add_boundary_condition(self, bc: BoundaryCondition):
        """Add a boundary condition."""
        self.mathematical_model.boundary_conditions.append(bc)

    def add_numerical_method(self, method: NumericalMethod):
        """Add or update numerical method."""
        self.numerical_method = method

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MathSchema":
        """Create MathSchema from dictionary."""
        meta = MetaInfo(**data.get("meta", {}))

        model_data = data.get("mathematical_model", {})
        equations = [
            GoverningEquation(**e) for e in model_data.get("governing_equations", [])
        ]
        boundaries = []
        for bc_data in model_data.get("boundary_conditions", []):
            mo_data = bc_data.get("mathematical_object", {})
            components = [TensorComponent(**c) for c in mo_data.get("components", [])]
            mo = MathematicalObject(
                field=mo_data.get("field", ""),
                tensor_rank=mo_data.get("tensor_rank", 0),
                tensor_form=mo_data.get("tensor_form", ""),
                components=components,
                symmetry=mo_data.get("symmetry"),
                trace_condition=mo_data.get("trace_condition"),
            )
            bc = BoundaryCondition(
                id=bc_data.get("id", ""),
                type=bc_data.get("type", ""),
                domain=bc_data.get("domain", {}),
                mathematical_object=mo,
                software_implementation=bc_data.get("software_implementation", {}),
                dual_role=bc_data.get("dual_role"),
                equivalent_formulations=bc_data.get("equivalent_formulations", []),
            )
            boundaries.append(bc)

        # Parse parameter relationships
        param_relationships = [
            ParameterRelationship(**pr)
            for pr in model_data.get("parameter_relationships", [])
        ]

        math_model = MathematicalModel(
            governing_equations=equations,
            boundary_conditions=boundaries,
            initial_conditions=model_data.get("initial_conditions", []),
            constitutive_relations=model_data.get("constitutive_relations", []),
            coupling_conditions=model_data.get("coupling_conditions", []),
            parameter_relationships=param_relationships,
        )

        num_data = data.get("numerical_method", {})
        disc_data = num_data.get("discretization", {})
        discretization = Discretization(**disc_data)
        solver_data = num_data.get("solver", {})
        solver = Solver(**solver_data)
        numerical_method = NumericalMethod(
            discretization=discretization,
            solver=solver,
            parallelization=num_data.get("parallelization", {}),
        )

        cg_data = data.get("computational_graph", {})
        nodes = [ComputationalNode(**n) for n in cg_data.get("nodes", [])]

        # Handle edge dict mapping (from->from_node, to->to_node)
        edges = []
        for e in cg_data.get("edges", []):
            edge = ComputationalEdge(
                from_node=e.get("from", e.get("from_node", "")),
                to_node=e.get("to", e.get("to_node", "")),
                data_type=e.get("data_type", ""),
                dependency=e.get("dependency", ""),
            )
            edges.append(edge)

        computational_graph = ComputationalGraph(
            version=cg_data.get("version", "1.0"),
            nodes=nodes,
            edges=edges,
            execution_topology=cg_data.get("execution_topology", {}),
        )

        # Parse symbolic constraints
        constraints = [
            SymbolicConstraint(**c) for c in data.get("symbolic_constraints", [])
        ]

        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            meta=meta,
            mathematical_model=math_model,
            numerical_method=numerical_method,
            conservation_properties=data.get("conservation_properties", {}),
            computational_graph=computational_graph,
            raw_symbols=data.get("raw_symbols", {}),
            symbolic_constraints=constraints,
        )


class SchemaValidator:
    """Validator for Math Schema compliance."""

    REQUIRED_TOP_LEVEL_KEYS = [
        "schema_version",
        "meta",
        "mathematical_model",
        "numerical_method",
        "computational_graph",
    ]

    REQUIRED_META_KEYS = [
        "extracted_by",
        "extractor_version",
        "extracted_at",
    ]

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate a dictionary against Math Schema v1.0."""
        self.errors = []
        self.warnings = []

        # Check top-level keys
        for key in self.REQUIRED_TOP_LEVEL_KEYS:
            if key not in data:
                self.errors.append(f"Missing required key: {key}")

        if "meta" in data:
            meta = data["meta"]
            for key in self.REQUIRED_META_KEYS:
                if key not in meta:
                    self.errors.append(f"Missing required meta key: {key}")

        # Check schema version
        if "schema_version" in data:
            version = data["schema_version"]
            if not version.startswith("1."):
                self.warnings.append(
                    f"Schema version {version} may not be fully supported"
                )

        # Validate computational graph has explicit/implicit distinction
        if "computational_graph" in data:
            cg = data["computational_graph"]
            if "nodes" in cg:
                for i, node in enumerate(cg["nodes"]):
                    if "math_semantics" in node:
                        semantics = node["math_semantics"]
                        if "updates" in semantics:
                            if "mode" not in semantics["updates"]:
                                self.warnings.append(
                                    f"Node {node.get('id', i)} missing update mode (explicit/implicit)"
                                )

        return len(self.errors) == 0

    def validate_file(self, path: str) -> bool:
        """Validate a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self.validate(data)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON: {e}")
            return False
        except FileNotFoundError:
            self.errors.append(f"File not found: {path}")
            return False
