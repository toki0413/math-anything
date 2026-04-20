"""Schema Extension System - Plugin architecture for custom mathematical objects.

This module provides an extension mechanism allowing users to register
custom mathematical object types beyond the standard Math Schema.

Example:
    ```python
    from math_anything.schemas.extensions import SchemaExtension, ExtensionRegistry
    
    # Define a custom extension for machine learning potentials
    @ExtensionRegistry.register
    class MLPotentialExtension(SchemaExtension):
        name = "ml_potential"
        version = "1.0.0"
        
        def get_schema_definition(self):
            return {
                "type": "object",
                "properties": {
                    "architecture": {"type": "string"},
                    "layers": {"type": "array"},
                    "activation": {"type": "string"}
                }
            }
    
    # Use in Harness
    schema.extensions["ml_potential"] = {
        "architecture": "GAP",
        "layers": [64, 128, 64],
        "activation": "tanh"
    }
    ```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type, Callable
import json
from datetime import datetime


@dataclass
class ExtensionMetadata:
    """Metadata for a schema extension."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    compatible_schema_versions: List[str] = field(default_factory=lambda: ["1.0.0"])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "dependencies": self.dependencies,
            "compatible_schema_versions": self.compatible_schema_versions,
            "created_at": self.created_at,
        }


class SchemaExtension(ABC):
    """Base class for Math Schema extensions.
    
    Extensions allow harnesses to output custom mathematical objects
    that are not part of the standard Math Schema v1.0.
    
    Subclasses must implement:
    - name: Unique extension identifier
    - version: Extension version
    - get_schema_definition(): JSON Schema-like definition
    
    Example:
        ```python
        class PINNLossExtension(SchemaExtension):
            name = "pinn_loss"
            version = "1.0.0"
            
            def get_schema_definition(self):
                return {
                    "type": "object",
                    "properties": {
                        "residual_weight": {"type": "number"},
                        "boundary_weight": {"type": "number"},
                        "initial_weight": {"type": "number"}
                    }
                }
            
            def validate_data(self, data: Dict[str, Any]) -> bool:
                # Custom validation logic
                return all(w >= 0 for w in data.values())
        ```
    """
    
    name: str = ""
    version: str = "1.0.0"
    
    @abstractmethod
    def get_schema_definition(self) -> Dict[str, Any]:
        """Return JSON Schema-like definition for this extension.
        
        Returns:
            Dictionary describing the structure of extension data.
        """
        pass
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate extension data.
        
        Args:
            data: Extension data to validate
        
        Returns:
            True if valid
        """
        return True
    
    def get_metadata(self) -> ExtensionMetadata:
        """Get extension metadata."""
        return ExtensionMetadata(
            name=self.name,
            version=self.version,
        )
    
    def get_documentation(self) -> str:
        """Get human-readable documentation."""
        return f"Extension: {self.name} v{self.version}"


class ExtensionRegistry:
    """Registry for managing schema extensions.
    
    This registry allows dynamic discovery and loading of extensions.
    Extensions are automatically registered when decorated with @register.
    
    Example:
        ```python
        @ExtensionRegistry.register
        class MyExtension(SchemaExtension):
            name = "my_extension"
            ...
        
        # Later retrieval
        ext_class = ExtensionRegistry.get("my_extension")
        ext = ext_class()
        ```
    """
    
    _extensions: Dict[str, Type[SchemaExtension]] = {}
    _validators: Dict[str, Callable[[Any], bool]] = {}
    
    @classmethod
    def register(cls, extension_class: Type[SchemaExtension]):
        """Register an extension class.
        
        Args:
            extension_class: Extension class to register
        
        Returns:
            The registered class (for decorator use)
        """
        if not extension_class.name:
            raise ValueError("Extension must have a name")
        
        cls._extensions[extension_class.name] = extension_class
        return extension_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[SchemaExtension]]:
        """Get extension class by name.
        
        Args:
            name: Extension name
        
        Returns:
            Extension class or None
        """
        return cls._extensions.get(name)
    
    @classmethod
    def create(cls, name: str) -> Optional[SchemaExtension]:
        """Create extension instance by name.
        
        Args:
            name: Extension name
        
        Returns:
            Extension instance or None
        """
        ext_class = cls._extensions.get(name)
        if ext_class:
            return ext_class()
        return None
    
    @classmethod
    def list_extensions(cls) -> List[str]:
        """List all registered extension names."""
        return list(cls._extensions.keys())
    
    @classmethod
    def get_all_definitions(cls) -> Dict[str, Dict[str, Any]]:
        """Get schema definitions for all extensions."""
        definitions = {}
        for name, ext_class in cls._extensions.items():
            ext = ext_class()
            definitions[name] = {
                "metadata": ext.get_metadata().to_dict(),
                "schema": ext.get_schema_definition(),
            }
        return definitions
    
    @classmethod
    def validate_extension_data(cls, name: str, data: Dict[str, Any]) -> bool:
        """Validate data against extension schema.
        
        Args:
            name: Extension name
            data: Data to validate
        
        Returns:
            True if valid
        """
        ext = cls.create(name)
        if ext:
            return ext.validate_data(data)
        return False


class ExtendedMathSchema:
    """Extended Math Schema with support for custom extensions.
    
    This class wraps a standard MathSchema and adds support for
    extension fields. Extensions are validated before being added.
    
    Example:
        ```python
        from math_anything.schemas import MathSchema
        from math_anything.schemas.extensions import ExtendedMathSchema
        
        base_schema = MathSchema()
        extended = ExtendedMathSchema(base_schema)
        
        # Add extension data
        extended.add_extension("ml_potential", {
            "architecture": "GAP",
            "layers": [64, 128, 64]
        })
        
        # Convert to dict (includes extensions)
        data = extended.to_dict()
        ```
    """
    
    def __init__(self, base_schema: Any):
        """Initialize with base schema.
        
        Args:
            base_schema: Base MathSchema instance
        """
        self.base_schema = base_schema
        self.extensions: Dict[str, Dict[str, Any]] = {}
        self._extension_metadata: Dict[str, ExtensionMetadata] = {}
    
    def add_extension(self, name: str, data: Dict[str, Any], 
                      validate: bool = True) -> bool:
        """Add extension data.
        
        Args:
            name: Extension name
            data: Extension data
            validate: Whether to validate against extension schema
        
        Returns:
            True if added successfully
        
        Raises:
            ValueError: If extension not registered or validation fails
        """
        if validate:
            ext_class = ExtensionRegistry.get(name)
            if not ext_class:
                raise ValueError(f"Extension '{name}' not registered")
            
            ext = ext_class()
            if not ext.validate_data(data):
                raise ValueError(f"Extension data validation failed for '{name}'")
            
            self._extension_metadata[name] = ext.get_metadata()
        
        self.extensions[name] = data
        return True
    
    def get_extension(self, name: str) -> Optional[Dict[str, Any]]:
        """Get extension data by name.
        
        Args:
            name: Extension name
        
        Returns:
            Extension data or None
        """
        return self.extensions.get(name)
    
    def remove_extension(self, name: str) -> bool:
        """Remove extension.
        
        Args:
            name: Extension name
        
        Returns:
            True if removed
        """
        if name in self.extensions:
            del self.extensions[name]
            if name in self._extension_metadata:
                del self._extension_metadata[name]
            return True
        return False
    
    def list_extensions(self) -> List[str]:
        """List all extension names."""
        return list(self.extensions.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including extensions."""
        base_dict = self.base_schema.to_dict()
        
        if self.extensions:
            base_dict["extensions"] = {
                name: {
                    "data": data,
                    "metadata": self._extension_metadata.get(name, {}).to_dict() 
                                if name in self._extension_metadata else {},
                }
                for name, data in self.extensions.items()
            }
        
        return base_dict
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, path: str):
        """Save to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


# ── Pre-defined Extensions ───────────────────────────────────────

@ExtensionRegistry.register
class MLInteratomicPotentialExtension(SchemaExtension):
    """Extension for machine learning interatomic potentials.
    
    Captures the architecture and parameters of ML potentials like
    Gaussian Approximation Potential (GAP), SNAP, or neural networks.
    """
    
    name = "ml_interatomic_potential"
    version = "1.0.0"
    
    def get_schema_definition(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "architecture": {
                    "type": "string",
                    "enum": ["GAP", "SNAP", "NeuralNetwork", "DeepPotential", "Other"],
                    "description": "ML potential architecture type"
                },
                "descriptors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Structural descriptors used"
                },
                "cutoff_radius": {
                    "type": "number",
                    "description": "Cutoff radius in Angstroms"
                },
                "n_descriptors": {
                    "type": "integer",
                    "description": "Number of descriptor components"
                },
                "network_architecture": {
                    "type": "object",
                    "properties": {
                        "hidden_layers": {"type": "array", "items": {"type": "integer"}},
                        "activation": {"type": "string"},
                        "dropout_rate": {"type": "number"}
                    }
                },
                "training_data": {
                    "type": "object",
                    "properties": {
                        "n_configurations": {"type": "integer"},
                        "dft_package": {"type": "string"},
                        "xc_functional": {"type": "string"}
                    }
                }
            },
            "required": ["architecture"]
        }
    
    def get_documentation(self) -> str:
        return """
        Machine Learning Interatomic Potential Extension
        
        This extension captures the mathematical structure of ML potentials,
        which replace traditional analytical force fields with learned
        functions.
        
        Key mathematical aspects:
        - Descriptor space mapping: R -> D (structure to features)
        - Regression function: D -> E (features to energy)
        - Force computation: F = -∇E via automatic differentiation
        """


@ExtensionRegistry.register
class PINNLossExtension(SchemaExtension):
    """Extension for Physics-Informed Neural Network loss functions.
    
    Captures the multi-component loss structure in PINNs, including
    residual, boundary, and initial condition weights.
    """
    
    name = "pinn_loss_function"
    version = "1.0.0"
    
    def get_schema_definition(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "residual_weight": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Weight for PDE residual loss"
                },
                "boundary_weight": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Weight for boundary condition loss"
                },
                "initial_weight": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Weight for initial condition loss"
                },
                "data_weight": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Weight for data fitting loss"
                },
                "adaptive_weighting": {
                    "type": "boolean",
                    "description": "Whether to use adaptive weighting"
                },
                "weighting_scheme": {
                    "type": "string",
                    "enum": ["fixed", "gradient_statistics", "neural_tangent_kernel"]
                }
            }
        }
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Ensure weights are non-negative and sum is reasonable."""
        weight_keys = ["residual_weight", "boundary_weight", "initial_weight", "data_weight"]
        weights = [data.get(k, 0) for k in weight_keys]
        
        # All weights must be non-negative
        if any(w < 0 for w in weights):
            return False
        
        # At least one weight must be positive
        if sum(weights) == 0:
            return False
        
        return True
    
    def get_documentation(self) -> str:
        return """
        Physics-Informed Neural Network Loss Extension
        
        PINNs solve PDEs by minimizing a composite loss:
        
        L_total = w_r * L_residual + w_b * L_boundary + w_i * L_initial + w_d * L_data
        
        where:
        - L_residual = ||N[u] - f||² (PDE residual)
        - L_boundary = ||B[u] - g||² (Boundary condition)
        - L_initial = ||u(0,x) - u0||² (Initial condition)
        - L_data = ||u - u_obs||² (Data fitting)
        
        Adaptive weighting can help balance these competing objectives.
        """


@ExtensionRegistry.register
class GraphNeuralNetworkExtension(SchemaExtension):
    """Extension for Graph Neural Network structures in materials modeling.
    
    Captures the architecture of GNNs used for crystal structure prediction,
    molecular dynamics, or property prediction.
    """
    
    name = "graph_neural_network"
    version = "1.0.0"
    
    def get_schema_definition(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "gnn_type": {
                    "type": "string",
                    "enum": ["GCN", "GAT", "MPNN", "SchNet", "DimeNet", "Matformer"]
                },
                "n_layers": {"type": "integer", "minimum": 1},
                "hidden_dim": {"type": "integer", "minimum": 1},
                "edge_features": {"type": "boolean"},
                "message_passing_steps": {"type": "integer", "minimum": 1},
                "readout_function": {
                    "type": "string",
                    "enum": ["sum", "mean", "max", "attention"]
                },
                "aggregation": {
                    "type": "string",
                    "enum": ["add", "mean", "max", "softmax"]
                }
            },
            "required": ["gnn_type", "n_layers", "hidden_dim"]
        }
    
    def get_documentation(self) -> str:
        return """
        Graph Neural Network Extension for Materials
        
        GNNs operate on crystal/molecular graphs:
        - Nodes: Atoms with feature vectors
        - Edges: Bonds or neighborhood relations
        
        Message passing updates:
        h_i^(l+1) = UPDATE(h_i^(l), AGGREGATE({MESSAGE(h_i, h_j, e_ij)}))
        
        Common architectures:
        - SchNet: Continuous-filter convolution
        - DimeNet: Directional message passing
        - Matformer: Transformer for crystals
        """


# ── Extension Utilities ─────────────────────────────────────────

def get_available_extensions() -> List[str]:
    """Get list of all available extension names."""
    return ExtensionRegistry.list_extensions()


def get_extension_documentation(name: str) -> Optional[str]:
    """Get documentation for an extension.
    
    Args:
        name: Extension name
    
    Returns:
        Documentation string or None
    """
    ext = ExtensionRegistry.create(name)
    if ext:
        return ext.get_documentation()
    return None


def validate_with_extensions(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate schema data including extensions.
    
    Args:
        data: Full schema dictionary potentially with extensions
    
    Returns:
        Validation report
    """
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "extensions": {}
    }
    
    # Check if extensions present
    if "extensions" not in data:
        return report
    
    extensions = data["extensions"]
    
    for ext_name, ext_data in extensions.items():
        ext = ExtensionRegistry.create(ext_name)
        if not ext:
            report["warnings"].append(f"Unknown extension: {ext_name}")
            continue
        
        # Extract actual data (may be wrapped)
        actual_data = ext_data.get("data", ext_data)
        
        if not ext.validate_data(actual_data):
            report["errors"].append(f"Validation failed for extension: {ext_name}")
            report["valid"] = False
        else:
            report["extensions"][ext_name] = "valid"
    
    return report