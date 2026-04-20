"""Example: Schema Extensions - Custom Mathematical Objects

This example demonstrates how to use the extension system to handle
custom mathematical objects like ML potentials, PINN loss functions,
and Graph Neural Networks.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from math_anything.schemas import (
    MathSchema,
    ExtendedMathSchema,
    ExtensionRegistry,
    SchemaExtension,
    MLInteratomicPotentialExtension,
    PINNLossExtension,
    GraphNeuralNetworkExtension,
    get_available_extensions,
    get_extension_documentation,
    validate_with_extensions,
)


def example_1_builtin_extensions():
    """Example 1: Using built-in extensions."""
    print("=" * 70)
    print("EXAMPLE 1: Built-in Extensions")
    print("=" * 70)
    print()
    
    # List available extensions
    print("Available built-in extensions:")
    for name in get_available_extensions():
        print(f"  - {name}")
    print()
    
    # Get documentation
    print("ML Interatomic Potential Extension Documentation:")
    doc = get_extension_documentation("ml_interatomic_potential")
    print(doc[:500] + "...\n")


def example_2_ml_potential():
    """Example 2: Adding ML potential to schema."""
    print("=" * 70)
    print("EXAMPLE 2: Machine Learning Interatomic Potential")
    print("=" * 70)
    print()
    
    # Create base schema
    base_schema = MathSchema()
    extended = ExtendedMathSchema(base_schema)
    
    # Add ML potential data
    ml_data = {
        "architecture": "GAP",
        "descriptors": ["soap", "bispectrum"],
        "cutoff_radius": 5.0,
        "n_descriptors": 100,
        "training_data": {
            "n_configurations": 10000,
            "dft_package": "VASP",
            "xc_functional": "PBE",
        }
    }
    
    extended.add_extension("ml_interatomic_potential", ml_data)
    print("✓ Added ML potential extension")
    
    # Verify
    stored = extended.get_extension("ml_interatomic_potential")
    print(f"  Architecture: {stored['architecture']}")
    print(f"  Cutoff: {stored['cutoff_radius']} Å")
    print(f"  Training configs: {stored['training_data']['n_configurations']}")
    print()
    
    # Serialize
    data = extended.to_dict()
    print("JSON Structure (extensions section):")
    import json
    print(json.dumps(data.get("extensions", {}), indent=2)[:800])
    print("\n...\n")


def example_3_pinn_loss():
    """Example 3: PINN loss function weights."""
    print("=" * 70)
    print("EXAMPLE 3: Physics-Informed Neural Network Loss")
    print("=" * 70)
    print()
    
    base_schema = MathSchema()
    extended = ExtendedMathSchema(base_schema)
    
    # Add PINN loss configuration
    pinn_data = {
        "residual_weight": 1.0,
        "boundary_weight": 10.0,
        "initial_weight": 5.0,
        "data_weight": 0.1,
        "adaptive_weighting": True,
        "weighting_scheme": "gradient_statistics",
    }
    
    extended.add_extension("pinn_loss_function", pinn_data)
    print("✓ Added PINN loss extension")
    print()
    
    # Show documentation
    print("PINN Loss Extension Info:")
    ext = PINNLossExtension()
    print(f"  Name: {ext.name}")
    print(f"  Version: {ext.version}")
    print()
    
    # Try invalid data
    print("Testing validation:")
    invalid_data = {
        "residual_weight": 0,
        "boundary_weight": 0,  # All zeros - invalid!
    }
    
    try:
        extended.add_extension("pinn_loss_function", invalid_data, validate=True)
        print("  ✗ Should have rejected invalid data")
    except ValueError as e:
        print(f"  ✓ Correctly rejected: {e}")
    print()


def example_4_gnn():
    """Example 4: Graph Neural Network structure."""
    print("=" * 70)
    print("EXAMPLE 4: Graph Neural Network for Materials")
    print("=" * 70)
    print()
    
    base_schema = MathSchema()
    extended = ExtendedMathSchema(base_schema)
    
    gnn_data = {
        "gnn_type": "SchNet",
        "n_layers": 6,
        "hidden_dim": 128,
        "edge_features": True,
        "message_passing_steps": 3,
        "readout_function": "sum",
        "aggregation": "add",
    }
    
    extended.add_extension("graph_neural_network", gnn_data)
    print("✓ Added GNN extension")
    print(f"  Type: {gnn_data['gnn_type']}")
    print(f"  Layers: {gnn_data['n_layers']}")
    print(f"  Hidden dim: {gnn_data['hidden_dim']}")
    print()


def example_5_custom_extension():
    """Example 5: Creating custom extension."""
    print("=" * 70)
    print("EXAMPLE 5: Custom Extension (Quantum Circuit)")
    print("=" * 70)
    print()
    
    # Define custom extension
    @ExtensionRegistry.register
    class QuantumCircuitExtension(SchemaExtension):
        """Extension for quantum circuits in variational quantum algorithms."""
        name = "quantum_circuit"
        version = "1.0.0"
        
        def get_schema_definition(self):
            return {
                "type": "object",
                "properties": {
                    "n_qubits": {"type": "integer", "minimum": 1},
                    "depth": {"type": "integer", "minimum": 1},
                    "ansatz_type": {
                        "type": "string",
                        "enum": ["UCCSD", "HardwareEfficient", "ADAPT-VQE"]
                    },
                    "entanglement": {
                        "type": "string",
                        "enum": ["linear", "circular", "full"]
                    },
                    "parameters": {
                        "type": "array",
                        "items": {"type": "number"}
                    }
                },
                "required": ["n_qubits", "depth", "ansatz_type"]
            }
        
        def validate_data(self, data):
            """Ensure number of parameters matches circuit size."""
            n_qubits = data.get("n_qubits", 0)
            depth = data.get("depth", 0)
            parameters = data.get("parameters", [])
            
            # Rough estimate: each layer needs ~2*n_qubits parameters
            expected_min = depth * n_qubits
            
            return len(parameters) >= expected_min
    
    print("✓ Registered custom extension: quantum_circuit")
    
    # Use the custom extension
    base_schema = MathSchema()
    extended = ExtendedMathSchema(base_schema)
    
    qc_data = {
        "n_qubits": 4,
        "depth": 3,
        "ansatz_type": "HardwareEfficient",
        "entanglement": "linear",
        "parameters": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }
    
    extended.add_extension("quantum_circuit", qc_data)
    print("✓ Added quantum circuit extension")
    print(f"  Qubits: {qc_data['n_qubits']}")
    print(f"  Depth: {qc_data['depth']}")
    print(f"  Ansatz: {qc_data['ansatz_type']}")
    print()
    
    # Verify it's in the registry
    print(f"Now available extensions include: {get_available_extensions()[-3:]}")
    print()


def example_6_validation():
    """Example 6: Validating schemas with extensions."""
    print("=" * 70)
    print("EXAMPLE 6: Validation Report")
    print("=" * 70)
    print()
    
    # Valid schema
    valid_data = {
        "schema_version": "1.0.0",
        "meta": {
            "extracted_by": "test-harness",
            "extractor_version": "1.0.0",
        },
        "extensions": {
            "ml_interatomic_potential": {
                "data": {
                    "architecture": "GAP",
                    "cutoff_radius": 5.0,
                }
            },
            "pinn_loss_function": {
                "data": {
                    "residual_weight": 1.0,
                    "boundary_weight": 0.5,
                }
            }
        }
    }
    
    report = validate_with_extensions(valid_data)
    print("Valid schema report:")
    print(f"  Valid: {report['valid']}")
    print(f"  Errors: {len(report['errors'])}")
    print(f"  Warnings: {len(report['warnings'])}")
    print(f"  Extensions: {list(report['extensions'].keys())}")
    print()
    
    # Invalid schema
    invalid_data = {
        "schema_version": "1.0.0",
        "extensions": {
            "pinn_loss_function": {
                "data": {
                    "residual_weight": -1.0,  # Invalid!
                }
            }
        }
    }
    
    report = validate_with_extensions(invalid_data)
    print("Invalid schema report:")
    print(f"  Valid: {report['valid']}")
    print(f"  Errors: {report['errors']}")
    print()


def example_7_multiple_extensions():
    """Example 7: Combining multiple extensions."""
    print("=" * 70)
    print("EXAMPLE 7: Multi-Extension Schema")
    print("=" * 70)
    print()
    
    base_schema = MathSchema()
    extended = ExtendedMathSchema(base_schema)
    
    # Add ML potential
    extended.add_extension("ml_interatomic_potential", {
        "architecture": "NeuralNetwork",
        "network_architecture": {
            "hidden_layers": [64, 128, 64],
            "activation": "tanh",
        },
        "cutoff_radius": 6.0,
    })
    
    # Add GNN
    extended.add_extension("graph_neural_network", {
        "gnn_type": "Matformer",
        "n_layers": 12,
        "hidden_dim": 256,
        "readout_function": "attention",
    })
    
    print("✓ Schema with multiple extensions:")
    for name in extended.list_extensions():
        print(f"  - {name}")
    print()
    
    # Full JSON output
    import json
    data = extended.to_dict()
    print("Full schema JSON (extensions section):")
    print(json.dumps(data["extensions"], indent=2, default=str))
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "MATH ANYTHING - SCHEMA EXTENSIONS" + " " * 20 + "║")
    print("║" + " " * 12 + "Custom Mathematical Objects Demo" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    
    example_1_builtin_extensions()
    example_2_ml_potential()
    example_3_pinn_loss()
    example_4_gnn()
    example_5_custom_extension()
    example_6_validation()
    example_7_multiple_extensions()
    
    print("=" * 70)
    print("Extensions Demo Complete!")
    print("=" * 70)
    print("""
Key Takeaways:
1. Extensions allow harnesses to output custom mathematical objects
2. Built-in extensions cover ML potentials, PINN loss, GNNs
3. Users can register their own extensions via @ExtensionRegistry.register
4. Extensions are validated before being added to schema
5. Full JSON serialization preserves extension data

Next Steps:
- Create harness-specific extensions
- Share extension definitions across teams
- Version and migrate extension schemas
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
