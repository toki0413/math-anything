"""Test Math Anything with FEM simulation (Abaqus/Linear Elasticity).

This script demonstrates extracting mathematical structures from
a finite element input file containing beam bending analysis.
"""

import sys
from pathlib import Path

# Add core and harness to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'math-anything' / 'core'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'math-anything' / 'abaqus-harness'))

# Direct import from file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extractor",
    Path(__file__).parent.parent / 'math-anything' / 'abaqus-harness' / 'math_anything' / 'abaqus' / 'core' / 'extractor.py'
)
extractor_module = importlib.util.module_from_spec(spec)
sys.modules['extractor'] = extractor_module
spec.loader.exec_module(extractor_module)
AbaqusExtractor = extractor_module.AbaqusExtractor


def main():
    print("=" * 70)
    print("Math Anything - FEM Simulation Extraction Test")
    print("=" * 70)
    print()
    print("Testing with: Beam Bending (Linear Elasticity)")
    print("File: examples/fem_simulation/beam_bending.inp")
    print()
    
    # Initialize extractor
    extractor = AbaqusExtractor()
    print("✅ Loaded Abaqus Extractor v0.2.0")
    print()
    
    # Extract mathematical schema
    input_file = Path(__file__).parent / "fem_simulation" / "beam_bending.inp"
    
    print("🔍 Extracting mathematical structures...")
    print("-" * 70)
    
    files = {'input': str(input_file)}
    schema = extractor.extract(files)
    print()
    
    # Display results
    print("=" * 70)
    print("📊 Extraction Results")
    print("=" * 70)
    print()
    
    # Governing Equations
    equations = schema.mathematical_model.governing_equations
    print(f"🧮 Governing Equations: {len(equations)}")
    for eq in equations:
        print(f"   • {eq.name}")
        print(f"     Form: {eq.mathematical_form}")
        print(f"     Type: {eq.type}")
        print()
    
    # Parameter Relationships
    relationships = schema.mathematical_model.parameter_relationships
    print(f"📐 Parameter Relationships: {len(relationships)}")
    for rel in relationships:
        print(f"   • {rel.name}")
        print(f"     Expression: {rel.expression}")
        print(f"     Variables: {rel.variables}")
        if rel.physical_meaning:
            print(f"     Physical Meaning: {rel.physical_meaning}")
        print()
    
    # Symbolic Constraints
    constraints = schema.symbolic_constraints
    print(f"✅ Symbolic Constraints: {len(constraints)}")
    for c in constraints:
        print(f"   • {c.expression}")
        print(f"     Variables: {c.variables}")
        print(f"     Confidence: {c.confidence}")
        if c.inferred_from:
            print(f"     Source: {c.inferred_from}")
        print()
    
    # Constitutive Relations
    relations = schema.mathematical_model.constitutive_relations
    print(f"🔗 Constitutive Relations: {len(relations)}")
    for rel in relations:
        print(f"   • {rel['name']}: {rel['form']}")
        if 'parameters' in rel:
            print(f"     Parameters: {rel['parameters']}")
        print()
    
    # Numerical Method
    nm = schema.numerical_method
    if nm.solver.algorithm:
        print(f"🔢 Solver: {nm.solver.algorithm}")
        print(f"   Convergence: {nm.solver.convergence_criterion}")
        if nm.solver.tolerance:
            print(f"   Tolerance: {nm.solver.tolerance}")
    print()
    
    # Save schema
    output_file = Path(__file__).parent / "fem_simulation" / "beam_bending_schema.json"
    schema.save(str(output_file))
    print(f"💾 Schema saved to: {output_file}")
    
    print()
    print("=" * 70)
    print("✅ FEM Extraction Complete!")
    print("=" * 70)
    print()
    print("Key insights for LLM:")
    print("1. Governing PDE: ∇·σ + b = 0 (equilibrium)")
    print("2. Constitutive: σ = λ tr(ε) I + 2με (Hooke's law)")
    print("3. Material constraints: E > 0, -1 < ν < 0.5")
    print("4. Lamé relations: λ = Eν/((1+ν)(1-2ν)), μ = E/(2(1+ν))")
    print("5. FEM solves K u = F (algebraic system)")


if __name__ == "__main__":
    main()
