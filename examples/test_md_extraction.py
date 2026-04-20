"""Test Math Anything with real MD force field (Lennard-Jones).

This script demonstrates extracting mathematical structures from
a real LAMMPS input file containing Lennard-Jones potential.
"""

import sys
from pathlib import Path

# Add core and harness to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'math-anything' / 'core'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'math-anything' / 'lammps-harness'))

from math_anything.lammps.core.harness import LammpsHarness
from math_anything.schemas.math_schema import MathSchema, SymbolicConstraint


def main():
    print("=" * 70)
    print("Math Anything - Real MD Force Field Extraction Test")
    print("=" * 70)
    print()
    print("Testing with: Argon Lennard-Jones MD Simulation")
    print("File: examples/md_simulation/argon_lj.in")
    print()
    
    # Initialize harness
    harness = LammpsHarness()
    print(f"✅ Loaded harness: {harness.engine_name} v{harness.supported_schema_version}")
    print()
    
    # Extract mathematical schema
    input_file = Path(__file__).parent / "md_simulation" / "argon_lj.in"
    
    print("🔍 Extracting mathematical structures...")
    print("-" * 70)
    
    # LAMMPS harness expects a files dict
    files = {
        'input': str(input_file),
    }
    
    schema = harness.extract(files)
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
        if eq.symbolic_constraints:
            print(f"     Symbolic Constraints: {len(eq.symbolic_constraints)}")
        print()
    
    # Boundary Conditions
    bcs = schema.mathematical_model.boundary_conditions
    print(f"🔲 Boundary Conditions: {len(bcs)}")
    for bc in bcs:
        print(f"   • {bc.id}: {bc.type}")
        if bc.symbolic_constraints:
            print(f"     Constraints: {len(bc.symbolic_constraints)}")
    print()
    
    # Numerical Method
    nm = schema.numerical_method
    if nm.discretization.time_integrator:
        print(f"⏱️  Time Integrator: {nm.discretization.time_integrator}")
    if nm.discretization.time_step:
        print(f"   Time Step: {nm.discretization.time_step} fs")
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
    
    # Raw symbols
    print("=" * 70)
    print("📋 Extracted Raw Symbols")
    print("=" * 70)
    raw = schema.raw_symbols
    
    if 'pair_coeffs' in raw:
        print("\n🔵 Pair Coefficients (Force Field Parameters):")
        for i, pc in enumerate(raw['pair_coeffs']):
            print(f"   Type {i+1}: epsilon={pc.get('epsilon')}, sigma={pc.get('sigma')}, r_cut={pc.get('r_cut')}")
            # Show mathematical constraint
            if pc.get('epsilon', 0) > 0 and pc.get('sigma', 0) > 0:
                r_min = 2**(1/6) * pc['sigma']
                if pc.get('r_cut', 0) > r_min:
                    print(f"   ✅ Constraint satisfied: r_cut ({pc['r_cut']}) > 2^(1/6)*sigma ({r_min:.3f})")
    
    if 'fixes' in raw:
        print("\n🔧 Fixes (Time Integration & Constraints):")
        for fix in raw['fixes']:
            if isinstance(fix, dict):
                print(f"   • {fix.get('name', 'unknown')}: {fix.get('style', 'unknown')}")
                params = fix.get('params', [])
                if params:
                    print(f"     Parameters: {params}")
            else:
                print(f"   • {fix}")
    
    if 'timestep' in raw:
        print(f"\n⏱️  Timestep: {raw['timestep']} fs")
        # Check NVT stability constraint
        if 'fixes' in raw:
            for fix in raw['fixes']:
                if fix.get('style') == 'nvt' and 'temp' in str(fix.get('params', [])):
                    params = fix.get('params', [])
                    if len(params) >= 3:
                        tau_t = params[2]  # Third param is usually tau_T
                        dt = raw['timestep']
                        if dt < tau_t / 10:
                            print(f"   ✅ NVT stability: dt ({dt}) < tau_T/10 ({tau_t/10:.4f})")
                        else:
                            print(f"   ⚠️  NVT stability warning: dt ({dt}) >= tau_T/10 ({tau_t/10:.4f})")
    
    # Save schema
    output_file = Path(__file__).parent / "md_simulation" / "argon_lj_schema.json"
    schema.save(str(output_file))
    print()
    print(f"💾 Schema saved to: {output_file}")
    
    # LLM Context Protocol output
    print()
    print("=" * 70)
    print("🤖 LLM Context Protocol (Excerpt)")
    print("=" * 70)
    # LammpsHarness doesn't have get_llm_context, just print summary
    print(f"Engine: {harness.engine_name}")
    print(f"Schema Version: {harness.supported_schema_version}")
    print(f"Extractable Objects: {harness.list_extractable_objects()}")
    
    print()
    print("=" * 70)
    print("✅ Extraction Complete!")
    print("=" * 70)
    print()
    print("Key insights for LLM:")
    print("1. LJ potential form: U(r) = 4ε[(σ/r)¹² - (σ/r)⁶]")
    print("2. Physical constraints: ε>0, σ>0, r_cut > 2^(1/6)σ")
    print("3. NVT stability: dt < τ_T/10")
    print("4. Symplectic integrator preserves phase space volume")


if __name__ == "__main__":
    main()
