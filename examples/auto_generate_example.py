"""Example: Auto-generate Harness from LAMMPS-like source code.

Demonstrates how Math Anything's Harness Auto-Generator can:
1. Analyze C++ source code for command patterns
2. Extract parameter constraints from code
3. Map commands to mathematical semantics
4. Generate symbolic constraint expressions
5. Output ready-to-review Harness skeleton

This follows the "zero-intrusion, zero-judgment" principle:
- No manual regex writing required
- Automatic extraction from existing source
- Developer only reviews and微调
"""

import os
import sys
import tempfile
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'math-anything' / 'core'))

from math_anything.codegen import HarnessAutoGenerator
from math_anything.schemas.math_schema import (
    MathSchema, SymbolicConstraint, ParameterRelationship
)


def create_mock_lammps_source(temp_dir: str):
    """Create mock LAMMPS source files for demonstration."""
    
    # Create src directory structure
    src_dir = Path(temp_dir) / "src"
    src_dir.mkdir()
    
    # Create fix_nvt.cpp (simplified)
    fix_nvt_cpp = """
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   Fix NVT - Nose-Hoover thermostat
   
   Governing equation: dT/dt = (T_target - T) / tau_T
   Constraint: tau_T > 0 (thermostat damping must be positive)
------------------------------------------------------------------------- */

#include "fix_nvt.h"
#include <cmath>

FixNVT::FixNVT(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix nvt command");
  
  // Parse temperature parameters
  double t_start = atof(arg[3]);  // Initial temperature
  double t_stop = atof(arg[4]);   // Final temperature
  double t_damp = atof(arg[5]);   // Thermostat damping parameter
  
  // Validate constraints
  if (t_start <= 0) error->all(FLERR,"Temperature must be positive");
  if (t_stop <= 0) error->all(FLERR,"Temperature must be positive");
  if (t_damp <= 0) error->all(FLERR,"Thermostat damping must be positive");
  
  // CFL-like stability: dt must be less than tau/10 for stability
  double dt = update->dt;
  if (dt > t_damp / 10.0) {
    error->warning(FLERR,"Time step may be too large for NVT stability");
  }
  
  t_start_ = t_start;
  t_stop_ = t_stop;
  t_damp_ = t_damp;
}

void FixNVT::initial_integrate(int vflag)
{
  // Nose-Hoover thermostat integration
  // T_current = compute current temperature
  // T_target = t_start_ + (t_stop_ - t_start_) * fraction
  // dT_scale = 1.0 + (update->dt / t_damp_) * (T_target/T_current - 1.0)
  // v[i][0] *= dT_scale; v[i][1] *= dT_scale; v[i][2] *= dT_scale;
}
"""
    
    # Create fix_deform.cpp (simplified)
    fix_deform_cpp = """
/* ----------------------------------------------------------------------
   Fix Deform - Deform simulation box
   
   Mathematical form: epsilon = deformation rate * dt
   Constraint: strain rate < 0.1/dt (numerical stability)
   Tensor form: epsilon_ij = 1/2(du_i/dx_j + du_j/dx_i)
------------------------------------------------------------------------- */

#include "fix_deform.h"

FixDeform::FixDeform(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix deform command");
  
  // Parse deformation parameters
  int dim = atoi(arg[3]);  // 0=x, 1=y, 2=z
  char *style = arg[4];    // "erate", "trate", "volume", etc.
  double rate = atof(arg[5]);  // Deformation rate
  
  // Validate
  if (dim < 0 || dim > 2) error->all(FLERR,"Invalid dimension");
  if (rate > 0.1 / update->dt) {
    error->warning(FLERR,"Strain rate may be too large");
  }
  
  // Tensor boundary condition: epsilon_xx = rate * dt
  // Symmetric tensor: epsilon_ij = epsilon_ji
  strain_rate_ = rate;
  dim_ = dim;
}

void FixDeform::end_of_step()
{
  // Update box dimensions
  // L_new = L_old * (1.0 + strain_rate_ * update->dt)
  // Volume preservation: if volume style, Lx*Ly*Lz = constant
}
"""
    
    # Create pair_lj_cut.cpp (simplified)
    pair_lj_cpp = """
/* ----------------------------------------------------------------------
   Pair LJ Cut - Lennard-Jones 12-6 potential
   
   Mathematical form: 
   U(r) = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)
   F(r) = 24*epsilon*(2*(sigma/r)^12 - (sigma/r)^6)/r
   
   Constraints:
   - epsilon > 0 (energy well depth must be positive)
   - sigma > 0 (particle size must be positive)
   - r_cut > 0 (cutoff must be positive)
   - r_cut > 2^(1/6)*sigma for attractive tail
------------------------------------------------------------------------- */

#include "pair_lj_cut.h"

PairLJCut::PairLJCut(LAMMPS *lmp) : Pair(lmp) {}

void PairLJCut::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style lj/cut command");
  
  cut_global = atof(arg[0]);
  if (cut_global <= 0) error->all(FLERR,"Cutoff must be positive");
}

void PairLJCut::coeff(int narg, char **arg)
{
  if (narg < 4) error->all(FLERR,"Illegal pair_coeff command");
  
  double epsilon = atof(arg[2]);  // Energy well depth
  double sigma = atof(arg[3]);    // Length scale
  double r_cut = narg > 4 ? atof(arg[4]) : cut_global;
  
  // Physical constraints
  if (epsilon <= 0) error->all(FLERR,"Epsilon must be positive");
  if (sigma <= 0) error->all(FLERR,"Sigma must be positive");
  if (r_cut <= 0) error->all(FLERR,"Cutoff must be positive");
  
  // Constraint: r_cut should be > 2^(1/6) * sigma for attractive region
  double r_min = pow(2.0, 1.0/6.0) * sigma;
  if (r_cut < r_min) {
    error->warning(FLERR,"Cutoff less than minimum energy distance");
  }
  
  epsilon_[i][j] = epsilon;
  sigma_[i][j] = sigma;
  cut_[i][j] = r_cut;
}

double PairLJCut::compute_single(double r, int i, int j)
{
  double r2inv = 1.0/(r*r);
  double r6inv = r2inv*r2inv*r2inv;
  double sig6 = pow(sigma_[i][j], 6.0);
  
  // U = 4*eps*(sig^12/r^12 - sig^6/r^6)
  double lj1 = 4.0 * epsilon_[i][j] * sig6 * sig6;
  double lj2 = 4.0 * epsilon_[i][j] * sig6;
  
  return r6inv * (lj1*r6inv - lj2);
}
"""
    
    # Write files
    (src_dir / "fix_nvt.cpp").write_text(fix_nvt_cpp)
    (src_dir / "fix_deform.cpp").write_text(fix_deform_cpp)
    (src_dir / "pair_lj_cut.cpp").write_text(pair_lj_cpp)
    
    return str(src_dir)


def main():
    """Run auto-generation example."""
    
    print("=" * 70)
    print("Math Anything - Harness Auto-Generator Example")
    print("=" * 70)
    print()
    print("This example demonstrates automatic Harness generation from")
    print("simulation software source code (LAMMPS-like).")
    print()
    
    # Create mock source
    with tempfile.TemporaryDirectory() as temp_dir:
        print("📁 Creating mock LAMMPS source files...")
        src_dir = create_mock_lammps_source(temp_dir)
        print(f"   Created: {src_dir}")
        print()
        
        # Initialize generator
        print("🚀 Initializing Harness Auto-Generator...")
        generator = HarnessAutoGenerator()
        print()
        
        # Generate from source
        print("🔍 Analyzing source code...")
        template = generator.generate_from_source(
            source_dir=src_dir,
            engine_name="lammps",
            engine_version="2023.08.02",
            file_patterns=["*.cpp"],
            entry_point_hints=["fix", "pair", "compute", "dump"],
        )
        print()
        
        # Display results
        print("=" * 70)
        print("📊 Analysis Results")
        print("=" * 70)
        print()
        
        print(f"Engine: {template.engine_name} v{template.engine_version}")
        print(f"Source: {template.source_path}")
        print()
        
        print(f"📋 Extracted Commands: {len(template.extracted_commands)}")
        for cmd in template.extracted_commands:
            print(f"   • {cmd['name']}")
            print(f"     Pattern: {cmd['pattern']}")
            print(f"     Params: {[p['name'] for p in cmd['parameters']]}")
            print()
        
        print(f"🧮 Mathematical Mappings:")
        print(f"   Equations: {len(template.mathematical_mappings.get('equations', []))}")
        for eq in template.mathematical_mappings.get('equations', [])[:3]:
            print(f"   • {eq.get('name', 'Unknown')}: {eq.get('form', 'N/A')[:60]}...")
        print()
        
        print(f"📐 Symbolic Constraints: {len(template.constraint_expressions)}")
        for constraint in template.constraint_expressions[:5]:
            print(f"   • {constraint}")
        print()
        
        # Save generated harness
        output_dir = Path(__file__).parent / "generated_harness"
        print(f"💾 Saving generated Harness to: {output_dir}")
        generator.save_harness(template, str(output_dir))
        print()
        
        # Show review checklist
        print("=" * 70)
        print("⚠️  Review Checklist (Auto-Generated)")
        print("=" * 70)
        for item in template.review_checklist[:10]:
            print(f"   {item}")
        if len(template.review_checklist) > 10:
            print(f"   ... and {len(template.review_checklist) - 10} more items")
        print()
        
        # Demonstrate symbolic constraint extraction
        print("=" * 70)
        print("🔬 Symbolic Constraint Extraction Demo")
        print("=" * 70)
        print()
        
        # Example: Extract constraints from code snippet
        from math_anything.codegen import extract_symbolic_constraints
        
        code_snippet = """
        if (t_damp <= 0) error->all(FLERR,"Thermostat damping must be positive");
        if (dt > t_damp / 10.0) error->warning(FLERR,"Time step too large");
        """
        
        constraints = extract_symbolic_constraints(
            code_snippet,
            parameters=["t_damp", "dt"],
        )
        
        print("Code snippet:")
        print(code_snippet)
        print()
        print("Extracted constraints:")
        for c in constraints:
            print(f"   • {c.expression}")
            print(f"     Variables: {c.variables}")
            print(f"     Confidence: {c.confidence}")
        print()
        
        # Build a schema with symbolic constraints
        print("=" * 70)
        print("📐 Building MathSchema with Symbolic Constraints")
        print("=" * 70)
        print()
        
        schema = MathSchema(
            schema_version="1.0.0",
            meta=None,  # Will be auto-created
        )
        
        # Add extracted constraints
        from math_anything.schemas.math_schema import (
            GoverningEquation, BoundaryCondition, MathematicalObject,
            TensorComponent, NumericalMethod, Discretization
        )
        
        # Add governing equation with constraints
        eq = GoverningEquation(
            id="nvt_thermostat",
            type="thermostat",
            name="Nose-Hoover Thermostat",
            mathematical_form="dT/dt = (T_target - T) / tau_T",
            variables=["T", "T_target", "tau_T", "t"],
        )
        
        # Add symbolic constraints to equation
        eq.add_constraint(SymbolicConstraint(
            expression="tau_T > 0",
            description="Thermostat damping must be positive",
            variables=["tau_T"],
            confidence=1.0,
            inferred_from="fix_nvt.cpp:38",
        ))
        
        eq.add_constraint(SymbolicConstraint(
            expression="dt < tau_T / 10",
            description="Stability constraint for NVT integration",
            variables=["dt", "tau_T"],
            confidence=0.8,
            inferred_from="fix_nvt.cpp:42",
        ))
        
        schema.add_governing_equation(eq)
        
        # Add parameter relationship
        schema.add_parameter_relationship(ParameterRelationship(
            name="LJ_potential",
            expression="U(r) = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)",
            variables=["U", "r", "epsilon", "sigma"],
            relation_type="equality",
            description="Lennard-Jones 12-6 potential",
            physical_meaning="Van der Waals interaction energy",
        ))
        
        # Add another constraint
        schema.add_symbolic_constraint(SymbolicConstraint(
            expression="r_cut > 2^(1/6) * sigma",
            description="Cutoff must include attractive region",
            variables=["r_cut", "sigma"],
            confidence=0.9,
            inferred_from="pair_lj_cut.cpp:55",
        ))
        
        print("Schema Summary:")
        print(f"   Equations: {len(schema.mathematical_model.governing_equations)}")
        print(f"   Symbolic Constraints: {len(schema.symbolic_constraints)}")
        print(f"   Parameter Relationships: {len(schema.mathematical_model.parameter_relationships)}")
        print()
        
        # Save schema as example
        schema_path = output_dir / "example_schema_with_constraints.json"
        schema.save(str(schema_path))
        print(f"   Saved example schema: {schema_path}")
        print()
        
        # Show schema JSON preview
        print("Schema JSON Preview:")
        print("-" * 40)
        json_str = schema.to_json(indent=2)
        print(json_str[:2000] + "..." if len(json_str) > 2000 else json_str)
        print()
        
        print("=" * 70)
        print("✅ Example Complete!")
        print("=" * 70)
        print()
        print("Generated files:")
        print(f"   {output_dir / 'lammps-harness'}")
        print()
        print("Next steps:")
        print("   1. Review ANALYSIS_SUMMARY.md")
        print("   2. Complete REVIEW_CHECKLIST.md")
        print("   3. Implement TODO sections in harness.py")
        print("   4. Test with real input files")
        print("   5. Uncomment HarnessRegistry.register() after validation")
        print()


if __name__ == "__main__":
    main()
