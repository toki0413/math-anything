"""
Voxel Mathematical Structure Extraction Example
==============================================

Demonstrates how Math Anything extracts mathematical semantics
from voxel-based simulations like LBM, FDTD, and Cartesian FVM.

Key concepts demonstrated:
- Voxel grid as discretization domain
- Scale mapping (index ↔ physical space)
- Boundary condition numerical implementations
- Interpolation rules for continuous reconstruction
"""

import os
import sys

import numpy as np

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "voxel-harness"))

from math_anything.extractors.lbm_boundary_extractor import \
    LBMBoundaryExtractor
from math_anything.extractors.voxel_extractor import VoxelMathExtractor
from math_anything.voxel.core.harness import VoxelHarness


def create_sample_lbm_data():
    """Create sample LBM simulation data."""
    # 50x30x20 grid with D3Q19 populations
    nx, ny, nz = 50, 30, 20

    # Voxel data: population field (nx, ny, nz, 19)
    populations = np.random.rand(nx, ny, nz, 19).astype(np.float32)
    populations = populations / populations.sum(axis=-1, keepdims=True)

    # Flag field: 0=fluid, 1=wall, 2=inlet, 3=outlet
    flags = np.zeros((nx, ny, nz), dtype=np.int32)

    # Walls on y and z boundaries
    flags[:, 0, :] = 1
    flags[:, -1, :] = 1
    flags[:, :, 0] = 1
    flags[:, :, -1] = 1

    # Inlet at x=0
    flags[0, :, :] = 2

    # Outlet at x=max
    flags[-1, :, :] = 3

    return populations, flags


def example_voxel_harness():
    """Example 1: Using VoxelHarness to extract math schema."""
    print("=" * 60)
    print("Example 1: Voxel Harness - LBM Extraction")
    print("=" * 60)

    # Create sample data
    populations, flags = create_sample_lbm_data()

    # Save to temp files
    np.save("sample_populations.npy", populations)
    np.save("sample_flags.npy", flags)

    # Use harness
    harness = VoxelHarness()

    schema = harness.extract_math(
        voxel_file="sample_populations.npy",
        flag_file="sample_flags.npy",
        physical_origin=(0.0, 0.0, 0.0),
        voxel_size=0.001,  # 1mm voxels
        simulation_type="lattice_boltzmann",
    )

    print(f"\nExtracted Schema:")
    print(f"  Engine: {schema.engine} {schema.engine_version}")
    print(f"  Governing Equations: {len(schema.governing_equations)}")
    for eq in schema.governing_equations:
        print(f"    - {eq.name}: {eq.mathematical_form[:50]}...")

    print(f"\n  Boundary Conditions: {len(schema.boundary_conditions)}")
    for bc in schema.boundary_conditions[:3]:
        print(f"    - {bc.id}: {bc.physical_meaning}")

    print(f"\n  Mathematical Objects: {len(schema.mathematical_objects)}")
    for obj in schema.mathematical_objects:
        print(f"    - {obj.name} (rank-{obj.tensor_rank})")

    print("\n✓ Voxel harness extraction complete\n")

    # Cleanup
    os.remove("sample_populations.npy")
    os.remove("sample_flags.npy")


def example_voxel_extractor():
    """Example 2: Direct use of VoxelMathExtractor."""
    print("=" * 60)
    print("Example 2: Direct Voxel Math Extraction")
    print("=" * 60)

    # Create sample data
    voxel_data = np.random.rand(32, 32, 32).astype(np.float32)

    extractor = VoxelMathExtractor()

    result = extractor.extract(
        voxel_data=voxel_data,
        physical_origin=(0.0, 0.0, 0.0),
        voxel_size=0.01,
        simulation_type="lattice_boltzmann",
    )

    print(f"\nGrid Information:")
    grid_info = result["grid_info"]
    print(f"  Dimensions: {grid_info['dimensions']}")
    print(f"  Total Voxels: {grid_info['num_voxels']:,}")
    print(f"  Physical Size: {grid_info['physical_dimensions']}")
    print(f"  Total Volume: {grid_info['total_volume']:.6f} m³")

    print(f"\nScale Mapping:")
    scale = result["scale_mapping"]
    print(f"  Type: {scale['transformation_type']}")
    print(f"  Index→Physical: {scale['mathematical_expression']['index_to_physical']}")
    print(f"  Physical→Index: {scale['mathematical_expression']['physical_to_index']}")

    print(f"\nDiscretization:")
    disc = result["discretization"]
    print(f"  Method: {disc.get('method', 'Voxel-based')}")
    print(f"  Approach: {disc.get('discretization_approach', 'finite_volume_like')}")

    print(f"\nInterpolation Rules:")
    interp = result["interpolation_rules"]
    print(f"  Available methods: {len(interp['available_methods'])}")
    for method in interp["available_methods"][:2]:
        print(f"    - {method['name']}: {method['formula'][:40]}...")

    print("\n✓ Voxel extraction complete\n")


def example_lbm_boundaries():
    """Example 3: LBM Boundary Condition Extraction."""
    print("=" * 60)
    print("Example 3: LBM Boundary Condition Analysis")
    print("=" * 60)

    # Create flag field
    nx, ny, nz = 40, 20, 20
    flags = np.zeros((nx, ny, nz), dtype=np.int32)

    # Set up boundaries
    flags[:, 0, :] = 1  # Wall at ymin
    flags[:, -1, :] = 1  # Wall at ymax
    flags[:, :, 0] = 1  # Wall at zmin
    flags[:, :, -1] = 1  # Wall at zmax
    flags[0, :, :] = 2  # Inlet at xmin
    flags[-1, :, :] = 3  # Outlet at xmax

    extractor = LBMBoundaryExtractor()

    boundaries = extractor.extract_boundaries(
        flag_field=flags,
        lattice_model="D3Q19",
        collision_model="BGK",
    )

    print(f"\nDetected {len(boundaries)} boundaries:")
    for bc in boundaries:
        print(f"\n  {bc.boundary_type.value.upper()} at {bc.location}")
        print(f"    Mathematical: {bc.mathematical_form}")
        print(f"    Accuracy: {bc.accuracy_order}st/nd order")
        print(f"    Affected directions: {len(bc.lattice_directions)}")

        if bc.implementation_details:
            print(
                f"    Implementation: {bc.implementation_details.get('description', 'N/A')}"
            )

    print("\n✓ LBM boundary analysis complete\n")


def example_llm_prompt():
    """Example 4: Generate LLM-friendly prompt from voxel data."""
    print("=" * 60)
    print("Example 4: LLM Prompt Generation")
    print("=" * 60)

    # Create sample LBM data
    voxel_data = np.random.rand(20, 20, 20, 19).astype(np.float32)

    extractor = VoxelMathExtractor()

    result = extractor.extract(
        voxel_data=voxel_data,
        physical_origin=(0.0, 0.0, 0.0),
        voxel_size=1e-4,  # 0.1mm
        simulation_type="lattice_boltzmann",
    )

    # Generate prompt
    prompt = extractor.generate_llm_prompt(result)

    print("\nGenerated LLM Prompt:")
    print("-" * 40)
    print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
    print("-" * 40)

    print("\n✓ Prompt generation complete\n")


def example_scale_mapping():
    """Example 5: Scale mapping demonstration."""
    print("=" * 60)
    print("Example 5: Index-Physical Scale Mapping")
    print("=" * 60)

    from math_anything.extractors.voxel_extractor import ScaleMapping

    # Define mapping
    mapping = ScaleMapping(
        physical_origin=(0.0, 0.0, 0.0),
        voxel_size=(0.001, 0.001, 0.001),
        physical_dimensions=(0.05, 0.03, 0.02),
        index_to_physical_matrix=np.eye(4),
        physical_to_index_matrix=np.eye(4),
    )

    # Convert indices to physical
    test_indices = [(0, 0, 0), (10, 5, 3), (50, 30, 20)]

    print("\nIndex → Physical Coordinate Mapping:")
    for i, j, k in test_indices:
        phys = mapping.index_to_physical(i, j, k)
        print(
            f"  ({i:3d}, {j:3d}, {k:3d}) → ({phys[0]:.4f}, {phys[1]:.4f}, {phys[2]:.4f}) m"
        )

    # Convert physical to index
    test_physical = [(0.0, 0.0, 0.0), (0.01, 0.005, 0.003), (0.05, 0.03, 0.02)]

    print("\nPhysical → Index Mapping:")
    for x, y, z in test_physical:
        idx = mapping.physical_to_index(x, y, z)
        print(
            f"  ({x:.4f}, {y:.4f}, {z:.4f}) m → ({idx[0]:3d}, {idx[1]:3d}, {idx[2]:3d})"
        )

    print("\n✓ Scale mapping demonstration complete\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Math Anything - Voxel Mathematical Structure Examples")
    print("=" * 60 + "\n")

    try:
        example_voxel_harness()
        example_voxel_extractor()
        example_lbm_boundaries()
        example_llm_prompt()
        example_scale_mapping()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 60)
    print("All voxel examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
