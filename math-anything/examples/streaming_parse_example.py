"""
Streaming Parser Example - Handling TB-scale MD Trajectories
===========================================================

This example demonstrates how to use the streaming parser to efficiently
process large molecular dynamics trajectory files without loading them
entirely into memory.

Features demonstrated:
1. Uniform sampling for statistical analysis
2. Adaptive sampling based on structural changes
3. Keyframe extraction at specific timesteps
4. Checkpoint/resume for interrupted parsing
5. Statistical feature extraction
"""

import sys
import os

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from math_anything.utils.streaming_parser import (
    StreamingParser,
    DumpSampler,
    SamplingConfig,
    SamplingStrategy,
    FileFormat,
)


def create_sample_dump_file(filepath: str, num_frames: int = 100):
    """Create a sample LAMMPS dump file for testing."""
    with open(filepath, 'w') as f:
        for frame in range(num_frames):
            timestep = frame * 100
            f.write(f"ITEM: TIMESTEP\n")
            f.write(f"{timestep}\n")
            f.write(f"ITEM: NUMBER OF ATOMS\n")
            f.write(f"10\n")
            f.write(f"ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"0.0 10.0\n")
            f.write(f"0.0 10.0\n")
            f.write(f"0.0 10.0\n")
            f.write(f"ITEM: ATOMS id type x y z vx vy vz\n")
            
            for atom_id in range(1, 11):
                # Simulate some motion
                x = 1.0 + atom_id + 0.1 * (frame % 10)
                y = 2.0 + atom_id * 0.5 + 0.05 * frame
                z = 3.0 + (atom_id % 3) * 0.3
                vx = 0.1 * atom_id
                vy = -0.05 * atom_id
                vz = 0.0
                f.write(f"{atom_id} 1 {x:.3f} {y:.3f} {z:.3f} {vx:.3f} {vy:.3f} {vz:.3f}\n")


def example_uniform_sampling():
    """Example 1: Uniform sampling for statistical analysis."""
    print("=" * 60)
    print("Example 1: Uniform Sampling")
    print("=" * 60)
    
    dump_file = "sample_trajectory.dump"
    
    # Create sample file if it doesn't exist
    if not os.path.exists(dump_file):
        print(f"Creating sample dump file: {dump_file}")
        create_sample_dump_file(dump_file, num_frames=1000)
    
    # Sample every 100 frames, max 10 frames
    print("\nSampling every 100th frame...")
    stats = DumpSampler.sample_uniform(
        filepath=dump_file,
        interval=100,
        max_frames=10,
    )
    
    print(f"Total frames in file: {stats.total_frames}")
    print(f"Sampled frames: {stats.sampled_frames}")
    print(f"Frame indices: {stats.frame_indices}")
    
    if stats.temperature_mean:
        print(f"\nTemperature statistics:")
        print(f"  Mean: {stats.temperature_mean:.2f} K")
        print(f"  Std:  {stats.temperature_std:.2f} K")
        print(f"  Min:  {stats.temperature_min:.2f} K")
        print(f"  Max:  {stats.temperature_max:.2f} K")
    
    if stats.density_mean:
        print(f"\nDensity statistics:")
        print(f"  Mean: {stats.density_mean:.6f} atoms/Å³")
        print(f"  Std:  {stats.density_std:.6f}")
    
    print("\n✓ Uniform sampling complete\n")


def example_adaptive_sampling():
    """Example 2: Adaptive sampling based on structural changes."""
    print("=" * 60)
    print("Example 2: Adaptive Sampling")
    print("=" * 60)
    
    dump_file = "sample_trajectory.dump"
    
    # Sample adaptively with 0.05 threshold
    print("\nSampling adaptively (threshold=0.05)...")
    stats = DumpSampler.sample_adaptive(
        filepath=dump_file,
        threshold=0.05,
        max_frames=20,
    )
    
    print(f"Total frames in file: {stats.total_frames}")
    print(f"Sampled frames: {stats.sampled_frames}")
    print(f"Frame indices: {stats.frame_indices}")
    
    print("\n✓ Adaptive sampling complete\n")


def example_keyframe_sampling():
    """Example 3: Extract specific timesteps (keyframes)."""
    print("=" * 60)
    print("Example 3: Keyframe Sampling")
    print("=" * 60)
    
    dump_file = "sample_trajectory.dump"
    
    # Extract specific timesteps
    keyframes = [0, 5000, 10000, 20000, 50000]
    print(f"\nExtracting keyframes at timesteps: {keyframes}")
    
    stats = DumpSampler.sample_keyframes(
        filepath=dump_file,
        keyframe_steps=keyframes,
    )
    
    print(f"Total frames in file: {stats.total_frames}")
    print(f"Sampled frames: {stats.sampled_frames}")
    print(f"Frame indices: {stats.frame_indices}")
    
    print("\n✓ Keyframe sampling complete\n")


def example_with_checkpoint():
    """Example 4: Checkpoint/resume capability."""
    print("=" * 60)
    print("Example 4: Checkpoint/Resume")
    print("=" * 60)
    
    dump_file = "sample_trajectory.dump"
    checkpoint_path = "streaming_checkpoint.pkl"
    
    # Remove existing checkpoint to simulate fresh start
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print("\nParsing with checkpointing enabled...")
    
    parser = StreamingParser()
    
    # Set progress callback
    def progress_callback(current, total):
        if current % 500 == 0:
            print(f"  Progress: {current} frames processed...")
    
    parser.set_progress_callback(progress_callback)
    
    config = SamplingConfig(
        strategy=SamplingStrategy.UNIFORM,
        interval=50,
        max_frames=10,
    )
    
    stats = parser.parse(
        filepath=dump_file,
        format=FileFormat.LAMMPS_DUMP,
        sampling=config,
        checkpoint_path=checkpoint_path,
        resume=True,
        extract_features=True,
    )
    
    print(f"\nTotal frames: {stats.total_frames}")
    print(f"Sampled frames: {stats.sampled_frames}")
    
    # Check if checkpoint was cleaned up
    if not os.path.exists(checkpoint_path):
        print("\n✓ Checkpoint cleaned up after successful completion\n")
    else:
        print(f"\n! Checkpoint file still exists: {checkpoint_path}\n")


def example_begin_middle_end():
    """Example 5: Extract beginning, middle, and end frames."""
    print("=" * 60)
    print("Example 5: Begin/Middle/End Extraction")
    print("=" * 60)
    
    dump_file = "sample_trajectory.dump"
    
    print("\nExtracting BME (begin, middle, end) frames...")
    begin, middle, end = DumpSampler.extract_begin_middle_end(dump_file)
    
    print(f"\nBegin frame:")
    print(f"  Timestep: {begin.timestep}")
    print(f"  Number of atoms: {begin.num_atoms}")
    print(f"  Box bounds: {begin.box_bounds}")
    
    print(f"\nMiddle frame:")
    print(f"  Timestep: {middle.timestep}")
    print(f"  Number of atoms: {middle.num_atoms}")
    
    print(f"\nEnd frame:")
    print(f"  Timestep: {end.timestep}")
    print(f"  Number of atoms: {end.num_atoms}")
    
    # Compute RMSD between begin and end
    positions_begin = begin.get_positions()
    positions_end = end.get_positions()
    
    import numpy as np
    rmsd = np.sqrt(np.mean((positions_end - positions_begin)**2))
    print(f"\nRMSD (begin → end): {rmsd:.4f} Å")
    
    print("\n✓ BME extraction complete\n")


def example_advanced_usage():
    """Example 6: Advanced usage with custom extractor."""
    print("=" * 60)
    print("Example 6: Advanced Custom Extraction")
    print("=" * 60)
    
    from math_anything.utils.streaming_parser import LammpsDumpExtractor
    
    dump_file = "sample_trajectory.dump"
    
    print("\nUsing low-level extractor for frame-by-frame processing...")
    
    extractor = LammpsDumpExtractor()
    extractor.open(dump_file)
    
    # Process first 5 frames
    frame_count = 0
    for frame in extractor.frames():
        if frame_count >= 5:
            break
        
        print(f"\nFrame {frame_count}:")
        print(f"  Timestep: {frame.timestep}")
        print(f"  Atoms: {frame.num_atoms}")
        
        # Get velocity statistics
        velocities = frame.get_velocities()
        if velocities is not None:
            speed = np.linalg.norm(velocities, axis=1)
            print(f"  Mean speed: {np.mean(speed):.4f} Å/fs")
        
        frame_count += 1
    
    extractor.close()
    
    print(f"\n✓ Processed {frame_count} frames\n")


def cleanup():
    """Clean up sample files."""
    files_to_remove = [
        "sample_trajectory.dump",
        "streaming_checkpoint.pkl",
    ]
    
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned up: {f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Math Anything - Streaming Parser Examples")
    print("=" * 60 + "\n")
    
    try:
        example_uniform_sampling()
        example_adaptive_sampling()
        example_keyframe_sampling()
        example_with_checkpoint()
        example_begin_middle_end()
        
        # Import numpy only for advanced example
        global np
        import numpy as np
        example_advanced_usage()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("=" * 60)
        print("Cleaning up...")
        cleanup()
        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)


if __name__ == "__main__":
    main()
