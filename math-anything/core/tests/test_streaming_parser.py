"""Tests for streaming parser module."""

import gzip
import os
import pickle
import sys
import tempfile
import unittest

import numpy as np

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from math_anything.utils.streaming_parser import (
    AtomData,
    Checkpoint,
    DumpSampler,
    FileFormat,
    FrameData,
    LammpsDumpExtractor,
    SamplingConfig,
    SamplingStrategy,
    StreamingParser,
    TrajectoryStats,
)


class TestLammpsDumpExtractor(unittest.TestCase):
    """Test LAMMPS dump file extraction."""

    def setUp(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.dump_file = os.path.join(self.temp_dir, "test.dump")

        # Create sample dump file
        with open(self.dump_file, "w") as f:
            # Frame 1
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write("5\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write("0.0 10.0\n")
            f.write("0.0 10.0\n")
            f.write("0.0 10.0\n")
            f.write("ITEM: ATOMS id type x y z vx vy vz\n")
            f.write("1 1 1.0 2.0 3.0 0.1 0.2 0.3\n")
            f.write("2 1 2.0 3.0 4.0 0.2 0.3 0.4\n")
            f.write("3 2 3.0 4.0 5.0 0.3 0.4 0.5\n")
            f.write("4 2 4.0 5.0 6.0 0.4 0.5 0.6\n")
            f.write("5 1 5.0 6.0 7.0 0.5 0.6 0.7\n")

            # Frame 2
            f.write("ITEM: TIMESTEP\n")
            f.write("100\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write("5\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write("0.0 10.0\n")
            f.write("0.0 10.0\n")
            f.write("0.0 10.0\n")
            f.write("ITEM: ATOMS id type x y z vx vy vz\n")
            f.write("1 1 1.1 2.1 3.1 0.1 0.2 0.3\n")
            f.write("2 1 2.1 3.1 4.1 0.2 0.3 0.4\n")
            f.write("3 2 3.1 4.1 5.1 0.3 0.4 0.5\n")
            f.write("4 2 4.1 5.1 6.1 0.4 0.5 0.6\n")
            f.write("5 1 5.1 6.1 7.1 0.5 0.6 0.7\n")

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_extractor_open_close(self):
        """Test opening and closing extractor."""
        extractor = LammpsDumpExtractor()
        extractor.open(self.dump_file)
        self.assertIsNotNone(extractor.file_handle)
        extractor.close()
        self.assertIsNone(extractor.file_handle)

    def test_read_frame(self):
        """Test reading a single frame."""
        extractor = LammpsDumpExtractor()
        extractor.open(self.dump_file)

        frame = extractor._read_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.timestep, 0)
        self.assertEqual(frame.num_atoms, 5)
        self.assertEqual(len(frame.atoms), 5)

        # Check box bounds
        self.assertEqual(frame.box_bounds, [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)])

        extractor.close()

    def test_frame_iteration(self):
        """Test iterating over all frames."""
        extractor = LammpsDumpExtractor()
        extractor.open(self.dump_file)

        frames = list(extractor.frames())
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].timestep, 0)
        self.assertEqual(frames[1].timestep, 100)

        extractor.close()

    def test_atom_data(self):
        """Test atom data parsing."""
        extractor = LammpsDumpExtractor()
        extractor.open(self.dump_file)

        frame = extractor._read_frame()
        atom = frame.atoms[0]

        self.assertEqual(atom.id, 1)
        self.assertEqual(atom.type, 1)
        self.assertAlmostEqual(atom.x, 1.0)
        self.assertAlmostEqual(atom.y, 2.0)
        self.assertAlmostEqual(atom.z, 3.0)
        self.assertAlmostEqual(atom.vx, 0.1)
        self.assertAlmostEqual(atom.vy, 0.2)
        self.assertAlmostEqual(atom.vz, 0.3)

        extractor.close()

    def test_get_positions(self):
        """Test getting positions array."""
        extractor = LammpsDumpExtractor()
        extractor.open(self.dump_file)

        frame = extractor._read_frame()
        positions = frame.get_positions()

        self.assertEqual(positions.shape, (5, 3))
        self.assertAlmostEqual(positions[0, 0], 1.0)
        self.assertAlmostEqual(positions[0, 1], 2.0)
        self.assertAlmostEqual(positions[0, 2], 3.0)

        extractor.close()

    def test_get_velocities(self):
        """Test getting velocities array."""
        extractor = LammpsDumpExtractor()
        extractor.open(self.dump_file)

        frame = extractor._read_frame()
        velocities = frame.get_velocities()

        self.assertIsNotNone(velocities)
        self.assertEqual(velocities.shape, (5, 3))
        self.assertAlmostEqual(velocities[0, 0], 0.1)

        extractor.close()

    def test_compute_density(self):
        """Test density computation."""
        extractor = LammpsDumpExtractor()
        extractor.open(self.dump_file)

        frame = extractor._read_frame()
        density = frame.compute_density()

        # Expected: 5 atoms / (10*10*10) = 0.005
        expected = 5.0 / 1000.0
        self.assertAlmostEqual(density, expected)

        extractor.close()


class TestStreamingParser(unittest.TestCase):
    """Test streaming parser functionality."""

    def setUp(self):
        """Create test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.dump_file = os.path.join(self.temp_dir, "test.dump")

        # Create 20-frame dump file
        with open(self.dump_file, "w") as f:
            for i in range(20):
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{i * 100}\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write("4\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                f.write("0.0 5.0\n")
                f.write("0.0 5.0\n")
                f.write("0.0 5.0\n")
                f.write("ITEM: ATOMS id type x y z\n")
                # Vary positions slightly per frame
                offset = i * 0.1
                f.write(f"1 1 {1.0+offset} 1.0 1.0\n")
                f.write(f"2 1 {2.0+offset} 2.0 2.0\n")
                f.write(f"3 2 {3.0+offset} 3.0 3.0\n")
                f.write(f"4 2 {4.0+offset} 4.0 4.0\n")

    def tearDown(self):
        """Clean up."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_uniform_sampling(self):
        """Test uniform frame sampling."""
        parser = StreamingParser()
        config = SamplingConfig(
            strategy=SamplingStrategy.UNIFORM,
            interval=5,
        )

        stats = parser.parse(
            filepath=self.dump_file,
            format=FileFormat.LAMMPS_DUMP,
            sampling=config,
        )

        self.assertEqual(stats.total_frames, 20)
        self.assertEqual(stats.sampled_frames, 4)  # 0, 5, 10, 15
        self.assertEqual(stats.frame_indices, [0, 5, 10, 15])

    def test_keyframe_sampling(self):
        """Test keyframe sampling."""
        parser = StreamingParser()
        config = SamplingConfig(
            strategy=SamplingStrategy.KEYFRAME,
            keyframe_steps=[0, 500, 1000],
        )

        stats = parser.parse(
            filepath=self.dump_file,
            format=FileFormat.LAMMPS_DUMP,
            sampling=config,
        )

        # Frames with timesteps 0, 500, 1000 should be sampled
        self.assertIn(0, stats.frame_indices)  # timestep 0
        self.assertIn(5, stats.frame_indices)  # timestep 500
        self.assertIn(10, stats.frame_indices)  # timestep 1000

    def test_max_frames_limit(self):
        """Test max frames limit."""
        parser = StreamingParser()
        config = SamplingConfig(
            strategy=SamplingStrategy.UNIFORM,
            interval=1,
            max_frames=5,
        )

        stats = parser.parse(
            filepath=self.dump_file,
            format=FileFormat.LAMMPS_DUMP,
            sampling=config,
        )

        self.assertEqual(stats.sampled_frames, 5)

    def test_density_stats(self):
        """Test density statistics extraction."""
        parser = StreamingParser()
        config = SamplingConfig(
            strategy=SamplingStrategy.UNIFORM,
            interval=5,
        )

        stats = parser.parse(
            filepath=self.dump_file,
            format=FileFormat.LAMMPS_DUMP,
            sampling=config,
            extract_features=True,
        )

        # Density should be 4 atoms / 125 = 0.032
        self.assertIsNotNone(stats.density_mean)
        self.assertAlmostEqual(stats.density_mean, 0.032, places=5)


class TestDumpSampler(unittest.TestCase):
    """Test DumpSampler convenience class."""

    def setUp(self):
        """Create test file."""
        self.temp_dir = tempfile.mkdtemp()
        self.dump_file = os.path.join(self.temp_dir, "test.dump")

        with open(self.dump_file, "w") as f:
            for i in range(10):
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{i}\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write("2\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                f.write("0.0 5.0\n")
                f.write("0.0 5.0\n")
                f.write("0.0 5.0\n")
                f.write("ITEM: ATOMS id type x y z\n")
                f.write("1 1 1.0 1.0 1.0\n")
                f.write("2 1 2.0 2.0 2.0\n")

    def tearDown(self):
        """Clean up."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_sample_uniform(self):
        """Test uniform sampling convenience method."""
        stats = DumpSampler.sample_uniform(
            filepath=self.dump_file,
            interval=2,
        )

        self.assertEqual(stats.total_frames, 10)
        self.assertEqual(stats.sampled_frames, 5)

    def test_extract_begin_middle_end(self):
        """Test BME extraction."""
        begin, middle, end = DumpSampler.extract_begin_middle_end(self.dump_file)

        self.assertIsNotNone(begin)
        self.assertIsNotNone(middle)
        self.assertIsNotNone(end)

        self.assertEqual(begin.timestep, 0)
        self.assertEqual(middle.timestep, 5)  # Middle of 10 frames
        self.assertEqual(end.timestep, 9)


class TestCompressedFiles(unittest.TestCase):
    """Test compressed file handling."""

    def setUp(self):
        """Create compressed test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.dump_content = b"""ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 5.0
0.0 5.0
0.0 5.0
ITEM: ATOMS id type x y z
1 1 1.0 1.0 1.0
2 1 2.0 2.0 2.0
"""

    def tearDown(self):
        """Clean up."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_gzip_file(self):
        """Test reading gzip compressed file."""
        gz_file = os.path.join(self.temp_dir, "test.dump.gz")
        with gzip.open(gz_file, "wb") as f:
            f.write(self.dump_content)

        extractor = LammpsDumpExtractor()
        extractor.open(gz_file)

        frames = list(extractor.frames())
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].num_atoms, 2)

        extractor.close()


class TestCheckpoint(unittest.TestCase):
    """Test checkpoint/resume functionality."""

    def setUp(self):
        """Create test file and checkpoint."""
        self.temp_dir = tempfile.mkdtemp()
        self.dump_file = os.path.join(self.temp_dir, "test.dump")
        self.checkpoint_file = os.path.join(self.temp_dir, "checkpoint.pkl")

        # Create 50-frame dump file
        with open(self.dump_file, "w") as f:
            for i in range(50):
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{i * 100}\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write("3\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                f.write("0.0 5.0\n")
                f.write("0.0 5.0\n")
                f.write("0.0 5.0\n")
                f.write("ITEM: ATOMS id type x y z\n")
                f.write("1 1 1.0 1.0 1.0\n")
                f.write("2 1 2.0 2.0 2.0\n")
                f.write("3 2 3.0 3.0 3.0\n")

    def tearDown(self):
        """Clean up."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_checkpoint_save_load(self):
        """Test checkpoint serialization."""
        checkpoint = Checkpoint(
            file_path=self.dump_file,
            file_hash="abc123",
            last_frame=25,
            last_position=1000,
            timestamp=1234567890.0,
            metadata={"test": "data"},
        )

        # Save
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(checkpoint, f)

        # Load
        with open(self.checkpoint_file, "rb") as f:
            loaded = pickle.load(f)

        self.assertEqual(loaded.file_path, self.dump_file)
        self.assertEqual(loaded.last_frame, 25)
        self.assertEqual(loaded.metadata["test"], "data")


def run_tests():
    """Run all tests."""
    unittest.main(argv=[""], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
