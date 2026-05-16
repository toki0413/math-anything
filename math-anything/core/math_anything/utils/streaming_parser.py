"""Streaming parser for large MD trajectory files.

Handles TB-scale MD dump files with:
- Frame sampling (time-based or step-based)
- Statistical feature extraction
- Resume capability for interrupted parsing
- Memory-efficient streaming
"""

import bz2
import gzip
import hashlib
import lzma
import pickle
import re
import struct
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np


class SamplingStrategy(Enum):
    """Frame sampling strategies."""

    UNIFORM = "uniform"  # Sample every N frames
    RANDOM = "random"  # Random sampling
    ADAPTIVE = "adaptive"  # Adaptive based on change magnitude
    KEYFRAME = "keyframe"  # Keyframe-based (physics-informed)
    BEGIN_MIDDLE_END = "bme"  # Beginning, middle, end


class FileFormat(Enum):
    """Supported trajectory file formats."""

    LAMMPS_DUMP = "lammps_dump"
    LAMMPS_BINARY = "lammps_binary"
    XYZ = "xyz"
    DCD = "dcd"
    NETCDF = "netcdf"
    HDF5 = "hdf5"


@dataclass
class FrameRange:
    """Range of frames to parse."""

    start: int = 0
    end: Optional[int] = None  # None means until end
    step: int = 1


@dataclass
class SamplingConfig:
    """Configuration for frame sampling."""

    strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    interval: int = 1  # For UNIFORM: sample every N frames
    sample_rate: float = 0.1  # For RANDOM: fraction to sample
    threshold: float = 0.05  # For ADAPTIVE: change threshold
    keyframe_steps: List[int] = field(default_factory=list)  # For KEYFRAME
    max_frames: Optional[int] = None


@dataclass
class Checkpoint:
    """Checkpoint for resume capability."""

    file_path: str
    file_hash: str
    last_frame: int
    last_position: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomData:
    """Single atom data from a trajectory frame."""

    id: int
    type: int
    x: float
    y: float
    z: float
    vx: Optional[float] = None
    vy: Optional[float] = None
    vz: Optional[float] = None
    fx: Optional[float] = None
    fy: Optional[float] = None
    fz: Optional[float] = None
    q: Optional[float] = None  # Charge
    attributes: Dict[str, float] = field(default_factory=dict)


@dataclass
class FrameData:
    """Complete frame data from trajectory."""

    timestep: int
    num_atoms: int
    box_bounds: List[Tuple[float, float]]  # [(xlo, xhi), (ylo, yhi), (zlo, zhi)]
    atoms: List[AtomData]
    tilt_factors: Optional[Tuple[float, float, float]] = None  # xy, xz, yz

    # Statistical features (computed on-demand)
    _temperature: Optional[float] = None
    _pressure: Optional[float] = None
    _density: Optional[float] = None

    def get_positions(self) -> np.ndarray:
        """Get positions as Nx3 array."""
        return np.array([[a.x, a.y, a.z] for a in self.atoms])

    def get_velocities(self) -> Optional[np.ndarray]:
        """Get velocities as Nx3 array if available."""
        if self.atoms and self.atoms[0].vx is not None:
            return np.array([[a.vx, a.vy, a.vz] for a in self.atoms])
        return None

    def get_forces(self) -> Optional[np.ndarray]:
        """Get forces as Nx3 array if available."""
        if self.atoms and self.atoms[0].fx is not None:
            return np.array([[a.fx, a.fy, a.fz] for a in self.atoms])
        return None

    def compute_temperature(self, mass_map: Optional[Dict[int, float]] = None) -> float:
        """Compute temperature from kinetic energy."""
        velocities = self.get_velocities()
        if velocities is None:
            return 0.0

        if mass_map:
            masses = np.array([mass_map.get(a.type, 1.0) for a in self.atoms])
            ke = 0.5 * np.sum(masses[:, None] * velocities**2)
            dof = 3 * len(self.atoms) - 3  # Minus 3 for center of mass
        else:
            # Assume unit mass
            ke = 0.5 * np.sum(velocities**2)
            dof = 3 * len(self.atoms) - 3

        # T = 2*KE / (dof * k_B), using LAMMPS metal units (Boltzmann in eV/K)
        k_B = 8.617333262e-5  # eV/K
        self._temperature = 2 * ke / (dof * k_B)
        return self._temperature

    def compute_density(self) -> float:
        """Compute number density."""
        volume = np.prod([hi - lo for lo, hi in self.box_bounds])
        self._density = self.num_atoms / volume if volume > 0 else 0.0
        return self._density


@dataclass
class TrajectoryStats:
    """Statistical summary of trajectory."""

    total_frames: int = 0
    sampled_frames: int = 0
    frame_indices: List[int] = field(default_factory=list)

    # Temperature statistics
    temperature_mean: Optional[float] = None
    temperature_std: Optional[float] = None
    temperature_min: Optional[float] = None
    temperature_max: Optional[float] = None

    # Density statistics
    density_mean: Optional[float] = None
    density_std: Optional[float] = None

    # Energy drift (if available)
    energy_drift: Optional[float] = None

    # Structural evolution
    rmsd_progression: List[float] = field(default_factory=list)

    # Velocity distribution moments
    velocity_mean: Optional[np.ndarray] = None
    velocity_cov: Optional[np.ndarray] = None


class StreamingFrameExtractor(ABC):
    """Abstract base class for streaming frame extraction."""

    @abstractmethod
    def open(self, filepath: str) -> None:
        """Open the trajectory file."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the trajectory file."""
        pass

    @abstractmethod
    def frames(self) -> Iterator[FrameData]:
        """Yield frames from the trajectory."""
        pass

    @abstractmethod
    def seek_frame(self, frame_index: int) -> bool:
        """Seek to a specific frame. Returns success status."""
        pass

    @abstractmethod
    def estimate_total_frames(self) -> Optional[int]:
        """Estimate total number of frames (may be expensive)."""
        pass


class LammpsDumpExtractor(StreamingFrameExtractor):
    """Streaming extractor for LAMMPS dump files.

    Supports text, gz, bz2, and xz compressed files.
    Handles triclinic boxes with tilt factors.
    """

    # Attribute mapping for common dump formats
    ATTRIBUTE_MAP = {
        "id": "id",
        "type": "type",
        "element": "element",
        "x": "x",
        "y": "y",
        "z": "z",
        "xs": "xs",
        "ys": "ys",
        "zs": "zs",
        "xu": "xu",
        "yu": "yu",
        "zu": "zu",
        "vx": "vx",
        "vy": "vy",
        "vz": "vz",
        "fx": "fx",
        "fy": "fy",
        "fz": "fz",
        "q": "q",
    }

    def __init__(self):
        self.filepath: Optional[str] = None
        self.file_handle = None
        self.attributes: List[str] = []
        self.frame_count = 0
        self.current_position = 0
        self._is_compressed = False

    def open(self, filepath: str) -> None:
        """Open a LAMMPS dump file (handles compression)."""
        self.filepath = filepath
        path = Path(filepath)

        # Detect compression and open appropriately
        if filepath.endswith(".gz"):
            self.file_handle = gzip.open(filepath, "rt", encoding="utf-8")
            self._is_compressed = True
        elif filepath.endswith(".bz2"):
            self.file_handle = bz2.open(filepath, "rt", encoding="utf-8")
            self._is_compressed = True
        elif filepath.endswith(".xz"):
            self.file_handle = lzma.open(filepath, "rt", encoding="utf-8")
            self._is_compressed = True
        else:
            self.file_handle = open(filepath, "r", encoding="utf-8")
            self._is_compressed = False

        self.frame_count = 0
        self.current_position = 0

    def close(self) -> None:
        """Close the file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def _read_frame(self) -> Optional[FrameData]:
        """Read a single frame from the file."""
        if not self.file_handle:
            return None

        lines = []

        # Read until we find "ITEM: TIMESTEP"
        line = self.file_handle.readline()
        if not line:
            return None  # EOF

        if not line.strip().startswith("ITEM: TIMESTEP"):
            # Try to find next frame
            while line and not line.strip().startswith("ITEM: TIMESTEP"):
                line = self.file_handle.readline()
            if not line:
                return None

        lines.append(line.strip())

        # Read timestep
        line = self.file_handle.readline()
        if not line:
            return None
        timestep = int(line.strip())
        lines.append(line.strip())

        # Read number of atoms
        line = self.file_handle.readline()  # "ITEM: NUMBER OF ATOMS"
        if not line:
            return None
        lines.append(line.strip())

        line = self.file_handle.readline()
        if not line:
            return None
        num_atoms = int(line.strip())
        lines.append(line.strip())

        # Read box bounds
        line = self.file_handle.readline()  # "ITEM: BOX BOUNDS ..."
        if not line:
            return None
        bounds_line = line.strip()
        lines.append(bounds_line)

        # Check for triclinic box
        tilt_factors = None
        parts = bounds_line.split()
        if len(parts) >= 6 and parts[4] == "xy" and parts[5] == "xz":
            # Triclinic box
            xlo_bound = float(self.file_handle.readline().strip().split()[0])
            xhi_bound = float(self.file_handle.readline().strip().split()[1])
            ylo_bound = float(self.file_handle.readline().strip().split()[0])
            yhi_bound = float(self.file_handle.readline().strip().split()[1])
            zlo = float(self.file_handle.readline().strip().split()[0])
            zhi = float(self.file_handle.readline().strip().split()[1])
            xy = float(self.file_handle.readline().strip().split()[0])
            xz = float(self.file_handle.readline().strip().split()[1])
            yz = float(self.file_handle.readline().strip().split()[2])
            tilt_factors = (xy, xz, yz)

            # Convert bounds to actual box dimensions
            xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
            xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
            ylo = ylo_bound - min(0.0, yz)
            yhi = yhi_bound - max(0.0, yz)

            box_bounds = [(xlo, xhi), (ylo, yhi), (zlo, zhi)]
        else:
            # Orthogonal box
            box_bounds = []
            for _ in range(3):
                line = self.file_handle.readline()
                if not line:
                    return None
                parts = line.strip().split()
                box_bounds.append((float(parts[0]), float(parts[1])))
                lines.append(line.strip())

        # Read atom attributes header
        line = self.file_handle.readline()
        if not line:
            return None
        header = line.strip()
        lines.append(header)

        # Parse attributes from header
        if header.startswith("ITEM: ATOMS"):
            self.attributes = header[11:].strip().split()
        else:
            self.attributes = ["id", "type", "x", "y", "z"]  # Default

        # Read atoms
        atoms = []
        for _ in range(num_atoms):
            line = self.file_handle.readline()
            if not line:
                break
            parts = line.strip().split()

            atom_data = {}
            for i, attr in enumerate(self.attributes):
                if i < len(parts):
                    try:
                        atom_data[attr] = float(parts[i])
                    except ValueError:
                        atom_data[attr] = parts[i]

            atom = AtomData(
                id=int(atom_data.get("id", 0)),
                type=int(atom_data.get("type", 1)),
                x=atom_data.get("x", atom_data.get("xs", 0.0)),
                y=atom_data.get("y", atom_data.get("ys", 0.0)),
                z=atom_data.get("z", atom_data.get("zs", 0.0)),
                vx=atom_data.get("vx"),
                vy=atom_data.get("vy"),
                vz=atom_data.get("vz"),
                fx=atom_data.get("fx"),
                fy=atom_data.get("fy"),
                fz=atom_data.get("fz"),
                q=atom_data.get("q"),
            )

            # Handle scaled coordinates
            if "xs" in atom_data:
                xlo, xhi = box_bounds[0]
                atom.x = xlo + atom_data["xs"] * (xhi - xlo)
            if "ys" in atom_data:
                ylo, yhi = box_bounds[1]
                atom.y = ylo + atom_data["ys"] * (yhi - ylo)
            if "zs" in atom_data:
                zlo, zhi = box_bounds[2]
                atom.z = zlo + atom_data["zs"] * (zhi - zlo)

            atoms.append(atom)

        self.frame_count += 1

        return FrameData(
            timestep=timestep,
            num_atoms=num_atoms,
            box_bounds=box_bounds,
            atoms=atoms,
            tilt_factors=tilt_factors,
        )

    def frames(self) -> Iterator[FrameData]:
        """Yield frames from the dump file."""
        while True:
            frame = self._read_frame()
            if frame is None:
                break
            yield frame

    def seek_frame(self, frame_index: int) -> bool:
        """Seek to a specific frame (linear scan, expensive)."""
        if frame_index < self.frame_count:
            # Need to reopen and scan
            self.close()
            self.open(self.filepath)

        while self.frame_count < frame_index:
            frame = self._read_frame()
            if frame is None:
                return False

        return True

    def estimate_total_frames(self) -> Optional[int]:
        """Estimate total frames by file size / first frame size."""
        if not self.filepath:
            return None

        try:
            file_size = Path(self.filepath).stat().st_size

            # Read first frame and estimate
            first_frame = self._read_frame()
            if first_frame is None:
                return 0

            # Estimate based on average frame size (very rough)
            # Reopen to get accurate count
            self.close()
            self.open(self.filepath)

            count = 0
            for _ in self.frames():
                count += 1

            self.close()
            self.open(self.filepath)

            return count
        except Exception:
            return None


class StreamingParser:
    """Main streaming parser for large MD trajectory files.

    Features:
    - Memory-efficient streaming
    - Frame sampling strategies
    - Statistical feature extraction
    - Checkpoint/resume capability
    - Progress callbacks

    Example:
        ```python
        parser = StreamingParser()

        config = SamplingConfig(
            strategy=SamplingStrategy.UNIFORM,
            interval=100,  # Sample every 100 frames
            max_frames=1000,
        )

        results = parser.parse(
            "trajectory.dump",
            format=FileFormat.LAMMPS_DUMP,
            sampling=config,
            checkpoint_path="checkpoint.pkl",
        )
        ```
    """

    def __init__(self):
        self.extractor: Optional[StreamingFrameExtractor] = None
        self.stats = TrajectoryStats()
        self._progress_callback: Optional[Callable[[int, int], None]] = None
        self._checkpoint_path: Optional[str] = None

    def set_progress_callback(self, callback: Callable[[int, int], None]) -> None:
        """Set a callback for progress updates.

        Args:
            callback: Function called with (current_frame, total_frames or -1)
        """
        self._progress_callback = callback

    def parse(
        self,
        filepath: str,
        format: FileFormat = FileFormat.LAMMPS_DUMP,
        sampling: Optional[SamplingConfig] = None,
        checkpoint_path: Optional[str] = None,
        resume: bool = True,
        extract_features: bool = True,
    ) -> TrajectoryStats:
        """Parse a trajectory file with streaming.

        Args:
            filepath: Path to trajectory file
            format: File format
            sampling: Sampling configuration
            checkpoint_path: Path to save/load checkpoint
            resume: Whether to resume from checkpoint
            extract_features: Whether to compute statistical features

        Returns:
            TrajectoryStats with extracted information
        """
        self._checkpoint_path = checkpoint_path
        sampling = sampling or SamplingConfig()

        # Create appropriate extractor
        self.extractor = self._create_extractor(format)
        self.extractor.open(filepath)

        # Check for checkpoint
        start_frame = 0
        if resume and checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = self._load_checkpoint(checkpoint_path)
            if checkpoint and checkpoint.file_hash == self._compute_file_hash(filepath):
                start_frame = checkpoint.last_frame
                self.stats = checkpoint.metadata.get("stats", TrajectoryStats())
                print(f"Resuming from frame {start_frame}")

        # Seek if needed
        if start_frame > 0:
            self.extractor.seek_frame(start_frame)

        # Parse frames
        self.stats = self._parse_frames(
            sampling=sampling,
            start_frame=start_frame,
            extract_features=extract_features,
        )

        self.extractor.close()

        # Clean up checkpoint if completed
        if checkpoint_path and Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()

        return self.stats

    def _create_extractor(self, format: FileFormat) -> StreamingFrameExtractor:
        """Create frame extractor for the given format."""
        if format == FileFormat.LAMMPS_DUMP:
            return LammpsDumpExtractor()
        # Add more formats as needed
        raise ValueError(f"Unsupported format: {format}")

    def _parse_frames(
        self,
        sampling: SamplingConfig,
        start_frame: int,
        extract_features: bool,
    ) -> TrajectoryStats:
        """Parse frames with sampling."""
        stats = TrajectoryStats()
        frame_idx = start_frame
        sampled_count = 0

        temperatures = []
        densities = []
        velocities = []
        positions_ref = None

        # Initialize sampling state
        last_keyframe_data = None

        for frame in self.extractor.frames():
            # Check max frames
            if sampling.max_frames and sampled_count >= sampling.max_frames:
                break

            # Apply sampling strategy
            should_sample = self._should_sample_frame(
                frame_idx=frame_idx,
                sampling=sampling,
                frame=frame,
                last_keyframe=last_keyframe_data,
            )

            if should_sample:
                sampled_count += 1
                stats.frame_indices.append(frame_idx)

                # Extract features
                if extract_features:
                    # Temperature
                    temp = frame.compute_temperature()
                    if temp > 0:
                        temperatures.append(temp)

                    # Density
                    density = frame.compute_density()
                    densities.append(density)

                    # Velocities for distribution
                    vel = frame.get_velocities()
                    if vel is not None:
                        velocities.append(vel)

                    # RMSD from reference (first sampled frame)
                    if positions_ref is None:
                        positions_ref = frame.get_positions()
                    else:
                        positions = frame.get_positions()
                        if positions.shape == positions_ref.shape:
                            rmsd = np.sqrt(np.mean((positions - positions_ref) ** 2))
                            stats.rmsd_progression.append(rmsd)

                last_keyframe_data = frame

            # Progress callback
            if self._progress_callback and frame_idx % 100 == 0:
                self._progress_callback(frame_idx, -1)

            # Save checkpoint periodically
            if (
                self._checkpoint_path
                and frame_idx % 1000 == 0
                and frame_idx > start_frame
            ):
                self._save_checkpoint(frame_idx, stats)

            frame_idx += 1

        # Finalize statistics
        stats.total_frames = frame_idx
        stats.sampled_frames = sampled_count

        if temperatures:
            stats.temperature_mean = float(np.mean(temperatures))
            stats.temperature_std = float(np.std(temperatures))
            stats.temperature_min = float(np.min(temperatures))
            stats.temperature_max = float(np.max(temperatures))

        if densities:
            stats.density_mean = float(np.mean(densities))
            stats.density_std = float(np.std(densities))

        if velocities:
            all_velocities = np.concatenate(velocities, axis=0)
            stats.velocity_mean = np.mean(all_velocities, axis=0).tolist()
            stats.velocity_cov = np.cov(all_velocities.T).tolist()

        return stats

    def _should_sample_frame(
        self,
        frame_idx: int,
        sampling: SamplingConfig,
        frame: FrameData,
        last_keyframe: Optional[FrameData],
    ) -> bool:
        """Determine if a frame should be sampled."""
        strategy = sampling.strategy

        if strategy == SamplingStrategy.UNIFORM:
            return frame_idx % sampling.interval == 0

        elif strategy == SamplingStrategy.BEGIN_MIDDLE_END:
            # Will sample first, middle, and last
            # This is handled separately
            return False

        elif strategy == SamplingStrategy.KEYFRAME:
            if sampling.keyframe_steps:
                return frame.timestep in sampling.keyframe_steps
            return frame_idx % sampling.interval == 0

        elif strategy == SamplingStrategy.RANDOM:
            import random

            return random.random() < sampling.sample_rate

        elif strategy == SamplingStrategy.ADAPTIVE:
            if last_keyframe is None:
                return True
            # Sample if change exceeds threshold
            return (
                self._compute_frame_difference(frame, last_keyframe)
                > sampling.threshold
            )

        return True

    def _compute_frame_difference(
        self,
        frame1: FrameData,
        frame2: FrameData,
    ) -> float:
        """Compute normalized difference between two frames."""
        pos1 = frame1.get_positions()
        pos2 = frame2.get_positions()

        if pos1.shape != pos2.shape:
            return 1.0  # Max difference

        rmsd = np.sqrt(np.mean((pos1 - pos2) ** 2))
        # Normalize by typical atomic spacing (~1 Angstrom)
        return min(rmsd, 1.0)

    def _compute_file_hash(self, filepath: str) -> str:
        """Compute hash of file for checkpoint validation."""
        hasher = hashlib.md5(usedforsecurity=False)
        stat = Path(filepath).stat()
        # Use size and mtime as lightweight hash
        hasher.update(f"{stat.st_size}:{stat.st_mtime}".encode())
        return hasher.hexdigest()

    def _save_checkpoint(self, frame_idx: int, stats: TrajectoryStats) -> None:
        """Save checkpoint for resume."""
        if not self._checkpoint_path:
            return

        checkpoint = Checkpoint(
            file_path=self.extractor.filepath,
            file_hash=self._compute_file_hash(self.extractor.filepath),
            last_frame=frame_idx,
            last_position=0,  # Not used for text files
            timestamp=__import__("time").time(),
            metadata={"stats": stats},
        )

        with open(self._checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

    def _load_checkpoint(self, path: str) -> Optional[Checkpoint]:
        """Load checkpoint."""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None


class DumpSampler:
    """High-level interface for sampling dump files.

    Convenience class for common sampling patterns.
    """

    @staticmethod
    def sample_uniform(
        filepath: str,
        interval: int = 100,
        max_frames: Optional[int] = None,
    ) -> TrajectoryStats:
        """Sample uniformly spaced frames."""
        parser = StreamingParser()
        config = SamplingConfig(
            strategy=SamplingStrategy.UNIFORM,
            interval=interval,
            max_frames=max_frames,
        )
        return parser.parse(filepath, sampling=config)

    @staticmethod
    def sample_keyframes(
        filepath: str,
        keyframe_steps: List[int],
    ) -> TrajectoryStats:
        """Sample specific timesteps."""
        parser = StreamingParser()
        config = SamplingConfig(
            strategy=SamplingStrategy.KEYFRAME,
            keyframe_steps=keyframe_steps,
        )
        return parser.parse(filepath, sampling=config)

    @staticmethod
    def sample_adaptive(
        filepath: str,
        threshold: float = 0.05,
        max_frames: Optional[int] = None,
    ) -> TrajectoryStats:
        """Sample adaptively based on change magnitude."""
        parser = StreamingParser()
        config = SamplingConfig(
            strategy=SamplingStrategy.ADAPTIVE,
            threshold=threshold,
            max_frames=max_frames,
        )
        return parser.parse(filepath, sampling=config)

    @staticmethod
    def extract_begin_middle_end(
        filepath: str,
    ) -> Tuple[FrameData, FrameData, FrameData]:
        """Extract beginning, middle, and end frames."""
        parser = StreamingParser()
        extractor = LammpsDumpExtractor()
        extractor.open(filepath)

        # First pass: count frames
        total = 0
        for _ in extractor.frames():
            total += 1

        # Second pass: extract specific frames
        extractor.close()
        extractor.open(filepath)

        begin = None
        middle = None
        end = None

        for idx, frame in enumerate(extractor.frames()):
            if idx == 0:
                begin = frame
            elif idx == total // 2:
                middle = frame
            elif idx == total - 1:
                end = frame
                break

        extractor.close()
        return begin, middle, end


# Backward compatibility alias
LammpsDumpStreamer = LammpsDumpExtractor
