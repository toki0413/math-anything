"""VASP KPOINTS file parser for Brillouin zone sampling.

Parses KPOINTS files and extracts:
- k-point mesh (Monkhorst-Pack or Gamma-centered)
- k-point paths for band structure
- Line mode specifications
- Explicit k-point lists
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class KpointsMode(Enum):
    """KPOINTS generation mode."""
    AUTOMATIC = "automatic"  # Monkhorst-Pack or Gamma
    GAMMA = "gamma"  # Gamma-centered
    MONKHORST_PACK = "monkhorst-pack"  # Monkhorst-Pack
    LINE_MODE = "line-mode"  # Band structure path
    EXPLICIT = "explicit"  # Explicit k-point list
    EXPLICIT_BAND = "explicit-band"  # Explicit with band weights


@dataclass
class Kpoint:
    """Single k-point in reciprocal space."""
    coordinates: np.ndarray  # 3D coordinates in reciprocal space
    weight: float = 1.0
    label: str = ""  # High-symmetry point label (e.g., "GAMMA", "L", "X")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinates": self.coordinates.tolist(),
            "weight": self.weight,
            "label": self.label,
        }


@dataclass
class KpointsMesh:
    """Automatic k-point mesh (Monkhorst-Pack or Gamma)."""
    subdivisions: List[int]  # [nx, ny, nz]
    shift: List[float]  # [sx, sy, sz]
    mode: KpointsMode
    
    @property
    def total_kpoints(self) -> int:
        """Calculate total number of k-points."""
        return self.subdivisions[0] * self.subdivisions[1] * self.subdivisions[2]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subdivisions": self.subdivisions,
            "shift": self.shift,
            "mode": self.mode.value,
            "total_kpoints": self.total_kpoints,
        }


@dataclass
class KpointsPath:
    """k-point path for band structure calculation."""
    segments: List[Dict[str, Any]]  # List of path segments
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": self.segments,
            "num_segments": len(self.segments),
        }


@dataclass
class KpointsData:
    """Complete KPOINTS data."""
    comment: str
    mode: KpointsMode
    num_kpoints: int  # 0 for automatic, actual number for explicit
    mesh: Optional[KpointsMesh] = None
    path: Optional[KpointsPath] = None
    explicit_kpoints: List[Kpoint] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "comment": self.comment,
            "mode": self.mode.value,
            "num_kpoints": self.num_kpoints,
            "mesh": self.mesh.to_dict() if self.mesh else None,
            "path": self.path.to_dict() if self.path else None,
            "explicit_kpoints": [kp.to_dict() for kp in self.explicit_kpoints],
        }


class KpointsParser:
    """Parser for VASP KPOINTS files.
    
    Example:
        parser = KpointsParser()
        kpoints = parser.parse("KPOINTS")
        
        if kpoints.mode == KpointsMode.AUTOMATIC:
            print(f"Mesh: {kpoints.mesh.subdivisions}")
            print(f"Total k-points: {kpoints.mesh.total_kpoints}")
    """
    
    # Standard high-symmetry points in reciprocal space
    STANDARD_POINTS = {
        "GAMMA": [0.0, 0.0, 0.0],
        "G": [0.0, 0.0, 0.0],
        "X": [0.5, 0.0, 0.0],
        "L": [0.5, 0.5, 0.5],
        "W": [0.5, 0.25, 0.75],
        "K": [0.375, 0.375, 0.75],
        "U": [0.625, 0.25, 0.625],
        "M": [0.5, 0.5, 0.0],
        "A": [0.5, 0.5, 0.5],
        "R": [0.5, 0.5, 0.5],
        "H": [0.5, -0.5, 0.5],
    }
    
    def parse(self, filepath: str) -> KpointsData:
        """Parse KPOINTS file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        line_idx = 0
        
        # Line 1: Comment
        comment = lines[line_idx].strip()
        line_idx += 1
        
        # Line 2: Number of k-points (0 for automatic)
        num_kpoints = int(lines[line_idx].strip().split()[0])
        line_idx += 1
        
        # Determine mode
        mode_line = lines[line_idx].strip().lower()
        
        if num_kpoints == 0:
            # Automatic mode (Monkhorst-Pack or Gamma)
            return self._parse_automatic(lines, comment, num_kpoints, line_idx)
        elif "line" in mode_line:
            # Line mode for band structure
            return self._parse_line_mode(lines, comment, num_kpoints, line_idx)
        else:
            # Explicit k-point list
            return self._parse_explicit(lines, comment, num_kpoints, line_idx)
    
    def _parse_automatic(self, lines: List[str], comment: str, 
                        num_kpoints: int, line_idx: int) -> KpointsData:
        """Parse automatic k-point mesh."""
        mode_line = lines[line_idx].strip().lower()
        line_idx += 1
        
        # Determine if Gamma or Monkhorst-Pack
        if mode_line.startswith('g'):
            mode = KpointsMode.GAMMA
        else:
            mode = KpointsMode.MONKHORST_PACK
        
        # Read subdivisions
        subdiv_parts = lines[line_idx].strip().split()
        subdivisions = [int(x) for x in subdiv_parts[:3]]
        line_idx += 1
        
        # Read shift
        shift_parts = lines[line_idx].strip().split()
        shift = [float(x) for x in shift_parts[:3]]
        
        mesh = KpointsMesh(
            subdivisions=subdivisions,
            shift=shift,
            mode=mode
        )
        
        return KpointsData(
            comment=comment,
            mode=mode,
            num_kpoints=0,
            mesh=mesh
        )
    
    def _parse_line_mode(self, lines: List[str], comment: str,
                        num_kpoints: int, line_idx: int) -> KpointsData:
        """Parse line mode for band structure.
        
        Format:
          num_kpoints    (number of intersections)
          Line-mode
          [rec|cart]     (optional coordinate system)
          k-point pairs...
        """
        line_idx += 1  # Skip "Line-mode" line
        
        # Check for optional coordinate system (rec or cart)
        coord_line = lines[line_idx].strip().lower()
        if coord_line.startswith(('rec', 'cart', 'k ', 'c ')):
            line_idx += 1
        
        segments = []
        
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if not line:
                line_idx += 1
                continue
            
            # Read start point
            parts1 = line.split()
            coords1 = [float(x) for x in parts1[:3]]
            label1 = parts1[3] if len(parts1) > 3 else ""
            line_idx += 1
            
            # Read end point
            if line_idx >= len(lines):
                break
            parts2 = lines[line_idx].strip().split()
            coords2 = [float(x) for x in parts2[:3]]
            label2 = parts2[3] if len(parts2) > 3 else ""
            line_idx += 1
            
            segments.append({
                "start": {"coordinates": coords1, "label": label1},
                "end": {"coordinates": coords2, "label": label2},
                "num_points": num_kpoints,  # num_kpoints is points per segment in line-mode
            })
        
        path = KpointsPath(segments=segments)
        
        return KpointsData(
            comment=comment,
            mode=KpointsMode.LINE_MODE,
            num_kpoints=len(segments) * num_kpoints,
            path=path
        )
    
    def _parse_explicit(self, lines: List[str], comment: str,
                       num_kpoints: int, line_idx: int) -> KpointsData:
        """Parse explicit k-point list."""
        # Skip coordinate system line if present
        coord_line = lines[line_idx].strip().lower()
        if coord_line in ['c', 'k', 'cartesian', 'd', 'direct']:
            line_idx += 1
        
        kpoints = []
        
        for i in range(num_kpoints):
            if line_idx >= len(lines):
                break
            
            parts = lines[line_idx].strip().split()
            if len(parts) < 3:
                continue
            
            coords = [float(x) for x in parts[:3]]
            weight = float(parts[3]) if len(parts) > 3 else 1.0
            
            kpoint = Kpoint(
                coordinates=np.array(coords),
                weight=weight
            )
            kpoints.append(kpoint)
            line_idx += 1
        
        return KpointsData(
            comment=comment,
            mode=KpointsMode.EXPLICIT,
            num_kpoints=len(kpoints),
            explicit_kpoints=kpoints
        )
    
    def get_brillouin_zone_info(self, kpoints: KpointsData) -> Dict[str, Any]:
        """Extract Brillouin zone sampling information."""
        info = {
            "sampling_method": kpoints.mode.value,
            "total_kpoints": kpoints.num_kpoints,
        }
        
        if kpoints.mesh:
            info["mesh_density"] = kpoints.mesh.subdivisions
            info["k_spacing"] = self._estimate_k_spacing(kpoints.mesh)
        
        return info
    
    def _estimate_k_spacing(self, mesh: KpointsMesh) -> float:
        """Estimate k-point spacing (approximate)."""
        # Approximate: assume cubic reciprocal cell
        # Real implementation would need actual reciprocal lattice
        max_subdiv = max(mesh.subdivisions)
        if max_subdiv > 0:
            return 1.0 / max_subdiv  # Rough estimate in units of 2π/a
        return 0.0
    
    def validate_sampling(self, kpoints: KpointsData, structure_volume: float) -> List[Dict[str, Any]]:
        """Validate k-point sampling quality."""
        constraints = []
        
        if kpoints.mesh:
            total_kpoints = kpoints.mesh.total_kpoints
            
            # Minimum k-point density
            constraints.append({
                "expression": "N_k >= 1",
                "description": "At least one k-point required",
                "satisfied": total_kpoints >= 1,
                "value": total_kpoints,
            })
            
            # Metal vs insulator sampling
            constraints.append({
                "expression": "ISMEAR appropriate for system type",
                "description": "Gaussian smearing (ISMEAR=0) for metals, tetrahedron for insulators",
                "satisfied": None,  # Need INCAR info
            })
        
        return constraints


# Convenience function
def parse_kpoints(filepath: str) -> KpointsData:
    """Parse KPOINTS file."""
    parser = KpointsParser()
    return parser.parse(filepath)
