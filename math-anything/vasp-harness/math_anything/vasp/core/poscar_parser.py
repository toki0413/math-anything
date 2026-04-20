"""VASP POSCAR file parser for crystal structures.

Parses POSCAR/CONTCAR files and extracts:
- Lattice vectors (unit cell)
- Atomic positions (direct or Cartesian)
- Atomic types and counts
- Symmetry information
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Lattice:
    """Crystal lattice definition."""
    vectors: np.ndarray  # 3x3 matrix, each row is a lattice vector
    scale: float = 1.0
    
    def __post_init__(self):
        if isinstance(self.vectors, list):
            self.vectors = np.array(vectors)
    
    @property
    def volume(self) -> float:
        """Calculate unit cell volume."""
        return abs(np.linalg.det(self.vectors))
    
    @property
    def reciprocal_vectors(self) -> np.ndarray:
        """Calculate reciprocal lattice vectors."""
        volume = self.volume
        a1, a2, a3 = self.vectors
        b1 = 2 * np.pi * np.cross(a2, a3) / volume
        b2 = 2 * np.pi * np.cross(a3, a1) / volume
        b3 = 2 * np.pi * np.cross(a1, a2) / volume
        return np.array([b1, b2, b3])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vectors": self.vectors.tolist(),
            "scale": self.scale,
            "volume": float(self.volume),
            "reciprocal_vectors": self.reciprocal_vectors.tolist(),
        }


@dataclass
class Atom:
    """Single atom in the structure."""
    symbol: str
    position: np.ndarray  # 3D position
    position_type: str  # "direct" or "cartesian"
    index: int
    
    def to_cartesian(self, lattice: Lattice) -> np.ndarray:
        """Convert to Cartesian coordinates."""
        if self.position_type == "cartesian":
            return self.position * lattice.scale
        else:
            return np.dot(self.position, lattice.vectors) * lattice.scale
    
    def to_dict(self, lattice: Optional[Lattice] = None) -> Dict[str, Any]:
        result = {
            "symbol": self.symbol,
            "position": self.position.tolist(),
            "position_type": self.position_type,
            "index": self.index,
        }
        if lattice:
            result["cartesian_position"] = self.to_cartesian(lattice).tolist()
        return result


@dataclass
class CrystalStructure:
    """Complete crystal structure."""
    comment: str
    lattice: Lattice
    atom_types: List[str]
    atom_counts: List[int]
    atoms: List[Atom]
    coordinate_system: str  # "Direct" or "Cartesian"
    selective_dynamics: bool = False
    
    @property
    def chemical_formula(self) -> str:
        """Generate chemical formula."""
        parts = []
        for symbol, count in zip(self.atom_types, self.atom_counts):
            if count == 1:
                parts.append(symbol)
            else:
                parts.append(f"{symbol}{count}")
        return "".join(parts)
    
    @property
    def num_atoms(self) -> int:
        """Total number of atoms."""
        return sum(self.atom_counts)
    
    @property
    def density(self) -> float:
        """Calculate density (requires atomic masses)."""
        # Simplified - would need atomic masses
        return 0.0
    
    def get_atomic_positions(self, cartesian: bool = True) -> np.ndarray:
        """Get all atomic positions as array."""
        if cartesian:
            return np.array([atom.to_cartesian(self.lattice) for atom in self.atoms])
        else:
            return np.array([atom.position for atom in self.atoms])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "comment": self.comment,
            "chemical_formula": self.chemical_formula,
            "num_atoms": self.num_atoms,
            "lattice": self.lattice.to_dict(),
            "atom_types": self.atom_types,
            "atom_counts": self.atom_counts,
            "coordinate_system": self.coordinate_system,
            "selective_dynamics": self.selective_dynamics,
            "atoms": [atom.to_dict(self.lattice) for atom in self.atoms],
        }


class PoscarParser:
    """Parser for VASP POSCAR/CONTCAR files.
    
    Example:
        parser = PoscarParser()
        structure = parser.parse("POSCAR")
        
        print(f"Formula: {structure.chemical_formula}")
        print(f"Volume: {structure.lattice.volume} Å³")
        print(f"Atoms: {structure.num_atoms}")
    """
    
    def parse(self, filepath: str) -> CrystalStructure:
        """Parse POSCAR file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        line_idx = 0
        
        # Line 1: Comment
        comment = lines[line_idx].strip()
        line_idx += 1
        
        # Line 2: Scale factor
        scale = float(lines[line_idx].strip())
        line_idx += 1
        
        # Lines 3-5: Lattice vectors
        lattice_vectors = []
        for i in range(3):
            parts = lines[line_idx].strip().split()
            vector = [float(x) for x in parts[:3]]
            lattice_vectors.append(vector)
            line_idx += 1
        
        lattice = Lattice(vectors=np.array(lattice_vectors), scale=scale)
        
        # Line 6: Atom types
        atom_types = lines[line_idx].strip().split()
        line_idx += 1
        
        # Line 7: Atom counts
        atom_counts = [int(x) for x in lines[line_idx].strip().split()]
        line_idx += 1
        
        # Check for selective dynamics
        selective_dynamics = False
        coord_line = lines[line_idx].strip()
        if coord_line[0].lower() == 's':
            selective_dynamics = True
            line_idx += 1
            coord_line = lines[line_idx].strip()
        
        # Coordinate system
        coordinate_system = "Cartesian" if coord_line[0].lower() == 'c' or coord_line[0].lower() == 'k' else "Direct"
        line_idx += 1
        
        # Read atomic positions
        atoms = []
        atom_idx = 0
        for type_idx, (symbol, count) in enumerate(zip(atom_types, atom_counts)):
            for _ in range(count):
                parts = lines[line_idx].strip().split()
                position = np.array([float(x) for x in parts[:3]])
                
                atom = Atom(
                    symbol=symbol,
                    position=position,
                    position_type=coordinate_system.lower(),
                    index=atom_idx
                )
                atoms.append(atom)
                atom_idx += 1
                line_idx += 1
        
        return CrystalStructure(
            comment=comment,
            lattice=lattice,
            atom_types=atom_types,
            atom_counts=atom_counts,
            atoms=atoms,
            coordinate_system=coordinate_system,
            selective_dynamics=selective_dynamics
        )
    
    def extract_symmetry_info(self, structure: CrystalStructure) -> Dict[str, Any]:
        """Extract symmetry-related information."""
        # Simplified symmetry detection
        # Real implementation would use spglib
        
        return {
            "crystal_system": "unknown",  # Would need analysis
            "space_group": "unknown",
            "point_group": "unknown",
            "symmetry_operations": [],
            "primitive_cell": False,
            "conventional_cell": True,
        }
    
    def calculate_distances(self, structure: CrystalStructure, 
                          max_distance: float = 5.0) -> List[Dict[str, Any]]:
        """Calculate interatomic distances."""
        positions = structure.get_atomic_positions(cartesian=True)
        distances = []
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= max_distance:
                    distances.append({
                        "atom1": structure.atoms[i].symbol,
                        "atom2": structure.atoms[j].symbol,
                        "distance": float(dist),
                        "index1": i,
                        "index2": j,
                    })
        
        return distances


# Convenience function
def parse_poscar(filepath: str) -> CrystalStructure:
    """Parse POSCAR file."""
    parser = PoscarParser()
    return parser.parse(filepath)
