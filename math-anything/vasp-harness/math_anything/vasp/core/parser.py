"""VASP input and output file parsers.

Parses VASP files (INCAR, POSCAR, KPOINTS, OUTCAR) to extract
computational parameters, crystal structure, and calculation results.
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class VaspCalculationType(Enum):
    """Types of VASP calculations."""
    SCF = "single_point"           # Single point calculation
    RELAX = "relax"                # Ionic relaxation
    MD = "md"                      # Molecular dynamics
    BANDS = "bands"                # Band structure
    DOS = "dos"                    # Density of states
    NEB = "neb"                    # Nudged elastic band
    DFT_PLUS_U = "dft_plus_u"      # DFT+U calculation
    GW = "gw"                      # GW approximation
    HSE = "hse"                    # HSE06 hybrid functional


@dataclass
class KpointGrid:
    """K-point grid specification."""
    mode: str                       # Automatic, Gamma, Monkhorst-Pack
    grid: List[int]                 # [nx, ny, nz]
    shift: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "grid": self.grid,
            "shift": self.shift,
        }


@dataclass
class CrystalStructure:
    """Crystal structure from POSCAR."""
    system_name: str
    scale: float
    lattice_vectors: np.ndarray     # 3x3 matrix
    coord_type: str                 # Direct or Cartesian
    atom_types: List[str]
    atom_counts: List[int]
    positions: np.ndarray           # Nx3 array
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_name": self.system_name,
            "scale": self.scale,
            "lattice_vectors": self.lattice_vectors.tolist(),
            "coord_type": self.coord_type,
            "atom_types": self.atom_types,
            "atom_counts": self.atom_counts,
            "positions": self.positions.tolist(),
            "n_atoms": sum(self.atom_counts),
        }


@dataclass
class ElectronicParameters:
    """Electronic structure calculation parameters."""
    encut: float                    # Plane wave cutoff energy
    ismear: int                     # Smearing method
    sigma: float                    # Smearing width
    ispin: int                      # Spin polarization (1 or 2)
    ibrion: int                     # Ion relaxation method
    nsw: int                        # Number of ionic steps
    nelm: int                       # Max electronic steps
    nelmin: int                     # Min electronic steps
    ediff: float                    # Electronic convergence criterion
    ediffg: float                   # Ionic convergence criterion
    algo: str                       # Algorithm (Normal, Fast, All, etc.)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "encut": self.encut,
            "ismear": self.ismear,
            "sigma": self.sigma,
            "ispin": self.ispin,
            "ibrion": self.ibrion,
            "nsw": self.nsw,
            "nelm": self.nelm,
            "nelmin": self.nelmin,
            "ediff": self.ediff,
            "ediffg": self.ediffg,
            "algo": self.algo,
        }


@dataclass
class SCFIteration:
    """Single SCF iteration data."""
    step: int
    energy: float
    fermi_energy: Optional[float]
    band_occurrences: Optional[List[float]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "energy": self.energy,
            "fermi_energy": self.fermi_energy,
        }


@dataclass
class CalculationResults:
    """Results from VASP calculation."""
    total_energy: Optional[float]
    fermi_energy: Optional[float]
    scf_iterations: List[SCFIteration] = field(default_factory=list)
    converged: bool = False
    n_scf_steps: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_energy": self.total_energy,
            "fermi_energy": self.fermi_energy,
            "converged": self.converged,
            "n_scf_steps": self.n_scf_steps,
            "scf_iterations": [s.to_dict() for s in self.scf_iterations[:5]],
        }


class VaspInputParser:
    """Parser for VASP input files (INCAR, POSCAR, KPOINTS)."""
    
    def __init__(self):
        self.incar_params: Dict[str, Any] = {}
        self.structure: Optional[CrystalStructure] = None
        self.kpoints: Optional[KpointGrid] = None
    
    def parse_incar(self, filepath: str) -> Dict[str, Any]:
        """Parse INCAR file.
        
        Args:
            filepath: Path to INCAR file
        
        Returns:
            Dictionary of parameters
        """
        params = {}
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#') or line.startswith('!'):
                    continue
                
                # Parse key = value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip().upper()
                    value = value.split('#')[0].split('!')[0].strip()
                    
                    # Try to convert to appropriate type
                    params[key] = self._convert_value(value)
        
        self.incar_params = params
        return params
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Try int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try bool
        if value.lower() in ('.true.', 'true', 't'):
            return True
        if value.lower() in ('.false.', 'false', 'f'):
            return False
        
        # Return as string
        return value
    
    def parse_poscar(self, filepath: str) -> CrystalStructure:
        """Parse POSCAR/CONTCAR file.
        
        Args:
            filepath: Path to POSCAR file
        
        Returns:
            CrystalStructure object
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # System name
        system_name = lines[0].strip()
        
        # Scale factor
        scale = float(lines[1].strip())
        
        # Lattice vectors
        lattice = np.zeros((3, 3))
        for i in range(3):
            lattice[i] = [float(x) for x in lines[2+i].split()]
        
        # Atom types and counts
        atom_types = lines[5].split()
        atom_counts = [int(x) for x in lines[6].split()]
        
        # Coordinate type
        coord_type = lines[7].strip()[0].upper()  # D or C
        coord_type = "Direct" if coord_type == "D" else "Cartesian"
        
        # Positions
        n_atoms = sum(atom_counts)
        positions = np.zeros((n_atoms, 3))
        for i in range(n_atoms):
            positions[i] = [float(x) for x in lines[8+i].split()[:3]]
        
        self.structure = CrystalStructure(
            system_name=system_name,
            scale=scale,
            lattice_vectors=lattice,
            coord_type=coord_type,
            atom_types=atom_types,
            atom_counts=atom_counts,
            positions=positions,
        )
        
        return self.structure
    
    def parse_kpoints(self, filepath: str) -> KpointGrid:
        """Parse KPOINTS file.
        
        Args:
            filepath: Path to KPOINTS file
        
        Returns:
            KpointGrid object
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip title
        kpt_mode = lines[1].strip().lower()
        
        if kpt_mode == "auto":
            # Automatic k-point generation
            grid_line = lines[2].strip().split()
            grid = [int(x) for x in grid_line[:3]]
            shift = [0.0, 0.0, 0.0]
            
            if len(grid_line) > 3:
                shift = [float(x) for x in grid_line[3:6]]
            
            self.kpoints = KpointGrid(
                mode="Automatic",
                grid=grid,
                shift=shift,
            )
        
        elif kpt_mode in ("gamma", "monkhorst-pack", "mp"):
            # Gamma or Monkhorst-Pack grid
            grid = [int(x) for x in lines[2].split()[:3]]
            shift_line = lines[3].strip().split()
            shift = [float(x) for x in shift_line[:3]] if len(shift_line) >= 3 else [0.0, 0.0, 0.0]
            
            mode = "Gamma" if kpt_mode == "gamma" else "Monkhorst-Pack"
            
            self.kpoints = KpointGrid(
                mode=mode,
                grid=grid,
                shift=shift,
            )
        
        else:
            # Explicit k-points (not fully implemented)
            self.kpoints = KpointGrid(
                mode="Explicit",
                grid=[0, 0, 0],
            )
        
        return self.kpoints
    
    def get_electronic_parameters(self) -> ElectronicParameters:
        """Extract electronic parameters from parsed INCAR."""
        return ElectronicParameters(
            encut=self.incar_params.get("ENCUT", 300.0),
            ismear=self.incar_params.get("ISMEAR", 1),
            sigma=self.incar_params.get("SIGMA", 0.2),
            ispin=self.incar_params.get("ISPIN", 1),
            ibrion=self.incar_params.get("IBRION", -1),
            nsw=self.incar_params.get("NSW", 0),
            nelm=self.incar_params.get("NELM", 60),
            nelmin=self.incar_params.get("NELMIN", 4),
            ediff=self.incar_params.get("EDIFF", 1e-4),
            ediffg=self.incar_params.get("EDIFFG", -0.01),
            algo=self.incar_params.get("ALGO", "Normal"),
        )
    
    def get_calculation_type(self) -> VaspCalculationType:
        """Determine calculation type from parameters."""
        ibrion = self.incar_params.get("IBRION", -1)
        nsw = self.incar_params.get("NSW", 0)
        
        if ibrion in (1, 2, 3) and nsw > 0:
            return VaspCalculationType.RELAX
        elif ibrion == 0 and nsw > 0:
            return VaspCalculationType.MD
        elif self.incar_params.get("LHFCALC", False):
            return VaspCalculationType.HSE
        elif self.incar_params.get("LGW", False):
            return VaspCalculationType.GW
        elif self.incar_params.get("LDAU", False):
            return VaspCalculationType.DFT_PLUS_U
        else:
            return VaspCalculationType.SCF


class VaspOutputParser:
    """Parser for VASP output files (OUTCAR)."""
    
    def __init__(self):
        self.results = CalculationResults()
    
    def parse_outcar(self, filepath: str) -> CalculationResults:
        """Parse OUTCAR file.
        
        Args:
            filepath: Path to OUTCAR file
        
        Returns:
            CalculationResults object
        """
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract total energy
        energy_match = re.search(r"free energy\s+TOTEN\s+=\s+([-\d.]+)", content)
        if energy_match:
            self.results.total_energy = float(energy_match.group(1))
        
        # Extract Fermi energy
        fermi_match = re.search(r"E-fermi\s*:\s*([-\d.]+)", content)
        if fermi_match:
            self.results.fermi_energy = float(fermi_match.group(1))
        
        # Check convergence
        self.results.converged = "reached required accuracy" in content
        
        # Count SCF iterations
        scf_matches = re.findall(r"LOOP:\s*\+?\s*CPU time", content)
        self.results.n_scf_steps = len(scf_matches)
        
        # Extract SCF cycle energies
        scf_pattern = r"LOOP:\s+.+?E=(.+?)\s+dE="
        energies = re.findall(scf_pattern, content)
        
        for i, e_str in enumerate(energies[:50]):  # Limit to first 50
            try:
                energy = float(e_str.replace("+", "").strip())
                self.results.scf_iterations.append(SCFIteration(
                    step=i+1,
                    energy=energy,
                    fermi_energy=None,
                ))
            except ValueError:
                pass
        
        return self.results
    
    def extract_eigenvalues(self, filepath: str) -> Optional[np.ndarray]:
        """Extract eigenvalues from OUTCAR (band structure)."""
        eigenvalues = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        in_band_section = False
        for line in lines:
            if "k-point" in line and "band" in line:
                in_band_section = True
            elif in_band_section and line.strip().startswith("band"):
                # Parse eigenvalue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        eigenvalues.append(float(parts[1]))
                    except ValueError:
                        pass
        
        return np.array(eigenvalues) if eigenvalues else None
    
    def extract_forces(self, filepath: str) -> Optional[np.ndarray]:
        """Extract forces from OUTCAR."""
        forces = []
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Find TOTAL-FORCE sections
        force_sections = re.findall(
            r"TOTAL-FORCE.*?\n(.*?)(?=\n\s*\n|\Z)", 
            content, 
            re.DOTALL
        )
        
        if force_sections:
            # Parse last force section
            last_section = force_sections[-1]
            for line in last_section.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 6 and parts[0].replace('.', '').isdigit():
                    forces.append([float(parts[3]), float(parts[4]), float(parts[5])])
        
        return np.array(forces) if forces else None