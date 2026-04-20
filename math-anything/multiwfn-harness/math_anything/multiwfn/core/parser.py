"""Parsers for Multiwfn file formats.

Supports:
- WFN/WFX wavefunction files
- Gaussian CUBE files
- Multiwfn input scripts
"""

import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class MultiwfnInputParser:
    """Parser for Multiwfn input scripts."""
    
    def __init__(self):
        self.commands: List[Dict[str, Any]] = []
        
    def parse(self, content: str) -> Dict[str, Any]:
        """Parse Multiwfn input script.
        
        Args:
            content: Input script content
            
        Returns:
            Dictionary with parsed commands
        """
        self.commands = []
        lines = content.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('!'):
                continue
            
            # Parse command
            parts = line.split()
            if parts:
                self.commands.append({
                    'command': parts[0].lower(),
                    'args': parts[1:],
                    'line': line_num,
                })
        
        return {
            'commands': self.commands,
            'num_commands': len(self.commands),
        }
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse Multiwfn input file."""
        with open(filepath, 'r') as f:
            content = f.read()
        return self.parse(content)


class WfnFileParser:
    """Parser for WFN (AIM/ACIS wavefunction) files.
    
    Supports both AIM/ACIS WFN format and extended WFX format.
    """
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        
    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse WFN/WFX wavefunction file.
        
        Args:
            filepath: Path to .wfn or .wfx file
            
        Returns:
            Dictionary with wavefunction data
        """
        with open(filepath, 'r') as f:
            content = f.read()
        
        self.data = {
            'title': '',
            'num_atoms': 0,
            'num_molecular_orbitals': 0,
            'num_primitive_gaussians': 0,
            'method': 'unknown',
            'basis_set': 'unknown',
            'atoms': [],
            'basis_functions': [],
            'molecular_orbitals': [],
            'energy': 0.0,
            'virial_ratio': 0.0,
        }
        
        lines = content.split('\n')
        
        # Parse header
        if lines:
            self.data['title'] = lines[0].strip()
        
        # Parse molecule info line
        for line in lines:
            if line.strip().startswith('GAUSSIAN'):
                parts = line.split()
                if len(parts) >= 6:
                    self.data['num_molecular_orbitals'] = int(parts[1])
                    self.data['num_primitive_gaussians'] = int(parts[4])
                    self.data['num_atoms'] = int(parts[5])
                break
        
        # Parse atoms
        atom_pattern = re.compile(
            r'(\w+)\s+([\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)'
        )
        
        for line in lines:
            match = atom_pattern.match(line.strip())
            if match and len(self.data['atoms']) < self.data['num_atoms']:
                self.data['atoms'].append({
                    'symbol': match.group(1),
                    'charge': float(match.group(2)),
                    'x': float(match.group(3)),
                    'y': float(match.group(4)),
                    'z': float(match.group(5)),
                })
        
        # Parse basis function centers
        center_pattern = re.compile(r'CENTRE\s+(\d+)\s+(\w+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)')
        for line in lines:
            match = center_pattern.match(line.strip())
            if match:
                self.data['basis_functions'].append({
                    'center': int(match.group(1)),
                    'symbol': match.group(2),
                    'x': float(match.group(3)),
                    'y': float(match.group(4)),
                    'z': float(match.group(5)),
                })
        
        # Parse molecular orbitals
        self._parse_molecular_orbitals(lines)
        
        # Parse energy and virial
        for line in lines:
            if 'TOTAL ENERGY' in line:
                match = re.search(r'([-\d\.]+)', line)
                if match:
                    self.data['energy'] = float(match.group(1))
            elif 'VIRIAL RATIO' in line:
                match = re.search(r'([-\d\.]+)', line)
                if match:
                    self.data['virial_ratio'] = float(match.group(1))
        
        return self.data
    
    def _parse_molecular_orbitals(self, lines: List[str]):
        """Parse molecular orbital coefficients and energies."""
        mo_pattern = re.compile(r'MO\s+(\d+)\s+([-\d\.]+)\s+OCC\s+([\d\.]+)')
        
        in_mo = False
        current_mo = None
        
        for line in lines:
            mo_match = mo_pattern.match(line.strip())
            if mo_match:
                in_mo = True
                current_mo = {
                    'index': int(mo_match.group(1)),
                    'energy': float(mo_match.group(2)),
                    'occupation': float(mo_match.group(3)),
                    'coefficients': [],
                }
                self.data['molecular_orbitals'].append(current_mo)
            elif in_mo and current_mo is not None:
                # Parse coefficients
                coeffs = [float(x) for x in line.split() if self._is_float(x)]
                current_mo['coefficients'].extend(coeffs)
                
                # Check if MO is complete
                if len(current_mo['coefficients']) >= self.data['num_primitive_gaussians']:
                    in_mo = False
    
    def _is_float(self, s: str) -> bool:
        """Check if string can be converted to float."""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse WFN file."""
        return self.parse(filepath)


class CubeFileParser:
    """Parser for Gaussian CUBE format files.
    
    CUBE files contain volumetric data on a regular 3D grid,
    commonly used for electron density, electrostatic potential, etc.
    """
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        
    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse CUBE file.
        
        Args:
            filepath: Path to .cube file
            
        Returns:
            Dictionary with cube data including:
            - title: File title
            - num_atoms: Number of atoms
            - origin: Grid origin (x, y, z)
            - axes: Grid axes (3x3 matrix)
            - atoms: List of atom data
            - data: 3D numpy array of values
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 6:
            raise ValueError("Invalid CUBE file: too few lines")
        
        self.data = {
            'title1': lines[0].strip(),
            'title2': lines[1].strip(),
            'num_atoms': 0,
            'origin': (0.0, 0.0, 0.0),
            'nx': 0, 'ny': 0, 'nz': 0,
            'xvec': (0.0, 0.0, 0.0),
            'yvec': (0.0, 0.0, 0.0),
            'zvec': (0.0, 0.0, 0.0),
            'atoms': [],
            'data': None,
        }
        
        # Parse grid specification (line 3)
        parts = lines[2].split()
        self.data['num_atoms'] = abs(int(parts[0]))  # Negative = orbital CUBE
        self.data['origin'] = (
            float(parts[1]), float(parts[2]), float(parts[3])
        )
        
        # Parse grid dimensions (lines 4-6)
        nx, x1, x2, x3 = lines[3].split()
        ny, y1, y2, y3 = lines[4].split()
        nz, z1, z2, z3 = lines[5].split()
        
        self.data['nx'] = int(nx)
        self.data['ny'] = int(ny)
        self.data['nz'] = int(nz)
        
        self.data['xvec'] = (float(x1), float(x2), float(x3))
        self.data['yvec'] = (float(y1), float(y2), float(y3))
        self.data['zvec'] = (float(z1), float(z2), float(z3))
        
        # Parse atoms
        line_idx = 6
        for i in range(abs(self.data['num_atoms'])):
            parts = lines[line_idx].split()
            if len(parts) >= 5:
                self.data['atoms'].append({
                    'atomic_number': int(parts[0]),
                    'charge': float(parts[1]),
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'z': float(parts[4]),
                })
            line_idx += 1
        
        # Parse volumetric data
        values = []
        for line in lines[line_idx:]:
            values.extend([float(x) for x in line.split()])
        
        # Reshape to 3D array
        expected_size = self.data['nx'] * self.data['ny'] * self.data['nz']
        if len(values) >= expected_size:
            self.data['data'] = np.array(values[:expected_size]).reshape(
                (self.data['nx'], self.data['ny'], self.data['nz'])
            )
        
        return self.data
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse CUBE file."""
        return self.parse(filepath)
    
    def get_grid_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get grid coordinate arrays.
        
        Returns:
            Tuple of (X, Y, Z) coordinate arrays
        """
        nx, ny, nz = self.data['nx'], self.data['ny'], self.data['nz']
        origin = self.data['origin']
        xvec = self.data['xvec']
        yvec = self.data['yvec']
        zvec = self.data['zvec']
        
        # Generate grid coordinates
        x = np.linspace(0, nx-1, nx) * xvec[0] + origin[0]
        y = np.linspace(0, ny-1, ny) * yvec[1] + origin[1]
        z = np.linspace(0, nz-1, nz) * zvec[2] + origin[2]
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        return X, Y, Z
    
    def compute_volume_element(self) -> float:
        """Compute volume element (dV) for integration."""
        xvec = np.array(self.data['xvec'])
        yvec = np.array(self.data['yvec'])
        zvec = np.array(self.data['zvec'])
        
        # Volume of parallelepiped
        volume = np.abs(np.dot(xvec, np.cross(yvec, zvec)))
        return float(volume)
