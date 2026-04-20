"""Parsers for SolidWorks Simulation file formats.
"""

import re
from typing import Dict, List, Any, Optional


class CWRParser:
    """Parser for SolidWorks CWR (CosmosWorks Results) files.
    
    CWR files contain simulation results.
    """
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        
    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse CWR results file.
        
        Args:
            filepath: Path to .cwr file
            
        Returns:
            Dictionary with results data
        """
        self.data = {
            'file_path': filepath,
            'study_name': '',
            'study_type': '',
            'num_nodes': 0,
            'num_elements': 0,
            'results': {},
            'fixed_constraints': [],
            'forces': [],
            'pressures': [],
            'element_quality': 'high',
        }
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse header info
            self._parse_header(content)
            
            # Parse results
            self._parse_results(content)
            
        except Exception as e:
            self.data['error'] = str(e)
        
        return self.data
    
    def _parse_header(self, content: str):
        """Parse file header."""
        lines = content.split('\n')
        
        for line in lines[:50]:  # Check first 50 lines
            line = line.strip()
            
            if 'Study Name:' in line:
                self.data['study_name'] = line.split(':', 1)[-1].strip()
            elif 'Study Type:' in line:
                self.data['study_type'] = line.split(':', 1)[-1].strip()
            elif 'Nodes:' in line or 'Number of Nodes' in line:
                match = re.search(r'(\d+)', line)
                if match:
                    self.data['num_nodes'] = int(match.group(1))
            elif 'Elements:' in line or 'Number of Elements' in line:
                match = re.search(r'(\d+)', line)
                if match:
                    self.data['num_elements'] = int(match.group(1))
    
    def _parse_results(self, content: str):
        """Parse results section."""
        # Look for stress results
        if 'Von Mises' in content or 'von Mises' in content:
            self.data['results']['stress_available'] = True
        
        if 'Displacement' in content:
            self.data['results']['displacement_available'] = True
        
        if 'Strain' in content:
            self.data['results']['strain_available'] = True
        
        if 'Factor of Safety' in content or 'Safety Factor' in content:
            self.data['results']['safety_factor_available'] = True
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse CWR file."""
        return self.parse(filepath)


class StudyParser:
    """Parser for SolidWorks Simulation study files."""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        
    def parse(self, content: str) -> Dict[str, Any]:
        """Parse study definition.
        
        Args:
            content: Study definition content
            
        Returns:
            Dictionary with study information
        """
        self.data = {
            'study_name': '',
            'study_type': '',
            'material': '',
            'fixtures': [],
            'external_loads': [],
            'mesh_settings': {},
        }
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Study:'):
                self.data['study_name'] = line.split(':', 1)[-1].strip()
            elif 'Type:' in line:
                self.data['study_type'] = line.split(':', 1)[-1].strip()
            elif 'Material:' in line:
                self.data['material'] = line.split(':', 1)[-1].strip()
            elif 'Fixture' in line or 'Fixed' in line:
                self.data['fixtures'].append(line)
            elif 'Force' in line or 'Pressure' in line or 'Load' in line:
                self.data['external_loads'].append(line)
        
        return self.data
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse study file."""
        with open(filepath, 'r') as f:
            content = f.read()
        return self.parse(content)


class MaterialParser:
    """Parser for SolidWorks material files."""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        
    def parse(self, content: str) -> Dict[str, Any]:
        """Parse material definition.
        
        Args:
            content: Material definition content
            
        Returns:
            Dictionary with material properties
        """
        self.data = {
            'name': '',
            'description': '',
            'properties': {},
            'elastic_modulus': 0.0,
            'poisson_ratio': 0.0,
            'density': 0.0,
            'yield_strength': 0.0,
            'tensile_strength': 0.0,
        }
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Name:'):
                self.data['name'] = line.split(':', 1)[-1].strip()
            elif 'Elastic Modulus' in line or 'Young' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    self.data['elastic_modulus'] = float(match.group(1))
            elif 'Poisson' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    self.data['poisson_ratio'] = float(match.group(1))
            elif 'Density' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    self.data['density'] = float(match.group(1))
            elif 'Yield' in line or 'Yield Strength' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    self.data['yield_strength'] = float(match.group(1))
            elif 'Tensile' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    self.data['tensile_strength'] = float(match.group(1))
        
        # Compute derived properties
        E = self.data['elastic_modulus']
        nu = self.data['poisson_ratio']
        
        if E > 0 and nu > 0:
            self.data['properties']['shear_modulus'] = E / (2 * (1 + nu))
            self.data['properties']['bulk_modulus'] = E / (3 * (1 - 2 * nu))
            self.data['properties']['lame_lambda'] = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        return self.data
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse material file."""
        with open(filepath, 'r') as f:
            content = f.read()
        return self.parse(content)
