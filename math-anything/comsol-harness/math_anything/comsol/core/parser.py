"""Parsers for COMSOL file formats.

Supports:
- MPH model files (basic metadata)
- Model Java files
- XML exports
"""

import re
import zipfile
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET


class MPHParser:
    """Parser for COMSOL MPH model files.
    
    MPH files are ZIP archives containing XML model definitions.
    """
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        
    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse MPH file.
        
        Args:
            filepath: Path to .mph file
            
        Returns:
            Dictionary with model metadata
        """
        self.data = {
            'file_path': filepath,
            'model_name': '',
            'version': '',
            'physics_interfaces': [],
            'study_types': [],
            'parameters': {},
            'variables': {},
            'num_dofs': 0,
            'element_order': 'quadratic',
        }
        
        try:
            with zipfile.ZipFile(filepath, 'r') as zf:
                # List files in archive
                files = zf.namelist()
                self.data['archive_contents'] = files
                
                # Parse model XML
                if 'model.xml' in files:
                    with zf.open('model.xml') as f:
                        content = f.read().decode('utf-8')
                        self._parse_model_xml(content)
                
                # Parse parameters
                if 'parameters.xml' in files:
                    with zf.open('parameters.xml') as f:
                        content = f.read().decode('utf-8')
                        self._parse_parameters_xml(content)
                        
        except Exception as e:
            self.data['error'] = str(e)
        
        return self.data
    
    def _parse_model_xml(self, content: str):
        """Parse model XML content."""
        try:
            root = ET.fromstring(content)
            
            # Extract model name
            if 'name' in root.attrib:
                self.data['model_name'] = root.attrib['name']
            
            # Find physics interfaces
            for elem in root.iter():
                tag = elem.tag.lower()
                if 'physics' in tag or 'interface' in tag:
                    name = elem.get('name', '')
                    if name:
                        self.data['physics_interfaces'].append(name)
                
                # Find study types
                if 'study' in tag:
                    study_type = elem.get('type', '')
                    if study_type:
                        self.data['study_types'].append(study_type)
                        
        except ET.ParseError:
            pass
    
    def _parse_parameters_xml(self, content: str):
        """Parse parameters XML content."""
        try:
            root = ET.fromstring(content)
            
            for param in root.iter('parameter'):
                name = param.get('name', '')
                value = param.get('value', '')
                if name:
                    self.data['parameters'][name] = value
                    
        except ET.ParseError:
            pass
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse MPH file."""
        return self.parse(filepath)


class ModelParser:
    """Parser for COMSOL Model XML files."""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        
    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse Model XML file."""
        self.data = {
            'model_name': '',
            'physics': [],
            'boundary_conditions': [],
            'materials': [],
        }
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Extract model name
            self.data['model_name'] = root.get('name', '')
            
            # Extract physics
            for physics in root.findall('.//physics'):
                self.data['physics'].append({
                    'name': physics.get('name', ''),
                    'type': physics.get('type', ''),
                })
            
            # Extract boundary conditions
            for bc in root.findall('.//boundarycondition'):
                self.data['boundary_conditions'].append({
                    'id': bc.get('id', ''),
                    'type': bc.get('type', ''),
                    'boundary': bc.get('boundary', ''),
                    'equation': bc.get('equation', ''),
                })
            
            # Extract materials
            for mat in root.findall('.//material'):
                self.data['materials'].append({
                    'name': mat.get('name', ''),
                    'properties': {},
                })
                
        except ET.ParseError as e:
            self.data['error'] = str(e)
        
        return self.data
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse Model XML file."""
        return self.parse(filepath)


class JavaParser:
    """Parser for COMSOL Java API files."""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        
    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse Java model file.
        
        Args:
            filepath: Path to .java file
            
        Returns:
            Dictionary with extracted information
        """
        self.data = {
            'class_name': '',
            'model_name': '',
            'physics_interfaces': [],
            'study_types': [],
            'parameters': {},
            'boundary_conditions': [],
            'materials': [],
        }
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Class name
            if line.startswith('public class'):
                match = re.search(r'public class (\w+)', line)
                if match:
                    self.data['class_name'] = match.group(1)
            
            # Model name
            if 'ModelUtil.create' in line:
                match = re.search(r'create\("([^"]+)"', line)
                if match:
                    self.data['model_name'] = match.group(1)
            
            # Physics interfaces
            if 'physics().create' in line:
                match = re.search(r'create\("([^"]+)",\s*"([^"]+)"', line)
                if match:
                    self.data['physics_interfaces'].append({
                        'tag': match.group(1),
                        'type': match.group(2),
                    })
            
            # Study types
            if 'study().create' in line:
                match = re.search(r'create\("([^"]+)",\s*"([^"]+)"', line)
                if match:
                    self.data['study_types'].append({
                        'tag': match.group(1),
                        'type': match.group(2),
                    })
            
            # Parameters
            if 'param().set' in line:
                match = re.search(r'set\("([^"]+)",\s*"([^"]+)"', line)
                if match:
                    self.data['parameters'][match.group(1)] = match.group(2)
            
            # Boundary conditions
            if 'create("fix' in line or 'create("load' in line:
                match = re.search(r'create\("([^"]+)"', line)
                if match:
                    self.data['boundary_conditions'].append(match.group(1))
            
            # Materials
            if 'material().create' in line:
                match = re.search(r'create\("([^"]+)"', line)
                if match:
                    self.data['materials'].append(match.group(1))
        
        return self.data
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse Java model file."""
        return self.parse(filepath)
    
    def extract_physics_sequence(self) -> List[str]:
        """Extract the sequence of physics operations."""
        sequence = []
        
        for phys in self.data.get('physics_interfaces', []):
            sequence.append(f"Create physics: {phys['type']}")
        
        for bc in self.data.get('boundary_conditions', []):
            sequence.append(f"Add BC: {bc}")
        
        return sequence
