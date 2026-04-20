"""VASP INCAR file parser with symbolic constraint extraction.

Parses VASP INCAR input files and extracts mathematical structures,
including symbolic constraints for physical parameters.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class IncarParameter:
    """A parameter in INCAR with its value and constraints."""
    name: str
    value: Any
    raw_string: str
    line_number: int
    description: str = ""
    constraints: List[str] = field(default_factory=list)


@dataclass
class IncarSection:
    """A section in INCAR (e.g., electronic, ionic, parallel)."""
    name: str
    parameters: List[IncarParameter]
    description: str = ""


class IncarParser:
    """Parse VASP INCAR files and extract symbolic constraints.
    
    Example:
        parser = IncarParser()
        result = parser.parse("INCAR")
        
        # Access parameters
        encut = result.get("ENCUT")
        print(f"ENCUT = {encut.value} eV")
        
        # Validate constraints
        constraints = result.validate_constraints()
        for c in constraints:
            print(f"{c['expression']}: {'✓' if c['satisfied'] else '✗'}")
    """
    
    # Parameter definitions with constraints
    PARAMETER_DEFS = {
        # System
        "SYSTEM": {"type": str, "constraints": []},
        
        # Electronic relaxation
        "ENCUT": {
            "type": float,
            "constraints": ["> 0"],
            "unit": "eV",
            "description": "Plane-wave cutoff energy"
        },
        "EDIFF": {
            "type": float,
            "constraints": ["> 0"],
            "description": "Electronic convergence criterion"
        },
        "NELM": {
            "type": int,
            "constraints": ["> 0"],
            "description": "Max electronic SC steps"
        },
        "NELMIN": {
            "type": int,
            "constraints": [">= 0"],
            "description": "Min electronic SC steps"
        },
        
        # Smearing
        "ISMEAR": {
            "type": int,
            "constraints": ["in [-5, -4, -3, -2, -1, 0, 1, 2]"],
            "description": "Smearing method",
            "valid_values": [-5, -4, -3, -2, -1, 0, 1, 2]
        },
        "SIGMA": {
            "type": float,
            "constraints": ["> 0"],
            "unit": "eV",
            "description": "Smearing width"
        },
        
        # Ionic relaxation
        "NSW": {
            "type": int,
            "constraints": [">= 0"],
            "description": "Max ionic steps"
        },
        "EDIFFG": {
            "type": float,
            "constraints": ["!= 0"],
            "description": "Ionic convergence criterion (negative for energy, positive for force)"
        },
        "IBRION": {
            "type": int,
            "constraints": ["in [-1, 0, 1, 2, 3, 5, 6, 7, 8, 44]"],
            "description": "Ionic relaxation algorithm",
            "valid_values": [-1, 0, 1, 2, 3, 5, 6, 7, 8, 44]
        },
        "ISIF": {
            "type": int,
            "constraints": ["in [0, 1, 2, 3, 4, 5, 6, 7]"],
            "description": "Stress/relaxation flags",
            "valid_values": [0, 1, 2, 3, 4, 5, 6, 7]
        },
        
        # Spin
        "ISPIN": {
            "type": int,
            "constraints": ["in [1, 2]"],
            "description": "Spin polarization",
            "valid_values": [1, 2]
        },
        "MAGMOM": {
            "type": str,
            "constraints": [],
            "description": "Magnetic moments"
        },
        
        # Parallelization
        "NCORE": {
            "type": int,
            "constraints": ["> 0"],
            "description": "Cores per orbital"
        },
        "KPAR": {
            "type": int,
            "constraints": ["> 0"],
            "description": "k-point parallelization"
        },
        
        # Output control
        "NWRITE": {
            "type": int,
            "constraints": ["in [0, 1, 2, 3, 4]"],
            "description": "Output verbosity",
            "valid_values": [0, 1, 2, 3, 4]
        },
        "LWAVE": {
            "type": bool,
            "constraints": [],
            "description": "Write WAVECAR"
        },
        "LCHARG": {
            "type": bool,
            "constraints": [],
            "description": "Write CHGCAR"
        },
    }
    
    def __init__(self):
        self.parameters: Dict[str, IncarParameter] = {}
        self.sections: List[IncarSection] = []
        self.raw_lines: List[str] = []
    
    def parse(self, filepath: str) -> "IncarResult":
        """Parse INCAR file and return structured result."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.raw_lines = f.readlines()
        
        self.parameters = {}
        current_section = "general"
        
        for line_num, line in enumerate(self.raw_lines, 1):
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or stripped.startswith('!'):
                continue
            
            # Parse parameter
            param = self._parse_line(stripped, line_num)
            if param:
                self.parameters[param.name] = param
        
        return IncarResult(self.parameters, self.raw_lines)
    
    def _parse_line(self, line: str, line_num: int) -> Optional[IncarParameter]:
        """Parse a single INCAR line."""
        # Remove inline comments
        if '#' in line:
            line = line.split('#')[0]
        if '!' in line:
            line = line.split('!')[0]
        
        line = line.strip()
        if not line or '=' not in line:
            return None
        
        # Parse key = value
        parts = line.split('=', 1)
        if len(parts) != 2:
            return None
        
        name = parts[0].strip().upper()
        value_str = parts[1].strip()
        
        # Get parameter definition
        param_def = self.PARAMETER_DEFS.get(name, {"type": str, "constraints": []})
        
        # Convert value to proper type
        try:
            if param_def["type"] == bool:
                value = value_str.upper() in ("TRUE", ".TRUE.", "T", "1", "YES")
            elif param_def["type"] == int:
                value = int(value_str)
            elif param_def["type"] == float:
                value = float(value_str.replace('d', 'e').replace('D', 'E'))
            else:
                value = value_str
        except ValueError:
            value = value_str
        
        # Create parameter
        return IncarParameter(
            name=name,
            value=value,
            raw_string=value_str,
            line_number=line_num,
            description=param_def.get("description", ""),
            constraints=param_def.get("constraints", [])
        )


class IncarResult:
    """Result of INCAR parsing with validation methods."""
    
    def __init__(self, parameters: Dict[str, IncarParameter], raw_lines: List[str]):
        self.parameters = parameters
        self.raw_lines = raw_lines
    
    def get(self, name: str, default=None) -> Optional[IncarParameter]:
        """Get parameter by name."""
        return self.parameters.get(name.upper(), default)
    
    def get_value(self, name: str, default=None):
        """Get parameter value directly."""
        param = self.get(name)
        return param.value if param else default
    
    def validate_constraints(self) -> List[Dict[str, Any]]:
        """Validate all symbolic constraints."""
        results = []
        
        for name, param in self.parameters.items():
            for constraint in param.constraints:
                result = self._check_constraint(name, param.value, constraint)
                results.append(result)
        
        return results
    
    def _check_constraint(self, name: str, value, constraint: str) -> Dict[str, Any]:
        """Check if a single constraint is satisfied."""
        result = {
            "parameter": name,
            "value": value,
            "expression": f"{name} {constraint}",
            "satisfied": False,
            "message": ""
        }
        
        try:
            if constraint == "> 0":
                result["satisfied"] = float(value) > 0
            elif constraint == ">= 0":
                result["satisfied"] = float(value) >= 0
            elif constraint == "!= 0":
                result["satisfied"] = float(value) != 0
            elif constraint.startswith("in ["):
                # Extract valid values
                valid = eval(constraint.replace("in ", ""))
                result["satisfied"] = value in valid
            elif constraint.startswith("in ("):
                valid = eval(constraint.replace("in ", ""))
                result["satisfied"] = value in valid
            else:
                result["message"] = f"Unknown constraint: {constraint}"
        except (ValueError, TypeError) as e:
            result["message"] = f"Error checking constraint: {e}"
        
        return result
    
    def extract_symbolic_constraints(self) -> List[Dict[str, Any]]:
        """Extract all symbolic constraints in MathSchema format."""
        constraints = []
        
        for name, param in self.parameters.items():
            for constraint in param.constraints:
                expr = f"{name} {constraint}"
                # Check if satisfied
                check = self._check_constraint(name, param.value, constraint)
                
                constraints.append({
                    "expression": expr,
                    "description": param.description,
                    "variables": [name],
                    "confidence": 1.0,
                    "satisfied": check["satisfied"],
                    "value": param.value,
                })
        
        # Add derived constraints
        encut = self.get_value("ENCUT")
        if encut:
            constraints.append({
                "expression": "ENCUT > max(ENMAX)",
                "description": "Cutoff must exceed maximum ENMAX from POTCARs",
                "variables": ["ENCUT", "ENMAX"],
                "confidence": 0.9,
                "satisfied": None,  # Cannot check without POTCAR
            })
        
        # Charge conservation
        constraints.append({
            "expression": "∫ n(r) dr = N_electrons",
            "description": "Charge conservation - electron density integrates to total electrons",
            "variables": ["n", "N_electrons"],
            "confidence": 1.0,
            "satisfied": True,  # Always true in DFT
        })
        
        return constraints
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameters": {
                name: {
                    "value": param.value,
                    "description": param.description,
                    "constraints": param.constraints,
                    "line": param.line_number,
                }
                for name, param in self.parameters.items()
            },
            "symbolic_constraints": self.extract_symbolic_constraints(),
            "validation": self.validate_constraints(),
        }


# Convenience function
def parse_incar(filepath: str) -> IncarResult:
    """Parse INCAR file and return result."""
    parser = IncarParser()
    return parser.parse(filepath)
