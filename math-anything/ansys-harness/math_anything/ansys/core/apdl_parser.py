"""Enhanced APDL parser with symbolic constraint support.

Parses Ansys APDL scripts and extracts mathematical structures
with symbolic constraints for FEM analysis.
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class AnalysisType(Enum):
    """Ansys analysis types."""
    STATIC = "static_structural"
    MODAL = "modal"
    HARMONIC = "harmonic"
    TRANSIENT = "transient_structural"
    BUCKLING = "buckling"
    THERMAL = "thermal"


@dataclass
class MaterialProperty:
    """Material property with constraints."""
    name: str
    material_id: int
    value: float
    unit: str = ""
    constraints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "material_id": self.material_id,
            "value": self.value,
            "unit": self.unit,
            "constraints": self.constraints,
        }


@dataclass
class APDLCommand:
    """Single APDL command."""
    command: str
    args: List[str]
    line_number: int
    raw: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "args": self.args,
            "line_number": self.line_number,
            "raw": self.raw,
        }


@dataclass
class APDLResults:
    """Results of APDL parsing with validation."""
    commands: List[APDLCommand]
    parameters: Dict[str, Any]
    materials: List[MaterialProperty]
    analysis_type: AnalysisType
    constraints: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_commands": len(self.commands),
            "parameters": self.parameters,
            "materials": [m.to_dict() for m in self.materials],
            "analysis_type": self.analysis_type.value,
            "constraints": self.constraints,
        }


class APDLSymbolicConstraints:
    """Symbolic constraints for APDL/FEM parameters."""
    
    # Material property constraints
    MATERIAL_CONSTRAINTS = {
        "EX": [  # Young's modulus
            ("> 0", "Young's modulus must be positive (critical)"),
            ("< 1e15", "Young's modulus should be physically reasonable (info)"),
        ],
        "PRXY": [  # Poisson's ratio
            ("> -1", "Poisson's ratio must be > -1 (critical)"),
            ("< 0.5", "Poisson's ratio must be < 0.5 for stability (critical)"),
        ],
        "DENS": [  # Density
            ("> 0", "Density must be positive (critical)"),
        ],
        "ALPX": [  # Thermal expansion
            ("> -1e-2", "Thermal expansion coefficient lower bound (warning)"),
            ("< 1e-2", "Thermal expansion coefficient upper bound (warning)"),
        ],
        "KXX": [  # Thermal conductivity
            ("> 0", "Thermal conductivity must be positive (critical)"),
        ],
        "C": [  # Specific heat
            ("> 0", "Specific heat must be positive (critical)"),
        ],
        "MU": [  # Coefficient of friction
            (">= 0", "Friction coefficient must be non-negative (critical)"),
        ],
    }
    
    # Mesh constraints
    MESH_CONSTRAINTS = {
        "ESIZE": [
            ("> 0", "Element size must be positive (critical)"),
            ("< characteristic_length / 10", "Element size should be small enough (warning)"),
        ],
        "SMRTSIZE": [
            (">= 1", "Smart size level should be >= 1 (info)"),
            ("<= 10", "Smart size level should be <= 10 (info)"),
        ],
    }
    
    # Solution constraints
    SOLUTION_CONSTRAINTS = {
        "TIME": [
            (">= 0", "Time must be non-negative (critical)"),
        ],
        "NSUBST": [
            ("> 0", "Number of substeps must be positive (critical)"),
            ("< 10000", "Too many substeps may be inefficient (warning)"),
        ],
        "DELTIM": [
            ("> 0", "Time step must be positive (critical)"),
            ("< TEND / 100", "Time step should be small enough (info)"),
        ],
    }
    
    @classmethod
    def validate_material_property(cls, prop_name: str, value: float) -> List[Dict[str, Any]]:
        """Validate a material property against constraints."""
        results = []
        constraints = cls.MATERIAL_CONSTRAINTS.get(prop_name, [])
        
        for constraint_expr, description in constraints:
            result = {
                "property": prop_name,
                "value": value,
                "constraint": constraint_expr,
                "description": description,
                "satisfied": False,
            }
            
            try:
                if constraint_expr.startswith(">="):
                    threshold = float(constraint_expr.replace(">=", "").strip())
                    result["satisfied"] = float(value) >= threshold
                elif constraint_expr.startswith(">"):
                    threshold = float(constraint_expr.replace(">", "").strip())
                    result["satisfied"] = float(value) > threshold
                elif constraint_expr.startswith("<="):
                    threshold = float(constraint_expr.replace("<=", "").strip())
                    result["satisfied"] = float(value) <= threshold
                elif constraint_expr.startswith("<"):
                    threshold = float(constraint_expr.replace("<", "").strip())
                    result["satisfied"] = float(value) < threshold
                elif ">=" in constraint_expr:
                    import re
                    match = re.search(r'>=\s*([-\d.eE+]+)', constraint_expr)
                    if match:
                        result["satisfied"] = float(value) >= float(match.group(1))
                elif ">" in constraint_expr:
                    import re
                    match = re.search(r'>\s*([-\d.eE+]+)', constraint_expr)
                    if match:
                        result["satisfied"] = float(value) > float(match.group(1))
                elif "<=" in constraint_expr:
                    import re
                    match = re.search(r'<=\s*([-\d.eE+]+)', constraint_expr)
                    if match:
                        result["satisfied"] = float(value) <= float(match.group(1))
                elif "<" in constraint_expr:
                    import re
                    match = re.search(r'<\s*([-\d.eE+]+)', constraint_expr)
                    if match:
                        result["satisfied"] = float(value) < float(match.group(1))
                else:
                    result["satisfied"] = True
            except (ValueError, TypeError):
                result["satisfied"] = False
            
            results.append(result)
        
        return results


class EnhancedAPDLParser:
    """Enhanced APDL parser with symbolic constraint support.
    
    Example:
        parser = EnhancedAPDLParser()
        result = parser.parse_file("model.inp")
        
        print(f"Analysis type: {result.analysis_type.value}")
        print(f"Materials: {len(result.materials)}")
        
        # Check constraints
        for c in result.constraints:
            print(f"{c['property']}: {c['constraint']} -> {c['satisfied']}")
    """
    
    # Command patterns for mathematical extraction
    COMMAND_PATTERNS = {
        # Preprocessor
        "ET": "element_type_definition",
        "KEYOPT": "element_option",
        "MP": "material_property",
        "MPDATA": "material_data",
        "R": "real_constant",
        "RMORE": "real_constant_more",
        "TYPE": "element_type_selection",
        "MAT": "material_selection",
        "REAL": "real_constant_selection",
        "ESYS": "coordinate_system",
        "SECNUM": "section_selection",
        
        # Meshing
        "N": "node_definition",
        "E": "element_definition",
        "EN": "element_by_nodes",
        "NGEN": "node_generation",
        "EGEN": "element_generation",
        "ESIZE": "element_size",
        "SMRTSIZE": "smart_size",
        "MSHAPE": "mesh_shape",
        "MSHKEY": "meshing_method",
        "AMESH": "area_mesh",
        "VMESH": "volume_mesh",
        
        # Solution
        "ANTYPE": "analysis_type",
        "TIME": "time",
        "NSUBST": "substeps",
        "DELTIM": "time_step",
        "KBC": "load_application",
        "AUTOTS": "auto_time_step",
        "LNSRCH": "line_search",
        "PRED": "predictor",
        "NEQIT": "equilibrium_iterations",
        "NCNV": "convergence_criteria",
        "SSTIF": "stress_stiffness",
        "NROPT": "newton_raphson",
        
        # Loads
        "D": "displacement_constraint",
        "F": "force_load",
        "SFA": "surface_load_area",
        "SFE": "surface_load_element",
        "SF": "surface_load_nodes",
        "BF": "body_force",
        "BFE": "body_force_element",
        "IC": "initial_condition",
        
        # Postprocessing
        "/POST1": "postprocessor",
        "SET": "load_step_selection",
        "PLDISP": "plot_displacement",
        "PLNSOL": "plot_nodal_solution",
        "PRESOL": "print_element_solution",
        "PRNSOL": "print_nodal_solution",
    }
    
    def __init__(self):
        self.commands: List[APDLCommand] = []
        self.parameters: Dict[str, Any] = {}
        self.materials: List[MaterialProperty] = []
        self.analysis_type: AnalysisType = AnalysisType.STATIC
        self.constraints: List[Dict[str, Any]] = []
    
    def parse(self, content: str) -> APDLResults:
        """Parse APDL script content."""
        self.commands = []
        self.parameters = {}
        self.materials = []
        self.constraints = []
        
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('!'):
                continue
            
            # Remove inline comments
            if '!' in line:
                line = line[:line.index('!')].strip()
            
            # Check for parameter assignment
            param_match = re.match(r'(\w+)\s*=\s*(.+)', line)
            if param_match:
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                # Try to convert to number
                try:
                    param_value = float(param_value)
                except ValueError:
                    pass
                self.parameters[param_name] = param_value
                continue
            
            # Parse APDL command
            parts = line.split(',')
            if parts:
                cmd = parts[0].strip().upper()
                args = [p.strip() for p in parts[1:]]
                
                apdl_cmd = APDLCommand(
                    command=cmd,
                    args=args,
                    line_number=line_num,
                    raw=line,
                )
                self.commands.append(apdl_cmd)
                
                # Extract specific information
                self._process_command(apdl_cmd)
        
        # Compile results
        return APDLResults(
            commands=self.commands,
            parameters=self.parameters,
            materials=self.materials,
            analysis_type=self.analysis_type,
            constraints=self.constraints,
        )
    
    def parse_file(self, filepath: str) -> APDLResults:
        """Parse APDL file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return self.parse(content)
    
    def _process_command(self, cmd: APDLCommand):
        """Process a command to extract mathematical information."""
        cmd_name = cmd.command
        args = cmd.args
        
        # Analysis type
        if cmd_name == "ANTYPE":
            self._parse_analysis_type(args)
        
        # Material properties
        elif cmd_name == "MP":
            self._parse_material_property(args)
        
        # Element size
        elif cmd_name == "ESIZE":
            self._parse_element_size(args)
        
        # Time/step settings
        elif cmd_name == "TIME":
            self._parse_time_setting(args)
        elif cmd_name == "NSUBST":
            self._parse_substeps(args)
        elif cmd_name == "DELTIM":
            self._parse_time_step(args)
    
    def _parse_analysis_type(self, args: List[str]):
        """Parse ANTYPE command."""
        if not args:
            return
        
        atype = args[0].upper()
        type_map = {
            "0": AnalysisType.STATIC,
            "STATIC": AnalysisType.STATIC,
            "2": AnalysisType.MODAL,
            "MODAL": AnalysisType.MODAL,
            "3": AnalysisType.HARMONIC,
            "HARMONIC": AnalysisType.HARMONIC,
            "4": AnalysisType.TRANSIENT,
            "TRANS": AnalysisType.TRANSIENT,
            "8": AnalysisType.BUCKLING,
            "BUCKLE": AnalysisType.BUCKLING,
        }
        
        self.analysis_type = type_map.get(atype, AnalysisType.STATIC)
    
    def _parse_material_property(self, args: List[str]):
        """Parse MP command for material properties."""
        if len(args) < 3:
            return
        
        prop_name = args[0].upper()
        try:
            mat_id = int(args[1])
            value = float(args[2])
        except (ValueError, IndexError):
            return
        
        # Create material property
        mat_prop = MaterialProperty(
            name=prop_name,
            material_id=mat_id,
            value=value,
            constraints=[c[0] for c in APDLSymbolicConstraints.MATERIAL_CONSTRAINTS.get(prop_name, [])],
        )
        self.materials.append(mat_prop)
        
        # Validate constraints
        validations = APDLSymbolicConstraints.validate_material_property(prop_name, value)
        self.constraints.extend(validations)
    
    def _parse_element_size(self, args: List[str]):
        """Parse ESIZE command."""
        if not args:
            return
        
        try:
            size = float(args[0])
            validations = APDLSymbolicConstraints.validate_material_property("ESIZE", size)
            for v in validations:
                v["command"] = "ESIZE"
            self.constraints.extend(validations)
        except ValueError:
            pass
    
    def _parse_time_setting(self, args: List[str]):
        """Parse TIME command."""
        if not args:
            return
        
        try:
            time = float(args[0])
            validations = APDLSymbolicConstraints.validate_material_property("TIME", time)
            for v in validations:
                v["command"] = "TIME"
            self.constraints.extend(validations)
        except ValueError:
            pass
    
    def _parse_substeps(self, args: List[str]):
        """Parse NSUBST command."""
        if not args:
            return
        
        try:
            nsubst = int(args[0])
            validations = APDLSymbolicConstraints.validate_material_property("NSUBST", nsubst)
            for v in validations:
                v["command"] = "NSUBST"
            self.constraints.extend(validations)
        except ValueError:
            pass
    
    def _parse_time_step(self, args: List[str]):
        """Parse DELTIM command."""
        if not args:
            return
        
        try:
            deltim = float(args[0])
            validations = APDLSymbolicConstraints.validate_material_property("DELTIM", deltim)
            for v in validations:
                v["command"] = "DELTIM"
            self.constraints.extend(validations)
        except ValueError:
            pass
    
    def extract_fem_mathematics(self) -> Dict[str, Any]:
        """Extract FEM mathematical structures."""
        return {
            "governing_equations": self._extract_governing_equations(),
            "discretization": self._extract_discretization(),
            "solver_settings": self._extract_solver_settings(),
            "boundary_conditions": self._extract_boundary_conditions(),
            "material_models": self._extract_material_models(),
        }
    
    def _extract_governing_equations(self) -> List[Dict[str, Any]]:
        """Extract governing equations based on analysis type."""
        equations = []
        
        if self.analysis_type == AnalysisType.STATIC:
            equations.append({
                "type": "equilibrium",
                "form": "∇·σ + f = 0",
                "description": "Static equilibrium equation",
                "variables": ["stress", "body_force"],
            })
        elif self.analysis_type == AnalysisType.MODAL:
            equations.append({
                "type": "eigenvalue",
                "form": "(K - ω²M)φ = 0",
                "description": "Modal analysis eigenvalue problem",
                "variables": ["stiffness", "mass", "frequency", "mode_shape"],
            })
        elif self.analysis_type == AnalysisType.TRANSIENT:
            equations.append({
                "type": "dynamics",
                "form": "Mü + Ců + Ku = F(t)",
                "description": "Transient dynamic equation",
                "variables": ["mass", "damping", "stiffness", "displacement", "force"],
            })
        elif self.analysis_type == AnalysisType.THERMAL:
            equations.append({
                "type": "heat_conduction",
                "form": "ρc∂T/∂t = ∇·(k∇T) + Q",
                "description": "Heat conduction equation",
                "variables": ["density", "specific_heat", "temperature", "conductivity", "heat_source"],
            })
        
        return equations
    
    def _extract_discretization(self) -> Dict[str, Any]:
        """Extract discretization information."""
        # Find element type commands
        element_types = []
        for cmd in self.commands:
            if cmd.command == "ET":
                if len(cmd.args) >= 2:
                    element_types.append({
                        "type_number": cmd.args[0],
                        "element_name": cmd.args[1],
                    })
        
        # Find mesh settings
        esize = None
        smartsize = None
        for cmd in self.commands:
            if cmd.command == "ESIZE" and cmd.args:
                try:
                    esize = float(cmd.args[0])
                except ValueError:
                    pass
            elif cmd.command == "SMRTSIZE" and cmd.args:
                try:
                    smartsize = int(cmd.args[0])
                except ValueError:
                    pass
        
        return {
            "method": "finite_element",
            "element_types": element_types,
            "element_size": esize,
            "smart_size": smartsize,
            "formulation": "weak_form",
        }
    
    def _extract_solver_settings(self) -> Dict[str, Any]:
        """Extract solver settings."""
        settings = {
            "newton_raphson": "full",
            "line_search": "off",
            "predictor": "on",
            "autots": "off",
        }
        
        for cmd in self.commands:
            if cmd.command == "NROPT" and cmd.args:
                settings["newton_raphson"] = cmd.args[0]
            elif cmd.command == "LNSRCH" and cmd.args:
                settings["line_search"] = cmd.args[0]
            elif cmd.command == "PRED" and cmd.args:
                settings["predictor"] = cmd.args[0]
            elif cmd.command == "AUTOTS" and cmd.args:
                settings["autots"] = cmd.args[0]
        
        return settings
    
    def _extract_boundary_conditions(self) -> List[Dict[str, Any]]:
        """Extract boundary conditions."""
        bcs = []
        
        for cmd in self.commands:
            if cmd.command == "D":
                # Displacement constraints
                if len(cmd.args) >= 2:
                    bcs.append({
                        "type": "displacement",
                        "node": cmd.args[0],
                        "dof": cmd.args[1],
                        "value": cmd.args[2] if len(cmd.args) > 2 else "0",
                    })
            elif cmd.command == "F":
                # Force loads
                if len(cmd.args) >= 2:
                    bcs.append({
                        "type": "force",
                        "node": cmd.args[0],
                        "dof": cmd.args[1],
                        "value": cmd.args[2] if len(cmd.args) > 2 else "0",
                    })
        
        return bcs
    
    def _extract_material_models(self) -> List[Dict[str, Any]]:
        """Extract material constitutive models."""
        models = []
        
        # Group materials by ID
        mat_by_id = {}
        for mat in self.materials:
            if mat.material_id not in mat_by_id:
                mat_by_id[mat.material_id] = []
            mat_by_id[mat.material_id].append(mat)
        
        for mat_id, props in mat_by_id.items():
            prop_names = [p.name for p in props]
            
            # Determine material model
            if "EX" in prop_names and "PRXY" in prop_names:
                # Isotropic elasticity
                ex = next((p.value for p in props if p.name == "EX"), None)
                prxy = next((p.value for p in props if p.name == "PRXY"), None)
                
                if ex and prxy is not None:
                    mu = ex / (2 * (1 + prxy))
                    lam = None
                    if abs(1 - 2 * prxy) > 1e-10:
                        lam = ex * prxy / ((1 + prxy) * (1 - 2 * prxy))
                    
                    model_dict = {
                        "material_id": mat_id,
                        "type": "isotropic_elastic",
                        "youngs_modulus": ex,
                        "poissons_ratio": prxy,
                        "shear_modulus": mu,
                        "stress_strain_relation": "σ = λtr(ε)I + 2με",
                    }
                    if lam is not None:
                        model_dict["lame_first"] = lam
                    else:
                        model_dict["lame_first"] = float('inf')
                        model_dict["incompressible_warning"] = "Poisson's ratio = 0.5 (incompressible limit)"
                    models.append(model_dict)
            
            elif "KXX" in prop_names:
                # Thermal material
                kxx = next((p.value for p in props if p.name == "KXX"), None)
                models.append({
                    "material_id": mat_id,
                    "type": "thermal",
                    "conductivity": kxx,
                })
        
        return models


# Convenience function
def parse_apdl(filepath: str) -> APDLResults:
    """Parse APDL file."""
    parser = EnhancedAPDLParser()
    return parser.parse_file(filepath)
