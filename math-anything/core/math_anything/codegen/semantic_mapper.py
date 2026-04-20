"""Semantic Mapper - Map extracted commands to mathematical concepts.

Maps software-specific commands to mathematical structures:
- Commands -> Governing equations
- Parameters -> Mathematical variables
- Constraints -> Symbolic inequalities
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MathMapping:
    """Mapping from software command to mathematical concept."""
    command_name: str
    math_type: str  # equation, boundary_condition, numerical_method, etc.
    mathematical_form: str
    variables: List[str]
    confidence: float
    description: str


class SemanticMapper:
    """Map software commands to mathematical semantics.
    
    Uses pattern matching and keyword detection to identify
    mathematical structures in software commands.
    
    Example:
        ```python
        mapper = SemanticMapper()
        mappings = mapper.map_commands(
            commands=[{'name': 'nvt', 'description': 'Nose-Hoover thermostat'}],
            parameters=[{'name': 'temp', 'type': 'float'}],
        )
        
        for m in mappings['equations']:
            print(f"{m['name']}: {m['form']}")
        ```
    """
    
    # Mathematical keyword mappings
    EQUATION_KEYWORDS = {
        'nvt': {
            'form': 'dT/dt = (T_target - T) / tau_T + thermal_noise',
            'type': 'thermostat',
            'description': 'Nose-Hoover thermostat equation',
        },
        'npt': {
            'form': 'dP/dt = (P_target - P) / tau_P + barostat_coupling',
            'type': 'barostat',
            'description': 'Nose-Hoover barostat equation',
        },
        'minimize': {
            'form': 'F = -∇E, x_{n+1} = x_n + alpha * F',
            'type': 'energy_minimization',
            'description': 'Conjugate gradient energy minimization',
        },
        'heat': {
            'form': '∂T/∂t = α ∇²T + Q_source',
            'type': 'heat_equation',
            'description': 'Heat transfer equation',
        },
        'deform': {
            'form': 'ε = ∇u + (∇u)^T, σ = C : ε',
            'type': 'mechanics',
            'description': 'Linear elastic deformation',
        },
    }
    
    BOUNDARY_KEYWORDS = {
        'fix': {'type': 'boundary_condition', 'field': 'displacement'},
        'velocity': {'type': 'boundary_condition', 'field': 'velocity'},
        'force': {'type': 'boundary_condition', 'field': 'force'},
        'temperature': {'type': 'boundary_condition', 'field': 'temperature'},
        'wall': {'type': 'boundary_condition', 'field': 'constraint'},
        'periodic': {'type': 'boundary_condition', 'field': 'periodicity'},
        'fix deform': {'type': 'boundary_condition', 'field': 'strain'},
    }
    
    NUMERICAL_KEYWORDS = {
        'verlet': {'method': 'velocity_verlet', 'order': 2, 'symplectic': True},
        'leapfrog': {'method': 'leapfrog', 'order': 2, 'symplectic': True},
        'rk4': {'method': 'runge_kutta_4', 'order': 4, 'symplectic': False},
        'cg': {'method': 'conjugate_gradient', 'type': 'optimizer'},
        'pcg': {'method': 'preconditioned_cg', 'type': 'optimizer'},
        'gmres': {'method': 'gmres', 'type': 'solver'},
    }
    
    PHYSICAL_PARAMETERS = {
        'temp': {'symbol': 'T', 'unit': 'K', 'physical': 'temperature'},
        'pressure': {'symbol': 'P', 'unit': 'Pa', 'physical': 'pressure'},
        'volume': {'symbol': 'V', 'unit': 'm³', 'physical': 'volume'},
        'density': {'symbol': 'ρ', 'unit': 'kg/m³', 'physical': 'density'},
        'mass': {'symbol': 'm', 'unit': 'kg', 'physical': 'mass'},
        'charge': {'symbol': 'q', 'unit': 'e', 'physical': 'charge'},
        'epsilon': {'symbol': 'ε', 'unit': 'kcal/mol', 'physical': 'energy_well_depth'},
        'sigma': {'symbol': 'σ', 'unit': 'Å', 'physical': 'length_scale'},
        'dt': {'symbol': 'Δt', 'unit': 's', 'physical': 'time_step'},
        'timestep': {'symbol': 'Δt', 'unit': 'fs', 'physical': 'time_step'},
    }
    
    def __init__(self):
        self.mappings: List[MathMapping] = []
    
    def map_commands(
        self,
        commands: List[Dict[str, Any]],
        parameters: List[Dict[str, Any]] = None,
        equations: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Map extracted commands to mathematical structures.
        
        Args:
            commands: List of extracted commands
            parameters: List of extracted parameters
            equations: List of equations from source
            
        Returns:
            Dictionary with mathematical mappings
        """
        parameters = parameters or []
        equations = equations or []
        
        mapped_equations = []
        mapped_boundaries = []
        mapped_numerical = []
        mapped_variables = []
        
        # Map commands to equations
        for cmd in commands:
            name = cmd.get('name', '').lower()
            description = cmd.get('description', '').lower()
            
            # Check for equation keywords
            for keyword, eq_info in self.EQUATION_KEYWORDS.items():
                if keyword in name or keyword in description:
                    mapping = MathMapping(
                        command_name=name,
                        math_type='governing_equation',
                        mathematical_form=eq_info['form'],
                        variables=self._extract_variables(eq_info['form']),
                        confidence=0.8 if keyword in name else 0.5,
                        description=eq_info['description'],
                    )
                    self.mappings.append(mapping)
                    mapped_equations.append({
                        'id': f'eq_{name}',
                        'name': eq_info['description'],
                        'type': eq_info['type'],
                        'form': eq_info['form'],
                        'source_command': name,
                        'confidence': mapping.confidence,
                    })
            
            # Check for boundary condition keywords
            for keyword, bc_info in self.BOUNDARY_KEYWORDS.items():
                if keyword in name or keyword in description:
                    mapped_boundaries.append({
                        'id': f'bc_{name}',
                        'type': bc_info['type'],
                        'field': bc_info['field'],
                        'source_command': name,
                    })
            
            # Check for numerical method keywords
            for keyword, num_info in self.NUMERICAL_KEYWORDS.items():
                if keyword in name or keyword in description:
                    mapped_numerical.append({
                        'method': num_info['method'],
                        'order': num_info.get('order'),
                        'symplectic': num_info.get('symplectic'),
                        'source_command': name,
                    })
        
        # Map parameters to physical variables
        for param in parameters:
            param_name = param.get('name', '').lower()
            
            for keyword, phys_info in self.PHYSICAL_PARAMETERS.items():
                if keyword in param_name:
                    mapped_variables.append({
                        'param_name': param_name,
                        'symbol': phys_info['symbol'],
                        'unit': phys_info['unit'],
                        'physical_quantity': phys_info['physical'],
                    })
        
        # Process explicit equations from source
        for eq in equations:
            if eq.get('form'):
                mapped_equations.append({
                    'id': f'eq_src_{len(mapped_equations)}',
                    'name': 'Extracted from source',
                    'type': eq.get('type', 'unknown'),
                    'form': eq['form'],
                    'source': eq.get('source_file', ''),
                    'confidence': 0.9,
                })
        
        return {
            'equations': mapped_equations,
            'boundary_conditions': mapped_boundaries,
            'numerical_methods': mapped_numerical,
            'variables': mapped_variables,
            'total_mappings': len(self.mappings),
        }
    
    def _extract_variables(self, expression: str) -> List[str]:
        """Extract variable names from mathematical expression."""
        # Simple regex to find variable-like tokens
        # Exclude common math functions and numbers
        math_functions = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'pi', 'e'}
        
        # Find word-like tokens
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', expression)
        
        # Filter out math functions and common words
        variables = [
            t for t in tokens
            if t not in math_functions
            and t not in ['d', 'dt', 'dx']  # Differentiation symbols
        ]
        
        return list(set(variables))
    
    def suggest_math_semantics(
        self,
        command_name: str,
        context: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Suggest mathematical semantics for a command.
        
        Args:
            command_name: Name of the command
            context: Additional context
            
        Returns:
            Suggested mathematical mapping or None
        """
        name_lower = command_name.lower()
        
        # Direct keyword match
        for keyword, eq_info in self.EQUATION_KEYWORDS.items():
            if keyword in name_lower:
                return {
                    'type': 'governing_equation',
                    'name': eq_info['description'],
                    'mathematical_form': eq_info['form'],
                    'confidence': 0.9,
                }
        
        # Pattern-based detection
        if 'fix' in name_lower:
            return {
                'type': 'boundary_condition',
                'name': f'Fixed constraint: {command_name}',
                'mathematical_form': 'u = u_boundary on Γ_D',
                'confidence': 0.7,
            }
        
        if 'compute' in name_lower:
            return {
                'type': 'derived_quantity',
                'name': f'Computed quantity: {command_name}',
                'mathematical_form': 'q = f(state)',
                'confidence': 0.6,
            }
        
        return None
    
    def build_computational_graph(
        self,
        commands: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build computational graph from command sequence.
        
        Analyzes command dependencies to construct graph.
        """
        nodes = []
        edges = []
        
        for i, cmd in enumerate(commands):
            node_id = f"node_{i}"
            
            nodes.append({
                'id': node_id,
                'type': cmd.get('name', 'unknown'),
                'math_semantics': self.suggest_math_semantics(
                    cmd.get('name', '')
                ) or {},
            })
            
            # Create edges based on dependencies
            if i > 0:
                edges.append({
                    'from': f"node_{i-1}",
                    'to': node_id,
                    'dependency': 'sequence',
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
        }


# Convenience function
def quick_map(commands: List[Dict]) -> Dict[str, Any]:
    """Quick mapping of commands to math."""
    mapper = SemanticMapper()
    return mapper.map_commands(commands)
