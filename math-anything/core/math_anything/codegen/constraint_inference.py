"""Constraint Inference - Extract symbolic mathematical constraints.

Infers mathematical relationships from code and documentation:
- Parameter bounds (x > 0, 0 <= y <= 1)
- Physical constraints (dt < dx^2 / 2D for CFL)
- Derived relationships (mu = E / (2*(1+nu)))

Key innovation: Preserves symbolic relationships for LLM reasoning,
not just isolated numerical values.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class InferredConstraint:
    """An inferred mathematical constraint."""
    expression: str  # Symbolic expression, e.g., "dt < dx^2 / (2*D)"
    constraint_type: str  # inequality, equality, bound
    variables: List[str]
    confidence: float
    source: str  # Where this was inferred from
    physical_meaning: Optional[str] = None


class ConstraintInference:
    """Infer symbolic constraints from code and documentation.
    
    Extracts mathematical relationships like:
    - "if (x > 0)" -> symbolic constraint "x > 0"
    - "must be positive" -> "x > 0"
    - "dt is limited by dx^2/2D" -> "dt < dx^2 / (2*D)"
    
    This enables LLMs to perform symbolic reasoning rather than
    just comparing numerical values.
    
    Example:
        ```python
        inferencer = ConstraintInference()
        
        # From code analysis
        constraints = inferencer.infer(
            parameters=[{'name': 'timestep', 'type': 'float'}],
            command_contexts={
                'fix_nvt': ['if (dt <= 0) error->all(...)', 'timestep must be positive']
            }
        )
        
        # Results: ['timestep > 0', 'timestep < stability_limit']
        ```
    """
    
    # Common constraint patterns in code
    CODE_CONSTRAINT_PATTERNS = {
        'comparison': [
            # if (x > 0)
            (r'if\s*\(\s*(\w+)\s*([<>=!]+)\s*([\d.]+)\s*\)', 
             lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}"),
            # x must be greater than 0
            (r'(\w+)\s+must be\s+(greater than|less than|at least|at most)\s+([\d.]+)',
             lambda m: f"{m.group(1)} {'>' if 'greater' in m.group(2) else '<'} {m.group(3)}"),
        ],
        'range': [
            # range: [0, 1] or 0 <= x <= 1
            (r'range\s*[:\s]+\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?',
             lambda m, var: f"{m.group(1)} <= {var} <= {m.group(2)}"),
        ],
        'physical': [
            # positive, negative, non-zero
            (r'(\w+)\s+is\s+(positive|negative|non-zero|non-negative)',
             lambda m: f"{m.group(1)} {'> 0' if m.group(2) == 'positive' else '< 0' if m.group(2) == 'negative' else '!= 0'}"),
        ],
    }
    
    # Documentation constraint patterns
    DOC_CONSTRAINT_PATTERNS = [
        # Must be / Should be / Requires
        (r'(?:must|should|requires?)\s+be\s+([\w\s<>/=]+)', 'inequality'),
        # Range notation
        (r'range(?:\s+is)?[:\s]+\[?\s*([\d.]+)\s*[,-]\s*([\d.]+)\s*\]?', 'bound'),
        # Interval notation
        (r'([\d.]+)\s*<\s*(\w+)\s*<\s*([\d.]+)', 'inequality'),
        # Greater/less than
        (r'(\w+)\s+is\s+(greater|less)\s+than\s+([\d.]+)', 'inequality'),
        # Constraint: expression
        (r'constraint[:\s]+([^\n.]+)', 'inequality'),
    ]
    
    # Physical relationship patterns
    PHYSICAL_RELATIONSHIPS = {
        'cfl_condition': {
            'patterns': [
                r'(?:cfl|stability|Courant)[^\n]*dt[^\n]*dx',
                r'dt\s*<\s*dx\^2\s*/\s*\(?\s*2\s*\*?\s*D\s*\)?',
            ],
            'expression': 'dt < dx^2 / (2*D)',
            'variables': ['dt', 'dx', 'D'],
            'meaning': 'CFL stability condition for diffusion',
        },
        'elastic_modulus': {
            'patterns': [
                r'(?:shear modulus|mu|G)[^\n]*(?:Young|modulus|E)',
                r'mu\s*=\s*E\s*/\s*\(?\s*2\s*\*?\s*\(?\s*1\s*\+\s*nu\s*\)?',
            ],
            'expression': 'mu = E / (2*(1+nu))',
            'variables': ['mu', 'E', 'nu'],
            'meaning': 'Shear modulus from Young modulus and Poisson ratio',
        },
        'wave_speed': {
            'patterns': [
                r'(?:wave speed|c)[^\n]*sqrt[^\n]*(?:bulk|modulus)',
            ],
            'expression': 'c = sqrt(K/rho)',
            'variables': ['c', 'K', 'rho'],
            'meaning': 'Acoustic wave speed',
        },
    }
    
    def __init__(self):
        self.constraints: List[InferredConstraint] = []
    
    def infer(
        self,
        parameters: List[Dict[str, Any]],
        command_contexts: Dict[str, List[str]] = None,
    ) -> List[str]:
        """Infer constraints from code analysis.
        
        Args:
            parameters: List of extracted parameters
            command_contexts: Code contexts for each command
            
        Returns:
            List of symbolic constraint expressions
        """
        command_contexts = command_contexts or {}
        constraints = []
        
        for cmd_name, contexts in command_contexts.items():
            for context in contexts:
                # Extract from code comparison patterns
                for pattern_group in self.CODE_CONSTRAINT_PATTERNS.values():
                    for pattern, extractor in pattern_group:
                        for match in re.finditer(pattern, context, re.IGNORECASE):
                            try:
                                # For range patterns, need variable name
                                if 'range' in str(extractor):
                                    for param in parameters:
                                        var_name = param['name']
                                        constraint = extractor(match, var_name)
                                        if constraint:
                                            constraints.append(constraint)
                                else:
                                    constraint = extractor(match)
                                    if constraint:
                                        constraints.append(constraint)
                            except Exception:
                                continue
        
        # Check for physical relationships in all contexts
        all_context = ' '.join([
            ' '.join(ctxs) for ctxs in command_contexts.values()
        ])
        
        for rel_name, rel_info in self.PHYSICAL_RELATIONSHIPS.items():
            for pattern in rel_info['patterns']:
                if re.search(pattern, all_context, re.IGNORECASE):
                    constraints.append(rel_info['expression'])
                    self.constraints.append(InferredConstraint(
                        expression=rel_info['expression'],
                        constraint_type='equality',
                        variables=rel_info['variables'],
                        confidence=0.7,
                        source=f'physical_relationship:{rel_name}',
                        physical_meaning=rel_info['meaning'],
                    ))
        
        return list(set(constraints))  # Deduplicate
    
    def infer_from_docs(
        self,
        parameters: List[Dict[str, Any]],
        doc_sections: List[Dict[str, Any]] = None,
    ) -> List[str]:
        """Infer constraints from documentation.
        
        Args:
            parameters: List of parameters
            doc_sections: Sections of documentation
            
        Returns:
            List of symbolic constraint expressions
        """
        doc_sections = doc_sections or []
        constraints = []
        
        for section in doc_sections:
            content = section.get('content', '')
            
            for pattern, constraint_type in self.DOC_CONSTRAINT_PATTERNS:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    if constraint_type == 'bound':
                        # Range pattern: extract variable from context
                        lower = match.group(1)
                        upper = match.group(2)
                        for param in parameters:
                            if param['name'] in content:
                                constraint = f"{lower} <= {param['name']} <= {upper}"
                                constraints.append(constraint)
                    elif constraint_type == 'inequality':
                        constraint = match.group(1).strip()
                        # Clean up the constraint
                        constraint = re.sub(r'\s+', ' ', constraint)
                        if len(constraint) > 3:
                            constraints.append(constraint)
        
        return constraints
    
    def infer_from_equations(
        self,
        equations: List[Dict[str, Any]],
    ) -> List[InferredConstraint]:
        """Infer constraints from governing equations.
        
        Analyzes equations to extract inherent constraints like:
        - Conservation laws
        - Symmetry requirements
        - Stability conditions
        """
        constraints = []
        
        for eq in equations:
            form = eq.get('form', '')
            
            # Check for conservation properties
            if '∂ρ/∂t' in form or 'continuity' in eq.get('type', ''):
                constraints.append(InferredConstraint(
                    expression='∫ρ dV = constant',
                    constraint_type='conservation',
                    variables=['ρ'],
                    confidence=0.9,
                    source=eq.get('source_file', ''),
                    physical_meaning='Mass conservation',
                ))
            
            # Check for energy conservation
            if 'energy' in eq.get('type', '').lower() or 'E_total' in form:
                constraints.append(InferredConstraint(
                    expression='dE_total/dt = 0 (isolated system)',
                    constraint_type='conservation',
                    variables=['E_total'],
                    confidence=0.9,
                    source=eq.get('source_file', ''),
                    physical_meaning='Energy conservation',
                ))
            
            # Entropy constraints (thermodynamics)
            if 'entropy' in eq.get('type', '').lower() or 'S' in form:
                constraints.append(InferredConstraint(
                    expression='dS/dt >= 0',
                    constraint_type='inequality',
                    variables=['S'],
                    confidence=0.9,
                    source=eq.get('source_file', ''),
                    physical_meaning='Second law of thermodynamics',
                ))
        
        return constraints
    
    def build_relationship_graph(
        self,
        constraints: List[InferredConstraint],
    ) -> Dict[str, Any]:
        """Build graph of parameter relationships.
        
        Creates a graph showing which parameters are related
        through symbolic constraints.
        """
        variables = set()
        relationships = []
        
        for constraint in constraints:
            vars_in_constraint = set(constraint.variables)
            variables.update(vars_in_constraint)
            
            # Create pairwise relationships
            var_list = list(vars_in_constraint)
            for i in range(len(var_list)):
                for j in range(i + 1, len(var_list)):
                    relationships.append({
                        'from': var_list[i],
                        'to': var_list[j],
                        'constraint': constraint.expression,
                        'type': constraint.constraint_type,
                    })
        
        return {
            'variables': list(variables),
            'relationships': relationships,
            'constraint_count': len(constraints),
        }
    
    def generate_llm_prompt(
        self,
        constraints: List[InferredConstraint],
    ) -> str:
        """Generate LLM prompt with symbolic constraints.
        
        Creates a prompt that exposes the mathematical relationships
        for LLM symbolic reasoning.
        """
        prompt = "Mathematical constraints extracted from source:\n\n"
        
        # Group by type
        by_type: Dict[str, List[InferredConstraint]] = {}
        for c in constraints:
            by_type.setdefault(c.constraint_type, []).append(c)
        
        for constraint_type, type_constraints in by_type.items():
            prompt += f"## {constraint_type.upper()} Constraints\n"
            for c in type_constraints:
                prompt += f"- `{c.expression}`"
                if c.physical_meaning:
                    prompt += f" ({c.physical_meaning})"
                prompt += f" [confidence: {c.confidence:.2f}]\n"
            prompt += "\n"
        
        # Add relationship graph
        graph = self.build_relationship_graph(constraints)
        if graph['relationships']:
            prompt += "## Parameter Relationships\n"
            for rel in graph['relationships'][:10]:  # Limit to first 10
                prompt += f"- {rel['from']} <-> {rel['to']} via `{rel['constraint']}`\n"
        
        return prompt
    
    def validate_constraint(
        self,
        constraint: str,
        test_values: Dict[str, float],
    ) -> Tuple[bool, Optional[str]]:
        """Validate a constraint expression with test values.
        
        Args:
            constraint: Symbolic constraint expression
            test_values: Dictionary of variable values
            
        Returns:
            (is_valid, error_message)
        """
        try:
            # Simple validation - substitute and eval
            # In production, use sympy or similar
            eval_constraint = constraint
            for var, value in test_values.items():
                eval_constraint = eval_constraint.replace(var, str(value))
            
            # Check if it's a valid Python expression
            result = eval(eval_constraint)
            return bool(result), None
            
        except Exception as e:
            return False, str(e)
    
    def _extract_variables(self, expression: str) -> List[str]:
        """Extract variable names from mathematical expression.
        
        Args:
            expression: Mathematical expression string
            
        Returns:
            List of variable names
        """
        # Simple regex to find potential variable names
        # Exclude common math functions and numbers
        math_functions = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'pi', 'e', 'and', 'or', 'not'}
        
        # Find word-like tokens that could be variables
        tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)
        
        # Filter out math functions, single letters (usually operators), and common words
        variables = [
            t for t in tokens
            if t not in math_functions
            and len(t) > 1  # Exclude single letters like 'd' in 'dx'
            and not t.isdigit()
        ]
        
        return list(set(variables))


# Convenience functions
def quick_infer(contexts: List[str]) -> List[str]:
    """Quick constraint inference from contexts."""
    inferencer = ConstraintInference()
    return inferencer.infer(
        parameters=[],
        command_contexts={'default': contexts},
    )


def extract_symbolic_constraints(
    code_snippet: str,
    parameters: List[str],
) -> List[InferredConstraint]:
    """Extract symbolic constraints from a code snippet.
    
    Args:
        code_snippet: Source code to analyze
        parameters: Parameter names to look for
        
    Returns:
        List of inferred constraints
    """
    inferencer = ConstraintInference()
    
    param_dicts = [{'name': p} for p in parameters]
    constraints = inferencer.infer(
        parameters=param_dicts,
        command_contexts={'snippet': [code_snippet]},
    )
    
    return [
        InferredConstraint(
            expression=c,
            constraint_type='inferred',
            variables=[p for p in parameters if p in c],
            confidence=0.6,
            source='code_snippet',
        )
        for c in constraints
    ]
