"""Visualization module for Math Anything.

Generates diagrams of mathematical structures using Mermaid and Graphviz.

Example:
    >>> from math_anything.visualization import Visualizer
    >>> viz = Visualizer()
    >>> mermaid = viz.render(schema, format="mermaid")
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    show_constraints: bool = True
    show_approximations: bool = True
    show_dependencies: bool = True
    max_depth: int = 10


class Visualizer:
    """Generate visualizations of mathematical structures."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
    
    def render(self, schema: Dict[str, Any], format: str = "mermaid") -> str:
        """Render schema to visualization format.
        
        Args:
            schema: Mathematical schema
            format: Output format ("mermaid", "graphviz", "text")
            
        Returns:
            Visualization string
        """
        if format == "mermaid":
            return self._to_mermaid(schema)
        elif format == "graphviz":
            return self._to_graphviz(schema)
        elif format == "text":
            return self._to_text(schema)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _to_mermaid(self, schema: Dict[str, Any]) -> str:
        """Generate Mermaid flowchart."""
        lines = ["graph TD"]
        
        # Root node
        engine = schema.get("engine", "unknown")
        lines.append(f'    Root[\"{engine.upper()}\\nMathematical Structure\"]')
        
        # Mathematical structure
        math_struct = schema.get("mathematical_structure", {})
        if math_struct:
            problem_type = math_struct.get("problem_type", "unknown")
            canonical = math_struct.get("canonical_form", "N/A")
            
            lines.append(f'    Root --> ProblemType[\"Problem Type\\n{problem_type}\"]')
            lines.append(f'    Root --> Canonical[\"{canonical}\"]')
        
        # Variable dependencies
        deps = schema.get("variable_dependencies", [])
        if deps and self.config.show_dependencies:
            lines.append(f'    Root --> Deps[\"Variable Dependencies\"]')
            for i, dep in enumerate(deps[:5], 1):
                var = dep.get("variable", f"var_{i}")
                lines.append(f'    Deps --> Dep{i}[\"{var}\"]')
        
        # Approximations hierarchy
        approxs = schema.get("approximations", [])
        if approxs and self.config.show_approximations:
            lines.append(f'    Root --> ApproxRoot[\"Approximation Hierarchy\n({len(approxs)} levels)\"]')
            prev_node = "ApproxRoot"
            for i, app in enumerate(approxs[:self.config.max_depth]):
                name = app.get("name", f"approx_{i}")
                node_id = f"App{i}"
                lines.append(f'    {prev_node} --> {node_id}[\"{name}\"]')
                prev_node = node_id
        
        # Constraints
        decoding = schema.get("mathematical_decoding", {})
        constraints = decoding.get("constraints", [])
        if constraints and self.config.show_constraints:
            lines.append(f'    Root --> ConstRoot[\"Constraints\\n({len(constraints)} total)\"]')
            satisfied = sum(1 for c in constraints if c.get("satisfied"))
            lines.append(f'    ConstRoot --> ConstStatus[\"{satisfied}/{len(constraints)} Satisfied\"]')
            
            for i, const in enumerate(constraints[:5]):
                expr = const.get("expression", "unknown")
                is_sat = const.get("satisfied", False)
                status = "✓" if is_sat else "✗"
                node_id = f"Const{i}"
                lines.append(f'    ConstRoot --> {node_id}[\"{status} {expr}\"]')
        
        # Styling
        lines.append("")
        lines.append("    classDef problem fill:#e1f5fe,stroke:#01579b,stroke-width:2px")
        lines.append("    classDef constraint fill:#f3e5f5,stroke:#4a148c")
        lines.append("    classDef approximation fill:#e8f5e9,stroke:#1b5e20")
        lines.append("    classDef error fill:#ffebee,stroke:#b71c1c")
        
        lines.append(f"    class ProblemType,Canonical problem")
        lines.append(f"    class ConstRoot,ConstStatus constraint")
        lines.append(f"    class ApproxRoot approximation")
        
        return "\\n".join(lines)
    
    def _to_graphviz(self, schema: Dict[str, Any]) -> str:
        """Generate Graphviz DOT format."""
        lines = ["digraph MathStructure {"]
        lines.append('    rankdir=TB;')
        lines.append('    node [shape=box, style="rounded,filled", fontname="Arial"];')
        lines.append('')
        
        engine = schema.get("engine", "unknown")
        lines.append(f'    root [label="{engine.upper()}\\nMathematical Structure", fillcolor="#e3f2fd"];')
        
        # Mathematical structure
        math_struct = schema.get("mathematical_structure", {})
        if math_struct:
            problem_type = math_struct.get("problem_type", "unknown")
            canonical = math_struct.get("canonical_form", "N/A")
            
            lines.append(f'    problem [label="{problem_type}", fillcolor="#e1f5fe"];')
            lines.append(f'    canonical [label="{canonical}", fillcolor="#e1f5fe"];')
            lines.append('    root -> problem;')
            lines.append('    root -> canonical;')
        
        # Approximations
        approxs = schema.get("approximations", [])
        if approxs and self.config.show_approximations:
            lines.append(f'    approx [label="Approximations ({len(approxs)})", fillcolor="#e8f5e9"];')
            lines.append('    root -> approx;')
            
            prev = "approx"
            for i, app in enumerate(approxs[:self.config.max_depth]):
                name = app.get("name", f"approx_{i}")
                node_id = f"app{i}"
                lines.append(f'    {node_id} [label="{name}", fillcolor="#c8e6c9"];')
                lines.append(f'    {prev} -> {node_id};')
                prev = node_id
        
        # Constraints
        decoding = schema.get("mathematical_decoding", {})
        constraints = decoding.get("constraints", [])
        if constraints and self.config.show_constraints:
            satisfied = sum(1 for c in constraints if c.get("satisfied"))
            lines.append(f'    const [label="Constraints\\n{satisfied}/{len(constraints)} OK", fillcolor="#f3e5f5"];')
            lines.append('    root -> const;')
        
        lines.append("}")
        return "\\n".join(lines)
    
    def _to_text(self, schema: Dict[str, Any]) -> str:
        """Generate text-based tree visualization."""
        lines = []
        lines.append("=" * 60)
        lines.append("MATHEMATICAL STRUCTURE VISUALIZATION")
        lines.append("=" * 60)
        
        engine = schema.get("engine", "unknown")
        lines.append(f"\\nEngine: {engine.upper()}")
        lines.append("-" * 40)
        
        # Mathematical structure
        math_struct = schema.get("mathematical_structure", {})
        if math_struct:
            lines.append(f"\\n[PROBLEM TYPE]")
            lines.append(f"  {math_struct.get('problem_type', 'N/A')}")
            lines.append(f"\\n[CANONICAL FORM]")
            lines.append(f"  {math_struct.get('canonical_form', 'N/A')}")
        
        # Approximations tree
        approxs = schema.get("approximations", [])
        if approxs:
            lines.append(f"\\n[APPROXIMATION HIERARCHY]")
            for i, app in enumerate(approxs):
                indent = "  " + "  " * i
                name = app.get("name", f"Level {i}")
                lines.append(f"{indent}└─ {name}")
        
        # Constraints
        decoding = schema.get("mathematical_decoding", {})
        constraints = decoding.get("constraints", [])
        if constraints:
            lines.append(f"\\n[CONSTRAINTS]")
            for const in constraints:
                expr = const.get("expression", "unknown")
                satisfied = const.get("satisfied", False)
                status = "✓" if satisfied else "✗"
                lines.append(f"  [{status}] {expr}")
        
        lines.append("\\n" + "=" * 60)
        return "\\n".join(lines)
    
    def generate_html(
        self,
        schema: Dict[str, Any],
        interactive: bool = True
    ) -> str:
        """Generate HTML visualization with Mermaid.js.
        
        Args:
            schema: Mathematical schema
            interactive: Whether to include interactive features
            
        Returns:
            HTML string
        """
        mermaid = self._to_mermaid(schema)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Math Anything - {schema.get('engine', 'Unknown')}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1565c0;
            border-bottom: 2px solid #1565c0;
            padding-bottom: 10px;
        }}
        .mermaid {{
            text-align: center;
        }}
        .info {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Math Anything Visualization</h1>
        <div class="mermaid">
{mermaid}
        </div>
        <div class="info">
            <p><strong>Engine:</strong> {schema.get('engine', 'Unknown').upper()}</p>
            <p><strong>Problem Type:</strong> {schema.get('mathematical_structure', {}).get('problem_type', 'N/A')}</p>
        </div>
    </div>
    <script>
        mermaid.initialize({{startOnLoad: true}});
    </script>
</body>
</html>"""
        
        return html


# Convenience functions
def to_mermaid(schema: Dict[str, Any]) -> str:
    """Quick convert to Mermaid."""
    viz = Visualizer()
    return viz.render(schema, format="mermaid")


def to_graphviz(schema: Dict[str, Any]) -> str:
    """Quick convert to Graphviz."""
    viz = Visualizer()
    return viz.render(schema, format="graphviz")


def save_html(schema: Dict[str, Any], filepath: str):
    """Save visualization to HTML file."""
    viz = Visualizer()
    html = viz.generate_html(schema)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
