"""Ansys input file extractor for mathematical schema generation."""

import os
import sys
from typing import Any, Dict, List

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from math_anything.schemas import (
    ComputationalEdge,
    ComputationalGraph,
    ComputationalNode,
    GoverningEquation,
    MathematicalModel,
    MathSchema,
    MetaInfo,
    NumericalMethod,
    Solver,
    SymbolicConstraint,
    UpdateMode,
)

from .apdl_parser import EnhancedAPDLParser


class AnsysInputExtractor:
    """Extract MathSchema from Ansys APDL input files."""

    def __init__(self):
        self.parser = EnhancedAPDLParser()

    def extract(self, files: Dict[str, str], options: Dict[str, Any] = None) -> MathSchema:
        input_path = files.get("input")
        if not input_path:
            raise ValueError("Input file required")

        result = self.parser.parse_file(input_path)
        raw = self._build_raw_symbols(result)

        schema = MathSchema(
            schema_version="1.0.0",
            meta=MetaInfo(
                extracted_by="math-anything-ansys",
                extractor_version="0.1.0",
                source_files={"input": [input_path]},
            ),
            mathematical_model=self._extract_mathematical_model(raw),
            numerical_method=self._extract_numerical_method(raw),
            computational_graph=self._extract_computational_graph(raw),
            conservation_properties={},
            raw_symbols=raw,
        )
        schema.symbolic_constraints = self._extract_symbolic_constraints(raw)
        return schema

    def _build_raw_symbols(self, result: Any) -> Dict[str, Any]:
        """Convert APDLResults to parameter dictionary."""
        raw: Dict[str, Any] = {
            "analysis_type": result.analysis_type.value,
            "parameters": result.parameters,
            "commands": [c.to_dict() for c in result.commands],
            "constraints": result.constraints,
        }

        # Convert materials to standard format
        materials = []
        mat_props: Dict[int, Dict[str, Any]] = {}
        for m in result.materials:
            mid = m.material_id
            if mid not in mat_props:
                mat_props[mid] = {"id": mid, "properties": {}}
            mat_props[mid]["properties"][m.name] = m.value

        for mid, props in mat_props.items():
            p = props["properties"]
            materials.append({
                "id": mid,
                "youngs_modulus": p.get("EX"),
                "poisson_ratio": p.get("PRXY"),
                "density": p.get("DENS"),
                "thermal_expansion": p.get("ALPX"),
                "thermal_conductivity": p.get("KXX"),
                "specific_heat": p.get("C"),
                "friction_coeff": p.get("MU"),
                "model_type": "elastic" if "EX" in p else "unknown",
            })

        raw["materials"] = materials

        # Extract element types from ET commands
        elements = []
        for cmd in result.commands:
            if cmd.command == "ET" and len(cmd.args) >= 2:
                elements.append(cmd.args[1])
        raw["elements"] = elements

        # Extract boundary conditions from D and F commands
        bcs = []
        for cmd in result.commands:
            if cmd.command == "D" and len(cmd.args) >= 3:
                bcs.append({
                    "node_set": cmd.args[0],
                    "dof": cmd.args[1],
                    "value": float(cmd.args[2]) if len(cmd.args) > 2 else 0.0,
                    "bc_type": "displacement",
                })
            elif cmd.command == "F" and len(cmd.args) >= 3:
                bcs.append({
                    "node_set": cmd.args[0],
                    "dof": cmd.args[1],
                    "value": float(cmd.args[2]) if len(cmd.args) > 2 else 0.0,
                    "bc_type": "force",
                })
        raw["boundary_conditions"] = bcs

        # Extract step settings
        steps = []
        time_val = None
        nsubst = None
        deltim = None
        for cmd in result.commands:
            if cmd.command == "TIME" and cmd.args:
                try:
                    time_val = float(cmd.args[0])
                except ValueError:
                    pass
            elif cmd.command == "NSUBST" and cmd.args:
                try:
                    nsubst = int(cmd.args[0])
                except ValueError:
                    pass
            elif cmd.command == "DELTIM" and cmd.args:
                try:
                    deltim = float(cmd.args[0])
                except ValueError:
                    pass

        if time_val is not None or nsubst is not None:
            steps.append({
                "analysis_type": result.analysis_type.value,
                "total_time": time_val,
                "max_increments": nsubst,
                "initial_inc": deltim,
            })
        raw["steps"] = steps

        # Mesh info
        esize = None
        for cmd in result.commands:
            if cmd.command == "ESIZE" and cmd.args:
                try:
                    esize = float(cmd.args[0])
                except ValueError:
                    pass
        raw["element_size"] = esize

        # Nonlinear geometry
        nlgeom = any(
            c.command == "NLGEOM" and c.args and c.args[0].upper() == "ON"
            for c in result.commands
        )
        raw["nlgeom"] = nlgeom

        return raw

    def _extract_mathematical_model(self, raw: Dict[str, Any]) -> MathematicalModel:
        model = MathematicalModel()
        analysis = raw.get("analysis_type", "static_structural")

        if "thermal" in analysis:
            model.governing_equations = [
                GoverningEquation(
                    id="heat_conduction",
                    type="partial_differential_equation",
                    name="Heat Conduction",
                    mathematical_form="div(k grad(T)) + q = rho c dT/dt",
                    variables=["temperature", "thermal_conductivity", "heat_source"],
                    parameters={"form": "scalar_pde"},
                ),
            ]
        else:
            model.governing_equations = [
                GoverningEquation(
                    id="equilibrium",
                    type="partial_differential_equation",
                    name="Equilibrium Equation",
                    mathematical_form="div(sigma) + b = rho a",
                    variables=["stress", "body_force", "displacement"],
                    parameters={"form": "vector_pde"},
                ),
                GoverningEquation(
                    id="strain_displacement",
                    type="kinematic_relation",
                    name="Strain-Displacement",
                    mathematical_form="epsilon = 1/2 (grad(u) + grad(u)^T)",
                    variables=["strain", "displacement"],
                    parameters={"linearity": "linear"},
                ),
            ]
            if raw.get("materials") and raw["materials"][0].get("model_type") == "elastic":
                model.governing_equations.append(
                    GoverningEquation(
                        id="constitutive",
                        type="constitutive_relation",
                        name="Hooke's Law",
                        mathematical_form="sigma = lambda tr(epsilon) I + 2 mu epsilon",
                        variables=["stress", "strain"],
                        parameters={"isotropic": True},
                    )
                )

        return model

    def _extract_numerical_method(self, raw: Dict[str, Any]) -> NumericalMethod:
        method = NumericalMethod()
        elements = raw.get("elements", [])
        if elements:
            method.discretization.space_discretization = f"FEM_{elements[0]}"

        analysis = raw.get("analysis_type", "static_structural")
        nlgeom = raw.get("nlgeom", False)

        if "static" in analysis and not nlgeom:
            method.solver.algorithm = "direct_sparse"
            method.solver.convergence_criterion = "residual_norm"
        else:
            method.solver.algorithm = "newton_raphson"
            method.solver.convergence_criterion = "force_residual"

        return method

    def _extract_computational_graph(self, raw: Dict[str, Any]) -> ComputationalGraph:
        graph = ComputationalGraph()
        graph.add_node(ComputationalNode(
            id="assembly", type="matrix_assembly",
            math_semantics={"operator_type": "stiffness_assembly", "updates": {"target": "K", "mode": UpdateMode.EXPLICIT_UPDATE.value}},
        ))
        graph.add_node(ComputationalNode(
            id="solver", type="linear_solver",
            math_semantics={"operator_type": "sparse_direct_solver", "updates": {"target": "displacement", "mode": UpdateMode.EXPLICIT_UPDATE.value}},
        ))
        graph.add_node(ComputationalNode(
            id="post_process", type="stress_recovery",
            math_semantics={"operator_type": "strain_stress_computation", "updates": {"target": "stress", "mode": UpdateMode.EXPLICIT_UPDATE.value}},
        ))
        graph.add_edge(ComputationalEdge(from_node="assembly", to_node="solver", data_type="sparse_matrix", dependency="solve"))
        graph.add_edge(ComputationalEdge(from_node="solver", to_node="post_process", data_type="displacement_vector", dependency="compute"))
        return graph

    def _extract_symbolic_constraints(self, raw: Dict[str, Any]) -> List[SymbolicConstraint]:
        constraints = []
        for mat in raw.get("materials", []):
            E = mat.get("youngs_modulus")
            nu = mat.get("poisson_ratio")
            if E is not None and E > 0:
                constraints.append(SymbolicConstraint(
                    expression=f"E ({E}) > 0 ok",
                    description="Young's modulus positive",
                    variables=["E"], confidence=1.0, inferred_from="validation",
                ))
            if nu is not None and -1 < nu < 0.5:
                constraints.append(SymbolicConstraint(
                    expression=f"nu ({nu}) in range ok",
                    description="Poisson's ratio valid",
                    variables=["nu"], confidence=1.0, inferred_from="validation",
                ))
        return constraints
