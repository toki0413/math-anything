"""COMSOL mathematical structure extractor."""

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

from .parser import ComsolJavaParser


class ComsolExtractor:
    """Extract MathSchema from COMSOL parameter files."""

    def __init__(self):
        self.parser = ComsolJavaParser()

    def extract(self, files: Dict[str, str], options: Dict[str, Any] = None) -> MathSchema:
        input_path = files.get("input")
        if not input_path:
            raise ValueError("Input file required")

        cards = self.parser.parse_file(input_path)
        raw = self._build_raw_symbols(cards)

        schema = MathSchema(
            schema_version="1.0.0",
            meta=MetaInfo(
                extracted_by="math-anything-comsol",
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

    def _build_raw_symbols(self, cards: Dict[str, List[str]]) -> Dict[str, Any]:
        """Convert parsed cards to parameter dictionary."""
        raw: Dict[str, Any] = {"cards": list(cards.keys())}

        # Physics
        physics_lines = cards.get("PHYSICS", [])
        physics_type = "solid_mechanics"
        for line in physics_lines:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "type":
                physics_type = parts[1]
        raw["physics_type"] = physics_type

        # Materials
        materials = []
        mat_props: Dict[str, float] = {}
        for line in cards.get("MATERIAL", []):
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].lower()
                val_str = parts[1]
                # Handle units like 200e9[Pa]
                val_str = val_str.split("[")[0]
                try:
                    mat_props[key] = float(val_str)
                except ValueError:
                    pass

        if mat_props:
            materials.append({
                "youngs_modulus": mat_props.get("youngs_modulus"),
                "poisson_ratio": mat_props.get("poisson_ratio"),
                "density": mat_props.get("density"),
                "thermal_conductivity": mat_props.get("thermal_conductivity"),
                "specific_heat": mat_props.get("specific_heat"),
                "model_type": "elastic" if "youngs_modulus" in mat_props else "unknown",
            })
        raw["materials"] = materials

        # Mesh
        mesh_props: Dict[str, Any] = {}
        for line in cards.get("MESH", []):
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].lower()
                val = parts[1]
                if key == "element_type":
                    mesh_props["element_type"] = val
                elif key == "max_element_size":
                    val = val.split("[")[0]
                    try:
                        mesh_props["max_element_size"] = float(val)
                    except ValueError:
                        mesh_props["max_element_size"] = val
        raw["mesh"] = mesh_props

        # Study
        study_props = {}
        for line in cards.get("STUDY", []):
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "analysis_type":
                study_props["analysis_type"] = parts[1]
        raw["study"] = study_props

        # Boundary conditions
        bcs = []
        for line in cards.get("BOUNDARY", []):
            parts = line.split()
            if len(parts) >= 3:
                bc_type = parts[0].lower()
                node_set = parts[1]
                dof = parts[2]
                value = float(parts[3]) if len(parts) > 3 else 0.0
                bcs.append({
                    "node_set": node_set,
                    "dof": dof,
                    "value": value,
                    "bc_type": "displacement" if bc_type == "fixed" else "force",
                })
        raw["boundary_conditions"] = bcs

        # Analysis type normalization
        analysis = study_props.get("analysis_type", "stationary")
        if analysis in ("stationary", "static"):
            raw["analysis_type"] = "static"
        elif analysis == "transient":
            raw["analysis_type"] = "dynamic"
        elif analysis == "eigenfrequency":
            raw["analysis_type"] = "modal"
        elif "thermal" in physics_type:
            raw["analysis_type"] = "heat"
        else:
            raw["analysis_type"] = "static"

        raw["nlgeom"] = False
        raw["nodes"] = 0
        raw["elements"] = [mesh_props.get("element_type", "unspecified")]

        return raw

    def _extract_mathematical_model(self, raw: Dict[str, Any]) -> MathematicalModel:
        model = MathematicalModel()
        physics = raw.get("physics_type", "solid_mechanics")

        if "thermal" in physics:
            model.governing_equations = [
                GoverningEquation(
                    id="heat_conduction",
                    type="partial_differential_equation",
                    name="Heat Conduction",
                    mathematical_form="div(k grad(T)) + q = rho c dT/dt",
                    variables=["temperature", "thermal_conductivity"],
                    parameters={"form": "scalar_pde"},
                ),
            ]
        elif "fluid" in physics:
            model.governing_equations = [
                GoverningEquation(
                    id="navier_stokes",
                    type="partial_differential_equation",
                    name="Navier-Stokes",
                    mathematical_form="rho (du/dt + u . grad(u)) = -grad(p) + mu div(grad(u)) + f",
                    variables=["velocity", "pressure", "density", "viscosity"],
                    parameters={"form": "vector_pde"},
                ),
            ]
        elif "electromagnetic" in physics:
            model.governing_equations = [
                GoverningEquation(
                    id="maxwell",
                    type="partial_differential_equation",
                    name="Maxwell's Equations",
                    mathematical_form="curl(curl(E)) - k^2 E = 0",
                    variables=["electric_field", "wave_number"],
                    parameters={"form": "vector_pde"},
                ),
            ]
        else:
            model.governing_equations = [
                GoverningEquation(
                    id="equilibrium",
                    type="partial_differential_equation",
                    name="Equilibrium Equation",
                    mathematical_form="div(sigma) + b = 0",
                    variables=["stress", "body_force"],
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

        return model

    def _extract_numerical_method(self, raw: Dict[str, Any]) -> NumericalMethod:
        method = NumericalMethod()
        elements = raw.get("elements", [])
        if elements:
            method.discretization.space_discretization = f"FEM_{elements[0]}"

        analysis = raw.get("analysis_type", "static")
        if analysis == "static":
            method.solver.algorithm = "direct_sparse"
        elif analysis == "dynamic":
            method.solver.algorithm = "bdf_time_stepping"
        else:
            method.solver.algorithm = "direct_sparse"

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
                    expression=f"E ({E}) > 0 ok", description="Young's modulus positive",
                    variables=["E"], confidence=1.0, inferred_from="validation",
                ))
            if nu is not None and -1 < nu < 0.5:
                constraints.append(SymbolicConstraint(
                    expression=f"nu ({nu}) in range ok", description="Poisson's ratio valid",
                    variables=["nu"], confidence=1.0, inferred_from="validation",
                ))
        return constraints
