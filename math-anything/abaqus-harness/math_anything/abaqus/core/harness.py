"""Abaqus Harness - Math Anything implementation for Abaqus.

Extracts mathematical structures from Abaqus finite element simulations.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from math_anything.core.harness import HarnessRegistry, MathAnythingHarness
from math_anything.schemas import (BoundaryCondition, GoverningEquation,
                                   MathematicalModel, MathematicalObject,
                                   MathSchema, MetaInfo, NumericalMethod)


class AbaqusHarness(MathAnythingHarness):
    """Abaqus harness for Math Anything.

    Extracts mathematical structures from Abaqus finite element analysis,
    including elasticity equations, boundary conditions, and numerical methods.
    """

    @property
    def engine_name(self) -> str:
        return "abaqus"

    @property
    def supported_schema_version(self) -> str:
        return "1.0.0"

    def extract(self, files: dict, options: dict = None) -> MathSchema:
        """Extract mathematical structures from Abaqus files."""
        options = options or {}
        self.validate_files(files)

        # Create simplified schema for now
        schema = MathSchema(
            schema_version="1.0.0",
            meta=MetaInfo(
                extracted_by="math-anything-abaqus",
                extractor_version="0.1.0",
                source_files={"input": [files.get("input", "")]},
            ),
            mathematical_model=self._create_mathematical_model(),
            numerical_method=self._create_numerical_method(),
        )

        return schema

    def _create_mathematical_model(self) -> MathematicalModel:
        """Create FEM mathematical model."""
        model = MathematicalModel()

        # Equilibrium equation
        model.governing_equations.append(
            GoverningEquation(
                id="equilibrium",
                type="pde",
                name="Equilibrium Equation",
                mathematical_form="∇·σ + f = 0",
                variables=["stress", "body_force"],
            )
        )

        # Constitutive relation
        model.governing_equations.append(
            GoverningEquation(
                id="constitutive",
                type="tensor_equation",
                name="Constitutive Relation",
                mathematical_form="σ = C : ε",
                variables=["stress", "strain", "stiffness_tensor"],
            )
        )

        # Strain-displacement
        model.governing_equations.append(
            GoverningEquation(
                id="strain_displacement",
                type="pde",
                name="Strain-Displacement Relation",
                mathematical_form="ε = ½(∇u + ∇uᵀ)",
                variables=["strain", "displacement"],
            )
        )

        return model

    def _create_numerical_method(self) -> NumericalMethod:
        """Create FEM numerical method."""
        nm = NumericalMethod()
        nm.discretization.space_discretization = "finite_element_method"
        nm.discretization.time_integrator = "newton_raphson"
        nm.solver.algorithm = "direct_sparse_solver"
        nm.solver.convergence_criterion = "force_residual"
        return nm

    def list_extractable_objects(self) -> list:
        return ["governing_equations", "boundary_conditions", "numerical_method"]


HarnessRegistry.register(AbaqusHarness)
