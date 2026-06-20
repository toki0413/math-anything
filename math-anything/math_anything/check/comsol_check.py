"""COMSOL check engine.

Leverages FEM domain template; adds COMSOL-specific checks.
"""

from typing import Any, Dict, List

from ..schemas import MathSchema
from ..templates import FEMCheckTemplate, FEMParamExtractor
from .base import CheckEngine, CheckResult


class ComsolCheckEngine(CheckEngine):
    """Validate COMSOL input parameters."""

    @property
    def engine_name(self) -> str:
        return "comsol"

    def check(self, schema: MathSchema) -> List[CheckResult]:
        params = FEMParamExtractor.extract(schema, engine="comsol")
        tpl = FEMCheckTemplate(params)
        results = tpl.to_check_results()
        results.extend(self._check_comsol_specific(params))
        return results

    def _check_comsol_specific(self, params: Dict[str, Any]) -> List[CheckResult]:
        results = []
        physics = params.get("physics_type", "solid_mechanics")
        mesh = params.get("mesh", {})
        materials = params.get("materials", [])
        study = params.get("study", {})

        # Check physics-study compatibility
        analysis = study.get("analysis_type", "stationary")
        if "thermal" in physics and analysis == "eigenfrequency":
            results.append(
                CheckResult(
                    rule="Physics-study mismatch",
                    severity="warning",
                    message="Thermal physics with eigenfrequency study is unusual.",
                    suggestion="Verify that the eigenfrequency study is intended for thermal modes (rare).",
                )
            )

        # Mesh quality
        max_size = mesh.get("max_element_size")
        if max_size is not None:
            try:
                size_val = float(max_size)
                if size_val <= 0:
                    results.append(
                        CheckResult(
                            rule="Invalid mesh size",
                            severity="error",
                            message=f"Maximum element size = {size_val} is not positive.",
                            suggestion="Set a positive maximum element size.",
                        )
                    )
            except (ValueError, TypeError):
                pass

        # Multiphysics-specific material checks
        if "thermal" in physics and "solid" in physics:
            for mat in materials:
                if mat.get("thermal_expansion") is None:
                    results.append(
                        CheckResult(
                            rule="Missing thermal expansion",
                            severity="warning",
                            message="Thermomechanical analysis but no thermal expansion coefficient found.",
                            suggestion="Add thermal expansion coefficient (alpha) for coupled analysis.",
                        )
                    )

        return results
