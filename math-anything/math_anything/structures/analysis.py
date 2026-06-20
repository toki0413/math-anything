"""Numerical Analysis structure family — re-export 模块.

NumericalAnalysis is the meta-structure of discretization.
It bridges continuous mathematical structures to their computable discrete approximations.

Core concepts:
  - Consistency: truncation error → 0 as mesh refines
  - Stability: discrete solution remains bounded
  - Convergence: discrete solution → continuous solution
  - Lax Equivalence Theorem: consistent + stable ⇒ convergent

These are THE fundamental theorems that make computation possible.

实现拆分为：
  - analysis_stability: NumericalAnalysis, ConsistencyCondition, StabilityCondition,
                        VonNeumannStability, EnergyStability, LaxRichtmyerStability,
                        LaxEquivalenceTheorem, CFL_Condition, HyperbolicCFL, ParabolicCFL, GeneralCFL
  - analysis_convergence: ErrorEstimate, APrioriEstimate, APosterioriEstimate,
                          SuperconvergenceEstimate, ConvergenceOrder, AlgebraicConvergence,
                          ExponentialConvergence, SublinearConvergence
"""

from __future__ import annotations

from .analysis_convergence import (
    AlgebraicConvergence,
    APosterioriEstimate,
    APrioriEstimate,
    ConvergenceOrder,
    ErrorEstimate,
    ExponentialConvergence,
    SublinearConvergence,
    SuperconvergenceEstimate,
)
from .analysis_stability import (
    CFL_Condition,
    ConsistencyCondition,
    EnergyStability,
    GeneralCFL,
    HyperbolicCFL,
    LaxEquivalenceTheorem,
    LaxRichtmyerStability,
    NumericalAnalysis,
    ParabolicCFL,
    StabilityCondition,
    VonNeumannStability,
)

__all__ = [
    # stability
    "NumericalAnalysis",
    "ConsistencyCondition",
    "StabilityCondition",
    "VonNeumannStability",
    "EnergyStability",
    "LaxRichtmyerStability",
    "LaxEquivalenceTheorem",
    "CFL_Condition",
    "HyperbolicCFL",
    "ParabolicCFL",
    "GeneralCFL",
    # convergence
    "ErrorEstimate",
    "APrioriEstimate",
    "APosterioriEstimate",
    "SuperconvergenceEstimate",
    "ConvergenceOrder",
    "AlgebraicConvergence",
    "ExponentialConvergence",
    "SublinearConvergence",
]
