"""微分几何结构家族 — re-export 模块。

Differential Geometry = 光滑流形上的张量场、联络、曲率。

涵盖：
  - 光滑流形与切丛（微分拓扑）
  - 张量场与黎曼度量（Riemann 几何）
  - 联络与曲率（Cartan 几何）
  - Lie 导数与变形映射（连续介质力学）

实现拆分为：
  - geometry_manifold: SmoothManifold, TangentBundle, VectorField, CovectorField, TensorField
  - geometry_riemannian: Metric, Connection, LeviCivitaConnection, Curvature, RiemannCurvature,
                         RicciCurvature, ScalarCurvature, LieDerivative
  - geometry_continuum: DeformationMapping, RightCauchyGreen, GreenLagrangeStrain, PolarDecomposition
"""

from __future__ import annotations

from .geometry_continuum import (
    DeformationMapping,
    GreenLagrangeStrain,
    PolarDecomposition,
    RightCauchyGreen,
)
from .geometry_manifold import (
    CovectorField,
    SmoothManifold,
    TangentBundle,
    TensorField,
    VectorField,
)
from .geometry_riemannian import (
    Connection,
    Curvature,
    LeviCivitaConnection,
    LieDerivative,
    Metric,
    MetricFunction,
    RicciCurvature,
    RiemannCurvature,
    ScalarCurvature,
    flat_metric,
    lie_derivative_metric,
    lie_derivative_scalar,
    lie_derivative_vector_field,
    schwarzschild_metric,
    spherical_metric,
)

__all__ = [
    # manifold
    "SmoothManifold",
    "TangentBundle",
    "VectorField",
    "CovectorField",
    "TensorField",
    # riemannian
    "Metric",
    "MetricFunction",
    "Connection",
    "LeviCivitaConnection",
    "Curvature",
    "RiemannCurvature",
    "RicciCurvature",
    "ScalarCurvature",
    "LieDerivative",
    # predefined metrics
    "schwarzschild_metric",
    "flat_metric",
    "spherical_metric",
    # numerical Lie derivatives
    "lie_derivative_vector_field",
    "lie_derivative_scalar",
    "lie_derivative_metric",
    # continuum
    "DeformationMapping",
    "RightCauchyGreen",
    "GreenLagrangeStrain",
    "PolarDecomposition",
]
