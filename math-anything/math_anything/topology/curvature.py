"""Discrete curvature for morphism loops and bridge to Riemannian geometry."""

from __future__ import annotations

import math
from typing import Any

from math_anything.topology.loop import Loop


def holonomy(loop: Loop, loss_weights: dict[str, float]) -> float:
    """Compute discrete holonomy of a loop as product of morphism loss factors.

    A loss factor of 0 means the morphism preserves everything (factor 1).
    Higher loss reduces the product. The empty loop has holonomy 1.
    """
    product = 1.0
    for edge in loop.edges:
        weight = loss_weights.get(edge, 0.0)
        product *= math.exp(-weight)
    return float(product)


def discrete_curvature(loop: Loop, loss_weights: dict[str, float]) -> float:
    """Discrete curvature as deviation from identity holonomy.

    Returns 0 for a flat loop (holonomy == 1) and approaches 1 as the loop
    accumulates irreversible losses.
    """
    h = holonomy(loop, loss_weights)
    return float(abs(1.0 - h))


def riemannian_curvature_bridge(
    metric: Any,
    coords: dict[str, float],
    reference: float = 1.0,
) -> float:
    """Bridge topology curvature to Riemannian scalar curvature.

    Normalizes the absolute scalar curvature against a reference value so the
    result is comparable with discrete_curvature.
    """
    scalar = float(metric.scalar_curvature_at(coords))
    if reference == 0.0:
        return 0.0 if scalar == 0.0 else 1.0
    normalized = abs(scalar) / abs(reference)
    return float(min(normalized, 1.0))


def compute_curvature_map(
    loops: list[Loop],
    loss_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return a mapping from loop canonical form to discrete curvature.

    ``loss_weights`` maps morphism names to irreversibility loss factors.  When
    omitted, every morphism is treated as lossless and all curvatures are zero.
    """
    weights = loss_weights or {}
    return {
        loop.canonical_form: round(discrete_curvature(loop, weights), 4)
        for loop in loops
    }
