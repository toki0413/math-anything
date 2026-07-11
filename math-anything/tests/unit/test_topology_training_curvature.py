import numpy as np

from math_anything.topology.training_curvature import (
    OptimizationState,
    trajectory_curvature,
)


def test_straight_trajectory_has_zero_curvature():
    states = [
        OptimizationState(step=0, loss=1.0, weights=np.array([0.0])),
        OptimizationState(step=1, loss=0.5, weights=np.array([1.0])),
        OptimizationState(step=2, loss=0.0, weights=np.array([2.0])),
    ]
    curvatures = trajectory_curvature(states)
    assert all(abs(c) < 1e-6 for c in curvatures)


def test_curved_trajectory_has_nonzero_curvature():
    states = [
        OptimizationState(step=0, loss=1.0, weights=np.array([0.0])),
        OptimizationState(step=1, loss=0.3, weights=np.array([1.0])),
        OptimizationState(step=2, loss=0.5, weights=np.array([1.5])),
        OptimizationState(step=3, loss=0.1, weights=np.array([2.0])),
    ]
    curvatures = trajectory_curvature(states)
    assert any(abs(c) > 1e-3 for c in curvatures)
