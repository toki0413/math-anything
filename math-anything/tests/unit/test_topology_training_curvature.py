import numpy as np

from math_anything.topology import OptimizationState, trajectory_curvature
from math_anything.topology.training_curvature import (
    OptimizationState as OptimizationStateFromModule,
)
from math_anything.topology.training_curvature import (
    trajectory_curvature as trajectory_curvature_from_module,
)


def test_public_package_exports_optimization_state_and_trajectory_curvature():
    assert OptimizationState is OptimizationStateFromModule
    assert trajectory_curvature is trajectory_curvature_from_module


def test_short_trajectory_returns_empty_curvature():
    assert trajectory_curvature([]) == []
    assert trajectory_curvature([OptimizationState(step=0, loss=1.0, weights=np.array([0.0]))]) == []
    assert (
        trajectory_curvature(
            [
                OptimizationState(step=0, loss=1.0, weights=np.array([0.0])),
                OptimizationState(step=1, loss=0.5, weights=np.array([1.0])),
            ]
        )
        == []
    )


def test_duplicate_state_has_zero_curvature():
    states = [
        OptimizationState(step=0, loss=1.0, weights=np.array([0.0])),
        OptimizationState(step=1, loss=1.0, weights=np.array([0.0])),
        OptimizationState(step=2, loss=0.0, weights=np.array([2.0])),
    ]
    curvatures = trajectory_curvature(states)
    assert curvatures == [0.0]


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


def test_train_and_capture_produces_curvature():
    from math_anything.structures.neural_network import (
        ActivationMorphism,
        LinearMorphism,
        LossMorphism,
        SequentialNetwork,
    )
    from math_anything.topology.training_curvature import (
        train_and_capture,
        training_result_curvature,
    )

    net = SequentialNetwork(
        [
            LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
            ActivationMorphism(name="relu_1", activation="relu"),
            LinearMorphism(name="linear_2", input_dim=2, output_dim=1),
        ]
    )
    loss_fn = LossMorphism(name="mse", loss="mse")
    dataset = [(np.array([x]), np.array([2 * x + 1])) for x in [-1.0, 0.0, 1.0]]

    result = train_and_capture(net, dataset, loss_fn, epochs=10, lr=0.05)
    assert len(result.states) > 0
    curvatures = training_result_curvature(result)
    assert isinstance(curvatures, list)
