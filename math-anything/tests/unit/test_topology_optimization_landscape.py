import numpy as np

from math_anything.structures.neural_network import (
    ActivationMorphism,
    LinearMorphism,
    LossMorphism,
    SequentialNetwork,
)
from math_anything.topology.optimization_landscape import (
    TrainingPathMorphism,
    build_training_path,
    training_paths_homotopic,
)
from math_anything.topology.training_curvature import train_and_capture


def _tiny_dataset():
    xs = [np.array([x]) for x in [-1.0, 0.0, 1.0]]
    ys = [np.array([2 * x + 1]) for x in [-1.0, 0.0, 1.0]]
    return list(zip(xs, ys))


def test_build_training_path_returns_engine_and_path():
    net = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
        ActivationMorphism(name="relu_1", activation="relu"),
        LinearMorphism(name="linear_2", input_dim=2, output_dim=1),
    ])
    result = train_and_capture(net, _tiny_dataset(), LossMorphism(name="mse", loss="mse"), epochs=3, lr=0.05)

    engine, path = build_training_path(result, name_prefix="run_a")
    assert len(path) == 3
    assert all(name.startswith("run_a_") for name in path)
    assert engine.morphism_links[0].source_structure == "params_initial"
    assert engine.morphism_links[-1].target_structure == "params_final"


def test_identical_training_runs_are_homotopic():
    loss_fn = LossMorphism(name="mse", loss="mse")
    dataset = _tiny_dataset()

    net_a = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
        ActivationMorphism(name="relu_1", activation="relu"),
        LinearMorphism(name="linear_2", input_dim=2, output_dim=1),
    ])
    result_a = train_and_capture(net_a, dataset, loss_fn, epochs=3, lr=0.05)

    net_b = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
        ActivationMorphism(name="relu_1", activation="relu"),
        LinearMorphism(name="linear_2", input_dim=2, output_dim=1),
    ])
    result_b = train_and_capture(net_b, dataset, loss_fn, epochs=3, lr=0.05)

    witness = training_paths_homotopic(result_a, result_b)
    assert isinstance(witness.equivalent, bool)
    assert 0.0 <= witness.confidence <= 1.0
    assert witness.source == "params_initial"
    assert witness.target == "params_final"
