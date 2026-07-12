import numpy as np
import pytest

from math_anything.structures.neural_network import (
    ActivationMorphism,
    LinearMorphism,
    LossMorphism,
    SequentialNetwork,
)
from math_anything.structures.transfer import (
    WeightSpaceTransfer,
    flatten_network_weights,
    set_network_weights,
    transfer_learn,
    transfer_weights,
)


def test_transfer_weights_shape():
    adapter = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    source = [1.0, 2.0]
    target = transfer_weights(source, adapter)
    assert len(target) == 3
    assert target == [1.0, 2.0, 3.0]


def test_flatten_and_set_network_weights_roundtrip():
    net = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
        ActivationMorphism(name="relu_1", activation="relu"),
    ])
    original = flatten_network_weights(net)
    set_network_weights(net, [w + 0.1 for w in original])
    restored = flatten_network_weights(net)
    assert restored == pytest.approx([w + 0.1 for w in original])


def test_set_network_weights_rejects_wrong_length():
    net = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
    ])
    with pytest.raises(ValueError):
        set_network_weights(net, [0.0, 0.0])  # needs 4 values (2 weights + 2 biases)


def test_weight_space_transfer_rejects_bad_shape():
    with pytest.raises(ValueError):
        WeightSpaceTransfer(2, 3, matrix=[[1.0, 0.0], [0.0, 1.0]])


def test_transfer_learn_trains_target():
    source = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
        ActivationMorphism(name="relu_1", activation="relu"),
        LinearMorphism(name="linear_2", input_dim=2, output_dim=1),
    ])
    target = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=2),
        ActivationMorphism(name="relu_1", activation="relu"),
        LinearMorphism(name="linear_2", input_dim=2, output_dim=1),
    ])
    loss_fn = LossMorphism(name="mse", loss="mse")
    dataset = [(np.array([x]), np.array([2 * x + 1])) for x in [-1.0, 0.0, 1.0]]

    adapter = WeightSpaceTransfer(
        len(flatten_network_weights(source)),
        len(flatten_network_weights(target)),
    ).matrix
    result = transfer_learn(source, target, dataset, loss_fn, adapter, epochs=2, lr=0.05)
    assert len(result.states) == 2
    assert result.final_loss < result.states[0].loss
