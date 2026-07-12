import numpy as np
import pytest

from math_anything.structures.neural_network import (
    ActivationMorphism,
    LinearMorphism,
    LossMorphism,
    SequentialNetwork,
)


def test_linear_morphism_shape():
    m = LinearMorphism(name="linear_1", input_dim=3, output_dim=2)
    x = np.array([1.0, 2.0, 3.0])
    y = m.apply(x)
    assert y.shape == (2,)


def test_activation_relu():
    m = ActivationMorphism(name="relu_1", activation="relu")
    x = np.array([-1.0, 0.0, 2.0])
    assert np.allclose(m.apply(x), [0.0, 0.0, 2.0])


def test_activation_tanh():
    m = ActivationMorphism(name="tanh_1", activation="tanh")
    x = np.array([0.0, 1.0, -1.0])
    expected = np.tanh(x)
    assert np.allclose(m.apply(x), expected)


def test_activation_sigmoid():
    m = ActivationMorphism(name="sigmoid_1", activation="sigmoid")
    x = np.array([0.0, 1.0, -1.0])
    expected = 1.0 / (1.0 + np.exp(-x))
    assert np.allclose(m.apply(x), expected)


def test_loss_mse():
    m = LossMorphism(name="mse_loss", loss="mse")
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.5, 2.5, 2.5])
    loss = m.apply((y_pred, y_true))
    assert pytest.approx(loss) == ((0.5**2 + 0.5**2 + 0.5**2) / 3)


def test_loss_mae():
    m = LossMorphism(name="mae_loss", loss="mae")
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.5, 2.5, 2.5])
    loss = m.apply((y_pred, y_true))
    assert pytest.approx(loss) == ((0.5 + 0.5 + 0.5) / 3)



def test_sequential_network_forward_shape():
    net = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=2, output_dim=3),
        ActivationMorphism(name="relu_1", activation="relu"),
        LinearMorphism(name="linear_2", input_dim=3, output_dim=1),
    ])
    out = net.forward(np.array([1.0, -1.0]))
    assert out.shape == (1,)


def test_sequential_network_training_reduces_loss():
    net = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=4),
        ActivationMorphism(name="relu_1", activation="relu"),
        LinearMorphism(name="linear_2", input_dim=4, output_dim=1),
    ])
    loss_fn = LossMorphism(name="mse", loss="mse")

    xs = [np.array([x]) for x in [-1.0, 0.0, 1.0]]
    ys = [np.array([2 * x + 1]) for x in [-1.0, 0.0, 1.0]]

    initial_loss = None
    for epoch in range(50):
        epoch_loss = 0.0
        for x, y in zip(xs, ys):
            y_pred = net.forward(x)
            epoch_loss += loss_fn.apply((y_pred, y))
            grads = net.backward(x, y, loss_fn)
            net.sgd_step(grads, lr=0.05)
        if initial_loss is None:
            initial_loss = epoch_loss

    assert epoch_loss < initial_loss


def test_sequential_network_training_reduces_loss_tanh():
    from math_anything.structures.neural_network import (
        ActivationMorphism,
        LinearMorphism,
        LossMorphism,
        SequentialNetwork,
    )

    net = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=4),
        ActivationMorphism(name="tanh_1", activation="tanh"),
        LinearMorphism(name="linear_2", input_dim=4, output_dim=1),
    ])
    loss_fn = LossMorphism(name="mse", loss="mse")

    xs = [np.array([x]) for x in [-1.0, 0.0, 1.0]]
    ys = [np.array([2 * x + 1]) for x in [-1.0, 0.0, 1.0]]

    initial_loss = None
    for epoch in range(50):
        epoch_loss = 0.0
        for x, y in zip(xs, ys):
            y_pred = net.forward(x)
            epoch_loss += loss_fn.apply((y_pred, y))
            grads = net.backward(x, y, loss_fn)
            net.sgd_step(grads, lr=0.05)
        if initial_loss is None:
            initial_loss = epoch_loss

    assert epoch_loss < initial_loss


def test_sequential_network_training_reduces_loss_sigmoid():
    from math_anything.structures.neural_network import (
        ActivationMorphism,
        LinearMorphism,
        LossMorphism,
        SequentialNetwork,
    )

    net = SequentialNetwork([
        LinearMorphism(name="linear_1", input_dim=1, output_dim=4),
        ActivationMorphism(name="sigmoid_1", activation="sigmoid"),
        LinearMorphism(name="linear_2", input_dim=4, output_dim=1),
    ])
    loss_fn = LossMorphism(name="mse", loss="mse")

    xs = [np.array([x]) for x in [-1.0, 0.0, 1.0]]
    ys = [np.array([2 * x + 1]) for x in [-1.0, 0.0, 1.0]]

    initial_loss = None
    for epoch in range(50):
        epoch_loss = 0.0
        for x, y in zip(xs, ys):
            y_pred = net.forward(x)
            epoch_loss += loss_fn.apply((y_pred, y))
            grads = net.backward(x, y, loss_fn)
            net.sgd_step(grads, lr=0.05)
        if initial_loss is None:
            initial_loss = epoch_loss

    assert epoch_loss < initial_loss
