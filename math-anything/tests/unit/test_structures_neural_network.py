import numpy as np
import pytest

from math_anything.structures.neural_network import (
    ActivationMorphism,
    LinearMorphism,
    LossMorphism,
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
