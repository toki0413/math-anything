import numpy as np

from math_anything.structures.functor import (
    MatrixFunctor,
    NaturalTransformation,
    is_natural_transformation,
)


def test_identity_matrix_functor_maps_morphism():
    d = 3
    F = MatrixFunctor(np.eye(d))
    M = np.diag([1.0, 2.0, 3.0])
    mapped = F.map_morphism(M)
    np.testing.assert_allclose(mapped, M)


def test_natural_transformation_identity_is_valid():
    d = 2
    T = np.array([[2.0, 0.0], [0.0, 3.0]])
    F = MatrixFunctor(T)
    G = MatrixFunctor(T)
    eta = NaturalTransformation({d: np.eye(d)})

    M = np.array([[1.0, 1.0], [0.0, 1.0]])
    valid, reason = is_natural_transformation(
        F, G, eta, test_morphisms=[(d, d, M)]
    )
    assert valid, reason


def test_non_natural_transformation_is_invalid():
    d = 2
    F = MatrixFunctor(np.eye(d))
    G = MatrixFunctor(np.diag([2.0, 3.0]))
    eta = NaturalTransformation({d: np.eye(d)})

    M = np.array([[1.0, 1.0], [0.0, 1.0]])
    valid, reason = is_natural_transformation(
        F, G, eta, test_morphisms=[(d, d, M)]
    )
    assert not valid
