"""Functors and natural transformations for concrete matrix categories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class Functor(ABC):
    """Abstract functor between two categories."""

    @abstractmethod
    def map_object(self, obj: Any) -> Any:
        """Map an object to an object."""

    @abstractmethod
    def map_morphism(self, morphism: Any) -> Any:
        """Map a morphism to a morphism."""


class MatrixFunctor(Functor):
    """Functor on a matrix category defined by a square invertible matrix T.

    Objects are integer dimensions; morphisms are square matrices of that size.
    A morphism M is mapped to T @ M @ T^{-1}.
    """

    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=float)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("MatrixFunctor requires a square matrix")
        self._inv = np.linalg.inv(self.matrix)
        self._dim = self.matrix.shape[0]

    def map_object(self, obj: Any) -> Any:
        if obj != self._dim:
            raise ValueError(f"Object {obj} cannot be mapped by this functor")
        return self._dim

    def map_morphism(self, morphism: Any) -> np.ndarray:
        M = np.asarray(morphism, dtype=float)
        if M.shape != (self._dim, self._dim):
            raise ValueError(f"Morphism must be a {self._dim}x{self._dim} matrix, got {M.shape}")
        return self.matrix @ M @ self._inv


@dataclass
class NaturalTransformation:
    """A natural transformation eta: F => G, given by components eta_X."""

    components: dict[Any, Any]


def is_natural_transformation(
    F: Functor,
    G: Functor,
    eta: NaturalTransformation,
    test_morphisms: list[tuple[Any, Any, Any]],
    atol: float = 1e-8,
) -> tuple[bool, str]:
    """Verify G(f) ∘ eta_X == eta_Y ∘ F(f) for each test morphism f: X -> Y.

    Returns (True, "") if the square commutes for every test morphism,
    otherwise (False, diagnostic message).
    """
    for source_obj, target_obj, morphism in test_morphisms:
        source_component = eta.components.get(source_obj)
        target_component = eta.components.get(target_obj)
        if source_component is None or target_component is None:
            return False, f"Missing component for {source_obj} or {target_obj}"

        eta_source = np.asarray(source_component, dtype=float)
        eta_target = np.asarray(target_component, dtype=float)

        left = G.map_morphism(morphism) @ eta_source
        right = eta_target @ F.map_morphism(morphism)

        if not np.allclose(left, right, atol=atol):
            return False, (
                f"Square fails for morphism {source_obj} -> {target_obj}: "
                f"max deviation {float(np.max(np.abs(left - right)))}"
            )

    return True, ""
