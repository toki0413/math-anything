"""Category theory structures — re-export 模块.

Category = objects + morphisms + composition.
These are the foundational structures that unify all branches of mathematics.

Category theory provides the language for:
  - Functor: structure-preserving map between categories
  - Natural transformation: map between functors
  - Adjunction: optimal correspondence between functors
  - Monad: algebraic structure arising from adjunctions
  - Limit: universal construction over a diagram
  - Kan extension: optimal extension of a functor along another

实现拆分为：
  - category_basic: Category, Functor, NaturalTransformation
  - category_advanced: Adjunction, Monad, Limit, Product, Pullback, Equalizer, KanExtension
"""

from __future__ import annotations

from .category_advanced import (
    Adjunction,
    Equalizer,
    KanExtension,
    Limit,
    Monad,
    Product,
    Pullback,
)
from .category_basic import (
    Category,
    Functor,
    NaturalTransformation,
)

__all__ = [
    # basic
    "Category",
    "Functor",
    "NaturalTransformation",
    # advanced
    "Adjunction",
    "Monad",
    "Limit",
    "Product",
    "Pullback",
    "Equalizer",
    "KanExtension",
]
