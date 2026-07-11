"""Bourbaki Domain Instantiation Layer.

Each domain represents a physics discipline as a specific configuration
of mathematical structures (conservation fields + morphism chains).

Architecture:
  foundation/  →  structures/  →  domains/
  (algorithms)    (types)         (physics)

A domain is a *fiber* over the base of mathematical structures:
  DFT, CFD, MD, FEM ... are all sections of the same sheaf.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from math_anything.morphisms import Morphism
    from math_anything.structures.base import AbstractMathematicalStructure

DOMAIN_REGISTRY: dict[str, type] = {}
"""Registry of domain instantiations.

Keys are domain names (e.g. "dft", "cfd", "md", "fem").
Values are domain classes that will be populated in Phase 4.
"""


def register_domain(name: str):
    """Decorator to register a domain class."""

    def decorator(cls):
        DOMAIN_REGISTRY[name] = cls
        return cls

    return decorator


def list_domains() -> list[str]:
    """List all registered domain names."""
    return sorted(DOMAIN_REGISTRY.keys())


def get_domain(name: str):
    """Retrieve a domain class by name."""
    if name not in DOMAIN_REGISTRY:
        available = list_domains()
        raise KeyError(f"Domain '{name}' not found. Available: {available}")
    return DOMAIN_REGISTRY[name]


from math_anything.domains.base import Domain, DomainAnalysis
from math_anything.domains.cfd import CFDDomain
from math_anything.domains.dft import DFTDomain
from math_anything.domains.em import EMDomain
from math_anything.domains.fem import FEMDomain
from math_anything.domains.md import MDDomain
from math_anything.domains.ml import SupervisedLearningDomain
from math_anything.domains.phase_field import PhaseFieldDomain
from math_anything.domains.qc import QCDomain

__all__ = [
    "DOMAIN_REGISTRY",
    "register_domain",
    "list_domains",
    "get_domain",
    "Domain",
    "DomainAnalysis",
    "DFTDomain",
    "CFDDomain",
    "MDDomain",
    "FEMDomain",
    "EMDomain",
    "QCDomain",
    "PhaseFieldDomain",
    "SupervisedLearningDomain",
]
