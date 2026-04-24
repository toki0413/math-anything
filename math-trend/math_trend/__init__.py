"""Math Trend - Dynamical systems analysis for research trends."""

__version__ = "0.1.0"

from .dynamics.cross_domain_transfer import (
    CrossDomainTransferDetector,
    KnowledgeTransfer,
)

__all__ = [
    "CrossDomainTransferDetector",
    "KnowledgeTransfer",
]
