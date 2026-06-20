"""Unified mathematical semantic template layer.

New engines only need to provide a parameter map + domain-specific knowledge cards.
"""

from .base import (
    CheckTemplate,
    DraftTemplate,
    InsightTemplate,
    MathNarrativeTemplate,
)
from .dft_domain import DFTCheckTemplate, DFTDraftTemplate, DFTInsightTemplate
from .fem_domain import FEMCheckTemplate, FEMDraftTemplate, FEMInsightTemplate, FEMParamExtractor
from .md_domain import MDCheckTemplate, MDDraftTemplate, MDInsightTemplate

__all__ = [
    "MathNarrativeTemplate",
    "InsightTemplate",
    "DraftTemplate",
    "CheckTemplate",
    "DFTInsightTemplate",
    "DFTDraftTemplate",
    "DFTCheckTemplate",
    "FEMInsightTemplate",
    "FEMDraftTemplate",
    "FEMCheckTemplate",
    "FEMParamExtractor",
    "MDInsightTemplate",
    "MDDraftTemplate",
    "MDCheckTemplate",
]
