"""Default Built-in Knowledge Base Trio for Math Anything.

Provides transparent, privacy-preserving access to:
- arXiv: Mathematical and numerical methods literature
- Wikidata: Mathematical concepts, units, and constants
- NIST Interatomic Potentials: Potential function mathematical forms

Privacy-First Design:
- Only abstract mathematical fingerprints are sent to external services
- No raw model parameters or proprietary data ever leaves the system
- Local caching for offline availability
- User-controllable with clear visibility
"""

from .arxiv_client import ArxivClient
from .local_cache import LocalKnowledgeCache
from .math_fingerprint import MathFingerprintExtractor
from .nist_potentials_client import NISTPotentialsClient
from .trio_manager import KnowledgeTrioManager
from .wikidata_client import WikidataClient

__version__ = "1.0.0"

__all__ = [
    "KnowledgeTrioManager",
    "ArxivClient",
    "WikidataClient",
    "NISTPotentialsClient",
    "MathFingerprintExtractor",
    "LocalKnowledgeCache",
]
