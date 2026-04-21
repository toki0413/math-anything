"""Knowledge Trio Manager for Math Anything.

Integrates the three default knowledge bases:
- arXiv: Mathematical and numerical methods literature
- Wikidata: Mathematical concepts, units, and constants
- NIST Interatomic Potentials: Potential function forms

Transparent and User-Controllable:
- Clear visibility of what data is shared
- Ability to disable any or all knowledge bases
- Offline mode with local cache
- Privacy-preserving mathematical fingerprints only
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .arxiv_client import ArxivClient
from .local_cache import LocalKnowledgeCache
from .math_fingerprint import MathFingerprintExtractor
from .nist_potentials_client import NISTPotentialsClient
from .wikidata_client import WikidataClient


class KnowledgeSource(Enum):
    """Knowledge base sources."""

    ARXIV = "arxiv"
    WIKIDATA = "wikidata"
    NIST_POTENTIALS = "nist_potentials"


@dataclass
class KnowledgeQuery:
    """A query to the knowledge trio."""

    fingerprint: Dict[str, Any]
    sources: List[KnowledgeSource] = field(
        default_factory=lambda: [
            KnowledgeSource.ARXIV,
            KnowledgeSource.WIKIDATA,
            KnowledgeSource.NIST_POTENTIALS,
        ]
    )
    max_results_per_source: int = 5


@dataclass
class KnowledgeResult:
    """Result from knowledge base query."""

    source: str
    title: str
    content: str
    relevance: float
    uri: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeTrioManager:
    """Manager for the default built-in knowledge trio.

    Provides transparent, privacy-preserving access to mathematical knowledge.

    Transparency Features:
    - User can see what data is being queried
    - All queries use mathematical fingerprints only
    - No raw simulation data ever sent
    - Clear logging of all external requests

    User Control:
    - Enable/disable individual knowledge sources
    - Enable/disable entire trio
    - Offline mode (cache only)
    - Clear cache on demand

    Example:
        ```python
        # Initialize with defaults
        trio = KnowledgeTrioManager()

        # Check transparency settings
        print(trio.get_transparency_report())

        # Query with schema
        schema = harness.extract_math(...)
        results = trio.query(schema)

        # Control sources
        trio.disable_source(KnowledgeSource.ARXIV)
        trio.enable_offline_mode()
        ```
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        enabled_sources: Optional[List[KnowledgeSource]] = None,
    ):
        """Initialize knowledge trio manager.

        Args:
            cache_dir: Directory for local cache
            enabled_sources: List of initially enabled sources (default: all)
        """
        # Initialize cache
        self.cache = LocalKnowledgeCache(cache_dir=cache_dir)

        # Initialize clients
        self.arxiv = ArxivClient(cache=self.cache)
        self.wikidata = WikidataClient(cache=self.cache)
        self.nist = NISTPotentialsClient(cache=self.cache)

        # Fingerprint extractor
        self.fingerprint_extractor = MathFingerprintExtractor()

        # Source status
        self._enabled_sources = set(enabled_sources or list(KnowledgeSource))
        self._trio_enabled = True

        # Query logging for transparency
        self._query_log: List[Dict[str, Any]] = []
        self._max_log_entries = 100

    @property
    def is_enabled(self) -> bool:
        """Check if knowledge trio is enabled."""
        return self._trio_enabled

    @property
    def enabled_sources(self) -> List[KnowledgeSource]:
        """Get list of enabled knowledge sources."""
        return list(self._enabled_sources)

    def enable(self):
        """Enable the entire knowledge trio."""
        self._trio_enabled = True

    def disable(self):
        """Disable the entire knowledge trio."""
        self._trio_enabled = False

    def enable_source(self, source: KnowledgeSource):
        """Enable a specific knowledge source."""
        self._enabled_sources.add(source)

    def disable_source(self, source: KnowledgeSource):
        """Disable a specific knowledge source."""
        self._enabled_sources.discard(source)

    def enable_offline_mode(self):
        """Enable offline mode (cache only)."""
        self.cache.set_offline_mode(True)

    def disable_offline_mode(self):
        """Disable offline mode (allow network requests)."""
        self.cache.set_offline_mode(False)

    @property
    def is_offline_mode(self) -> bool:
        """Check if operating in offline mode."""
        return self.cache.is_offline_mode

    def query(self, schema, max_results: int = 5) -> List[KnowledgeResult]:
        """Query knowledge bases using schema.

        Args:
            schema: MathSchema to extract fingerprints from
            max_results: Maximum results per source

        Returns:
            List of knowledge results
        """
        if not self._trio_enabled:
            return []

        # Extract mathematical fingerprints
        fingerprints = self.fingerprint_extractor.extract_from_schema(schema)

        # Log query for transparency
        self._log_query(schema, fingerprints)

        results = []

        # Query each enabled source
        if (
            KnowledgeSource.ARXIV in self._enabled_sources
            and not self.cache.is_offline_mode
        ):
            arxiv_results = self._query_arxiv(schema, max_results)
            results.extend(arxiv_results)

        if (
            KnowledgeSource.WIKIDATA in self._enabled_sources
            and not self.cache.is_offline_mode
        ):
            wikidata_results = self._query_wikidata(schema, max_results)
            results.extend(wikidata_results)

        if (
            KnowledgeSource.NIST_POTENTIALS in self._enabled_sources
            and not self.cache.is_offline_mode
        ):
            nist_results = self._query_nist(schema, max_results)
            results.extend(nist_results)

        # Sort by relevance
        results.sort(key=lambda x: x.relevance, reverse=True)

        return results

    def query_by_fingerprint(
        self,
        fingerprint: Dict[str, Any],
        sources: Optional[List[KnowledgeSource]] = None,
        max_results: int = 5,
    ) -> List[KnowledgeResult]:
        """Query knowledge bases using pre-computed fingerprint.

        Args:
            fingerprint: Mathematical fingerprint
            sources: Specific sources to query (default: enabled)
            max_results: Maximum results per source

        Returns:
            List of knowledge results
        """
        if not self._trio_enabled:
            return []

        sources = sources or list(self._enabled_sources)
        results = []

        # Log for transparency
        self._log_query(None, fingerprint)

        if KnowledgeSource.ARXIV in sources:
            arxiv_fp = self.fingerprint_extractor.extract_for_arxiv(None)
            arxiv_fp.update(fingerprint)
            papers = self.arxiv.query_by_fingerprint(arxiv_fp, max_results)
            for paper in papers:
                results.append(
                    KnowledgeResult(
                        source="arXiv",
                        title=paper.title,
                        content=(
                            paper.abstract[:500] + "..."
                            if len(paper.abstract) > 500
                            else paper.abstract
                        ),
                        relevance=paper.relevance_score,
                        uri=paper.pdf_url,
                        metadata={
                            "authors": paper.authors,
                            "categories": paper.categories,
                            "published": paper.published,
                        },
                    )
                )

        if KnowledgeSource.WIKIDATA in sources:
            wiki_results = self.wikidata.query_by_fingerprint(fingerprint)
            for result in wiki_results:
                results.append(
                    KnowledgeResult(
                        source="Wikidata",
                        title=result.get("name", ""),
                        content=result.get("description", ""),
                        relevance=0.8,
                        uri=result.get("uri"),
                        metadata={"type": result.get("type")},
                    )
                )

        if KnowledgeSource.NIST_POTENTIALS in sources:
            nist_results = self.nist.query_by_fingerprint(fingerprint)
            for result in nist_results:
                results.append(
                    KnowledgeResult(
                        source="NIST Potentials",
                        title=result.get("name", ""),
                        content=result.get("description", ""),
                        relevance=0.9,
                        metadata={
                            "mathematical_form": result.get("mathematical_form"),
                            "parameters": result.get("parameters"),
                            "range_type": result.get("range_type"),
                        },
                    )
                )

        return results

    def _query_arxiv(self, schema, max_results: int) -> List[KnowledgeResult]:
        """Query arXiv using schema fingerprint."""
        fingerprint = self.fingerprint_extractor.extract_for_arxiv(schema)

        papers = self.arxiv.query_by_fingerprint(fingerprint, max_results)

        results = []
        for paper in papers:
            results.append(
                KnowledgeResult(
                    source="arXiv",
                    title=paper.title,
                    content=(
                        paper.abstract[:500] + "..."
                        if len(paper.abstract) > 500
                        else paper.abstract
                    ),
                    relevance=0.7 + (0.3 if paper.matched_concepts else 0),
                    uri=paper.pdf_url,
                    metadata={
                        "authors": paper.authors[:3],
                        "categories": paper.categories,
                        "arxiv_id": paper.id,
                    },
                )
            )

        return results

    def _query_wikidata(self, schema, max_results: int) -> List[KnowledgeResult]:
        """Query Wikidata using schema fingerprint."""
        fingerprint = self.fingerprint_extractor.extract_for_wikidata(schema)

        wiki_results = self.wikidata.query_by_fingerprint(fingerprint)

        results = []
        for result in wiki_results[:max_results]:
            results.append(
                KnowledgeResult(
                    source="Wikidata",
                    title=result.get("name", ""),
                    content=result.get("description", ""),
                    relevance=0.85,
                    uri=result.get("uri"),
                    metadata={"type": result.get("type")},
                )
            )

        return results

    def _query_nist(self, schema, max_results: int) -> List[KnowledgeResult]:
        """Query NIST potentials using schema fingerprint."""
        fingerprint = self.fingerprint_extractor.extract_for_nist(schema)

        nist_results = self.nist.query_by_fingerprint(fingerprint)

        results = []
        for result in nist_results[:max_results]:
            results.append(
                KnowledgeResult(
                    source="NIST Potentials",
                    title=result.get("name", ""),
                    content=result.get("description", ""),
                    relevance=0.9 if result.get("mathematical_form") else 0.7,
                    metadata={
                        "mathematical_form": result.get("mathematical_form"),
                        "parameters": result.get("parameters", []),
                        "range_type": result.get("range_type"),
                    },
                )
            )

        return results

    def _log_query(self, schema, fingerprints: List[Any]):
        """Log query for transparency."""
        entry = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "schema_type": type(schema).__name__ if schema else "fingerprint_only",
            "num_fingerprints": (
                len(fingerprints) if isinstance(fingerprints, list) else 1
            ),
            "fingerprint_types": (
                [fp.fingerprint_type.value for fp in fingerprints]
                if isinstance(fingerprints, list)
                else ["dict"]
            ),
            "enabled_sources": [s.value for s in self._enabled_sources],
            "offline_mode": self.cache.is_offline_mode,
        }

        self._query_log.append(entry)

        # Trim log if needed
        if len(self._query_log) > self._max_log_entries:
            self._query_log = self._query_log[-self._max_log_entries :]

    def get_transparency_report(self) -> Dict[str, Any]:
        """Get transparency report showing what data is shared.

        Returns:
            Dictionary with transparency information
        """
        return {
            "trio_enabled": self._trio_enabled,
            "enabled_sources": [s.value for s in self._enabled_sources],
            "offline_mode": self.cache.is_offline_mode,
            "cache_stats": self.cache.get_stats(),
            "recent_queries": self._query_log[-10:],
            "privacy_guarantees": [
                "Only mathematical fingerprints are sent to external services",
                "No raw simulation parameters are ever transmitted",
                "No proprietary model data leaves your system",
                "Query fingerprints are abstract mathematical structures only",
            ],
            "data_shared_examples": {
                "arXiv": 'Equation patterns (e.g., "[COEFF]*∇²u"), numerical method names',
                "Wikidata": "Mathematical concept names, unit types",
                "NIST": "Potential function types, material classes",
            },
            "data_not_shared": [
                "Specific parameter values",
                "Raw simulation data",
                "Proprietary model configurations",
                "User identification information",
            ],
        }

    def get_settings(self) -> Dict[str, Any]:
        """Get current settings."""
        return {
            "trio_enabled": self._trio_enabled,
            "enabled_sources": [s.value for s in self._enabled_sources],
            "disabled_sources": [
                s.value for s in KnowledgeSource if s not in self._enabled_sources
            ],
            "offline_mode": self.cache.is_offline_mode,
            "cache_dir": str(self.cache.cache_dir),
        }

    def clear_cache(self):
        """Clear all cached knowledge base data."""
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
