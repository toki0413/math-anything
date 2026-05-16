"""Provenance tracking — trace every mathematical conclusion to its source.

Inspired by AiiDA's provenance tracking, each extracted constraint,
mathematical structure, or verification result carries a Provenance
record that tells you exactly where it came from.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Provenance:
    """Provenance record for a mathematical conclusion.

    Tracks the complete lineage from source file → extraction → analysis.
    """

    source_file: str = ""
    engine: str = ""
    parameter: str = ""
    line_number: int = 0
    extraction_method: str = ""
    confidence: float = 1.0
    literature_ref: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    parent_id: Optional[str] = None
    node_id: str = ""

    def __post_init__(self):
        if not self.node_id:
            self.node_id = self._compute_id()

    def _compute_id(self) -> str:
        raw = f"{self.source_file}:{self.engine}:{self.parameter}:{self.line_number}:{self.extraction_method}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "node_id": self.node_id,
            "engine": self.engine,
            "parameter": self.parameter,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }
        if self.source_file:
            d["source_file"] = self.source_file
        if self.line_number:
            d["line_number"] = self.line_number
        if self.literature_ref:
            d["literature_ref"] = self.literature_ref
        if self.parent_id:
            d["parent_id"] = self.parent_id
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Provenance":
        return cls(
            source_file=d.get("source_file", ""),
            engine=d.get("engine", ""),
            parameter=d.get("parameter", ""),
            line_number=d.get("line_number", 0),
            extraction_method=d.get("extraction_method", ""),
            confidence=d.get("confidence", 1.0),
            literature_ref=d.get("literature_ref"),
            timestamp=d.get("timestamp", time.time()),
            parent_id=d.get("parent_id"),
            node_id=d.get("node_id", ""),
        )


@dataclass
class ProvenanceChain:
    """Chain of provenance records showing derivation path.

    Example: POSCAR → spglib → space_group → character_table → selection_rule
    """

    nodes: List[Provenance] = field(default_factory=list)

    def add(self, provenance: Provenance) -> Provenance:
        if self.nodes:
            provenance.parent_id = self.nodes[-1].node_id
        if not provenance.node_id or provenance.node_id == hashlib.sha256(b"").hexdigest()[:12]:
            provenance.__post_init__()
        self.nodes.append(provenance)
        return provenance

    def trace(self, node_id: str) -> List[Provenance]:
        """Trace the full derivation chain leading to a node."""
        target_idx = None
        for i, node in enumerate(self.nodes):
            if node.node_id == node_id:
                target_idx = i
                break
        if target_idx is None:
            return []

        chain = [self.nodes[target_idx]]
        current = self.nodes[target_idx]
        while current.parent_id:
            for node in self.nodes:
                if node.node_id == current.parent_id:
                    chain.append(node)
                    current = node
                    break
            else:
                break
        chain.reverse()
        return chain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "length": len(self.nodes),
        }


class ProvenanceTracker:
    """Track provenance of mathematical conclusions across the pipeline.

    Usage:
        tracker = ProvenanceTracker()
        p1 = tracker.record("INCAR", "vasp", "ENCUT", 15, "regex_parse")
        p2 = tracker.record_derived("constraint_check", p1, "ENCUT > max(ENMAX)")
    """

    def __init__(self):
        self._chains: Dict[str, ProvenanceChain] = {}

    def record(
        self,
        source_file: str,
        engine: str,
        parameter: str,
        line_number: int = 0,
        extraction_method: str = "parse",
        confidence: float = 1.0,
        literature_ref: Optional[str] = None,
        chain_key: str = "default",
    ) -> Provenance:
        """Record a provenance entry for a directly extracted fact."""
        if chain_key not in self._chains:
            self._chains[chain_key] = ProvenanceChain()

        p = Provenance(
            source_file=source_file,
            engine=engine,
            parameter=parameter,
            line_number=line_number,
            extraction_method=extraction_method,
            confidence=confidence,
            literature_ref=literature_ref,
        )
        self._chains[chain_key].add(p)
        return p

    def record_derived(
        self,
        derivation_method: str,
        parent: Provenance,
        result_description: str = "",
        confidence: Optional[float] = None,
        chain_key: str = "default",
    ) -> Provenance:
        """Record a provenance entry for a derived conclusion."""
        if chain_key not in self._chains:
            self._chains[chain_key] = ProvenanceChain()

        if confidence is None:
            confidence = parent.confidence * 0.95

        p = Provenance(
            engine=parent.engine,
            parameter=result_description or f"derived_from_{parent.parameter}",
            extraction_method=derivation_method,
            confidence=confidence,
            parent_id=parent.node_id,
        )
        self._chains[chain_key].add(p)
        return p

    def trace(self, node_id: str, chain_key: str = "default") -> List[Provenance]:
        """Trace the full derivation chain for a node."""
        chain = self._chains.get(chain_key)
        if chain is None:
            return []
        return chain.trace(node_id)

    def get_chain(self, chain_key: str = "default") -> Optional[ProvenanceChain]:
        return self._chains.get(chain_key)

    def all_chains(self) -> Dict[str, ProvenanceChain]:
        return dict(self._chains)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in self._chains.items()}
