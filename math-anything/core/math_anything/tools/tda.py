"""Topological Data Analysis tool — persistent homology and Betti numbers.

Pipeline: point cloud / volumetric data → persistent homology → Betti numbers
→ topological features (persistence entropy, bottleneck distance).

Requires optional dependencies: ripser, gudhi, persim.
Falls back to numpy-based basic analysis when unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PersistenceDiagram:
    """Persistence diagram from homology computation."""

    dim_0: List[Tuple[float, float]] = field(default_factory=list)
    dim_1: List[Tuple[float, float]] = field(default_factory=list)
    dim_2: List[Tuple[float, float]] = field(default_factory=list)
    max_dimension: int = 2
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for dim, pairs in [(0, self.dim_0), (1, self.dim_1), (2, self.dim_2)]:
            if pairs:
                d[f"H{dim}"] = [{"birth": b, "death": d} for b, d in pairs]
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class BettiNumbers:
    """Betti numbers β_k for k = 0, 1, 2."""

    beta_0: int = 0
    beta_1: int = 0
    beta_2: int = 0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {"beta_0": self.beta_0, "beta_1": self.beta_1, "beta_2": self.beta_2}
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class TopologyResult:
    """Complete TDA analysis result."""

    persistence: Optional[PersistenceDiagram] = None
    betti: Optional[BettiNumbers] = None
    persistence_entropy: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.persistence:
            d["persistence"] = self.persistence.to_dict()
        if self.betti:
            d["betti_numbers"] = self.betti.to_dict()
        if self.persistence_entropy is not None:
            d["persistence_entropy"] = self.persistence_entropy
        if self.description:
            d["description"] = self.description
        return d


class TDAAnalyzer:
    """Topological Data Analysis — persistent homology and features.

    Uses ripser for fast persistent homology, gudhi for cubical complexes
    (volumetric data), and persim for diagram comparison.
    Falls back to numpy-based distance matrix analysis.
    """

    def persistent_homology(
        self,
        point_cloud: np.ndarray,
        max_dim: int = 2,
    ) -> PersistenceDiagram:
        """Compute persistent homology of a point cloud.

        Args:
            point_cloud: (N, D) array of N points in D dimensions
            max_dim: Maximum homology dimension to compute
        """
        try:
            from ripser import ripser as ripser_compute

            result = ripser_compute(point_cloud, maxdim=max_dim)
            dgms = result["dgms"]

            dim_0 = [(b, d) for b, d in dgms[0] if np.isfinite(d)]
            dim_1 = (
                [(b, d) for b, d in dgms[1] if np.isfinite(d)] if len(dgms) > 1 else []
            )
            dim_2 = (
                [(b, d) for b, d in dgms[2] if np.isfinite(d)] if len(dgms) > 2 else []
            )

            return PersistenceDiagram(
                dim_0=dim_0,
                dim_1=dim_1,
                dim_2=dim_2,
                max_dimension=max_dim,
                description=f"Persistent homology via ripser ({len(point_cloud)} points, max_dim={max_dim})",
            )
        except ImportError:
            return self._fallback_homology(point_cloud, max_dim)

    def cubical_homology(
        self,
        volume_data: np.ndarray,
        max_dim: int = 2,
    ) -> PersistenceDiagram:
        """Compute persistent homology of volumetric data using cubical complex.

        Args:
            volume_data: 3D array of scalar values (e.g. electron density)
            max_dim: Maximum homology dimension
        """
        try:
            import gudhi

            cc = gudhi.CubicalComplex(top_dimensional_cells=volume_data)
            cc.compute_persistence()

            dgms = {0: [], 1: [], 2: []}
            for dim, (b, d) in cc.persistence():
                if dim <= max_dim and np.isfinite(d):
                    dgms[dim].append((b, d))

            return PersistenceDiagram(
                dim_0=dgms[0],
                dim_1=dgms[1],
                dim_2=dgms[2],
                max_dimension=max_dim,
                description=f"Cubical complex homology via gudhi (shape={volume_data.shape})",
            )
        except ImportError:
            return self._fallback_cubical(volume_data, max_dim)

    def betti_numbers(self, persistence: PersistenceDiagram) -> BettiNumbers:
        """Extract Betti numbers from persistence diagram.

        β_k = number of features with infinite persistence in dimension k.
        For finite data, use a threshold.
        """
        all_pairs = persistence.dim_0 + persistence.dim_1 + persistence.dim_2

        if not all_pairs:
            return BettiNumbers(description="No persistence data available")

        max_val = max(d for _, d in all_pairs) if all_pairs else 1.0
        threshold = max_val * 0.9

        beta_0 = sum(
            1 for b, d in persistence.dim_0 if d > threshold or d == float("inf")
        )
        beta_1 = sum(
            1 for b, d in persistence.dim_1 if d > threshold or d == float("inf")
        )
        beta_2 = sum(
            1 for b, d in persistence.dim_2 if d > threshold or d == float("inf")
        )

        if beta_0 == 0 and persistence.dim_0:
            beta_0 = 1

        return BettiNumbers(
            beta_0=beta_0,
            beta_1=beta_1,
            beta_2=beta_2,
            description=f"β₀={beta_0} (connected), β₁={beta_1} (loops), β₂={beta_2} (voids)",
        )

    def persistence_entropy(self, persistence: PersistenceDiagram) -> float:
        """Compute persistence entropy — a measure of topological complexity.

        S = -Σ p_i log(p_i), where p_i = (d_i - b_i) / Σ(d_j - b_j)
        """
        all_pairs = persistence.dim_0 + persistence.dim_1 + persistence.dim_2
        if not all_pairs:
            return 0.0

        persistences = [d - b for b, d in all_pairs if d > b]
        if not persistences:
            return 0.0

        total = sum(persistences)
        if total < 1e-15:
            return 0.0

        probs = [p / total for p in persistences]
        entropy = -sum(p * np.log(p) for p in probs if p > 1e-15)

        return float(entropy)

    def bottleneck_distance(
        self, pd1: PersistenceDiagram, pd2: PersistenceDiagram, dim: int = 0
    ) -> float:
        """Compute bottleneck distance between two persistence diagrams.

        Args:
            pd1, pd2: Persistence diagrams to compare
            dim: Homology dimension to compare (0, 1, or 2)
        """
        pairs1 = getattr(pd1, f"dim_{dim}", [])
        pairs2 = getattr(pd2, f"dim_{dim}", [])

        if not pairs1 and not pairs2:
            return 0.0

        try:
            from persim import bottleneck

            dgm1 = np.array(pairs1) if pairs1 else np.empty((0, 2))
            dgm2 = np.array(pairs2) if pairs2 else np.empty((0, 2))
            return float(bottleneck(dgm1, dgm2))
        except ImportError:
            return self._fallback_bottleneck(pairs1, pairs2)

    def analyze(
        self,
        data: np.ndarray,
        data_type: str = "point_cloud",
        max_dim: int = 2,
    ) -> TopologyResult:
        """Full TDA analysis pipeline.

        Args:
            data: Input data (point cloud or volumetric)
            data_type: "point_cloud" or "volume"
            max_dim: Maximum homology dimension
        """
        if data_type == "volume":
            persistence = self.cubical_homology(data, max_dim)
        else:
            persistence = self.persistent_homology(data, max_dim)

        betti = self.betti_numbers(persistence)
        entropy = self.persistence_entropy(persistence)

        return TopologyResult(
            persistence=persistence,
            betti=betti,
            persistence_entropy=entropy,
            description=(
                f"TDA analysis: β₀={betti.beta_0}, β₁={betti.beta_1}, β₂={betti.beta_2}, "
                f"entropy={entropy:.4f}"
            ),
        )

    def _fallback_homology(
        self, point_cloud: np.ndarray, max_dim: int
    ) -> PersistenceDiagram:
        """Basic fallback: compute pairwise distances and estimate H0."""
        n = len(point_cloud)
        if n < 2:
            return PersistenceDiagram(
                dim_0=[(0.0, float("inf"))],
                description="Single point: trivial H0",
            )

        from scipy.spatial.distance import pdist

        distances = pdist(point_cloud)
        if len(distances) == 0:
            return PersistenceDiagram(description="No distances computed")

        sorted_dists = np.sort(distances)
        n_components = 1
        threshold = sorted_dists[len(sorted_dists) // 4]

        dim_0 = [(0.0, float("inf"))]
        for d in sorted_dists[: min(10, len(sorted_dists))]:
            if d > threshold:
                break
            dim_0.append((0.0, float(d)))

        return PersistenceDiagram(
            dim_0=dim_0,
            dim_1=[],
            dim_2=[],
            max_dimension=0,
            description=f"Basic H0 analysis (install ripser for full persistent homology)",
        )

    def _fallback_cubical(
        self, volume_data: np.ndarray, max_dim: int
    ) -> PersistenceDiagram:
        """Fallback for volumetric data without gudhi."""
        flat = volume_data.flatten()
        if len(flat) == 0:
            return PersistenceDiagram(description="Empty volume data")

        min_val, max_val = float(np.min(flat)), float(np.max(flat))
        return PersistenceDiagram(
            dim_0=[(min_val, float("inf"))],
            description=f"Basic analysis (install gudhi for cubical complex homology)",
        )

    @staticmethod
    def _fallback_bottleneck(
        pairs1: List[Tuple[float, float]], pairs2: List[Tuple[float, float]]
    ) -> float:
        """Approximate bottleneck distance without persim."""
        if not pairs1 and not pairs2:
            return 0.0
        if not pairs1 or not pairs2:
            return float("inf")

        diag1 = [(b, d) for b, d in pairs1]
        diag2 = [(b, d) for b, d in pairs2]

        max_dist = 0.0
        for b1, d1 in diag1:
            min_d = min(max(abs(b1 - b2), abs(d1 - d2)) for b2, d2 in diag2)
            max_dist = max(max_dist, min_d)

        return float(max_dist)
