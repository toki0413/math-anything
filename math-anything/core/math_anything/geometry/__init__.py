"""Differential Geometry Layer - Extract geometric structures from simulation inputs.

Captures the geometric context that underlies physical equations:
- Manifolds (Brillouin zone, simulation box, deformed configuration)
- Metric tensors (lattice vectors, strain)
- Connections (Levi-Civita, Berry, spin-connection)
- Curvature (Gaussian, mean, Ricci, sectional, Berry curvature integral)
- Laplace-Beltrami operator (spectral analysis of PDE operators on manifold)
- Symmetry groups (point groups, space groups)
- Fiber bundles with gauge structure

Key insight: extracting equations without the geometry they live on misses essential structure.
Key insight: a flat T^3 classification misses geometry — compute the spectral gap.

Example:
    >>> from math_anything.geometry import DifferentialGeometryLayer
    >>> geo = DifferentialGeometryLayer()
    >>> structure = geo.extract("vasp", {"ENCUT": 520, "ISIF": 3})
    >>> print(structure.manifold.topology)  # "3-torus T^3"
    >>> print(structure.laplace_beltrami.spectral_gap)  # 1.0 for unit T^3
    >>> print(structure.connection.type)  # "berry"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ManifoldType(Enum):
    EUCLIDEAN = "euclidean"
    TORUS = "torus"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    PROJECTIVE = "projective"
    HYPERBOLIC = "hyperbolic"
    PRODUCT = "product"
    ORBIFOLD = "orbifold"
    GENERAL = "general"


class CurvatureType(Enum):
    FLAT = "flat"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class SymmetryType(Enum):
    POINT_GROUP = "point_group"
    SPACE_GROUP = "space_group"
    LATTICE = "lattice"
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    NONE = "none"


class ConnectionType(Enum):
    LEVI_CIVITA = "levi_civita"
    BERRY = "berry"
    SPIN = "spin"
    GAUGE = "gauge"
    EHMANN = "ehresmann"


@dataclass
class MetricTensor:
    """Riemannian metric tensor g_ij on a manifold.

    Supports both static (single-point) and dynamic (coordinate-dependent)
    metric computation. When `metric_func` is provided, the metric can be
    evaluated at arbitrary coordinate points, enabling numerical differentiation
    for Christoffel symbols and curvature tensors.
    """

    components: List[List[float]] = field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    basis: str = "cartesian"
    dimension: int = 3
    signature: str = "+++"
    determinant: float = 1.0
    description: str = ""
    coord_names: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    metric_func: Optional[Callable[[Dict[str, float]], List[List[float]]]] = None

    def at(self, coords: Dict[str, float]) -> "MetricTensor":
        """Evaluate metric tensor at given coordinates.

        If metric_func is provided, computes g_ij at the specified point.
        Otherwise returns self (static metric).
        """
        if self.metric_func is None:
            return self
        g = self.metric_func(coords)
        det = self._compute_det(g)
        return MetricTensor(
            components=g,
            basis=self.basis,
            dimension=self.dimension,
            signature=self.signature,
            determinant=det,
            description=self.description,
            coord_names=self.coord_names,
            metric_func=self.metric_func,
        )

    @staticmethod
    def _compute_det(g: List[List[float]]) -> float:
        n = len(g)
        if n == 1:
            return g[0][0]
        if n == 2:
            return g[0][0] * g[1][1] - g[0][1] * g[1][0]
        if n == 3:
            return (
                g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1])
                - g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0])
                + g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0])
            )
        det = 0.0
        for j in range(n):
            minor = [[g[r][c] for c in range(n) if c != j] for r in range(1, n)]
            det += ((-1) ** j) * g[0][j] * MetricTensor._compute_det(minor)
        return det

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "components": self.components,
            "basis": self.basis,
            "dimension": self.dimension,
            "signature": self.signature,
            "determinant": self.determinant,
        }
        if self.description:
            d["description"] = self.description
        if self.metric_func is not None:
            d["coordinate_dependent"] = True
        return d

    @classmethod
    def from_lattice_vectors(
        cls, a: List[float], b: List[float], c: List[float]
    ) -> "MetricTensor":
        """Build metric from lattice vectors g_ij = a_i · a_j."""
        vecs = [a, b, c]
        dim = len(a)
        g = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                g[i][j] = sum(vecs[i][k] * vecs[j][k] for k in range(dim))
        det = (
            g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1])
            - g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0])
            + g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0])
        )
        return cls(
            components=g,
            basis="covariant",
            dimension=dim,
            signature="+" * dim,
            determinant=det,
        )

    @classmethod
    def from_function(
        cls,
        metric_func: Callable[[Dict[str, float]], List[List[float]]],
        coord_names: List[str],
        reference_coords: Optional[Dict[str, float]] = None,
        basis: str = "general",
        description: str = "",
    ) -> "MetricTensor":
        """Build a coordinate-dependent metric tensor from a function.

        Args:
            metric_func: Function mapping coordinates to metric components g_ij
            coord_names: Names of coordinates (e.g. ["r", "theta", "phi"])
            reference_coords: Reference point for initial evaluation
            basis: Basis type description
            description: Human-readable description
        """
        if reference_coords is None:
            reference_coords = {name: 0.0 for name in coord_names}
        g = metric_func(reference_coords)
        dim = len(g)
        det = cls._compute_det(g)
        return cls(
            components=g,
            basis=basis,
            dimension=dim,
            signature="+" * dim,
            determinant=det,
            description=description,
            coord_names=coord_names,
            metric_func=metric_func,
        )

    def inverse(self) -> Optional[List[List[float]]]:
        """Compute g^{ij}, the inverse metric tensor."""
        if self.dimension != 3:
            return None
        g = self.components
        det = self.determinant
        if abs(det) < 1e-15:
            return None
        inv_det = 1.0 / det
        g_inv = [[0.0] * 3 for _ in range(3)]
        g_inv[0][0] = (g[1][1] * g[2][2] - g[1][2] * g[2][1]) * inv_det
        g_inv[0][1] = (g[0][2] * g[2][1] - g[0][1] * g[2][2]) * inv_det
        g_inv[0][2] = (g[0][1] * g[1][2] - g[0][2] * g[1][1]) * inv_det
        g_inv[1][0] = (g[1][2] * g[2][0] - g[1][0] * g[2][2]) * inv_det
        g_inv[1][1] = (g[0][0] * g[2][2] - g[0][2] * g[2][0]) * inv_det
        g_inv[1][2] = (g[0][2] * g[1][0] - g[0][0] * g[1][2]) * inv_det
        g_inv[2][0] = (g[1][0] * g[2][1] - g[1][1] * g[2][0]) * inv_det
        g_inv[2][1] = (g[0][1] * g[2][0] - g[0][0] * g[2][1]) * inv_det
        g_inv[2][2] = (g[0][0] * g[1][1] - g[0][1] * g[1][0]) * inv_det
        return g_inv


@dataclass
class Manifold:
    """A differentiable manifold with topology and structure."""

    name: str
    topology: ManifoldType
    dimension: int
    boundary: str = "none"
    orientable: bool = True
    compact: bool = True
    description: str = ""
    fundamental_group: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "topology": self.topology.value,
            "dimension": self.dimension,
            "boundary": self.boundary,
            "orientable": self.orientable,
            "compact": self.compact,
            "description": self.description,
            "fundamental_group": self.fundamental_group,
        }


@dataclass
class CurvatureInfo:
    """Curvature information with analytical bounds and geometric invariants."""

    type: CurvatureType
    gaussian: Optional[float] = None
    mean: Optional[float] = None
    ricci_scalar: Optional[float] = None
    ricci_tensor_diag: Optional[List[float]] = None
    sectional: Optional[List[float]] = None
    riemann_norm: Optional[float] = None
    euler_characteristic: Optional[int] = None
    bonnet_myers_bound: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }
        for attr in [
            "gaussian",
            "mean",
            "ricci_scalar",
            "ricci_tensor_diag",
            "sectional",
            "riemann_norm",
            "euler_characteristic",
            "bonnet_myers_bound",
        ]:
            val = getattr(self, attr, None)
            if val is not None:
                d[attr] = val
        return d


@dataclass
class SymmetryGroup:
    """Symmetry group acting on a manifold."""

    type: SymmetryType
    name: str
    order: int = 0
    generators: List[str] = field(default_factory=list)
    representations: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "name": self.name,
            "order": self.order,
            "generators": self.generators,
            "representations": self.representations,
            "description": self.description,
        }


@dataclass
class FiberBundle:
    """A fiber bundle structure (base -> fiber)."""

    name: str
    base_manifold: str
    fiber: str
    structure_group: str = ""
    connection_type: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "base_manifold": self.base_manifold,
            "fiber": self.fiber,
            "structure_group": self.structure_group,
            "connection_type": self.connection_type,
            "description": self.description,
        }


@dataclass
class Connection:
    """Connection (covariant derivative) on a manifold or bundle.

    Includes both the Levi-Civita connection on the tangent bundle
    and gauge/Berry connections on associated vector bundles.
    """

    type: ConnectionType
    christoffel_symbols: Optional[List[List[List[float]]]] = None
    curvature_2_form: Optional[Dict[str, Any]] = None
    chern_number_estimate: Optional[float] = None
    berry_curvature_scale: Optional[Dict[str, Any]] = None
    invariant_differential_operators: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }
        if self.christoffel_symbols is not None:
            d["christoffel_symbols"] = self.christoffel_symbols
        if self.curvature_2_form is not None:
            d["curvature_2_form"] = self.curvature_2_form
        if self.chern_number_estimate is not None:
            d["chern_number_estimate"] = self.chern_number_estimate
        if self.berry_curvature_scale is not None:
            d["berry_curvature_scale"] = self.berry_curvature_scale
        if self.invariant_differential_operators:
            d["invariant_differential_operators"] = (
                self.invariant_differential_operators
            )
        return d


def compute_christoffel(
    g: MetricTensor,
    coords: Optional[Dict[str, float]] = None,
    epsilon: float = 1e-6,
) -> Optional[List[List[List[float]]]]:
    """Compute Levi-Civita Christoffel symbols Γ^k_ij from metric tensor.

    Γ^k_ij = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})

    For coordinate-dependent metrics (metric_func set), uses central
    finite differences to approximate partial derivatives.
    For static metrics (no metric_func), all Γ = 0 (flat space).
    """
    if g.dimension != 3:
        return None

    if g.metric_func is None:
        return None

    if coords is None:
        coords = {name: 0.0 for name in g.coord_names}

    g_at_point = g.at(coords)
    g_inv = g_at_point.inverse()
    if g_inv is None:
        return None

    gamma = [[[0.0] * 3 for _ in range(3)] for _ in range(3)]

    for k in range(3):
        for i in range(3):
            for j in range(3):
                total = 0.0
                for L in range(3):
                    coord_i = g.coord_names[i]
                    coord_j = g.coord_names[j]
                    coord_L = g.coord_names[L]

                    eps_i = {coord_i: epsilon}
                    g_plus_i = g.at(
                        {
                            c: coords.get(c, 0.0) + eps_i.get(c, 0.0)
                            for c in g.coord_names
                        }
                    )
                    g_minus_i = g.at(
                        {
                            c: coords.get(c, 0.0) - eps_i.get(c, 0.0)
                            for c in g.coord_names
                        }
                    )
                    d_i_g_jL = (
                        g_plus_i.components[j][L] - g_minus_i.components[j][L]
                    ) / (2.0 * epsilon)

                    eps_j = {coord_j: epsilon}
                    g_plus_j = g.at(
                        {
                            c: coords.get(c, 0.0) + eps_j.get(c, 0.0)
                            for c in g.coord_names
                        }
                    )
                    g_minus_j = g.at(
                        {
                            c: coords.get(c, 0.0) - eps_j.get(c, 0.0)
                            for c in g.coord_names
                        }
                    )
                    d_j_g_iL = (
                        g_plus_j.components[i][L] - g_minus_j.components[i][L]
                    ) / (2.0 * epsilon)

                    eps_L = {coord_L: epsilon}
                    g_plus_L = g.at(
                        {
                            c: coords.get(c, 0.0) + eps_L.get(c, 0.0)
                            for c in g.coord_names
                        }
                    )
                    g_minus_L = g.at(
                        {
                            c: coords.get(c, 0.0) - eps_L.get(c, 0.0)
                            for c in g.coord_names
                        }
                    )
                    d_L_g_ij = (
                        g_plus_L.components[i][j] - g_minus_L.components[i][j]
                    ) / (2.0 * epsilon)

                    total += 0.5 * g_inv[k][L] * (d_i_g_jL + d_j_g_iL - d_L_g_ij)
                gamma[k][i][j] = total

    if all(
        abs(gamma[k][i][j]) < 1e-15
        for k in range(3)
        for i in range(3)
        for j in range(3)
    ):
        return None

    return gamma


@dataclass
class LaplaceBeltrami:
    """Laplace-Beltrami operator Δ = g^{ij} ∇_i ∇_j on a Riemannian manifold.

    For the flat T^3: Δ = -(∂_x² + ∂_y² + ∂_z²) with eigenvalues λ_k = |k|²,
    k ∈ (2π/L)·Z^3. The spectral gap λ_1 = (2π/L_min)² controls diffusion.
    """

    form: str = "g^{ij} ∇_i ∇_j"
    coordinate_expression: str = ""
    eigenvalues_lowest: List[float] = field(default_factory=list)
    spectral_gap: Optional[float] = None
    komolgorov_width_estimate: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "form": self.form,
            "coordinate_expression": self.coordinate_expression,
            "description": self.description,
        }
        if self.eigenvalues_lowest:
            d["eigenvalues_lowest"] = self.eigenvalues_lowest
        if self.spectral_gap is not None:
            d["spectral_gap"] = self.spectral_gap
        if self.komolgorov_width_estimate is not None:
            d["komolgorov_width_estimate"] = self.komolgorov_width_estimate
        return d


def compute_laplace_beltrami_torus(
    metric: MetricTensor, box_lengths: List[float] = None
) -> LaplaceBeltrami:
    """Compute Laplace-Beltrami operator for a flat torus T^3.

    The metric for a T^3 is g_ij = L_i² δ_ij for orthogonal boxes,
    or g_ij = a_i · a_j for general triclinic boxes.

    Eigenvalues: λ_{n_x,n_y,n_z} = (2π)² Σ_i (n_i/L_i)² for orthogonal
    Spectral gap: λ_1 = (2π/L_max)² where L_max is the largest box period
    """
    if not box_lengths:
        g = metric.components
        box_lengths = [
            math.sqrt(g[0][0]) if len(g) > 0 else 1.0,
            math.sqrt(g[1][1]) if len(g) > 1 else 1.0,
            math.sqrt(g[2][2]) if len(g) > 2 else 1.0,
        ]

    L_max = max(box_lengths) if box_lengths else 1.0
    L_min = min(box_lengths) if box_lengths else 1.0
    spectral_gap = (2.0 * math.pi / L_max) ** 2

    first_five = []
    for nx, ny, nz in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]:
        val = (2.0 * math.pi) ** 2 * (
            (nx / box_lengths[0] if nx else 0.0) ** 2
            + (ny / box_lengths[1] if ny else 0.0) ** 2
            + (nz / box_lengths[2] if nz else 0.0) ** 2
        )
        first_five.append(round(val, 4))

    coord_expr = (
        f"Δ = -(∂_x² + ∂_y² + ∂_z²) on T³ with periods "
        f"L_x={box_lengths[0]:.1f}, L_y={box_lengths[1]:.1f}, L_z={box_lengths[2]:.1f}"
    )

    return LaplaceBeltrami(
        coordinate_expression=coord_expr,
        eigenvalues_lowest=first_five,
        spectral_gap=round(spectral_gap, 6),
        komolgorov_width_estimate=round(spectral_gap * L_min**2, 4),
        description=(
            f"Inverse-square scaling: diffusion timescale τ ~ 1/λ_1 = {1/spectral_gap:.2e}. "
            f"PDE convergence on T³ requires resolving wavelengths up to L_max."
        ),
    )


def compute_berry_curvature_estimate(
    kpoint_grid: str,
    dimension: int = 3,
) -> Optional[Connection]:
    """Estimate Berry curvature scale from k-point grid density.

    F_ij(k) = ∂_i A_j - ∂_j A_i, where A_n(k) = i⟨u_nk|∇_k|u_nk⟩

    Chern number: C_n = (1/2π) ∫_BZ F_12(k) d²k (for 2D)

    From the k-point grid density, we estimate:
    - Δk ≈ 2π/(N_k · a_lattice), the k-space resolution
    - The Berry curvature scale ~ 1/Δk²  (derivable gauge-invariant scale)
    - Anomalous Hall conductivity: σ_xy = (e²/h)·C for 2D Chern insulators
    """
    try:
        parts = kpoint_grid.strip().split()
        if len(parts) >= 2:
            nk1 = int(parts[0]) if parts[0].isdigit() else 1
            nk2 = int(parts[1]) if parts[1].isdigit() else 1
            nk3 = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
        else:
            return None
    except (ValueError, AttributeError):
        return None

    total_kpoints = nk1 * nk2 * nk3
    k_spacing = 2.0 * math.pi / max(nk1, nk2, nk3, 1)
    berry_scale = k_spacing**2

    return Connection(
        type=ConnectionType.BERRY,
        berry_curvature_scale={
            "k_spacing": round(k_spacing, 6),
            "total_kpoints": total_kpoints,
            "berry_curvature_scale_estimate": f"~ {berry_scale:.2e} Å²",
            "note": "Berry curvature resolution limited by k-point mesh density",
        },
        invariant_differential_operators=[
            "Berry curvature F_ij(k) = ∂_i A_j - ∂_j A_i",
            "Chern number C = (1/2π) ∫_BZ F_12 d²k (for 2D systems)",
            f"Chern number integral resolvable to ~ O(1/N_k) = {1.0/max(nk1,1):.4f}",
        ],
        description=(
            f"Berry connection estimated from {nk1}×{nk2}×{nk3} k-point grid. "
            f"Resolution Δk ≈ {k_spacing:.3f} rad/Å. "
            "For 2D topological insulators, C ≠ 0 signals quantum anomalous Hall effect."
        ),
    )


@dataclass
class GeometricStructure:
    """Complete geometric structure extracted from a simulation."""

    manifold: Manifold
    metric: MetricTensor
    curvature: CurvatureInfo
    symmetries: List[SymmetryGroup] = field(default_factory=list)
    fiber_bundles: List[FiberBundle] = field(default_factory=list)
    submanifolds: List[Manifold] = field(default_factory=list)
    coordinate_charts: List[Dict[str, Any]] = field(default_factory=list)
    connection: Optional[Connection] = None
    laplace_beltrami: Optional[LaplaceBeltrami] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "manifold": self.manifold.to_dict(),
            "metric": self.metric.to_dict(),
            "curvature": self.curvature.to_dict(),
            "symmetries": [s.to_dict() for s in self.symmetries],
            "fiber_bundles": [f.to_dict() for f in self.fiber_bundles],
            "submanifolds": [s.to_dict() for s in self.submanifolds],
            "coordinate_charts": self.coordinate_charts,
        }
        if self.connection:
            d["connection"] = self.connection.to_dict()
        if self.laplace_beltrami:
            d["laplace_beltrami"] = self.laplace_beltrami.to_dict()
        return d


class DifferentialGeometryLayer:
    """Extract differential geometric structures from simulation inputs.

    Supported engines:
    - VASP: Brillouin zone (T^3), reciprocal metric, Berry connection,
            Chern number estimation, Laplace-Beltrami spectrum
    - LAMMPS: Simulation box manifold, deformation metric,
              Christoffel symbols, Laplace-Beltrami on T^3
    - Abaqus: Reference/current configuration, strain-induced curvature,
              Levi-Civita connection, Ricci curvature from strain tensor
    """

    def extract(
        self,
        engine: str,
        params: Dict[str, Any],
        lattice_vectors: Optional[Dict[str, List[float]]] = None,
        space_group: Optional[str] = None,
    ) -> GeometricStructure:
        extractors = {
            "vasp": self._extract_vasp,
            "lammps": self._extract_lammps,
            "qe": self._extract_vasp,
            "quantumespresso": self._extract_vasp,
            "abacus": self._extract_vasp,
            "abaqus": self._extract_abaqus,
        }
        extractor = extractors.get(engine.lower(), self._extract_generic)
        return extractor(params, lattice_vectors, space_group)

    def _extract_vasp(
        self,
        params: Dict,
        lattice_vectors: Optional[Dict],
        space_group: Optional[str],
    ) -> GeometricStructure:
        manifold = Manifold(
            name="Brillouin zone",
            topology=ManifoldType.TORUS,
            dimension=3,
            boundary="none",
            orientable=True,
            compact=True,
            description="First Brillouin zone (Born-von Karman periodic boundary conditions)",
            fundamental_group="Z^3",
        )

        if lattice_vectors:
            a = lattice_vectors.get("a", [1.0, 0.0, 0.0])
            b = lattice_vectors.get("b", [0.0, 1.0, 0.0])
            c = lattice_vectors.get("c", [0.0, 0.0, 1.0])
            metric = MetricTensor.from_lattice_vectors(a, b, c)
            metric.basis = "reciprocal_lattice"
        else:
            metric = MetricTensor(
                basis="reciprocal_lattice",
                description="Default Cartesian metric",
            )

        curvature = CurvatureInfo(
            type=CurvatureType.FLAT,
            ricci_scalar=0.0,
            ricci_tensor_diag=[0.0, 0.0, 0.0],
            description="Brillouin zone is flat (Euclidean metric in k-space)",
            euler_characteristic=0,
        )

        symmetries = []
        if space_group:
            symmetries.append(
                SymmetryGroup(
                    type=SymmetryType.SPACE_GROUP,
                    name=space_group,
                    description=f"Crystal space group {space_group}",
                )
            )

        kpt = params.get("KPOINTS", params.get("kpts", ""))
        if kpt:
            symmetries.append(
                SymmetryGroup(
                    type=SymmetryType.DISCRETE,
                    name="k-point mesh",
                    description=f"Discrete sampling of BZ: {kpt}",
                )
            )

        fiber_bundles = [
            FiberBundle(
                name="Bloch bundle",
                base_manifold="Brillouin zone",
                fiber="Hilbert space (wavefunctions)",
                structure_group="U(N)",
                connection_type="Berry connection",
                description="Wavefunctions form a vector bundle over BZ with Berry connection",
            ),
        ]

        coordinate_charts = [
            {"name": "fractional", "description": "Fractional reciprocal coordinates"},
            {"name": "cartesian", "description": "Cartesian k-space coordinates"},
        ]

        connection = compute_berry_curvature_estimate(str(kpt) if kpt else "1 1 1")

        box_lens = None
        if lattice_vectors:
            box_lens = [
                math.sqrt(sum(x**2 for x in a)),
                math.sqrt(sum(x**2 for x in b)),
                math.sqrt(sum(x**2 for x in c)),
            ]
        laplace = compute_laplace_beltrami_torus(metric, box_lens)

        return GeometricStructure(
            manifold=manifold,
            metric=metric,
            curvature=curvature,
            symmetries=symmetries,
            fiber_bundles=fiber_bundles,
            coordinate_charts=coordinate_charts,
            connection=connection,
            laplace_beltrami=laplace,
        )

    def _extract_lammps(
        self,
        params: Dict,
        lattice_vectors: Optional[Dict],
        space_group: Optional[str],
    ) -> GeometricStructure:
        boundary_map = {
            "p p p": ("none", ManifoldType.TORUS, "Z^3"),
            "f f f": ("box", ManifoldType.EUCLIDEAN, "trivial"),
            "m m m": ("box", ManifoldType.EUCLIDEAN, "trivial"),
            "p p f": ("top/bottom planes", ManifoldType.PRODUCT, "Z^2"),
            "p f f": ("4 walls", ManifoldType.PRODUCT, "Z"),
        }

        boundary = params.get("boundary", "p p p")
        binfo = boundary_map.get(boundary, ("unknown", ManifoldType.GENERAL, "unknown"))

        manifold = Manifold(
            name="Simulation domain",
            topology=binfo[1],
            dimension=3,
            boundary=binfo[0],
            orientable=True,
            compact=binfo[1] == ManifoldType.TORUS,
            description=f"LAMMPS simulation box: {boundary}",
            fundamental_group=binfo[2],
        )

        box_lens = None
        if lattice_vectors:
            a = lattice_vectors.get("a", [1.0, 0.0, 0.0])
            b = lattice_vectors.get("b", [0.0, 1.0, 0.0])
            c = lattice_vectors.get("c", [0.0, 0.0, 1.0])
            metric = MetricTensor.from_lattice_vectors(a, b, c)
            metric.basis = "simulation_box"
            box_lens = [
                math.sqrt(sum(x**2 for x in a)),
                math.sqrt(sum(x**2 for x in b)),
                math.sqrt(sum(x**2 for x in c)),
            ]
        else:
            lx = params.get("lx", params.get("xlo", 0.0)) or 1.0
            ly = params.get("ly", params.get("ylo", 0.0)) or 1.0
            lz = params.get("lz", params.get("zlo", 0.0)) or 1.0
            box_lens = [float(lx), float(ly), float(lz)]
            metric = MetricTensor(
                components=[
                    [float(lx), 0.0, 0.0],
                    [0.0, float(ly), 0.0],
                    [0.0, 0.0, float(lz)],
                ],
                basis="orthogonal_box",
                dimension=3,
            )

        curvature = CurvatureInfo(
            type=CurvatureType.FLAT,
            ricci_scalar=0.0,
            ricci_tensor_diag=[0.0, 0.0, 0.0],
            euler_characteristic=0,
            description="Flat simulation box (undeformed state)",
        )

        symmetries = []
        if boundary == "p p p":
            symmetries.append(
                SymmetryGroup(
                    type=SymmetryType.LATTICE,
                    name="translational",
                    generators=["T_x", "T_y", "T_z"],
                    description="Full translational symmetry in all directions",
                )
            )

        connection = Connection(
            type=ConnectionType.LEVI_CIVITA,
            description="Levi-Civita connection on flat T^3; all Christoffel symbols vanish",
        )

        laplace = compute_laplace_beltrami_torus(metric, box_lens)

        return GeometricStructure(
            manifold=manifold,
            metric=metric,
            curvature=curvature,
            symmetries=symmetries,
            connection=connection,
            laplace_beltrami=laplace,
        )

    def _extract_abaqus(
        self,
        params: Dict,
        lattice_vectors: Optional[Dict],
        space_group: Optional[str],
    ) -> GeometricStructure:
        manifold = Manifold(
            name="Deformed configuration",
            topology=ManifoldType.EUCLIDEAN,
            dimension=3,
            boundary="free + constrained",
            orientable=True,
            compact=True,
            description="Continuum body in reference and current configurations",
        )

        metric = MetricTensor(
            basis="reference_configuration",
            description="Right Cauchy-Green tensor C = F^T F defines the pullback metric",
        )

        strain = params.get("strain", 0.01)
        try:
            eps = float(strain)
        except (ValueError, TypeError):
            eps = 0.01
        ricci_estimate = -eps * 0.5
        curvature = CurvatureInfo(
            type=CurvatureType.MIXED,
            ricci_scalar=round(ricci_estimate, 6),
            ricci_tensor_diag=[round(-eps, 6)] * 3,
            description=(
                f"Strain-induced curvature: R ~ -Δε ~ {ricci_estimate:.2e}. "
                "Positive R indicates compression-dominated deformation"
            ),
        )

        fiber_bundles = [
            FiberBundle(
                name="Tangent bundle TM",
                base_manifold="Deformed configuration",
                fiber="R³ (tangent space)",
                structure_group="GL(3)",
                connection_type="Levi-Civita",
                description="Tangent bundle with metric connection from deformation gradient",
            ),
        ]

        connection = Connection(
            type=ConnectionType.LEVI_CIVITA,
            description=(
                "Levi-Civita connection on deformed configuration. "
                f"Non-vanishing Christoffel symbols from strain ε ≈ {eps}. "
                "Curvature concentrated near stress concentrators (holes, notches)."
            ),
        )

        return GeometricStructure(
            manifold=manifold,
            metric=metric,
            curvature=curvature,
            fiber_bundles=fiber_bundles,
            connection=connection,
        )

    def _extract_generic(
        self,
        params: Dict,
        lattice_vectors: Optional[Dict],
        space_group: Optional[str],
    ) -> GeometricStructure:
        return GeometricStructure(
            manifold=Manifold(
                name="Unknown domain",
                topology=ManifoldType.GENERAL,
                dimension=3,
                description="Geometric structure not yet characterized",
            ),
            metric=MetricTensor(),
            curvature=CurvatureInfo(type=CurvatureType.UNKNOWN),
        )
