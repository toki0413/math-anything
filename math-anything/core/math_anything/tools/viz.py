"""Interactive scientific visualization using Plotly.

Generates HTML-embeddable interactive charts for:
- Band structure and DOS
- Persistence diagrams (TDA)
- Phase portraits (dynamical systems)
- Berry curvature heatmaps
- Metric tensor surfaces

Requires optional dependency: plotly.
Falls back to text-based output when unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class InteractiveVisualizer:
    """Generate interactive scientific visualizations.

    Uses Plotly for browser-based interactive charts.
    All methods return either a Plotly Figure or a fallback dict.
    """

    def plot_dos(
        self,
        energies: List[float],
        dos: List[float],
        fermi_energy: Optional[float] = None,
        band_gap: Optional[float] = None,
        title: str = "Density of States",
    ) -> Dict[str, Any]:
        """Plot density of states.

        Returns dict with 'html' key containing embeddable HTML,
        or 'data' key with raw plot data as fallback.
        """
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=dos,
                    y=energies,
                    fill="tozerox" if dos else None,
                    mode="lines",
                    line=dict(color="#1565c0", width=2),
                    name="DOS",
                )
            )

            if fermi_energy is not None:
                fig.add_hline(
                    y=fermi_energy,
                    line_dash="dash",
                    line_color="#e53935",
                    annotation_text=f"E_F = {fermi_energy:.3f}",
                )

            if band_gap is not None and band_gap > 0.01:
                fig.add_vrect(
                    x0=0,
                    x1=max(dos) * 0.3 if dos else 1,
                    y0=fermi_energy - band_gap / 2 if fermi_energy else 0,
                    y1=fermi_energy + band_gap / 2 if fermi_energy else 0,
                    fillcolor="#fff9c4",
                    opacity=0.3,
                    annotation_text=f"gap = {band_gap:.3f}",
                )

            fig.update_layout(
                title=title,
                xaxis_title="DOS (states/eV)",
                yaxis_title="Energy (eV)",
                template="plotly_white",
                height=500,
            )

            return {"html": fig.to_html(include_plotlyjs="cdn", full_html=False)}

        except ImportError:
            return {
                "data": {"energies": energies[:100], "dos": dos[:100]},
                "fallback": True,
                "message": "Install plotly for interactive visualization",
            }

    def plot_persistence_diagram(
        self,
        persistence: Dict[str, List[Dict[str, float]]],
        title: str = "Persistence Diagram",
    ) -> Dict[str, Any]:
        """Plot persistence diagram from TDA results.

        Args:
            persistence: Dict with keys H0, H1, H2 containing
                         lists of {birth, death} pairs
        """
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            colors = {"H0": "#1565c0", "H1": "#e53935", "H2": "#2e7d32"}
            for dim_key, pairs in persistence.items():
                if not isinstance(pairs, list) or not pairs:
                    continue
                births = [p["birth"] for p in pairs]
                deaths = [p["death"] for p in pairs if p["death"] != float("inf")]
                if not births:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=births[: len(deaths)],
                        y=deaths,
                        mode="markers",
                        marker=dict(size=6, color=colors.get(dim_key, "#666")),
                        name=f"{dim_key} ({len(pairs)} features)",
                    )
                )

            all_vals = []
            for pairs in persistence.values():
                if not isinstance(pairs, list):
                    continue
                all_vals.extend([p["birth"] for p in pairs])
                all_vals.extend(
                    [p["death"] for p in pairs if p["death"] != float("inf")]
                )

            if all_vals:
                lo, hi = min(all_vals), max(all_vals)
                margin = (hi - lo) * 0.1 or 1.0
                fig.add_trace(
                    go.Scatter(
                        x=[lo - margin, hi + margin],
                        y=[lo - margin, hi + margin],
                        mode="lines",
                        line=dict(dash="dash", color="#999"),
                        showlegend=False,
                    )
                )

            fig.update_layout(
                title=title,
                xaxis_title="Birth",
                yaxis_title="Death",
                template="plotly_white",
                height=500,
            )

            return {"html": fig.to_html(include_plotlyjs="cdn", full_html=False)}

        except ImportError:
            return {
                "data": persistence,
                "fallback": True,
                "message": "Install plotly for interactive persistence diagrams",
            }

    def plot_phase_portrait(
        self,
        time_series: np.ndarray,
        delay: int = 1,
        title: str = "Phase Portrait",
    ) -> Dict[str, Any]:
        """Plot phase portrait (delay embedding) of a time series.

        Args:
            time_series: 1D array of observations
            delay: Delay embedding parameter τ
        """
        ts = np.asarray(time_series).flatten()
        if len(ts) < delay + 2:
            return {"error": "Time series too short for phase portrait"}

        x = ts[:-delay]
        y = ts[delay:]

        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    marker=dict(size=2, color=np.arange(len(x)), colorscale="Viridis"),
                    line=dict(width=1, color="#1565c0"),
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title=f"x(t)",
                yaxis_title=f"x(t+{delay})",
                template="plotly_white",
                height=500,
            )

            return {"html": fig.to_html(include_plotlyjs="cdn", full_html=False)}

        except ImportError:
            return {
                "data": {"x": x.tolist()[:200], "y": y.tolist()[:200]},
                "fallback": True,
                "message": "Install plotly for interactive phase portraits",
            }

    def plot_berry_curvature(
        self,
        kx: np.ndarray,
        ky: np.ndarray,
        berry: np.ndarray,
        title: str = "Berry Curvature",
    ) -> Dict[str, Any]:
        """Plot Berry curvature as a 2D heatmap.

        Args:
            kx: 1D array of kx values
            ky: 1D array of ky values
            berry: 2D array of Berry curvature values
        """
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=kx,
                    y=ky,
                    z=berry,
                    colorscale="RdBu_r",
                    colorbar=dict(title="Ω(k)"),
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title="kx",
                yaxis_title="ky",
                template="plotly_white",
                height=500,
            )

            return {"html": fig.to_html(include_plotlyjs="cdn", full_html=False)}

        except ImportError:
            return {
                "data": {"kx": kx.tolist(), "ky": ky.tolist(), "berry": berry.tolist()},
                "fallback": True,
                "message": "Install plotly for interactive Berry curvature plots",
            }

    def plot_band_structure(
        self,
        k_path: np.ndarray,
        eigenvalues: np.ndarray,
        fermi_energy: Optional[float] = None,
        k_labels: Optional[List[str]] = None,
        title: str = "Band Structure",
    ) -> Dict[str, Any]:
        """Plot band structure.

        Args:
            k_path: 1D array of k-point distances along path
            eigenvalues: (n_kpoints, n_bands) array
            fermi_energy: Optional Fermi energy line
            k_labels: Optional labels for high-symmetry points
        """
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            n_bands = eigenvalues.shape[1] if eigenvalues.ndim > 1 else 1

            for band_idx in range(n_bands):
                band_eigs = (
                    eigenvalues[:, band_idx] if eigenvalues.ndim > 1 else eigenvalues
                )
                fig.add_trace(
                    go.Scatter(
                        x=k_path,
                        y=band_eigs,
                        mode="lines",
                        line=dict(width=1.5, color="#1565c0"),
                        showlegend=False,
                    )
                )

            if fermi_energy is not None:
                fig.add_hline(
                    y=fermi_energy,
                    line_dash="dash",
                    line_color="#e53935",
                    annotation_text=f"E_F = {fermi_energy:.3f}",
                )

            fig.update_layout(
                title=title,
                xaxis_title="k-path",
                yaxis_title="Energy (eV)",
                template="plotly_white",
                height=500,
            )

            return {"html": fig.to_html(include_plotlyjs="cdn", full_html=False)}

        except ImportError:
            return {
                "data": {
                    "k_path": k_path.tolist(),
                    "eigenvalues": eigenvalues.tolist(),
                },
                "fallback": True,
                "message": "Install plotly for interactive band structure plots",
            }

    def plot_manifold(
        self,
        metric_tensor: List[List[float]],
        coord_range: List[float] = None,
        n_points: int = 50,
        title: str = "Riemannian Manifold",
    ) -> Dict[str, Any]:
        """Plot a 2D Riemannian manifold surface from metric tensor.

        Visualizes the embedding surface z = f(x,y) where the induced
        metric matches the given metric tensor at the origin.

        Args:
            metric_tensor: 2x2 metric tensor [[g_xx, g_xy], [g_xy, g_yy]]
            coord_range: [min, max] for coordinate grid
            n_points: Grid resolution
        """
        if coord_range is None:
            coord_range = [-2, 2]

        try:
            import plotly.graph_objects as go

            g = np.array(metric_tensor) if metric_tensor else np.eye(2)
            if g.shape == (2, 2):
                lo, hi = coord_range[0], coord_range[1]
                x = np.linspace(lo, hi, n_points)
                y = np.linspace(lo, hi, n_points)
                X, Y = np.meshgrid(x, y)

                det_g = g[0, 0] * g[1, 1] - g[0, 1] ** 2
                if det_g <= 0:
                    Z = np.zeros_like(X)
                else:
                    Z = 0.5 * (g[0, 0] * X**2 + 2 * g[0, 1] * X * Y + g[1, 1] * Y**2)

                fig = go.Figure()
                fig.add_trace(
                    go.Surface(
                        x=X,
                        y=Y,
                        z=Z,
                        colorscale="Viridis",
                        colorbar=dict(title="z"),
                    )
                )

                fig.update_layout(
                    title=title,
                    scene=dict(
                        xaxis_title="x",
                        yaxis_title="y",
                        zaxis_title="z",
                    ),
                    template="plotly_white",
                    height=600,
                )

                return {"html": fig.to_html(include_plotlyjs="cdn", full_html=False)}

            return {"error": "Metric tensor must be 2x2"}

        except ImportError:
            return {
                "data": {"metric_tensor": metric_tensor, "coord_range": coord_range},
                "fallback": True,
                "message": "Install plotly for interactive 3D manifold visualization",
            }

    def plot_brillouin_zone(
        self,
        lattice_vectors: List[List[float]],
        title: str = "Brillouin Zone",
    ) -> Dict[str, Any]:
        """Plot the first Brillouin zone from lattice vectors.

        Args:
            lattice_vectors: 3x3 list of lattice vectors (rows)
        """
        try:
            import plotly.graph_objects as go

            a = np.array(lattice_vectors) if lattice_vectors else np.eye(3)
            if a.shape != (3, 3):
                return {"error": "Need 3x3 lattice vectors"}

            b = 2 * np.pi * np.linalg.inv(a).T

            origin = np.zeros(3)
            vertices = []
            for i in range(3):
                for sign in [1, -1]:
                    v = np.zeros(3)
                    v[i] = sign
                    vertices.append(sign * b[i])

            fig = go.Figure()

            for v in vertices:
                fig.add_trace(
                    go.Scatter3d(
                        x=[0, v[0]],
                        y=[0, v[1]],
                        z=[0, v[2]],
                        mode="lines",
                        line=dict(width=2, color="#1565c0"),
                        showlegend=False,
                    )
                )

            for v in vertices:
                fig.add_trace(
                    go.Scatter3d(
                        x=[v[0]],
                        y=[v[1]],
                        z=[v[2]],
                        mode="markers",
                        marker=dict(size=4, color="#e53935"),
                        showlegend=False,
                    )
                )

            fig.add_trace(
                go.Scatter3d(
                    x=[0],
                    y=[0],
                    z=[0],
                    mode="markers",
                    marker=dict(size=6, color="#2e7d32"),
                    name="Γ",
                )
            )

            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="kx",
                    yaxis_title="ky",
                    zaxis_title="kz",
                    aspectmode="data",
                ),
                template="plotly_white",
                height=600,
            )

            return {"html": fig.to_html(include_plotlyjs="cdn", full_html=False)}

        except ImportError:
            return {
                "data": {"lattice_vectors": lattice_vectors},
                "fallback": True,
                "message": "Install plotly for interactive Brillouin zone visualization",
            }

    def plot_sindy_equations(
        self,
        equations: List[Dict[str, Any]],
        model_score: Optional[float] = None,
        complexity: Optional[int] = None,
        title: str = "SINDy Discovered Equations",
    ) -> Dict[str, Any]:
        """Plot SINDy discovered equations as coefficient heatmap.

        Args:
            equations: List of equation dicts with variable_name, coefficients, features
            model_score: Optional model R² score
            complexity: Optional number of active terms
        """
        try:
            import plotly.graph_objects as go

            if not equations:
                return {"error": "No equations to visualize"}

            all_features = set()
            for eq in equations:
                for f in eq.get("features", []):
                    all_features.add(f)
            all_features = sorted(all_features)

            var_names = [
                eq.get("variable_name", f"eq_{i}") for i, eq in enumerate(equations)
            ]

            z_data = []
            annotations = []
            for eq in equations:
                row = []
                feat_to_coeff = dict(
                    zip(eq.get("features", []), eq.get("coefficients", []))
                )
                for f in all_features:
                    c = feat_to_coeff.get(f, 0.0)
                    row.append(c)
                z_data.append(row)

            z_arr = np.array(z_data)

            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    z=z_arr,
                    x=all_features,
                    y=var_names,
                    colorscale="RdBu_r",
                    zmid=0,
                    colorbar=dict(title="Coefficient"),
                )
            )

            for i, eq in enumerate(equations):
                for j, f in enumerate(all_features):
                    val = z_arr[i, j]
                    if abs(val) > 1e-10:
                        fig.add_annotation(
                            x=f,
                            y=var_names[i],
                            text=f"{val:.3f}",
                            showarrow=False,
                            font=dict(size=9, color="black"),
                        )

            subtitle_parts = []
            if model_score is not None:
                subtitle_parts.append(f"R² = {model_score:.4f}")
            if complexity is not None:
                subtitle_parts.append(f"{complexity} active terms")
            if subtitle_parts:
                title += f" ({', '.join(subtitle_parts)})"

            fig.update_layout(
                title=title,
                xaxis_title="Feature",
                yaxis_title="Variable",
                template="plotly_white",
                height=max(300, 100 * len(equations)),
            )

            return {"html": fig.to_html(include_plotlyjs="cdn", full_html=False)}

        except ImportError:
            return {
                "data": {"equations": equations, "model_score": model_score},
                "fallback": True,
                "message": "Install plotly for SINDy equation visualization",
            }

    def to_html(self, fig_result: Dict[str, Any]) -> str:
        """Extract HTML from a visualization result dict."""
        if "html" in fig_result:
            return fig_result["html"]
        return f"<pre>{fig_result.get('message', 'No visualization available')}</pre>"
