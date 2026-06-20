"""LAMMPS methodology draft generator."""

from typing import Any, Dict

from ..schemas import MathSchema
from .base import DraftEngine


class LammpsDraftEngine(DraftEngine):
    """Generate publication methodology for LAMMPS MD simulations."""

    @property
    def engine_name(self) -> str:
        return "lammps"

    def generate(self, schema: MathSchema, fmt: str = "markdown") -> str:
        raw: Dict[str, Any] = schema.raw_symbols or {}
        settings = raw.get("settings", {})
        fixes = settings.get("fixes", [])
        pair_style = settings.get("pair_style", "")
        timestep = settings.get("timestep")
        units = settings.get("units", "metal")
        boundary = settings.get("boundary", "p p p")
        run_steps = settings.get("run_steps")

        ensemble = self._detect_ensemble(fixes)

        lines: list[str] = []
        if fmt == "markdown":
            lines.append("# Computational Details")
        else:
            lines.append("\\section{Computational Details}")
        lines.append("")

        lines.append(self._theory(ensemble, pair_style, fmt))
        lines.append(self._software(fmt))
        lines.append(self._potential(pair_style, fmt))
        lines.append(self._ensemble(ensemble, fixes, fmt))
        if timestep is not None and run_steps is not None:
            lines.append(self._simulation_protocol(timestep, run_steps, units, fmt))
        lines.append(self._boundary(boundary, fmt))

        return "\n".join(lines)

    def _detect_ensemble(self, fixes: list[Any]) -> str:
        styles = [f.get("style", "").lower() for f in fixes]
        if "npt" in styles:
            return "NPT"
        if "nvt" in styles:
            return "NVT"
        if "nve" in styles:
            return "NVE"
        if "nph" in styles:
            return "NPH"
        if "langevin" in styles:
            return "Langevin"
        return "unknown"

    def _theory(self, ensemble: str, pair_style: str, fmt: str) -> str:
        body = (
            "Classical molecular dynamics (MD) simulations were performed by numerically "
            "integrating Newton's equations of motion:"
        )
        if fmt == "latex":
            body += r"\begin{equation}\nm_i \frac{d^2 \mathbf{r}_i}{dt^2} = -\nabla_i U(\{\mathbf{r}_j\})\end{equation}"
        else:
            body += " $$m_i d^2 r_i/dt^2 = -\\nabla_i U({r_j})$$"
        body += " where $U$ is the potential energy surface derived from the chosen interatomic potential. "
        if ensemble == "NVE":
            body += "The simulations were conducted in the microcanonical ensemble (NVE), conserving the total Hamiltonian $H = \\sum_i p_i^2/2m_i + U$."
        elif ensemble == "NVT":
            body += "The canonical ensemble (NVT) was enforced using a Nose-Hoover thermostat, which introduces a fictitious dynamical variable coupling the system to a thermal reservoir at the target temperature."
        elif ensemble == "NPT":
            body += "The isothermal-isobaric ensemble (NPT) was maintained using Nose-Hoover thermostat and barostat, allowing both temperature and pressure to be controlled while permitting volume fluctuations."
        elif ensemble == "Langevin":
            body += "The Langevin thermostat was used to maintain constant temperature via stochastic damping and random forces, corresponding to coupling to an implicit solvent or heat bath."
        return self._section("Simulation Method", body, fmt)

    def _software(self, fmt: str) -> str:
        body = "All molecular dynamics simulations were carried out using the Large-scale Atomic/Molecular Massively Parallel Simulator (LAMMPS)."
        return self._section("Software", body, fmt)

    def _potential(self, pair_style: str, fmt: str) -> str:
        body = f"Interatomic interactions were described using the {pair_style} potential. "
        if "lj" in pair_style.lower():
            body += (
                "The Lennard-Jones (LJ) pair potential $U(r) = 4\\varepsilon[(\\sigma/r)^{12} - (\\sigma/r)^6]$ "
                "was truncated at the specified cutoff distance. "
                "Long-range dispersion corrections were applied unless otherwise noted."
            )
        elif "eam" in pair_style.lower():
            body += (
                "The embedded-atom method (EAM) potential accounts for the local electron density "
                "at each atomic site via an embedding energy functional, supplemented by pairwise repulsive terms."
            )
        elif "reax" in pair_style.lower():
            body += (
                "The reactive force field (ReaxFF) describes bond breaking and formation through "
                "a bond-order-dependent energy expression, enabling chemically reactive simulations."
            )
        else:
            body += "See potential parameter file for detailed functional form and fitted parameters."
        return self._section("Interatomic Potential", body, fmt)

    def _ensemble(self, ensemble: str, fixes: list[Any], fmt: str) -> str:
        body = f"The {ensemble} ensemble was employed. "
        temps = []
        presses = []
        for f in fixes:
            params = f.get("params", [])
            for i, p in enumerate(params):
                if isinstance(p, (int, float)) and i == 0 and ensemble in ("NVT", "NPT"):
                    temps.append(str(p))
                if isinstance(p, (int, float)) and i == 1 and ensemble == "NPT":
                    presses.append(str(p))
        if temps:
            body += f"Temperature was maintained at {temps[0]} K. "
        if presses:
            body += f"Pressure was maintained at {presses[0]} atm (or pressure units). "
        return self._section("Thermostat and Barostat", body, fmt)

    def _simulation_protocol(self, timestep: float, run_steps: int, units: str, fmt: str) -> str:
        unit_desc = {
            "metal": "fs",
            "real": "fs",
            "lj": "LJ time units",
        }
        tu = unit_desc.get(units, units)
        body = (
            f"The velocity-Verlet algorithm was used to integrate the equations of motion "
            f"with a time step of {timestep} {tu}. "
            f"A total of {run_steps} steps were performed. "
        )
        return self._section("Integration Protocol", body, fmt)

    def _boundary(self, boundary: str, fmt: str) -> str:
        parts = boundary.split()
        dims = ["x", "y", "z"]
        descs = []
        for dim, b in zip(dims, parts):
            if b == "p":
                descs.append(f"{dim}: periodic")
            elif b == "f":
                descs.append(f"{dim}: fixed (non-periodic)")
            elif b == "s":
                descs.append(f"{dim}: shrink-wrapped")
            else:
                descs.append(f"{dim}: {b}")
        body = "Boundary conditions: " + ", ".join(descs) + "."
        return self._section("Boundary Conditions", body, fmt)
