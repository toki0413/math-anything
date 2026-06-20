"""LAMMPS-specific mathematical insight generation.

Connects LAMMPS input commands to Hamiltonian dynamics,
thermostats, barostats, and numerical integrators.
"""

from typing import Any, Dict, List, Optional

from ..schemas import MathSchema
from .base import InsightBlock, InsightEngine


class LammpsInsightEngine(InsightEngine):
    """Generate mathematical insights for LAMMPS MD simulations."""

    @property
    def engine_name(self) -> str:
        return "lammps"

    def generate(self, schema: MathSchema) -> List[InsightBlock]:
        blocks: List[InsightBlock] = []
        raw: Dict[str, Any] = schema.raw_symbols or {}
        # LammpsExtractor stores raw data directly (not under 'settings')
        fixes_dict = raw.get("fixes", {})
        fixes = (
            [
                {"style": v.get("style", ""), "group": v.get("group", ""), "params": v.get("params", [])}
                for v in fixes_dict.values()
            ]
            if isinstance(fixes_dict, dict)
            else fixes_dict
        )
        pair_style = raw.get("pair_style", "")
        timestep = raw.get("timestep")
        units = raw.get("units", "lj")
        boundary = raw.get("boundary", "p p p")

        # ── Problem overview ──
        blocks.append(self._problem_overview(fixes, pair_style, units))

        # ── Integrator & ensemble ──
        blocks.append(self._ensemble_insight(fixes, timestep, units))

        # ── Stability ──
        if timestep is not None:
            blocks.append(self._stability_insight(timestep, units, pair_style))

        # ── Boundary conditions ──
        blocks.append(self._boundary_insight(boundary))

        # ── Potential warnings ──
        warn = self._potential_warnings(pair_style, raw)
        if warn:
            blocks.append(warn)

        return blocks

    def _problem_overview(self, fixes: List[Any], pair_style: str, units: str) -> InsightBlock:
        content = (
            "You are solving Hamilton's equations of motion:\n"
            "  dr_i/dt = v_i,    dv_i/dt = F_i({r_j})/m_i\n\n"
            "Mathematically, this is an initial-value ODE system.\n"
            "The forces F_i derive from a potential energy surface U({r_j}):\n"
            "  F_i = -∇_i U({r_j})\n\n"
        )
        if pair_style:
            content += f"Your potential: {pair_style}\n"
        content += (
            f"Units: {units}. "
            "This determines the numerical values of physical constants\n"
            "in the simulation (e.g., Boltzmann constant, Planck constant).\n\n"
            "Key mathematical distinction from DFT:\n"
            "  • MD is time evolution (dynamics), not energy minimization.\n"
            "  • The trajectory is deterministic (NVE) or stochastic (NVT/NPT).\n"
            "  • Energy is conserved in NVE; in NVT/NPT, a heat bath\n"
            "    couples the system to a reservoir."
        )
        return InsightBlock(
            title="Mathematical Problem",
            content=content,
            level="math",
        )

    def _ensemble_insight(self, fixes: List[Any], timestep: Optional[float], units: str) -> InsightBlock:
        fix_styles = [f.get("style", "").lower() for f in fixes]

        ensemble = "unknown"
        hamiltonian_extra = ""
        if "nve" in fix_styles:
            ensemble = "NVE (microcanonical)"
            hamiltonian_extra = (
                "H = Σ p_i^2/2m_i + U({r_j})  is strictly conserved.\n"
                "Mathematically: dH/dt = 0 (up to floating-point error)."
            )
        elif "nvt" in fix_styles:
            ensemble = "NVT (canonical)"
            hamiltonian_extra = (
                "Extended Hamiltonian with Nose-Hoover thermostat:\n"
                "  H' = Σ p_i^2/2m_i + U({r_j}) + p_eta^2/2Q + N_f k_B T eta\n"
                "where eta is a fictitious degree of freedom coupling\n"
                "the system to a thermal reservoir at temperature T."
            )
        elif "npt" in fix_styles:
            ensemble = "NPT (isothermal-isobaric)"
            hamiltonian_extra = (
                "Extended Hamiltonian with Nose-Hoover thermostat + barostat:\n"
                "  H' = Σ p_i^2/2m_i + U({r_j}) + thermostat terms + PV term\n"
                "The barostat rescales the simulation box to maintain\n"
                "the target pressure."
            )
        elif "nph" in fix_styles:
            ensemble = "NPH (isoenthalpic-isobaric)"
            hamiltonian_extra = (
                "No thermostat, but box volume fluctuates to maintain pressure.\nTotal enthalpy H + PV is conserved."
            )
        elif "langevin" in fix_styles:
            ensemble = "Langevin (stochastic thermostat)"
            hamiltonian_extra = (
                "Stochastic differential equation:\n"
                "  m dv = F dt - γ v dt + √(2γ k_B T) dW(t)\n"
                "where dW(t) is a Wiener process. The trajectory is\n"
                "a Markov chain in phase space."
            )
        else:
            hamiltonian_extra = "No standard thermostat/fix detected. Check your input."

        content = f"Ensemble: {ensemble}\n\n{hamiltonian_extra}\n\n"
        if timestep:
            content += (
                f"Time step: {timestep} ({units} time units).\n"
                f"This determines the discretization error of the integrator.\n"
            )

        return InsightBlock(
            title="Ensemble & Hamiltonian",
            content=content,
            level="math",
        )

    def _stability_insight(self, timestep: float, units: str, pair_style: str) -> InsightBlock:
        # Rough stability estimates in different units
        if units == "metal":
            # timestep in fs
            typical = 1.0
            warning = timestep > 2.0
            scale = "fs"
            physics = (
                "For metals with EAM potentials, the fastest vibrations\n"
                "are optical phonons (~10 THz -> period ~100 fs).\n"
                f"A time step of {timestep} fs is {'large' if warning else 'reasonable'}."
            )
        elif units == "real":
            typical = 1.0
            warning = timestep > 2.0
            scale = "fs"
            physics = (
                "For organic molecules, the fastest modes are C-H stretches\n"
                "(~3000 cm- -> period ~11 fs).\n"
                f"A time step of {timestep} fs is {'dangerous' if warning else 'acceptable'}."
            )
        elif units == "lj":
            typical = 0.005
            warning = timestep > 0.01
            scale = "τ* (reduced time)"
            physics = (
                "In LJ units, the natural time scale is τ = sigma√(m/epsilon).\nTypical stability limit: dt ≈ 0.005 τ."
            )
        else:
            typical = None
            warning = False
            scale = units
            physics = f"Cannot estimate stability for units='{units}'."

        content = (
            f"Time step: {timestep} {scale}\n\n"
            f"{physics}\n\n"
            "Mathematical stability criterion (Verlet integrator):\n"
            "  dt < 2/ω_max, where ω_max is the highest frequency in the system.\n"
            "In practice, dt < 0.1 × T_min (shortest vibrational period).\n\n"
        )
        if warning:
            content += (
                "⚠  WARNING: Your time step may be too large for numerical stability.\n"
                "   If the total energy drifts upward or the simulation crashes,\n"
                "   reduce dt by a factor of 2."
            )
        elif typical and timestep < typical * 0.5:
            content += (
                "Note: Your time step is conservative. This is safe but\n"
                "inefficient — you could increase dt by ~2× to halve walltime."
            )
        else:
            content += "Time step appears reasonable for this unit system."

        return InsightBlock(
            title="Numerical Stability",
            content=content,
            level="warning" if warning else "info",
            params=["timestep"],
        )

    def _boundary_insight(self, boundary: str) -> InsightBlock:
        parts = boundary.split()
        dims = ["x", "y", "z"]
        desc = []
        for dim, b in zip(dims, parts):
            if b == "p":
                desc.append(f"  {dim}: periodic — r(x + L_{dim}) = r(x). Momentum is conserved; no surface effects.")
            elif b == "f":
                desc.append(
                    f"  {dim}: fixed — rigid walls at box boundaries. "
                    "Atoms reflect elastically (momentum not conserved)."
                )
            elif b == "s":
                desc.append(f"  {dim}: shrink-wrapped — box bounds track extreme atoms. Useful for isolated clusters.")
            elif b == "m":
                desc.append(f"  {dim}: non-periodic with minimum image convention.")
            else:
                desc.append(f"  {dim}: unknown boundary '{b}'")

        return InsightBlock(
            title="Boundary Conditions",
            content="\n".join(desc),
            level="info",
        )

    def _potential_warnings(self, pair_style: str, settings: Dict[str, Any]) -> Optional[InsightBlock]:
        warnings: List[str] = []
        style_lower = pair_style.lower()

        if "lj" in style_lower:
            warnings.append(
                "Lennard-Jones potential: U(r) = 4epsilon[(sigma/r)^2 - (sigma/r)^6]\n"
                "Mathematical issue: the potential is truncated at r_cut.\n"
                "This introduces a discontinuity in the force unless\n"
                "you apply a smoothing tail (e.g., pair_modify shift yes)."
            )
        if "eam" in style_lower:
            warnings.append(
                "EAM potential: embedding energy depends on local density.\n"
                "Ensure your potential file matches the crystal structure\n"
                "(fcc, bcc, hcp) — the embedding function is not transferable\n"
                "across different coordinations."
            )
        if "reax" in style_lower:
            warnings.append(
                "ReaxFF: bond order is a continuous function of interatomic distance.\n"
                "The energy surface has many local minima. Use a small time step\n"
                "(0.1-0.25 fs) and consider energy minimization before dynamics."
            )

        if not warnings:
            return None

        return InsightBlock(
            title="Potential-Specific Notes",
            content="\n\n".join(f"{i + 1}. {w}" for i, w in enumerate(warnings)),
            level="tip",
        )
