"""LAMMPS pre-flight parameter consistency checks."""

from typing import Any, Dict, List

from ..schemas import MathSchema
from .base import CheckEngine, CheckResult


class LammpsCheckEngine(CheckEngine):
    """Validate LAMMPS input parameters before submission."""

    @property
    def engine_name(self) -> str:
        return "lammps"

    def check(self, schema: MathSchema) -> List[CheckResult]:
        raw: Dict[str, Any] = schema.raw_symbols or {}
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
        run_steps = raw.get("run_steps", 0)

        results: List[CheckResult] = []
        results.extend(self._check_timestep(timestep, units, pair_style))
        results.extend(self._check_ensemble(fixes, timestep))
        results.extend(self._check_boundary(boundary, fixes))
        results.extend(self._check_potential(pair_style, raw))
        results.extend(self._check_run_length(run_steps, timestep, units))
        return results

    def _check_timestep(self, timestep: Any, units: str, pair_style: str) -> List[CheckResult]:
        results: List[CheckResult] = []
        if timestep is None:
            results.append(
                CheckResult(
                    rule="TIMESTEP missing",
                    severity="warning",
                    message="No explicit timestep detected. LAMMPS will use default.",
                    suggestion="Explicitly set timestep. Typical values: 1 fs (metal/real), 0.005 (LJ).",
                )
            )
            return results

        if units == "metal":
            if timestep > 5.0:
                results.append(
                    CheckResult(
                        rule="TIMESTEP extremely large",
                        severity="error",
                        message=f"timestep = {timestep} fs is dangerously large for metal units.",
                        suggestion="Typical metal timestep: 1-2 fs. Reduce to <= 2 fs.",
                    )
                )
            elif timestep > 2.0:
                results.append(
                    CheckResult(
                        rule="TIMESTEP large",
                        severity="warning",
                        message=f"timestep = {timestep} fs may cause energy drift.",
                        suggestion="For stability, use timestep <= 1-2 fs. Test energy conservation in NVE first.",
                    )
                )
            elif timestep < 0.1:
                results.append(
                    CheckResult(
                        rule="TIMESTEP very small",
                        severity="info",
                        message=f"timestep = {timestep} fs is conservative.",
                        suggestion="You may increase timestep to 1 fs for most systems to reduce walltime.",
                    )
                )
        elif units == "real":
            if timestep > 2.0:
                results.append(
                    CheckResult(
                        rule="TIMESTEP large for organic systems",
                        severity="warning",
                        message=f"timestep = {timestep} fs may be too large for systems with C-H bonds.",
                        suggestion="For organic molecules, use 0.5-1 fs. For ReaxFF, 0.25 fs is common.",
                    )
                )
        elif units == "lj":
            if timestep > 0.01:
                results.append(
                    CheckResult(
                        rule="TIMESTEP large in LJ units",
                        severity="warning",
                        message=f"timestep = {timestep} is large for LJ units.",
                        suggestion="Typical LJ timestep: 0.005. Reduce if energy drifts.",
                    )
                )

        return results

    def _check_ensemble(self, fixes: List[Any], timestep: Any) -> List[CheckResult]:
        results: List[CheckResult] = []
        styles = [f.get("style", "").lower() for f in fixes]

        has_thermostat = any(s in styles for s in ("nvt", "npt", "langevin", "berendsen"))
        has_integrator = any(s in styles for s in ("nve", "nvt", "npt", "nph"))

        if not has_integrator:
            results.append(
                CheckResult(
                    rule="No integrator/fix detected",
                    severity="error",
                    message="No NVE, NVT, NPT, or equivalent fix found.",
                    suggestion="Add 'fix ... nvt' or 'fix ... nve' to perform dynamics. Without a fix, atoms will not move.",
                )
            )

        if "nve" in styles and has_thermostat:
            results.append(
                CheckResult(
                    rule="NVE combined with thermostat",
                    severity="warning",
                    message="NVE (microcanonical) combined with a thermostat contradicts energy conservation.",
                    suggestion="NVE should not use thermostats. If temperature control is needed, use NVT instead.",
                )
            )

        if "npt" in styles and timestep and timestep > 2.0:
            results.append(
                CheckResult(
                    rule="Large timestep with barostat",
                    severity="warning",
                    message=f"NPT with timestep = {timestep} fs may cause cell fluctuations to become unstable.",
                    suggestion="Barostats are sensitive to timestep. Use <= 1 fs for NPT, especially with anisotropic cells.",
                )
            )

        return results

    def _check_boundary(self, boundary: str, fixes: List[Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        parts = boundary.split()
        has_deform = any(f.get("style", "").lower() == "deform" for f in fixes)

        if has_deform:
            non_periodic = [b for b in parts if b != "p"]
            if non_periodic:
                results.append(
                    CheckResult(
                        rule="fix deform with non-periodic boundaries",
                        severity="error",
                        message=f"fix deform requires periodic boundaries, but boundary = '{boundary}'.",
                        suggestion="Set boundary p p p (or at least periodic in deformation directions).",
                    )
                )

        if "f" in parts and any(f.get("style", "").lower() == "npt" for f in fixes):
            results.append(
                CheckResult(
                    rule="NPT with fixed boundaries",
                    severity="error",
                    message="NPT barostat requires periodic boundaries, but fixed boundaries detected.",
                    suggestion="Use boundary p p p for NPT simulations.",
                )
            )

        return results

    def _check_potential(self, pair_style: str, raw: Dict[str, Any]) -> List[CheckResult]:
        results: List[CheckResult] = []
        if not pair_style:
            results.append(
                CheckResult(
                    rule="No pair_style defined",
                    severity="error",
                    message="pair_style is missing or not parsed.",
                    suggestion="Define interatomic potential (e.g., pair_style lj/cut, eam/alloy).",
                )
            )
            return results

        style_lower = pair_style.lower()
        if "lj" in style_lower:
            results.append(
                CheckResult(
                    rule="Lennard-Jones truncation",
                    severity="info",
                    message="LJ potential is truncated at r_cut.",
                    suggestion="Apply 'pair_modify shift yes' to avoid force discontinuity at cutoff.",
                )
            )

        if "reax" in style_lower:
            results.append(
                CheckResult(
                    rule="ReaxFF timestep recommendation",
                    severity="warning",
                    message="ReaxFF requires small timesteps due to bond-order discontinuities.",
                    suggestion="Use timestep <= 0.25 fs. Verify that total energy is conserved in a short NVE test.",
                )
            )

        return results

    def _check_run_length(self, run_steps: int, timestep: Any, units: str) -> List[CheckResult]:
        results: List[CheckResult] = []
        if run_steps == 0:
            results.append(
                CheckResult(
                    rule="No run command",
                    severity="error",
                    message="No 'run' command detected in input.",
                    suggestion="Add 'run N' to specify the number of MD steps.",
                )
            )
            return results

        if timestep:
            # Estimate physical time
            if units == "metal":
                total_fs = run_steps * timestep
                total_ps = total_fs / 1000.0
                if total_ps < 1:
                    results.append(
                        CheckResult(
                            rule="Very short simulation",
                            severity="info",
                            message=f"Total simulation time: {total_fs:.1f} fs ({run_steps} steps).",
                            suggestion="This may be insufficient for equilibrium or meaningful sampling.",
                        )
                    )
                elif total_ps > 1000:
                    results.append(
                        CheckResult(
                            rule="Long simulation",
                            severity="info",
                            message=f"Total simulation time: {total_ps:.1f} ps ({run_steps} steps).",
                            suggestion="Ensure trajectory storage (dump) frequency is appropriate for file size.",
                        )
                    )

        return results
