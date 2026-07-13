"""Engine Adapters — Thin translation layer from engine-specific parameters to domain parameters.

Each adapter is ~20-50 lines, mapping engine-specific parameter names
to domain-agnostic parameters. The mathematical structure is the same
regardless of which engine you use.
"""

from __future__ import annotations

from typing import Any

# Engine → Domain mapping
ENGINE_DOMAIN_MAP: dict[str, str] = {
    # DFT domain
    "vasp": "dft",
    "qe": "dft",
    "cp2k": "dft",
    "gaussian": "qc",
    "gamess": "qc",
    "nwchem": "qc",
    "multiwfn": "dft",
    # MD domain
    "lammps": "md",
    "gromacs": "md",
    "liggghts": "md",
    # FEM domain
    "abaqus": "fem",
    "ansys": "fem",
    "solidworks": "fem",
    "comsol": "fem",
    # CFD domain
    "openfoam": "cfd",
    "fluent": "cfd",
    "su2": "cfd",
    # EM domain
    "cst": "em",
    "hfss": "em",
    # Phase Field domain
    "micress": "phase_field",
    "openphase": "phase_field",
    # Optimization
    "dakota": "fem",
    # Visualization
    "voxel": "fem",
}


def translate_params(engine: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Translate engine-specific parameters to domain parameters.

    Args:
        engine: Engine name (lowercase)
        parameters: Engine-specific parameters

    Returns:
        Dict with 'domain', 'domain_params', 'engine', 'original_params'
    """
    engine = engine.lower()
    domain = ENGINE_DOMAIN_MAP.get(engine, "unknown")

    # Engine-specific translations
    translator = _TRANSLATORS.get(engine, _generic_translate)
    domain_params = translator(parameters)

    return {
        "engine": engine,
        "domain": domain,
        "domain_params": domain_params,
        "original_params": parameters,
    }


def _generic_translate(params: dict[str, Any]) -> dict[str, Any]:
    """Generic translator: pass through parameters as-is."""
    return dict(params)


def _translate_vasp(params: dict[str, Any]) -> dict[str, Any]:
    """VASP → DFT domain parameters."""
    result = {}
    if "ENCUT" in params:
        result["ecutwfc"] = params["ENCUT"]
    if "EDIFF" in params:
        result["scf_tol"] = params["EDIFF"]
    if "ISPIN" in params:
        result["n_spin"] = params["ISPIN"]
    if "ISMEAR" in params:
        result["smearing_type"] = params["ISMEAR"]
    if "SIGMA" in params:
        result["smearing_width"] = params["SIGMA"]
    if "KPOINTS" in params:
        result["k_grid"] = params["KPOINTS"]
    if "ALGO" in params:
        algo_map = {"Normal": "davidson", "All": "cg", "Damped": "rmmdiis"}
        result["algorithm"] = algo_map.get(params["ALGO"], params["ALGO"])
    if "LDAU" in params:
        result["dft_u"] = bool(params["LDAU"])
    if "LASPH" in params:
        result["l_asph"] = bool(params["LASPH"])
    # Pass through any unrecognized params
    for k, v in params.items():
        if k not in result:
            result[k.lower()] = v
    return result


def _translate_qe(params: dict[str, Any]) -> dict[str, Any]:
    """Quantum ESPRESSO → DFT domain parameters."""
    result = {}
    if "ecutwfc" in params:
        result["ecutwfc"] = params["ecutwfc"]
    if "ecutrho" in params:
        result["ecutrho"] = params["ecutrho"]
    if "conv_thr" in params:
        result["scf_tol"] = params["conv_thr"]
    if "nspin" in params:
        result["n_spin"] = params["nspin"]
    if "degauss" in params:
        result["smearing_width"] = params["degauss"]
    for k, v in params.items():
        if k not in result:
            result[k] = v
    return result


def _translate_lammps(params: dict[str, Any]) -> dict[str, Any]:
    """LAMMPS → MD domain parameters."""
    result = {}
    if "timestep" in params:
        result["dt"] = params["timestep"]
    if "run" in params:
        result["n_steps"] = params["run"]
    if "pair_style" in params:
        result["force_field"] = params["pair_style"]
    if "thermo" in params:
        result["thermo_style"] = params["thermo"]
    if "fix_npt" in params or "fix npt" in str(params):
        result["ensemble"] = "NPT"
    elif "fix_nvt" in params or "fix nvt" in str(params):
        result["ensemble"] = "NVT"
    elif "fix_nve" in params or "fix nve" in str(params):
        result["ensemble"] = "NVE"
    for k, v in params.items():
        if k not in result:
            result[k] = v
    return result


def _translate_gromacs(params: dict[str, Any]) -> dict[str, Any]:
    """GROMACS → MD domain parameters."""
    result = {}
    if "dt" in params:
        result["dt"] = params["dt"]
    if "nsteps" in params:
        result["n_steps"] = params["nsteps"]
    if "integrator" in params:
        result["integrator"] = params["integrator"]
    if "tcoupl" in params:
        result["thermostat"] = params["tcoupl"]
    if "pcoupl" in params:
        result["barostat"] = params["pcoupl"]
    for k, v in params.items():
        if k not in result:
            result[k] = v
    return result


def _translate_abaqus(params: dict[str, Any]) -> dict[str, Any]:
    """Abaqus → FEM domain parameters."""
    result = {}
    if "NLGEOM" in params:
        result["geometric_nonlinear"] = bool(params["NLGEOM"])
    if "STEP_TYPE" in params:
        result["step_type"] = params["STEP_TYPE"]
    if "ELEMENT_TYPE" in params:
        result["element_type"] = params["ELEMENT_TYPE"]
    if "SOLVER" in params:
        result["solver"] = params["SOLVER"]
    for k, v in params.items():
        if k not in result:
            result[k.lower()] = v
    return result


def _translate_openfoam(params: dict[str, Any]) -> dict[str, Any]:
    """OpenFOAM → CFD domain parameters."""
    result = {}
    if "deltaT" in params:
        result["dt"] = params["deltaT"]
    if "endTime" in params:
        result["end_time"] = params["endTime"]
    if "solver" in params:
        result["solver"] = params["solver"]
    if "turbulenceModel" in params:
        result["turbulence_model"] = params["turbulenceModel"]
    for k, v in params.items():
        if k not in result:
            result[k] = v
    return result


# Registry of all translators
_TRANSLATORS: dict[str, callable] = {  # type: ignore[valid-type]
    "vasp": _translate_vasp,
    "qe": _translate_qe,
    "lammps": _translate_lammps,
    "gromacs": _translate_gromacs,
    "abaqus": _translate_abaqus,
    "openfoam": _translate_openfoam,
}


def list_supported_engines() -> list[str]:
    """List all engines with dedicated adapters."""
    return list(_TRANSLATORS.keys())


def list_all_engines() -> list[str]:
    """List all known engines (including generic fallback)."""
    return list(ENGINE_DOMAIN_MAP.keys())
