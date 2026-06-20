"""Visualize command implementation.

Generates Python visualization scripts. User runs them locally.
We do NOT render images - we generate runnable code.
"""

from pathlib import Path

from ..llm_client import LLMClient, LLMError

SYSTEM_PROMPT = """You are a scientific visualization expert.
Generate Python scripts for visualizing computational materials science data.
Use matplotlib, ASE, pymatgen, ovito, or other standard packages.
NEVER include API keys. The script must be runnable by the user."""


def cmd_visualize(args):
    """Generate visualization script."""
    engine = args.engine
    files = args.files
    viz_type = args.type
    output = args.output

    print(f"Generating {viz_type} visualization script for {engine}...")

    # Extract schema for context
    try:
        schema = _extract_schema(engine, files)
        schema_dict = schema.to_dict()
    except Exception as e:
        print(f"Warning: Schema extraction failed ({e}). Generating generic script.")
        schema_dict = {}

    # Build prompt based on viz type
    prompts = {
        "structure": _prompt_structure(engine, schema_dict),
        "mesh": _prompt_mesh(engine, schema_dict),
        "band_structure": _prompt_band_structure(engine, schema_dict),
        "radial_distribution": _prompt_rdf(engine, schema_dict),
        "stress_field": _prompt_stress(engine, schema_dict),
    }

    prompt = prompts.get(viz_type, prompts["structure"])

    try:
        client = LLMClient()
        code = client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
    except LLMError as e:
        print(f"LLM Error: {e}")
        return 1

    # Output
    if output:
        Path(output).write_text(code, encoding="utf-8")
        print(f"Visualization script saved to: {output}")
        print("Run it with: python " + output)
    else:
        print("\n--- Visualization Script ---")
        print(code)
        print("--- End ---")
        print("\nTip: Save with --output viz.py, then run: python viz.py")

    return 0


def _prompt_structure(engine, schema):
    return f"""Generate a Python script to visualize the crystal/atomic structure from a {engine} input file.
Use ASE (ase.io.read, ase.visualize.plot) or pymatgen (Structure.from_file, plot).
The script should:
1. Read the structure from the input file
2. Plot a 3D ball-and-stick or space-filling model
3. Save the figure as PNG

Available schema info:
{str(schema.get("raw_symbols", {}))[:2000]}

Generate a complete, runnable Python script."""


def _prompt_mesh(engine, schema):
    return f"""Generate a Python script to visualize a FEM mesh quality from a {engine} model.
Use matplotlib to plot element quality metrics (aspect ratio, Jacobian).
The script should:
1. Read mesh data if available
2. Compute and plot a histogram of element quality
3. Highlight poor-quality elements

Available schema info:
{str(schema.get("raw_symbols", {}))[:2000]}

Generate a complete, runnable Python script."""


def _prompt_band_structure(engine, schema):
    return f"""Generate a Python script to plot a DFT band structure.
Use matplotlib and pymatgen (if VASP) or ASE (if QE).
The script should:
1. Read band structure data from calculation output
2. Plot bands with high-symmetry k-point labels
3. Highlight Fermi level

Available schema info:
{str(schema.get("raw_symbols", {}))[:2000]}

Generate a complete, runnable Python script."""


def _prompt_rdf(engine, schema):
    return f"""Generate a Python script to compute and plot the radial distribution function (RDF).
Use ASE or MDAnalysis to read a trajectory and compute g(r).
The script should:
1. Read trajectory from LAMMPS dump or similar
2. Compute RDF for selected atom pairs
3. Plot g(r) vs r

Available schema info:
{str(schema.get("raw_symbols", {}))[:2000]}

Generate a complete, runnable Python script."""


def _prompt_stress(engine, schema):
    return f"""Generate a Python script to visualize stress field from FEM results.
Use matplotlib to create a contour plot or heatmap of von Mises stress.
The script should:
1. Read stress data from result files
2. Plot stress distribution
3. Add colorbar and annotations

Available schema info:
{str(schema.get("raw_symbols", {}))[:2000]}

Generate a complete, runnable Python script."""


def _extract_schema(engine, files):
    if engine == "vasp":
        from vasp.core.extractor_v2 import VaspExtractor

        extractor = VaspExtractor()
        file_dict = {}
        for f in files:
            fl = f.lower()
            if "incar" in fl:
                file_dict["incar"] = f
            elif "poscar" in fl:
                file_dict["poscar"] = f
            elif "kpoints" in fl:
                file_dict["kpoints"] = f
            elif "potcar" in fl:
                file_dict["potcar"] = f
        return extractor.extract(file_dict)
    elif engine == "lammps":
        from lammps.core.extractor import LammpsExtractor

        return LammpsExtractor().extract({"input": files[0]})
    elif engine == "abaqus":
        from abaqus.core.extractor import AbaqusExtractor

        return AbaqusExtractor().extract({"input": files[0]})
    elif engine == "quantum_espresso":
        from qe.core.extractor import QuantumEspressoExtractor

        return QuantumEspressoExtractor().extract({"input": files[0]})
    elif engine == "ansys":
        from ansys.core.input_extractor import AnsysInputExtractor

        return AnsysInputExtractor().extract({"input": files[0]})
    elif engine == "comsol":
        from comsol.core.extractor import ComsolExtractor

        return ComsolExtractor().extract({"input": files[0]})
    raise ValueError(f"Unsupported engine: {engine}")
