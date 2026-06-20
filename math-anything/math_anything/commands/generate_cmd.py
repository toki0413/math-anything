"""Generate command implementation.

Generates executable scripts using user's LLM API key.
We do NOT execute the scripts - user runs them locally.
"""

import json
from pathlib import Path

from ..llm_client import LLMClient, LLMError

SYSTEM_PROMPT = """You are an expert computational materials science programmer.
Generate clean, well-commented, runnable scripts based on simulation parameters.
NEVER include API keys or credentials in generated code.
Always add comments explaining the physical/mathematical meaning of key parameters."""


def cmd_generate(args):
    """Generate executable script from simulation setup."""
    engine = args.engine
    files = args.files
    target = args.target
    output = args.output

    print(f"Generating {target} script for {engine}...")

    # Extract schema
    try:
        schema = _extract_schema(engine, files)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return 1

    # Build prompt from raw symbols (avoid serialization issues with custom objects)
    raw = getattr(schema, "raw_symbols", {}) or {}
    try:
        schema_text = json.dumps(raw, indent=2, ensure_ascii=False, default=str)[:6000]
    except Exception:
        schema_text = str(raw)[:6000]

    target_desc = {
        "python": "a Python script using ASE, pymatgen, numpy, and matplotlib",
        "jupyter": "a Jupyter notebook with markdown explanations and code cells",
        "shell": "a bash shell script for batch submission (PBS/Slurm)",
        "matlab": "a MATLAB script",
    }

    prompt = f"""Generate {target_desc.get(target, "a script")} for a {engine} simulation.

Input schema (extracted parameters):
{schema_text}

Requirements:
1. The script must be runnable by the user on their own machine
2. Include comments explaining physical/mathematical meaning of parameters
3. For Python: use standard scientific packages (numpy, matplotlib, ASE, pymatgen)
4. For shell: include job scheduling headers and convergence test loops
5. Do NOT include any API keys or credentials

Generate ONLY the script content."""

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
        print(f"Script saved to: {output}")
    else:
        print("\n--- Generated Script ---")
        print(code)
        print("--- End ---")
        print(f"\nTip: Save to file with --output script.{target}")

    return 0


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
