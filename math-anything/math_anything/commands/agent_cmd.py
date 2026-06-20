"""Agent mode implementation.

Goal-driven agent that plans and executes tasks using user's LLM API.
"""

import json
from typing import Any, Dict, List

from ..check import check_schema
from ..config import get_config
from ..draft import draft_schema
from ..insight import explain_schema
from ..llm_client import LLMClient, LLMError
from ..schemas import MathSchema
from ..utils.terminal import safe_print

SYSTEM_PROMPT = """You are an expert computational materials science assistant.
Your job is to help users achieve their goals by planning and executing tasks using the math-anything tool suite.

Available tools:
- check: Validate input parameters for errors/warnings
- explain: Explain the mathematical/physical meaning of parameters
- draft: Generate publication-ready methodology sections
- generate: Create executable scripts (Python/shell)
- visualize: Generate visualization scripts

When given a goal, respond with a JSON plan:
{
  "plan": [
    {"step": 1, "tool": "check", "reason": "..."},
    {"step": 2, "tool": "explain", "reason": "..."}
  ],
  "summary": "Brief description of the plan"
}

If the goal is ambiguous, ask clarifying questions instead of generating a plan.
"""


def cmd_agent(args):
    """Run agent mode."""
    goal = args.goal
    engine = args.engine
    files = args.files or []
    auto = args.auto or get_config().get("agent.auto_confirm", False)

    if not goal:
        print("Agent mode activated. Describe your goal:")
        goal = input("> ").strip()
        if not goal:
            print("No goal provided.")
            return 1

    print(f"Goal: {goal}")
    print("Planning...")

    try:
        client = LLMClient()
    except LLMError as e:
        print(f"LLM Error: {e}")
        return 1

    # Step 1: Plan
    plan = _generate_plan(client, goal, engine, files)
    if not plan:
        return 1

    print(f"\nPlan: {plan.get('summary', 'No summary')}")
    steps = plan.get("plan", [])
    for s in steps:
        print(f"  Step {s['step']}: {s['tool']} - {s['reason']}")

    if not auto:
        print("\nExecute this plan? [Y/n/modify]")
        resp = input("> ").strip().lower()
        if resp == "n":
            print("Cancelled.")
            return 0
        elif resp == "modify":
            print("Enter modified goal:")
            goal = input("> ").strip()
            return cmd_agent(type("Args", (), {"goal": goal, "engine": engine, "files": files, "auto": auto})())

    # Step 2: Execute
    print("\nExecuting plan...\n")
    schema = None
    if engine and files:
        try:
            schema = _extract_schema(engine, files)
        except Exception as e:
            print(f"Schema extraction failed: {e}")
            schema = None

    results = []
    for step in steps:
        tool = step.get("tool")
        print(f"\n[{step['step']}/{len(steps)}] Running: {tool}")
        print("-" * 60)

        try:
            if tool == "check" and schema:
                exit_code, report = check_schema(schema, engine)
                safe_print(report)
                results.append({"tool": tool, "status": "ok", "exit_code": exit_code})
            elif tool == "explain" and schema:
                output = explain_schema(schema, engine, fmt="text")
                safe_print(output)
                results.append({"tool": tool, "status": "ok"})
            elif tool == "draft" and schema:
                output = draft_schema(schema, engine, fmt="markdown")
                safe_print(output)
                results.append({"tool": tool, "status": "ok"})
            elif tool == "generate":
                _generate_code(client, goal, engine, schema)
                results.append({"tool": tool, "status": "ok"})
            elif tool == "visualize":
                _generate_viz(client, goal, engine, schema)
                results.append({"tool": tool, "status": "ok"})
            else:
                print(f"Tool '{tool}' not available or schema missing.")
                results.append({"tool": tool, "status": "skipped"})
        except Exception as e:
            print(f"Error: {e}")
            results.append({"tool": tool, "status": "error", "message": str(e)})

    # Step 3: Summarize
    print("\n" + "=" * 60)
    print("Agent Summary")
    print("=" * 60)
    for r in results:
        icon = "✓" if r["status"] == "ok" else "✗" if r["status"] == "error" else "○"
        print(f"  {icon} {r['tool']}: {r['status']}")

    return 0


def _generate_plan(client: LLMClient, goal: str, engine: str, files: List[str]) -> Dict[str, Any]:
    """Ask LLM to generate a plan."""
    prompt = f"""User goal: {goal}
Engine: {engine or "not specified"}
Files: {files or "not provided"}

Generate a JSON plan with at most 4 steps."""

    response = client.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    # Extract JSON from response
    try:
        # Try to find JSON block
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response
        return json.loads(json_str)
    except Exception:
        print("Failed to parse plan. Raw response:")
        print(response)
        return {}


def _generate_code(client: LLMClient, goal: str, engine: str, schema: MathSchema):
    """Generate executable script via LLM."""
    schema_text = json.dumps(schema.to_dict(), indent=2, ensure_ascii=False)[:4000]
    prompt = f"""Generate a Python script that sets up a {engine} simulation based on the following schema.
The script should be runnable by the user on their own machine.
Do NOT include API keys or credentials.

Schema:
{schema_text}

User goal: {goal}

Generate a clean, well-commented Python script."""

    code = client.chat(
        [
            {"role": "system", "content": "You are a computational materials science coding assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    print("\n--- Generated Python Script ---")
    print(code)
    print("--- End of Script ---")
    print("\nSave this to a file and run with your local Python environment.")


def _generate_viz(client: LLMClient, goal: str, engine: str, schema: MathSchema):
    """Generate visualization script via LLM."""
    schema_text = json.dumps(schema.to_dict(), indent=2, ensure_ascii=False)[:4000]
    prompt = f"""Generate a Python script using matplotlib/pymatgen/ASE to visualize a {engine} simulation setup.
The script should be runnable by the user on their own machine.

Schema:
{schema_text}

Generate a clean visualization script."""

    code = client.chat(
        [
            {"role": "system", "content": "You are a scientific visualization assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    print("\n--- Generated Visualization Script ---")
    print(code)
    print("--- End of Script ---")


def _extract_schema(engine, files):
    """Extract schema for agent."""
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
