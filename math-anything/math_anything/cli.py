"""Bourbaki CLI Entry Point.

Provides command-line interface for:
- Interactive REPL mode
- One-shot extraction
- Math diff comparison
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure engines/ is importable
_root = Path(__file__).parent.parent
_engines = _root / "engines"
if str(_engines) not in sys.path:
    sys.path.insert(0, str(_engines))


def _get_engine_names() -> list[str]:
    """Dynamically discover all registered engine names."""
    from math_anything.plugin import get_registry

    return get_registry().list_engines()


# 引擎名延迟到 create_parser 内才真正查询，避免 `import math_anything.cli`
# 或 `bourbaki --version` / `--help` 时触发所有引擎插件的发现与 import。
ENGINE_NAMES: list[str] | None = None


def _engine_choices() -> list[str]:
    """惰性获取引擎名列表，结果缓存到模块级 ENGINE_NAMES。"""
    global ENGINE_NAMES
    if ENGINE_NAMES is None:
        try:
            ENGINE_NAMES = _get_engine_names()
        except Exception:
            # 注册中心初始化失败时退化为空列表，避免 argparse 构造阶段崩溃
            ENGINE_NAMES = []
    return ENGINE_NAMES


from math_anything.check.base import get_check_engine
from math_anything.commands import (
    cmd_agent,
    cmd_config,
    cmd_generate,
    cmd_rag,
    cmd_visualize,
    cmd_watch,
)
from math_anything.draft.base import get_draft_engine
from math_anything.insight.base import get_insight_engine
from math_anything.repl import MathAnythingREPL, MathDiff
from math_anything.schemas import MathSchema
from math_anything.utils.terminal import safe_print


def _get_engine(engine_name: str):
    """Get an engine plugin by name."""
    from math_anything.plugin import get_engine

    return get_engine(engine_name)


def _extract_schema(engine_name: str, files: list[str]) -> MathSchema:
    """Generic schema extraction using plugin registry."""
    engine = _get_engine(engine_name)
    file_dict = {}
    for f in files:
        f_lower = f.lower()
        if "incar" in f_lower:
            file_dict["incar"] = f
        elif "poscar" in f_lower:
            file_dict["poscar"] = f
        elif "kpoints" in f_lower:
            file_dict["kpoints"] = f
        elif "potcar" in f_lower:
            file_dict["potcar"] = f
        elif "outcar" in f_lower:
            file_dict["outcar"] = f
        elif "input" in f_lower or f.endswith((".in", ".lammps", ".inp")):
            file_dict["input"] = f
        else:
            key = Path(f).suffix.lstrip(".") or "input"
            file_dict[key] = f
    if not file_dict:
        file_dict["input"] = files[0]
    return engine.extract(file_dict)  # type: ignore[no-any-return]


def create_parser(prog_name: str = "bourbaki") -> argparse.ArgumentParser:
    """Create argument parser."""
    # 版本号从包元数据动态读取，避免与 pyproject.toml 失同步
    try:
        from importlib.metadata import version as _pkg_version

        _version = _pkg_version("bourbaki")
    except Exception:
        try:
            from math_anything import __version__ as _version
        except Exception:
            _version = "0.0.0"

    parser = argparse.ArgumentParser(
        prog=prog_name,
        description="Bourbaki - Mathematical Structure Modeling for Computational Science",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Interactive REPL mode
  {prog_name} repl

  # Extract from VASP files
  {prog_name} extract vasp INCAR POSCAR KPOINTS --output schema.json

  # Compare two schemas
  {prog_name} diff schema1.json schema2.json

  # Cross-engine comparison (VASP vs QE)
  {prog_name} cross vasp_INCAR.json quantum_espresso

For more information: https://github.com/toki0413/math-anything
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # REPL mode
    repl_parser = subparsers.add_parser(
        "repl",
        help="Start interactive REPL mode",
    )
    repl_parser.add_argument(
        "--session",
        type=str,
        help="Load existing session file",
    )

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract mathematical schema from input files",
    )
    extract_parser.add_argument(
        "engine",
        choices=_engine_choices(),
        help="Computational engine type",
    )
    extract_parser.add_argument(
        "files",
        nargs="+",
        help="Input file(s)",
    )
    extract_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output JSON file",
    )
    extract_parser.add_argument(
        "--format",
        choices=["json", "yaml", "pretty"],
        default="pretty",
        help="Output format",
    )

    # Diff command
    diff_parser = subparsers.add_parser(
        "diff",
        help="Compare two mathematical schemas (math-anything diff)",
    )
    diff_parser.add_argument(
        "file1",
        help="First schema file (JSON)",
    )
    diff_parser.add_argument(
        "file2",
        help="Second schema file (JSON)",
    )
    diff_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output diff result to file",
    )
    diff_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    # Explain command
    explain_parser = subparsers.add_parser(
        "explain",
        help="Explain the mathematics behind simulation inputs",
    )
    explain_parser.add_argument(
        "engine",
        choices=_engine_choices(),
        help="Computational engine type",
    )
    explain_parser.add_argument(
        "files",
        nargs="+",
        help="Input file(s)",
    )
    explain_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    # Draft command
    draft_parser = subparsers.add_parser(
        "draft",
        help="Generate publication-ready methodology section",
    )
    draft_parser.add_argument(
        "engine",
        choices=_engine_choices(),
        help="Computational engine type",
    )
    draft_parser.add_argument(
        "files",
        nargs="+",
        help="Input file(s)",
    )
    draft_parser.add_argument(
        "--format",
        choices=["markdown", "latex", "docx", "notebook"],
        default="markdown",
        help="Output format (docx/notebook require --output)",
    )
    draft_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Save to file instead of stdout",
    )

    # Cross-engine comparison
    cross_parser = subparsers.add_parser(
        "cross",
        help="Cross-engine parameter mapping",
    )
    cross_parser.add_argument(
        "schema_file",
        help="Source schema file",
    )
    cross_parser.add_argument(
        "target_engine",
        help="Target engine (e.g., quantum_espresso, gromacs)",
    )

    # Loops command
    loops_parser = subparsers.add_parser(
        "loops",
        help="Detect topology loops in an engine's morphism graph",
    )
    loops_parser.add_argument(
        "engine_positional",
        nargs="?",
        metavar="engine",
        help="Engine name (positional; --engine takes precedence)",
    )
    loops_parser.add_argument(
        "--engine",
        choices=_engine_choices(),
        default=None,
        help="Engine name (flag form; overrides positional engine)",
    )
    loops_parser.add_argument(
        "files",
        nargs="*",
        help="Input files to extract schema from",
    )
    loops_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file",
    )
    loops_parser.add_argument(
        "--visualize",
        choices=["mermaid", "graphviz"],
        default=None,
        help="Emit Mermaid or Graphviz representation of the morphism graph",
    )

    # ML surrogate command
    ml_parser = subparsers.add_parser(
        "ml",
        help="Analyze a supervised-learning model as a morphism chain",
    )
    ml_parser.add_argument("--input-dim", type=int, default=2)
    ml_parser.add_argument("--output-dim", type=int, default=1)
    ml_parser.add_argument("--architecture", type=str, default="mlp")
    ml_parser.add_argument("--loss", type=str, default="mse")
    ml_parser.add_argument("--visualize", choices=["mermaid"], default=None)
    ml_parser.add_argument(
        "--compare-with",
        type=str,
        default=None,
        help="Cross-domain homotopy comparison target",
    )
    ml_parser.add_argument(
        "--compare-paths",
        action="store_true",
        help="Train two identical runs and report optimization-landscape homotopy",
    )
    ml_parser.add_argument(
        "--transfer",
        action="store_true",
        help="Demonstrate transfer learning as a natural transformation",
    )
    ml_parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "deepmd", "mace", "chgnet"],
        help="Surrogate backend to use (falls back to numpy if not installed)",
    )

    # Homotopy command
    homotopy_parser = subparsers.add_parser(
        "homotopy",
        help="Check whether two DFT-family engine configurations are homotopic",
    )
    homotopy_parser.add_argument(
        "engine_a",
        choices=["vasp", "qe", "cp2k"],
        help="Source engine",
    )
    homotopy_parser.add_argument(
        "engine_b",
        choices=["vasp", "qe", "cp2k"],
        help="Target engine",
    )
    homotopy_parser.add_argument(
        "--param-a",
        type=float,
        default=None,
        help="Canonical cutoff parameter for engine A (eV)",
    )
    homotopy_parser.add_argument(
        "--param-b",
        type=float,
        default=None,
        help="Canonical cutoff parameter for engine B (eV)",
    )
    homotopy_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output JSON file (must be relative to cwd)",
    )

    # Check command
    check_parser = subparsers.add_parser(
        "check",
        help="Pre-flight parameter consistency check",
    )
    check_parser.add_argument(
        "engine",
        choices=_engine_choices(),
        help="Computational engine type",
    )
    check_parser.add_argument(
        "files",
        nargs="+",
        help="Input file(s)",
    )

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Manage configuration and API keys",
    )
    config_parser.add_argument(
        "action",
        choices=["get", "set", "list", "path"],
        help="Config action",
    )
    config_parser.add_argument(
        "key",
        nargs="?",
        help="Config key (dot path, e.g. llm.api_key)",
    )
    config_parser.add_argument(
        "value",
        nargs="?",
        help="Value to set",
    )

    # Watch command
    watch_parser = subparsers.add_parser(
        "watch",
        help="Watch input files for changes and auto-run checks",
    )
    watch_parser.add_argument(
        "engine",
        choices=[
            "vasp",
            "lammps",
            "abaqus",
            "ansys",
            "comsol",
            "quantum_espresso",
            "gromacs",
            "multiwfn",
            "openfoam",
            "cp2k",
            "fluent",
            "su2",
            "gamess",
            "nwchem",
            "liggghts",
            "dakota",
            "solidworks",
            "voxel",
        ],
        help="Engine to watch",
    )
    watch_parser.add_argument(
        "files",
        nargs="+",
        help="Files to watch",
    )
    watch_parser.add_argument(
        "--auto-check",
        action="store_true",
        default=True,
        help="Auto-run check on file changes",
    )
    watch_parser.add_argument(
        "--auto-explain",
        action="store_true",
        default=False,
        help="Auto-run explain on file changes",
    )

    # Agent command
    agent_parser = subparsers.add_parser(
        "agent",
        help="Goal-driven agent mode (uses your LLM API key)",
    )
    agent_parser.add_argument(
        "goal",
        nargs="?",
        help="Your goal in natural language",
    )
    agent_parser.add_argument(
        "--engine",
        choices=[
            "vasp",
            "lammps",
            "abaqus",
            "ansys",
            "comsol",
            "quantum_espresso",
            "gromacs",
            "multiwfn",
            "openfoam",
            "cp2k",
            "fluent",
            "su2",
            "gamess",
            "nwchem",
            "liggghts",
            "dakota",
            "solidworks",
            "voxel",
        ],
        help="Primary engine context",
    )
    agent_parser.add_argument(
        "--files",
        nargs="+",
        help="Input files for context",
    )
    agent_parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-execute without confirmation",
    )

    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate executable scripts from simulation setup (uses your LLM API key)",
    )
    gen_parser.add_argument(
        "engine",
        choices=[
            "vasp",
            "lammps",
            "abaqus",
            "ansys",
            "comsol",
            "quantum_espresso",
            "gromacs",
            "multiwfn",
            "openfoam",
            "cp2k",
            "fluent",
            "su2",
            "gamess",
            "nwchem",
            "liggghts",
            "dakota",
            "solidworks",
            "voxel",
        ],
        help="Engine type",
    )
    gen_parser.add_argument(
        "files",
        nargs="+",
        help="Input files",
    )
    gen_parser.add_argument(
        "--target",
        choices=["python", "jupyter", "shell", "matlab"],
        default="python",
        help="Output script type",
    )
    gen_parser.add_argument(
        "--output",
        type=str,
        help="Output file path",
    )

    # RAG command
    rag_parser = subparsers.add_parser(
        "rag",
        help="Knowledge base operations (uses your embedding API key)",
    )
    rag_parser.add_argument(
        "action",
        choices=["index", "query", "status"],
        help="RAG action",
    )
    rag_parser.add_argument(
        "--docs",
        nargs="+",
        help="Document paths to index",
    )
    rag_parser.add_argument(
        "--query",
        type=str,
        help="Query string",
    )

    # Visualize command
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Generate visualization scripts (runs on your machine)",
    )
    viz_parser.add_argument(
        "engine",
        choices=[
            "vasp",
            "lammps",
            "abaqus",
            "ansys",
            "comsol",
            "quantum_espresso",
            "gromacs",
            "multiwfn",
            "openfoam",
            "cp2k",
            "fluent",
            "su2",
            "gamess",
            "nwchem",
            "liggghts",
            "dakota",
            "solidworks",
            "voxel",
        ],
        help="Engine type",
    )
    viz_parser.add_argument(
        "files",
        nargs="+",
        help="Input files",
    )
    viz_parser.add_argument(
        "--type",
        choices=["structure", "mesh", "band_structure", "radial_distribution", "stress_field"],
        default="structure",
        help="Visualization type",
    )
    viz_parser.add_argument(
        "--output",
        type=str,
        help="Output script file",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate symbolic constraints",
    )
    validate_parser.add_argument(
        "schema_file",
        help="Schema file to validate",
    )

    # Version
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {_version}",
    )

    return parser


def cmd_repl(args):
    """Start REPL mode."""
    repl = MathAnythingREPL()

    if args.session:
        try:
            from math_anything.repl.core import REPLSession

            repl.session = REPLSession.load(args.session)
            print(f"Loaded session: {repl.session.name}")
        except Exception as e:
            print(f"Error loading session: {e}")

    try:
        repl.cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")


def cmd_extract(args):
    """Extract schema from files — all engines via registry."""
    print(f"Extracting from {args.engine.upper()} files...")
    try:
        schema = _extract_schema(args.engine, args.files)
        if args.output:
            schema.save(args.output)
            print(f"Schema saved to {args.output}")
        elif args.format == "json":
            import json

            print(json.dumps(schema.to_dict(), indent=2))
        elif args.format == "yaml":
            try:
                import yaml
            except ImportError:
                print("Error: PyYAML not installed. Run `pip install pyyaml`.")
                return 1
            print(yaml.safe_dump(schema.to_dict(), sort_keys=False, allow_unicode=True))
        elif args.format == "pretty":
            _print_pretty_schema(schema)
        else:
            # 未知 format 兜底（避免静默无输出）
            _print_pretty_schema(schema)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_explain(args):
    """Explain mathematics behind input files — all engines via registry."""
    print(f"Analyzing {args.engine.upper()} inputs...")
    try:
        schema = _extract_schema(args.engine, args.files)
        engine = get_insight_engine(args.engine)
        output = engine.explain(schema, fmt=args.format)
        safe_print(output)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_draft(args):
    """Generate publication methodology — all engines via registry."""
    print(f"Drafting methodology for {args.engine.upper()}...")
    try:
        schema = _extract_schema(args.engine, args.files)
        if args.format in ("docx", "notebook") or (args.output and args.output.lower().endswith((".docx", ".ipynb"))):
            if not args.output:
                print(f"Error: {args.format} output requires --output path")
                return 1
            from math_anything.draft.docx_draft import markdown_to_docx
            from math_anything.draft.notebook_draft import markdown_to_notebook

            md = get_draft_engine(args.engine).generate(schema, fmt="markdown")
            if args.format == "docx":
                markdown_to_docx(md, args.output)
            else:
                markdown_to_notebook(md, args.output)
            print(f"Methodology saved to {args.output}")
        else:
            engine = get_draft_engine(args.engine)
            output = engine.generate(schema, fmt=args.format)
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
                print(f"Methodology saved to {args.output}")
            else:
                safe_print(output)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_check(args):
    """Pre-flight check — all engines via registry."""
    print(f"Checking {args.engine.upper()} inputs...")
    try:
        schema = _extract_schema(args.engine, args.files)
        engine = get_check_engine(args.engine)
        exit_code, report = engine.run(schema)
        safe_print(report)
        return exit_code
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_diff(args):
    """Compare two schemas."""
    print(f"Comparing {args.file1} vs {args.file2}...")

    try:
        # Load schemas
        schema1 = MathSchema.load(args.file1)
        schema2 = MathSchema.load(args.file2)

        # Compare
        diff = MathDiff.compare(schema1, schema2)

        # Output
        if args.output:
            import json

            with open(args.output, "w") as f:
                json.dump(diff.to_dict(), f, indent=2)
            print(f"Diff saved to {args.output}")
        else:
            if args.format == "json":
                import json

                print(json.dumps(diff.to_dict(), indent=2))
            else:
                _print_pretty_diff(diff, args.file1, args.file2)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_cross(args):
    """Cross-engine comparison."""
    print(f"Mapping {args.schema_file} to {args.target_engine}...")

    try:
        schema = MathSchema.load(args.schema_file)
        suggestions = MathDiff.cross_engine_compare(schema, args.target_engine)

        print("\n" + "=" * 60)
        print(f"Cross-Engine Mapping: {args.schema_file} -> {args.target_engine}")
        print("=" * 60)

        if suggestions["mappable_parameters"]:
            print("\n🔄 Mappable Parameters:")
            for m in suggestions["mappable_parameters"]:
                print(f"  {m['source']:15} -> {m['target']:15} ({m['meaning']})")
                print(f"    Value: {m['value']}")

        if suggestions["unmappable_parameters"]:
            print("\n⚠️ Unmapped Parameters (manual attention needed):")
            for p in suggestions["unmappable_parameters"]:
                print(f"  - {p}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_loops(args: argparse.Namespace) -> int:
    """Handle the loops subcommand."""
    import json

    from math_anything.categories.engine import CategoryEngine
    from math_anything.topology.classifier import LoopClassifier
    from math_anything.topology.loop_engine import LoopEngine

    # Phase 1–2 scaffold: only DFT-family engines have a demonstration graph.
    supported_engines = {"vasp", "qe", "cp2k"}
    if args.engine.lower() not in supported_engines:
        print(
            f"Error: engine '{args.engine}' is not yet supported by loops. "
            f"Supported engines: {sorted(supported_engines)}"
        )
        return 1

    schema = None
    if args.files:
        try:
            schema = _extract_schema(args.engine, args.files)
        except Exception as e:
            print(f"Error extracting schema: {e}")
            return 1

    try:
        # Build a default CategoryEngine for the engine/domain.
        ce = CategoryEngine()
        # NOTE: This is a demonstration scaffold. In future, populate from a
        # domain-specific morphism registry using args.engine and the schema.
        from math_anything.morphisms.approximations import (
            BornOppenheimerApproximation,
            KohnShamMapping,
            PlaneWaveTruncation,
        )

        ce.register_morphism(BornOppenheimerApproximation())
        ce.register_morphism(KohnShamMapping())
        ce.register_morphism(PlaneWaveTruncation(encut=520))
        ce.link("born_oppenheimer", "FullManyBody", "ElectronicSchrodinger")
        ce.link("kohn_sham", "ElectronicSchrodinger", "KohnSham_Full")
        ce.link("plane_wave_truncation", "KohnSham_Full", "KohnSham_Truncated")

        le = LoopEngine(ce)
        loops = le.find_loops()
        classifier = LoopClassifier()

        from math_anything.topology.curvature import compute_curvature_map

        loss_weights = {"born_oppenheimer": 0.0, "kohn_sham": 0.05, "plane_wave_truncation": 0.1}
        curvature_map = compute_curvature_map(loops, loss_weights)

        loops_data = [
            {
                "type": classifier.classify(loop).value,
                "nodes": list(loop.nodes),
                "edges": list(loop.edges),
                "directed": loop.is_directed,
                "canonical_form": loop.canonical_form,
                "curvature": curvature_map[loop.canonical_form],
            }
            for loop in loops
        ]

        report = {
            "engine": args.engine,
            "schema_present": schema is not None,
            "betti": le.betti_numbers(),
            "loops": loops_data,
            "curvature": curvature_map,
        }

        if args.visualize == "mermaid":
            from math_anything.topology.visualization import to_mermaid

            output = to_mermaid(ce, loops, curvature_map)
        elif args.visualize == "graphviz":
            from math_anything.topology.visualization import to_graphviz

            output = to_graphviz(ce, loops, curvature_map)
        else:
            output = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output:
            out_path = Path(args.output).resolve()
            if not out_path.is_relative_to(Path.cwd().resolve()):
                print("Error: --output must be inside the working directory")
                return 1
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output, encoding="utf-8")
        else:
            safe_print(output)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


DEFAULT_CUTOFF_EV = 520.0


def cmd_homotopy(args: argparse.Namespace) -> int:
    """Check homotopy between two DFT-family engine parameterizations."""
    import json

    from math_anything.categories.engine import CategoryEngine
    from math_anything.morphisms.approximations import (
        BornOppenheimerApproximation,
        KohnShamMapping,
        PlaneWaveTruncation,
    )
    from math_anything.topology.homotopy import are_paths_homotopic

    cutoff_a = args.param_a if args.param_a is not None else DEFAULT_CUTOFF_EV
    cutoff_b = args.param_b if args.param_b is not None else DEFAULT_CUTOFF_EV

    def build_engine(prefix: str, cutoff: float):
        ce = CategoryEngine()
        bo = BornOppenheimerApproximation()
        ks = KohnShamMapping()
        pw = PlaneWaveTruncation(encut=cutoff)
        for m in (bo, ks, pw):
            m.name = f"{prefix}_{m.name}"
            ce.register_morphism(m)
        ce.link(f"{prefix}_born_oppenheimer", "FullManyBody", "ElectronicSchrodinger")
        ce.link(f"{prefix}_kohn_sham", "ElectronicSchrodinger", "KohnSham_Full")
        ce.link(
            f"{prefix}_plane_wave_truncation",
            "KohnSham_Full",
            "KohnSham_Truncated",
        )
        path = [
            f"{prefix}_born_oppenheimer",
            f"{prefix}_kohn_sham",
            f"{prefix}_plane_wave_truncation",
        ]
        return ce, path

    try:
        ce_a, path_a = build_engine(args.engine_a, cutoff_a)
        ce_b, path_b = build_engine(args.engine_b, cutoff_b)

        # Compare paths in a single merged engine with namespaced morphisms.
        ce_merged = CategoryEngine()
        for ce in (ce_a, ce_b):
            for m in ce.morphisms.values():
                ce_merged.register_morphism(m)
            for link in ce.morphism_links:
                ce_merged.link(
                    link.morphism.name,
                    link.source_structure,
                    link.target_structure,
                )

        witness = are_paths_homotopic(ce_merged, path_a, path_b)

        report = {
            "engine_a": args.engine_a,
            "engine_b": args.engine_b,
            "cutoff_a_eV": cutoff_a,
            "cutoff_b_eV": cutoff_b,
            "witness": {
                "equivalent": witness.equivalent,
                "shared_invariants": witness.shared_invariants,
                "confidence": witness.confidence,
            },
        }

        output = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output:
            out_path = Path(args.output).resolve()
            if not out_path.is_relative_to(Path.cwd().resolve()):
                print("Error: --output must be inside the working directory")
                return 1
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output, encoding="utf-8")
        else:
            safe_print(output)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_ml(args: argparse.Namespace) -> int:
    """Analyze a supervised-learning model."""
    import json

    from math_anything.domains import DOMAIN_REGISTRY
    from math_anything.topology.visualization import to_mermaid

    try:
        domain = DOMAIN_REGISTRY["supervised_learning"](
            {
                "input_dim": args.input_dim,
                "output_dim": args.output_dim,
                "architecture": args.architecture,
                "loss": args.loss,
            }
        )
        analysis = domain.analyze()

        report = {
            "domain": analysis.domain_name,
            "architecture": args.architecture,
            "input_dim": args.input_dim,
            "output_dim": args.output_dim,
            "preserved": analysis.preserved,
            "lost": analysis.lost,
            "emerged": analysis.emerged,
            "morphism_chain": analysis.morphism_chain,
        }

        import numpy as np

        from math_anything.structures.surrogate_backend import SurrogateModel

        def _demo_with_backend(backend_name: str):
            model = SurrogateModel(
                backend=backend_name,
                input_dim=args.input_dim,
                output_dim=args.output_dim,
                hidden_dim=4,
            )
            dataset = [
                (np.array([x] * args.input_dim), np.array([2.0 * x + 1.0] * args.output_dim)) for x in [-1.0, 0.0, 1.0]
            ]
            model.fit(dataset, epochs=5, lr=0.05)
            return model.predict(np.array([0.5] * args.input_dim))

        backend_used = args.backend
        try:
            demo_pred = _demo_with_backend(args.backend)
            backend_available = True
        except ImportError:
            backend_used = "numpy"
            demo_pred = _demo_with_backend("numpy")
            backend_available = False

        report["backend_requested"] = args.backend
        report["backend_used"] = backend_used
        report["backend_available"] = backend_available
        report["surrogate_demo_prediction"] = demo_pred.tolist() if hasattr(demo_pred, "tolist") else demo_pred

        if args.compare_with:
            from math_anything.topology.cross_domain import cross_domain_homotopy

            witness = cross_domain_homotopy(
                args.compare_with,
                {"n_electrons": 2},
                "supervised_learning",
                {
                    "input_dim": args.input_dim,
                    "output_dim": args.output_dim,
                    "architecture": args.architecture,
                    "loss": args.loss,
                },
            )
            report["cross_domain_homotopy"] = {
                "equivalent": witness.equivalent,
                "shared_invariants": witness.shared_invariants,
                "confidence": witness.confidence,
            }

        if args.compare_paths:
            import numpy as np

            from math_anything.structures.neural_network import (
                ActivationMorphism,
                LinearMorphism,
                LossMorphism,
                SequentialNetwork,
            )
            from math_anything.topology.optimization_landscape import (
                training_paths_homotopic,
            )
            from math_anything.topology.training_curvature import train_and_capture

            loss_fn = LossMorphism(name="loss", loss=args.loss)
            dataset = [
                (np.array([x] * args.input_dim), np.array([2.0 * x + 1.0] * args.output_dim)) for x in [-1.0, 0.0, 1.0]
            ]

            def _make_network():
                return SequentialNetwork(
                    [
                        LinearMorphism(name="linear_1", input_dim=args.input_dim, output_dim=4),
                        ActivationMorphism(name="relu_1", activation="relu"),
                        LinearMorphism(name="linear_2", input_dim=4, output_dim=args.output_dim),
                    ]
                )

            result_a = train_and_capture(_make_network(), dataset, loss_fn, epochs=5, lr=0.05)
            result_b = train_and_capture(_make_network(), dataset, loss_fn, epochs=5, lr=0.05)

            witness = training_paths_homotopic(result_a, result_b)
            report["optimization_landscape_homotopy"] = {
                "equivalent": witness.equivalent,
                "shared_invariants": witness.shared_invariants,
                "confidence": witness.confidence,
            }

        if args.transfer:
            import numpy as np

            from math_anything.structures.functor import (
                MatrixFunctor,
                NaturalTransformation,
                is_natural_transformation,
            )
            from math_anything.structures.neural_network import (
                ActivationMorphism,
                LinearMorphism,
                LossMorphism,
                SequentialNetwork,
            )
            from math_anything.structures.transfer import (
                WeightSpaceTransfer,
                flatten_network_weights,
                transfer_learn,
            )

            loss_fn = LossMorphism(name="loss", loss=args.loss)
            dataset = [
                (np.array([x] * args.input_dim), np.array([2.0 * x + 1.0] * args.output_dim)) for x in [-1.0, 0.0, 1.0]
            ]

            def _make_network():
                return SequentialNetwork(
                    [
                        LinearMorphism(name="linear_1", input_dim=args.input_dim, output_dim=4),
                        ActivationMorphism(name="relu_1", activation="relu"),
                        LinearMorphism(name="linear_2", input_dim=4, output_dim=args.output_dim),
                    ]
                )

            source = _make_network()
            target = _make_network()
            source_dim = len(flatten_network_weights(source))
            adapter = WeightSpaceTransfer(source_dim, source_dim).matrix

            result = transfer_learn(source, target, dataset, loss_fn, adapter, epochs=3, lr=0.05)

            # Natural-transformation check: identity adapter + identical functors should commute.
            dim = source_dim
            F = MatrixFunctor(np.eye(dim))
            G = MatrixFunctor(np.eye(dim))
            eta = NaturalTransformation({dim: np.eye(dim)})
            sample_morphism = np.eye(dim)
            valid, reason = is_natural_transformation(F, G, eta, test_morphisms=[(dim, dim, sample_morphism)])

            report["transfer_learning"] = {
                "natural_transformation_valid": valid,
                "natural_transformation_reason": reason,
                "final_loss": result.final_loss,
                "epochs": 3,
            }

        if args.visualize == "mermaid":
            from math_anything.categories.engine import CategoryEngine

            ce = CategoryEngine()
            for step in analysis.morphism_chain:
                ce.register_morphism(
                    type(
                        "M",
                        (),
                        {
                            "name": step["name"],
                            "source_type": "MLState",
                            "target_type": "MLState",
                        },
                    )()
                )
            prev = "Input"
            for step in analysis.morphism_chain:
                ce.link(step["name"], prev, step["name"])
                prev = step["name"]
            output = to_mermaid(ce)
        else:
            output = json.dumps(report, indent=2, ensure_ascii=False)

        safe_print(output)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_validate(args):
    """Validate schema constraints."""
    print(f"Validating {args.schema_file}...")

    try:
        schema = MathSchema.load(args.schema_file)

        print("\n" + "=" * 60)
        print("Symbolic Constraint Validation")
        print("=" * 60)

        if hasattr(schema, "symbolic_constraints") and schema.symbolic_constraints:
            passed = 0
            failed = 0

            for constraint in schema.symbolic_constraints:
                is_satisfied = "SATISFIED" in constraint.description
                symbol = "✓" if is_satisfied else "✗"

                print(f"  {symbol} {constraint.expression:40}", end="")
                if constraint.description:
                    desc = constraint.description.replace(" [SATISFIED]", "").replace(" [VIOLATED]", "")
                    print(f" | {desc[:30]}")
                else:
                    print()

                if is_satisfied:
                    passed += 1
                else:
                    failed += 1

            print(f"\n  Total: {passed} passed, {failed} failed")
            return 0 if failed == 0 else 1
        else:
            print("  No symbolic constraints found in schema")
            return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def _print_pretty_schema(schema: MathSchema):
    """Print schema in human-readable format."""
    print("\n" + "=" * 60)
    print("Mathematical Schema")
    print("=" * 60)

    if schema.mathematical_model:
        if schema.mathematical_model.governing_equations:
            print("\n📐 Governing Equations:")
            for eq in schema.mathematical_model.governing_equations:
                print(f"  [{eq.type}] {eq.name}")
                print(f"    {eq.mathematical_form}")

        if schema.mathematical_model.boundary_conditions:
            print(f"\n📏 Boundary Conditions: {len(schema.mathematical_model.boundary_conditions)}")

    if hasattr(schema, "symbolic_constraints") and schema.symbolic_constraints:
        print(f"\n✓ Symbolic Constraints: {len(schema.symbolic_constraints)}")


def _print_pretty_diff(diff, file1: str, file2: str):
    """Print diff in human-readable format."""
    print("\n" + "=" * 60)
    print(f"Comparison: {file1} vs {file2}")
    print("=" * 60)

    similarity = diff.similarity_score * 100
    print(f"\nSimilarity: {similarity:.1f}%")

    if diff.common_equations:
        print("\n✓ Common Equations:")
        for eq in diff.common_equations:
            print(f"  - {eq}")

    if diff.unique_to_first:
        print(f"\n📌 Unique to {file1}:")
        for item in diff.unique_to_first:
            print(f"  - {item}")

    if diff.unique_to_second:
        print(f"\n📌 Unique to {file2}:")
        for item in diff.unique_to_second:
            print(f"  - {item}")

    if diff.parameter_mapping:
        print("\n🔄 Parameter Mappings:")
        for m in diff.parameter_mapping[:10]:
            conf = m.get("confidence", "medium")
            print(f"  {m['first']} <-> {m['second']} ({m.get('meaning', 'unknown')}) [{conf}]")

    print(f"\n💡 {diff.analysis_summary}")


def main():
    """Main entry point."""
    # Determine program name from how it was invoked
    prog_name = os.path.basename(sys.argv[0]) if sys.argv else "math-anything"
    parser = create_parser(prog_name=prog_name)
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Resolve the loops engine argument.  The --engine flag wins over a
    # positional engine for backward compatibility.
    if args.command == "loops":
        if args.engine is None and args.engine_positional is None:
            parser.error("the following arguments are required: engine or --engine")
        if args.engine is None:
            args.engine = args.engine_positional
        elif args.engine_positional is not None:
            args.files.insert(0, args.engine_positional)

    # Dispatch to command handler
    commands = {
        "repl": cmd_repl,
        "extract": cmd_extract,
        "explain": cmd_explain,
        "draft": cmd_draft,
        "check": cmd_check,
        "diff": cmd_diff,
        "cross": cmd_cross,
        "loops": cmd_loops,
        "homotopy": cmd_homotopy,
        "ml": cmd_ml,
        "validate": cmd_validate,
        "config": cmd_config,
        "watch": cmd_watch,
        "agent": cmd_agent,
        "generate": cmd_generate,
        "rag": cmd_rag,
        "visualize": cmd_visualize,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
