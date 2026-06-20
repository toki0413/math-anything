"""Watch mode implementation.

Monitors input files for changes and auto-runs check/explain.
"""

import os
import time
from pathlib import Path

from ..check import check_schema
from ..insight import explain_schema
from ..utils.terminal import safe_print


def cmd_watch(args):
    """Watch files for changes."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        print("Error: watchdog not installed. Run: pip install watchdog")
        return 1

    engine = args.engine
    files = args.files
    auto_check = args.auto_check
    auto_explain = args.auto_explain

    # Resolve absolute paths
    watch_paths = [str(Path(f).resolve()) for f in files]
    last_modified = {p: os.path.getmtime(p) for p in watch_paths}

    print(f"Watching {len(watch_paths)} file(s) for engine: {engine}")
    print("Press Ctrl+C to stop\n")

    def run_analysis(filepath):
        """Run check and optionally explain."""
        print(f"\n[CHANGE DETECTED] {filepath}")
        print("=" * 60)

        try:
            schema = _extract_schema(engine, [filepath])
        except Exception as e:
            print(f"Extraction failed: {e}")
            return

        if auto_check:
            try:
                exit_code, report = check_schema(schema, engine)
                safe_print(report)
            except Exception as e:
                print(f"Check failed: {e}")

        if auto_explain:
            try:
                output = explain_schema(schema, engine, fmt="text")
                safe_print(output)
            except Exception as e:
                print(f"Explain failed: {e}")

        print("=" * 60)
        print("Watching... (Ctrl+C to stop)\n")

    # Initial run
    if auto_check:
        for p in watch_paths:
            run_analysis(p)

    # Polling loop (watchdog optional but we use simple polling for reliability)
    try:
        while True:
            time.sleep(1)
            for p in watch_paths:
                if not os.path.exists(p):
                    continue
                mtime = os.path.getmtime(p)
                if mtime > last_modified.get(p, 0):
                    last_modified[p] = mtime
                    run_analysis(p)
    except KeyboardInterrupt:
        print("\nStopped watching.")
        return 0

    return 0


def _extract_schema(engine, files):
    """Extract schema for watch mode."""
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

        extractor = LammpsExtractor()
        return extractor.extract({"input": files[0]})

    elif engine == "abaqus":
        from abaqus.core.extractor import AbaqusExtractor

        extractor = AbaqusExtractor()
        return extractor.extract({"input": files[0]})

    elif engine == "quantum_espresso":
        from qe.core.extractor import QuantumEspressoExtractor

        extractor = QuantumEspressoExtractor()
        return extractor.extract({"input": files[0]})

    elif engine == "ansys":
        from ansys.core.input_extractor import AnsysInputExtractor

        extractor = AnsysInputExtractor()
        return extractor.extract({"input": files[0]})

    elif engine == "comsol":
        from comsol.core.extractor import ComsolExtractor

        extractor = ComsolExtractor()
        return extractor.extract({"input": files[0]})

    raise ValueError(f"Unsupported engine: {engine}")
