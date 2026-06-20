#!/usr/bin/env python3
"""Bourbaki VS Code extension — JSON-RPC language server over stdio.

The server exposes Bourbaki analysis capabilities through a minimal
JSON-RPC protocol so that the TypeScript client can request structure
analysis, equation verification, numerical solutions, and diagnostics
without depending on MCP.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Allow running against the in-repo package during development.
LS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = LS_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_anything.adapters import translate_params, list_supported_engines
from math_anything.domains import get_domain
from math_anything.dimensional import BuckinghamPiEngine, PhysicalQuantity
from math_anything.psrn.pse_engine import PSEEngine, PSEConfig
from math_anything.structures.conservation_field import ConservationMatrixField
from math_anything.structures.evolution import SymplecticIntegrator, ConservationLawSolver
from math_anything.structures.equilibrium import VariationalSolver
from math_anything.structures.spectral import EigenvalueSolver, SelfConsistentSolver
from math_anything.type_theory.verify import VerificationPipeline


class BourbakiLanguageServer:
    """Minimal JSON-RPC server over stdin/stdout."""

    def __init__(self) -> None:
        self._counter = 0
        self._running = True

    def _send(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        sys.stdout.write(f"Content-Length: {len(data.encode('utf-8'))}\r\n\r\n{data}")
        sys.stdout.flush()

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _reply(self, id_: int, result: Any) -> None:
        self._send({"jsonrpc": "2.0", "id": id_, "result": result})

    def _error(self, id_: int, code: int, message: str, data: Any = None) -> None:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self._send({"jsonrpc": "2.0", "id": id_, "error": err})

    def run(self) -> None:
        while self._running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                if not line.startswith("Content-Length:"):
                    continue
                length = int(line.split(":")[1].strip())
                # blank line
                sys.stdin.read(2)
                body = sys.stdin.read(length)
                self._handle(json.loads(body))
            except Exception:
                traceback.print_exc(file=sys.stderr)

    def _handle(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        id_ = msg.get("id")
        params = msg.get("params", {}) or {}

        try:
            if method == "initialize":
                self._reply(id_, self._initialize(params))
            elif method == "initialized":
                pass
            elif method == "shutdown":
                self._running = False
                self._reply(id_, None)
            elif method == "exit":
                self._running = False
            elif method == "textDocument/didOpen":
                self._on_open(params)
            elif method == "textDocument/didChange":
                self._on_change(params)
            elif method == "textDocument/didSave":
                self._on_save(params)
            elif method == "bourbaki/analyzeFile":
                self._reply(id_, self._analyze_file(params))
            elif method == "bourbaki/analyzeWorkspace":
                self._reply(id_, self._analyze_workspace(params))
            elif method == "bourbaki/verifyEquation":
                self._reply(id_, self._verify_equation(params))
            elif method == "bourbaki/solveNumerical":
                self._reply(id_, self._solve_numerical(params))
            elif method == "bourbaki/translateEngine":
                self._reply(id_, self._translate_engine(params))
            elif method == "bourbaki/dimensionalAnalyze":
                self._reply(id_, self._dimensional_analyze(params))
            elif method == "bourbaki/discoverEquations":
                self._reply(id_, self._discover_equations(params))
            elif method == "bourbaki/listDomains":
                self._reply(id_, self._list_domains(params))
            else:
                if id_ is not None:
                    self._error(id_, -32601, f"Method not found: {method}")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            if id_ is not None:
                self._error(id_, -32603, str(exc), data=traceback.format_exc())

    # ------------------------------------------------------------------
    # LSP lifecycle
    # ------------------------------------------------------------------
    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "capabilities": {
                "textDocumentSync": {"openClose": True, "change": 1, "save": True},
                "diagnosticProvider": {"interFileDependencies": False, "workspaceDiagnostics": False},
            },
            "serverInfo": {"name": "bourbaki-lsp", "version": "0.1.0"},
        }

    def _on_open(self, params: dict[str, Any]) -> None:
        text_doc = params.get("textDocument", {})
        self._publish_diagnostics(text_doc.get("uri"), text_doc.get("text", ""))

    def _on_change(self, params: dict[str, Any]) -> None:
        text_doc = params.get("textDocument", {})
        content = params.get("contentChanges", [{}])[0].get("text", "")
        self._publish_diagnostics(text_doc.get("uri"), content)

    def _on_save(self, params: dict[str, Any]) -> None:
        text_doc = params.get("textDocument", {})
        # The server keeps no state; diagnostics are refreshed on change.
        self._publish_diagnostics(text_doc.get("uri"), "")

    def _publish_diagnostics(self, uri: str, content: str) -> None:
        if not uri:
            return
        diagnostics = self._compute_diagnostics(uri, content)
        self._notify("textDocument/publishDiagnostics", {"uri": uri, "diagnostics": diagnostics})

    # ------------------------------------------------------------------
    # Bourbaki capabilities
    # ------------------------------------------------------------------
    def _analyze_file(self, params: dict[str, Any]) -> dict[str, Any]:
        file_path = params.get("filePath", "")
        engine = params.get("engine", "")
        content = params.get("content", "")
        if not content and file_path and os.path.exists(file_path):
            content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        if not engine:
            engine = self._infer_engine(file_path)
        try:
            translated = translate_params(engine, self._parse_params(engine, content))
            domain_name = translated.get("domain", "generic")
            domain_cls = get_domain(domain_name)
            domain = domain_cls(translated.get("parameters", {}))
            conservation = domain.build_conservation_field()
            morphism_chain = domain.build_morphism_chain()
            analysis = domain.analyze()
            return {
                "engine": engine,
                "domain": domain_name,
                "translatedParameters": translated,
                "conservationField": conservation,
                "morphismChain": morphism_chain,
                "analysis": {
                    "domainName": analysis.domain_name,
                    "preserved": analysis.preserved,
                    "weakened": analysis.weakened,
                    "lost": analysis.lost,
                    "emerged": analysis.emerged,
                },
            }
        except Exception as exc:
            return {"error": str(exc), "traceback": traceback.format_exc()}

    def _analyze_workspace(self, params: dict[str, Any]) -> dict[str, Any]:
        root = params.get("rootPath", "")
        results = []
        if root and os.path.isdir(root):
            for engine in list_supported_engines():
                files = self._find_engine_files(root, engine)
                if files:
                    results.append({"engine": engine, "files": files})
        return {"engines": results}

    def _verify_equation(self, params: dict[str, Any]) -> dict[str, Any]:
        statement = params.get("statement", "")
        proof = params.get("proof", "")
        assumptions = params.get("assumptions", [])
        goals = params.get("goals", [])
        pipeline = VerificationPipeline()
        report = pipeline.verify(statement, proof_text=proof, assumptions=assumptions, goals=goals)
        return {
            "passed": report.passed,
            "confidence": report.confidence,
            "layers": [
                {
                    "layer": layer.layer.name,
                    "passed": layer.passed,
                    "message": layer.message,
                    "confidence": layer.confidence,
                    "warnings": layer.warnings,
                    "details": layer.details,
                }
                for layer in report.layers
            ],
        }

    def _solve_numerical(self, params: dict[str, Any]) -> dict[str, Any]:
        solver_type = params.get("solverType", "")
        try:
            if solver_type == "symplectic":
                si = SymplecticIntegrator()
                result = si.integrate(
                    q0=params["q0"],
                    p0=params["p0"],
                    dt=params["dt"],
                    n_steps=params["nSteps"],
                    mass=params.get("mass", 1.0),
                )
                return {"solver": solver_type, "result": result}
            if solver_type == "variational":
                vs = VariationalSolver()
                result = vs.solve_poisson(
                    source_fn=params["sourceFn"],
                    n_elements=params["nElements"],
                    domain=params.get("domain", [0.0, 1.0]),
                )
                return {"solver": solver_type, "result": result}
            if solver_type == "eigenvalue":
                es = EigenvalueSolver()
                result = es.solve(matrix=params["matrix"])
                return {"solver": solver_type, "result": result}
            if solver_type == "scf":
                scf = SelfConsistentSolver(
                    n_states=params["nStates"],
                    max_iter=params.get("maxIter", 100),
                    tol=params.get("tol", 1e-6),
                    mixing=params.get("mixing", 0.5),
                )
                result = scf.solve(hamiltonian_builder=lambda rho: params["hamiltonian"])
                return {"solver": solver_type, "result": result}
            if solver_type == "conservation":
                cls = ConservationLawSolver()
                result = cls.lax_friedrichs_step(
                    U=params["U"],
                    F=params["F"],
                    dt=params["dt"],
                    dx=params["dx"],
                )
                return {"solver": solver_type, "result": result}
            return {"error": f"Unknown solver type: {solver_type}"}
        except Exception as exc:
            return {"error": str(exc), "traceback": traceback.format_exc()}

    def _translate_engine(self, params: dict[str, Any]) -> dict[str, Any]:
        engine = params.get("engine", "")
        parameters = params.get("parameters", {})
        return translate_params(engine, parameters)

    def _dimensional_analyze(self, params: dict[str, Any]) -> dict[str, Any]:
        quantities = params.get("quantities", [])
        try:
            engine = BuckinghamPiEngine()
            physical = [
                PhysicalQuantity(
                    name=q.get("name", ""),
                    symbol=q.get("symbol", ""),
                    dimensions=q.get("dimensions", {}),
                )
                for q in quantities
            ]
            pi_groups = engine.compute(physical)
            return {"pi_groups": [g.to_dict() for g in pi_groups]}
        except Exception as exc:
            return {"error": str(exc)}

    def _discover_equations(self, params: dict[str, Any]) -> dict[str, Any]:
        data = params.get("data", [])
        variables = params.get("variables", [])
        if not data or not variables:
            return {"error": "data and variables are required"}
        try:
            import numpy as np
            data_arr = np.asarray(data)
            if data_arr.ndim == 1:
                data_arr = data_arr.reshape(-1, 1)
            X = data_arr[:, :-1]
            y = data_arr[:, -1]
            cfg = PSEConfig()
            engine = PSEEngine(cfg)
            best_expr, pareto = engine.discover(X, y, variable_names=variables)
            return {"best_expression": best_expr, "pareto_front": pareto}
        except Exception as exc:
            return {"error": str(exc)}

    def _list_domains(self, params: dict[str, Any]) -> dict[str, Any]:
        from math_anything.domains import list_domains
        return {"domains": list_domains()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _infer_engine(self, file_path: str) -> str:
        name = os.path.basename(file_path).lower()
        if name in {"incar", "poscar", "kpoints"}:
            return "vasp"
        ext = Path(file_path).suffix.lower()
        stem = Path(file_path).stem.lower()
        mapping = {
            ".in": "qe",
            ".lmp": "lammps",
            ".inp": "abaqus",
            ".top": "gromacs",
            ".itp": "gromacs",
        }
        if stem in {"controldict", "fvschemes", "fvsolution", "transportproperties"}:
            return "openfoam"
        return mapping.get(ext, "generic")

    def _parse_params(self, engine: str, content: str) -> dict[str, Any]:
        """Best-effort parameter extraction from file content."""
        params: dict[str, Any] = {}
        if engine == "vasp":
            for line in content.splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    key, _, val = line.partition("=")
                    params[key.strip()] = val.strip()
        elif engine == "qe":
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2 and not parts[0].startswith("&") and parts[0] not in {"!", "#"}:
                    params[parts[0]] = " ".join(parts[1:])
        elif engine == "lammps":
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0].lower() in {"variable", "fix", "compute"}:
                    params[parts[1]] = " ".join(parts[2:])
                elif len(parts) >= 2 and parts[0].lower() in {"timestep", "pair_style", "units"}:
                    params[parts[0]] = parts[1]
        elif engine == "abaqus":
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("*") and not line.startswith("**"):
                    tokens = line[1:].split(",")
                    params[tokens[0].strip()] = [t.strip() for t in tokens[1:]]
        elif engine == "gromacs":
            current_section = None
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1].strip()
                    params[current_section] = params.get(current_section, [])
                elif current_section:
                    params[current_section].append(line)
        elif engine == "openfoam":
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                if line.startswith("{") or line.startswith("}"):
                    continue
                if " " in line:
                    key, _, val = line.partition(" ")
                    params[key.strip()] = val.strip().rstrip(";")
        return params

    def _find_engine_files(self, root: str, engine: str) -> list[str]:
        patterns = {
            "vasp": ["INCAR", "POSCAR", "KPOINTS"],
            "qe": ["*.in"],
            "lammps": ["*.lmp"],
            "abaqus": ["*.inp"],
            "openfoam": ["system/controlDict", "system/fvSchemes", "system/fvSolution", "constant/transportProperties"],
            "gromacs": ["*.top", "*.itp"],
        }
        found: list[str] = []
        for pat in patterns.get(engine, []):
            for p in Path(root).rglob(pat):
                if p.is_file():
                    found.append(str(p.relative_to(root)))
        return found

    def _compute_diagnostics(self, uri: str, content: str) -> list[dict[str, Any]]:
        diagnostics: list[dict[str, Any]] = []
        if not content:
            return diagnostics
        engine = self._infer_engine(uri)
        try:
            translated = translate_params(engine, self._parse_params(engine, content))
            required = translated.get("required_parameters", [])
            provided = set(translated.get("parameters", {}).keys())
            for req in required:
                if req not in provided:
                    diagnostics.append({
                        "severity": 2,
                        "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}},
                        "message": f"Missing recommended parameter '{req}' for {translated.get('domain', engine)}",
                        "source": "bourbaki",
                    })
        except Exception:
            pass
        return diagnostics


def main() -> None:
    BourbakiLanguageServer().run()


if __name__ == "__main__":
    main()
