"""Math Anything API Server - HTTP interface for mathematical structure extraction.

Start:
    python -m math_anything.server
    # or
    uvicorn math_anything.server:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /extract/{engine}              - Extract from parameters
    POST /extract/{engine}/file         - Extract from uploaded file
    POST /extract/{engine}/emergence    - Extract + emergence analysis from file
    POST /emergence/{engine}            - Phase transition & emergence analysis
    POST /geometry/{engine}             - Extract geometric structure
    POST /verify                        - Formal verification
    POST /proposition                   - Generate mathematical propositions
    POST /config                        - Configure API (LLM provider, key, model)
    GET  /config/status                 - Get current API configuration status
    GET  /engines                       - List supported engines
    GET  /flywheel/stats                - Flywheel statistics
    GET  /health                        - Health check
"""

import json
import tempfile
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from math_anything import (
    DataFlywheel,
    DifferentialGeometryLayer,
    FormalVerifier,
    MathAnything,
)
from math_anything.emergence import EmergenceLayer
from math_anything.proposition import MathematicalTask, PropositionGenerator, TaskType

app = FastAPI(
    title="Math Anything API",
    description="Mathematical structure extraction for computational materials",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ma = MathAnything()
_geo = DifferentialGeometryLayer()
_em = EmergenceLayer()
_fv = FormalVerifier()
_pg = PropositionGenerator()
_fw = DataFlywheel()


class ExtractRequest(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict)
    lattice_vectors: Optional[Dict[str, List[float]]] = None
    space_group: Optional[str] = None


class VerifyRequest(BaseModel):
    task_id: str = ""
    task_type: str = "proof"
    task_name: str = ""
    statement: str = ""
    assumptions: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    proof_text: str = ""
    engine: str = ""
    with_geometry: bool = False


class PropositionRequest(BaseModel):
    engine: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)


class ApiConfigRequest(BaseModel):
    provider: str = "openai"
    api_key: str = ""
    base_url: str = ""
    model: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/engines")
def list_engines():
    from math_anything import list_engines as _list
    return {"engines": _list()}


@app.post("/extract/{engine}")
def extract(engine: str, req: ExtractRequest):
    try:
        result = _ma.extract(engine, req.params)
        _fw.record("extract", {
            "engine": engine,
            "params": req.params,
            "success": result.success,
        }, success=result.success)

        return {
            "engine": engine,
            "schema": result.schema,
            "success": result.success,
            "errors": result.errors,
            "warnings": result.warnings,
        }
    except Exception as e:
        _fw.record("extract", {"engine": engine, "params": req.params},
                   success=False)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/extract/{engine}/file")
async def extract_file(engine: str, file: UploadFile = File(...)):
    content = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}")
    try:
        tmp.write(content)
        tmp.close()
        result = _ma.extract_file(engine, tmp.name)
        _fw.record("extract", {
            "engine": engine,
            "file": file.filename,
            "success": result.success,
        }, success=result.success)

        return {
            "engine": engine,
            "file": file.filename,
            "schema": result.schema,
            "success": result.success,
            "errors": result.errors,
            "warnings": result.warnings,
        }
    except Exception as e:
        _fw.record("extract", {"engine": engine, "file": file.filename},
                   success=False)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp.name)


@app.post("/geometry/{engine}")
def extract_geometry(engine: str, req: ExtractRequest):
    try:
        structure = _geo.extract(
            engine, req.params,
            lattice_vectors=req.lattice_vectors,
            space_group=req.space_group,
        )
        return structure.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/verify")
def verify(req: VerifyRequest):
    task = MathematicalTask(
        id=req.task_id or "api-verify",
        type=TaskType(req.task_type) if req.task_type in [t.value for t in TaskType] else TaskType.PROOF,
        name=req.task_name,
        statement=req.statement,
        assumptions=req.assumptions,
        goals=req.goals,
    )

    geo_ctx = None
    if req.with_geometry and req.engine:
        try:
            geo_ctx = _geo.extract(req.engine, {})
        except Exception:
            pass

    result = _fv.verify(task, req.proof_text, geometric_context=geo_ctx)

    _fw.record("verify", {
        "task_id": task.id,
        "engine": req.engine,
        "formal_status": result.formal_status.value,
        "confidence": result.overall_confidence,
    }, success=result.formal_status.value in ("verified", "inconclusive"))

    return result.to_dict()


@app.post("/proposition")
def generate_propositions(req: PropositionRequest):
    try:
        result = _ma.extract(req.engine, req.params)
        propositions = _pg.translate(result.schema)
        return {
            "engine": req.engine,
            "core_problem": propositions.core_problem,
            "proof_tasks": [t.to_dict() for t in propositions.proof_tasks],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/emergence/{engine}")
def analyze_emergence(engine: str, req: ExtractRequest):
    try:
        result = _ma.extract(engine, req.params)
        emergence = _em.extract(
            engine, req.params,
            schema=result.schema if result.success else None,
        )
        return emergence.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/extract/{engine}/emergence")
async def extract_with_emergence(engine: str, file: UploadFile = File(...)):
    content = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}")
    try:
        tmp.write(content)
        tmp.close()
        result = _ma.extract_file(engine, tmp.name)
        if not result.success:
            raise HTTPException(status_code=400, detail=result.errors[0] if result.errors else "extraction failed")

        emergence = _em.extract(
            engine, result.files.get("params", {}),
            schema=result.schema,
        )
        return {
            "engine": engine,
            "file": file.filename,
            "schema": result.schema,
            "success": True,
            "emergence": emergence.to_dict(),
            "warnings": result.warnings + emergence.warnings,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp.name)


@app.get("/flywheel/stats")
def flywheel_stats():
    stats = _fw.get_stats()
    degraded = {}
    for engine in stats.engine_performance:
        degraded[engine] = _fw.degradation.is_degraded(engine)
    return {
        "stats": stats.to_dict(),
        "degraded_engines": degraded,
    }


@app.post("/analyze/{engine}/log")
async def analyze_log(engine: str, file: UploadFile = File(...)):
    content = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_log_{file.filename}")
    try:
        tmp.write(content)
        tmp.close()
        import re
        traj: Dict[str, Any] = {"steps": [], "columns": [], "n_rows": 0}
        with open(tmp.name) as f:
            lines = f.readlines()
        header_found = False
        col_names = []
        for line in lines[:200]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "Step" in line and not header_found:
                col_names = line.split()
                header_found = True
                traj["columns"] = col_names
                continue
            if header_found:
                parts = line.split()
                if len(parts) == len(col_names):
                    try:
                        values = [float(p) for p in parts]
                        traj["steps"].append(dict(zip(col_names, values)))
                    except (ValueError, IndexError):
                        continue
        traj["n_rows"] = len(traj["steps"])
        if traj["steps"]:
            for c in col_names:
                key = c.lower()
                if key in ("temp",):
                    temps = [s[c] for s in traj["steps"]]
                    traj["temperature"] = {
                        "mean": round(sum(temps) / len(temps), 4),
                        "drift_pct": round((temps[-1] - temps[0]) / abs(temps[0]) * 100, 2) if abs(temps[0]) > 0.01 else 0,
                    }
                    traj["equilibrated"] = abs(traj["temperature"]["drift_pct"]) < 2.0
                if key in ("toteng", "etotal", "energy"):
                    energies = [s[c] for s in traj["steps"]]
                    mean_e = sum(energies) / len(energies)
                    var_e = sum((e - mean_e) ** 2 for e in energies) / len(energies)
                    t_mean = traj.get("temperature", {}).get("mean", 1)
                    if abs(t_mean) > 0.01:
                        traj["heat_capacity_estimate"] = round(var_e / t_mean ** 2, 6)
        return traj
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp.name)


@app.post("/config")
def configure_api(req: ApiConfigRequest):
    global _fv
    api_config = {
        "provider": req.provider,
        "api_key": req.api_key,
        "base_url": req.base_url,
        "model": req.model,
    }
    _fv = FormalVerifier(api_config=api_config)
    return {
        "status": "configured",
        "provider": req.provider,
        "model": req.model or "(default)",
        "has_key": bool(req.api_key),
    }


@app.get("/config/status")
def config_status():
    llm_config = getattr(_fv.llm_semantic, 'api_config', {})
    return {
        "provider": llm_config.get("provider", ""),
        "model": llm_config.get("model", ""),
        "base_url": llm_config.get("base_url", ""),
        "has_key": bool(llm_config.get("api_key", "")),
        "llm_enabled": bool(llm_config.get("api_key", "")),
    }


class CrossValRequest(BaseModel):
    methods: List[str]
    conclusions: List[str]


class PredictionRequest(BaseModel):
    predictions: List[Dict[str, str]]


class DualPerspectiveRequest(BaseModel):
    conclusion: str
    geometric_checks: List[str]
    analytic_checks: List[str]


@app.post("/validate/crossval")
async def validate_crossval(req: CrossValRequest):
    try:
        from .validation_toolkit import CrossValidationMatrix, ValidationStatus
    except ImportError:
        from math_anything.validation_toolkit import CrossValidationMatrix, ValidationStatus
    matrix = CrossValidationMatrix(methods=req.methods, conclusions=req.conclusions)
    return {"matrix": matrix.to_dict(), "report": matrix.report()}


@app.post("/validate/predictions")
async def validate_predictions(req: PredictionRequest):
    try:
        from .validation_toolkit import FalsifiablePredictionTable, PredictionStatus
    except ImportError:
        from math_anything.validation_toolkit import FalsifiablePredictionTable, PredictionStatus
    table = FalsifiablePredictionTable()
    for p in req.predictions:
        table.add(
            prediction_id=p.get("id", "?"),
            statement=p.get("statement", ""),
            mathematical_condition=p.get("condition", ""),
            test_method=p.get("method", ""),
        )
    return {"table": table.to_dict(), "report": table.report()}


@app.post("/validate/dual")
async def validate_dual(req: DualPerspectiveRequest):
    try:
        from .validation_toolkit import DualPerspectiveAnalyzer
    except ImportError:
        from math_anything.validation_toolkit import DualPerspectiveAnalyzer
    analyzer = DualPerspectiveAnalyzer(conclusion=req.conclusion)
    analyzer.set_geometric_checklist(req.geometric_checks)
    analyzer.set_analytic_checklist(req.analytic_checks)
    result = analyzer.evaluate()
    return {
        "result": {
            "conclusion": result.conclusion,
            "geometric_verdict": result.geometric_verdict,
            "analytic_verdict": result.analytic_verdict,
            "agreement": result.agreement,
        },
        "report": analyzer.report(),
    }


@app.get("/lean4/status")
async def lean4_status():
    try:
        try:
            from .lean4_bridge import Lean4Bridge
        except ImportError:
            from math_anything.lean4_bridge import Lean4Bridge
        bridge = Lean4Bridge()
        return {"available": bridge.is_available()}
    except Exception:
        return {"available": False}


@app.get("/advisor/{engine}")
def get_advisory(engine: str):
    try:
        from .advisor import MathAdvisor
    except ImportError:
        from math_anything.advisor import MathAdvisor
    advisor = MathAdvisor()
    return advisor.advise(engine)


_firewall = None

def _get_firewall():
    global _firewall
    if _firewall is None:
        try:
            from .firewall import DataFirewall
        except ImportError:
            from math_anything.firewall import DataFirewall
        _firewall = DataFirewall()
    return _firewall

@app.get("/firewall/status")
def firewall_status():
    fw = _get_firewall()
    return {"enabled": fw.is_enabled}

@app.post("/firewall/toggle")
def firewall_toggle():
    fw = _get_firewall()
    fw.toggle()
    return {"enabled": fw.is_enabled}


@app.get("/tools")
def list_tools():
    registry = _get_tool_registry()
    return {"tools": registry.list_tools()}


@app.websocket("/ws/agent")
async def agent_ws(websocket: WebSocket):
    await websocket.accept()
    registry = _get_tool_registry()
    loop = _make_agent_loop(registry)

    async def on_event(event):
        try:
            await websocket.send_json(event.to_dict())
        except Exception:
            pass

    try:
        while True:
            raw = await websocket.receive_json()
            msg_type = raw.get("type", "")

            if msg_type == "user_input":
                content = raw.get("content", "")
                context = raw.get("context", {})
                await loop.run(content, context, on_event)
            elif msg_type == "cancel":
                loop.cancel()
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


def _get_tool_registry():
    from .tool_registry import build_default_registry
    return build_default_registry()


class SymmetryRequest(BaseModel):
    lattice: Optional[List[List[float]]] = None
    positions: Optional[List[List[float]]] = None
    numbers: Optional[List[int]] = None
    space_group_hint: Optional[str] = None


class SpectralRequest(BaseModel):
    eigenvalues: List[float] = Field(default_factory=list)
    k_grid: Optional[List[List[float]]] = None
    weights: Optional[List[float]] = None
    occupied_bands: int = 0
    time_reversal: bool = True


class DynamicsRequest(BaseModel):
    time_series: List[float] = Field(default_factory=list)
    min_tsep: int = 10


class TDARequest(BaseModel):
    point_cloud: Optional[List[List[float]]] = None
    volume_data: Optional[Any] = None
    data_type: str = "point_cloud"
    max_dim: int = 2


class SINDyRequest(BaseModel):
    time_series: List[List[float]] = Field(default_factory=list)
    dt: float = 1.0
    poly_order: int = 3
    threshold: float = 0.1
    variable_names: Optional[List[str]] = None


class SandboxRequest(BaseModel):
    code: str = ""
    timeout_seconds: float = 10.0
    backend: str = "auto"


class GeometryVizRequest(BaseModel):
    viz_type: str = "manifold"
    metric_tensor: Optional[List[List[float]]] = None
    coord_range: Optional[List[float]] = None
    n_points: Optional[int] = None
    lattice_vectors: Optional[List[List[float]]] = None


@app.post("/analyze/symmetry")
def analyze_symmetry(req: SymmetryRequest):
    from .tools.symmetry import SymmetryAnalyzer
    analyzer = SymmetryAnalyzer()
    result = analyzer.analyze_structure(
        lattice=req.lattice,
        positions=req.positions,
        numbers=req.numbers,
        space_group_hint=req.space_group_hint,
    )
    return result.to_dict()


@app.post("/analyze/spectral")
def analyze_spectral(req: SpectralRequest):
    from .tools.spectral import SpectralAnalyzer
    import numpy as np
    analyzer = SpectralAnalyzer()
    eigs = np.array(req.eigenvalues)
    k_grid = np.array(req.k_grid) if req.k_grid else None
    weights = np.array(req.weights) if req.weights else None
    result = analyzer.analyze(eigs, k_grid, weights, req.occupied_bands, req.time_reversal)
    return result.to_dict()


@app.post("/analyze/dynamics")
def analyze_dynamics(req: DynamicsRequest):
    from .tools.dynamics import DynamicsAnalyzer
    import numpy as np
    analyzer = DynamicsAnalyzer()
    ts = np.array(req.time_series)
    result = analyzer.analyze(ts)
    return result.to_dict()


@app.post("/analyze/tda")
def analyze_tda(req: TDARequest):
    from .tools.tda import TDAAnalyzer
    import numpy as np
    analyzer = TDAAnalyzer()
    if req.point_cloud is not None:
        data = np.array(req.point_cloud)
    elif req.volume_data is not None:
        data = np.array(req.volume_data)
    else:
        return {"error": "Provide point_cloud or volume_data"}
    result = analyzer.analyze(data, req.data_type, req.max_dim)
    return result.to_dict()


@app.post("/viz/sindy")
def viz_sindy(req: SINDyRequest):
    from .tools.sindy import SINDyDiscoverer
    from .tools.viz import InteractiveVisualizer
    import numpy as np
    discoverer = SINDyDiscoverer()
    viz = InteractiveVisualizer()
    data = np.array(req.time_series)
    result = discoverer.discover_ode(
        data,
        dt=req.dt,
        poly_order=req.poly_order,
        threshold=req.threshold,
        variable_names=req.variable_names,
    )
    eq_dicts = [eq.to_dict() for eq in result.equations]
    return viz.plot_sindy_equations(
        eq_dicts,
        model_score=result.model_score,
        complexity=result.complexity,
    )


@app.post("/sandbox/execute")
def sandbox_execute(req: SandboxRequest):
    from .sandbox import SandboxExecutor, SandboxConfig
    config = SandboxConfig(
        timeout_seconds=req.timeout_seconds,
        backend=req.backend,
    )
    executor = SandboxExecutor(config)
    result = executor.execute(req.code)
    return result.to_dict()


@app.post("/analyze/sindy")
def analyze_sindy(req: SINDyRequest):
    from .tools.sindy import SINDyDiscoverer
    import numpy as np
    discoverer = SINDyDiscoverer()
    data = np.array(req.time_series)
    result = discoverer.discover_ode(
        data,
        dt=req.dt,
        poly_order=req.poly_order,
        threshold=req.threshold,
        variable_names=req.variable_names,
    )
    return result.to_dict()


@app.post("/viz/dos")
def viz_dos(req: SpectralRequest):
    from .tools.spectral import SpectralAnalyzer
    from .tools.viz import InteractiveVisualizer
    import numpy as np
    analyzer = SpectralAnalyzer()
    viz = InteractiveVisualizer()
    eigs = np.array(req.eigenvalues)
    dos_result = analyzer.dos_analysis(eigs)
    if dos_result.energies and dos_result.total_dos:
        return viz.plot_dos(
            dos_result.energies, dos_result.total_dos,
            dos_result.fermi_energy, dos_result.band_gap,
        )
    return {"error": "DOS computation failed"}


@app.post("/viz/phase")
def viz_phase(req: DynamicsRequest):
    from .tools.viz import InteractiveVisualizer
    import numpy as np
    viz = InteractiveVisualizer()
    ts = np.array(req.time_series)
    return viz.plot_phase_portrait(ts)


@app.post("/viz/persistence")
def viz_persistence(req: TDARequest):
    from .tools.tda import TDAAnalyzer
    from .tools.viz import InteractiveVisualizer
    import numpy as np
    analyzer = TDAAnalyzer()
    viz = InteractiveVisualizer()
    if req.point_cloud is not None:
        data = np.array(req.point_cloud)
    else:
        return {"error": "Provide point_cloud for persistence diagram"}
    persistence = analyzer.persistent_homology(data, req.max_dim)
    return viz.plot_persistence_diagram(persistence.to_dict())


@app.post("/viz/geometry")
def viz_geometry(req: GeometryVizRequest):
    from .tools.viz import InteractiveVisualizer
    viz = InteractiveVisualizer()
    if req.viz_type == "manifold":
        return viz.plot_manifold(
            req.metric_tensor or [],
            req.coord_range or [-2, 2],
            req.n_points or 50,
        )
    if req.viz_type == "brillouin":
        return viz.plot_brillouin_zone(
            req.lattice_vectors or [],
        )
    return {"error": f"Unknown viz_type: {req.viz_type}"}


def _make_agent_loop(registry):
    from .agent_loop import MathAgentLoop
    llm_cfg = {}
    if hasattr(_fv, 'llm_semantic') and hasattr(_fv.llm_semantic, 'api_config'):
        llm_cfg = _fv.llm_semantic.api_config or {}
    fw = _get_firewall()
    return MathAgentLoop(registry, llm_cfg, firewall=fw)


if __name__ == "__main__":
    import sys
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")
