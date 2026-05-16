import { useState, useCallback } from "react";
import { Triangle, Loader2, Box, RotateCcw } from "lucide-react";
import { api } from "../lib/api";

const ENGINES = ["vasp", "lammps", "abaqus", "quantum_espresso", "gromacs", "multiwfn", "ansys"];

type VizType = "none" | "manifold" | "brillouin";

export function GeometryPage() {
  const [engine, setEngine] = useState("vasp");
  const [params, setParams] = useState('{"ENCUT": 520}');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const [vizType, setVizType] = useState<VizType>("none");
  const [vizHtml, setVizHtml] = useState<string | null>(null);
  const [vizLoading, setVizLoading] = useState(false);

  const doExtract = useCallback(async () => {
    setLoading(true);
    setError("");
    setResult(null);
    setVizHtml(null);
    try {
      const parsed = JSON.parse(params);
      const data = await api(`/geometry/${engine}`, {
        method: "POST",
        body: JSON.stringify({ params: parsed }),
      });
      setResult(data as Record<string, unknown>);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [engine, params]);

  async function runViz() {
    setVizLoading(true);
    setVizHtml(null);
    try {
      const body: Record<string, unknown> = { viz_type: vizType };
      if (vizType === "manifold") {
        const mt = result?.metric_tensor;
        if (mt && typeof mt === "object") {
          const m = mt as Record<string, unknown>;
          const components = m.components as number[][][] | undefined;
          if (components && components.length > 0) {
            body.metric_tensor = components[0];
          }
        }
        if (!body.metric_tensor) {
          body.metric_tensor = [[1, 0], [0, 1]];
        }
      }
      if (vizType === "brillouin") {
        const mt = result?.metric_tensor;
        if (mt && typeof mt === "object") {
          const m = mt as Record<string, unknown>;
          const lattice = m.lattice_vectors as number[][] | undefined;
          if (lattice) {
            body.lattice_vectors = lattice;
          }
        }
        if (!body.lattice_vectors) {
          body.lattice_vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        }
      }
      const data = await api("/viz/geometry", {
        method: "POST",
        body: JSON.stringify(body),
      });
      const r = data as Record<string, unknown>;
      if (r.html) {
        setVizHtml(r.html as string);
      } else if (r.error) {
        setError(String(r.error));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setVizLoading(false);
    }
  }

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <h1 className="font-display text-2xl font-semibold mb-1">几何提取</h1>
      <p className="text-text-2 text-sm mb-6">流形、度量张量、曲率、纤维丛结构</p>

      <div className="bg-bg-card border border-border rounded-lg p-6 space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-text-3 mb-1">引擎</label>
            <select
              value={engine}
              onChange={(e) => setEngine(e.target.value)}
              className="w-full bg-bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text"
            >
              {ENGINES.map((e) => (
                <option key={e} value={e}>{e}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-text-3 mb-1">参数 (JSON)</label>
            <textarea
              value={params}
              onChange={(e) => setParams(e.target.value)}
              rows={3}
              className="w-full bg-bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text font-mono focus:outline-none focus:border-accent/50"
            />
          </div>
        </div>

        <button
          onClick={doExtract}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-accent text-bg rounded-lg text-sm font-medium hover:bg-accent/90 disabled:opacity-50 transition-colors"
        >
          {loading ? <Loader2 size={14} className="animate-spin" /> : <Triangle size={14} />}
          提取几何结构
        </button>
      </div>

      {error && (
        <div className="mt-4 px-4 py-2 bg-error-dim text-error text-sm rounded-lg border border-error/20">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-6 space-y-4">
          {(["manifold", "metric_tensor", "curvature", "fiber_bundle"] as const).map((key) => {
            const val = result[key];
            if (!val || (typeof val === "object" && Object.keys(val as object).length === 0)) return null;
            return (
              <div key={key} className="bg-bg-card border border-border rounded-lg p-4">
                <div className="text-xs text-accent font-semibold mb-2 uppercase tracking-wider">{key.replace(/_/g, " ")}</div>
                <pre className="text-xs text-text-2 whitespace-pre-wrap overflow-x-auto">
                  {JSON.stringify(val, null, 2).slice(0, 3000)}
                </pre>
              </div>
            );
          })}

          <div className="bg-bg-card border border-border rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="text-xs text-accent font-semibold uppercase tracking-wider">3D Visualization</div>
              <div className="flex items-center gap-2">
                <select
                  value={vizType}
                  onChange={(e) => setVizType(e.target.value as VizType)}
                  className="bg-bg-surface border border-border rounded px-2 py-1 text-xs"
                >
                  <option value="none">None</option>
                  <option value="manifold">Manifold Surface</option>
                  <option value="brillouin">Brillouin Zone</option>
                </select>
                {vizType !== "none" && (
                  <button
                    onClick={runViz}
                    disabled={vizLoading}
                    className="flex items-center gap-1 px-3 py-1 bg-accent text-bg rounded text-xs font-medium hover:bg-accent/90 disabled:opacity-50"
                  >
                    {vizLoading ? <Loader2 size={12} className="animate-spin" /> : <Box size={12} />}
                    Render
                  </button>
                )}
              </div>
            </div>

            {vizHtml ? (
              <div
                className="w-full border border-border rounded overflow-hidden"
                dangerouslySetInnerHTML={{ __html: vizHtml }}
              />
            ) : (
              <div className="h-48 bg-bg-surface border border-border rounded flex items-center justify-center text-text-3 text-sm">
                {vizType === "none" ? "Select a visualization type" : "Click Render to generate 3D view"}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
