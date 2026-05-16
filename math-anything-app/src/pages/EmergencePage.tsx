import { useState, useCallback } from "react";
import { Waves, Loader2, Upload } from "lucide-react";
import { api, apiUpload } from "../lib/api";
import { useAppStore } from "../stores/appStore";

const ENGINES = ["vasp", "lammps", "abaqus", "quantum_espresso", "gromacs", "multiwfn", "ansys"];

export function EmergencePage() {
  const [engine, setEngine] = useState("vasp");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const setCurrentSchema = useAppStore((s) => s.setCurrentSchema);

  const doAnalyze = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const data = await apiUpload(`/extract/${engine}/emergence`, fd);
      setResult(data as Record<string, unknown>);
      if ((data as Record<string, unknown>).schema) {
        setCurrentSchema((data as Record<string, Record<string, unknown>>).schema as Record<string, unknown>);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [engine, file, setCurrentSchema]);

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="font-display text-2xl font-semibold mb-1">涌现分析</h1>
      <p className="text-text-2 text-sm mb-6">相变检测、涌现结构识别、谱间隙分析</p>

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
            <label className="block text-xs text-text-3 mb-1">输入文件</label>
            <div className="flex items-center gap-2">
              <input
                type="file"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="flex-1 text-sm text-text-2 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:bg-accent-dim file:text-accent"
              />
            </div>
          </div>
        </div>

        <button
          onClick={doAnalyze}
          disabled={loading || !file}
          className="flex items-center gap-2 px-4 py-2 bg-accent text-bg rounded-lg text-sm font-medium hover:bg-accent/90 disabled:opacity-50 transition-colors"
        >
          {loading ? <Loader2 size={14} className="animate-spin" /> : <Waves size={14} />}
          分析涌现
        </button>
      </div>

      {error && (
        <div className="mt-4 px-4 py-2 bg-error-dim text-error text-sm rounded-lg border border-error/20">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-6 bg-bg-card border border-border rounded-lg p-4">
          <div className="text-xs text-text-3 font-semibold mb-2">涌现分析结果</div>
          <pre className="text-xs text-text-2 whitespace-pre-wrap overflow-x-auto max-h-96">
            {JSON.stringify(result, null, 2).slice(0, 5000)}
          </pre>
        </div>
      )}
    </div>
  );
}
