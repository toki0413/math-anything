import { useState, useCallback, useEffect } from "react";
import { api, apiUpload } from "../lib/api";
import { useAppStore } from "../stores/appStore";
import { Upload, FlaskConical, Lightbulb } from "lucide-react";

const ENGINES = ["vasp", "lammps", "abaqus", "ansys", "comsol", "gromacs", "multiwfn"];

export function ExtractPage() {
  const [engine, setEngine] = useState("vasp");
  const [params, setParams] = useState('{\n  "ENCUT": 520,\n  "SIGMA": 0.05\n}');
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [advisory, setAdvisory] = useState<{
    tools?: string[];
    math_disciplines?: string[];
    workflow_hint?: string;
    keywords?: string[];
  } | null>(null);
  const { setCurrentSchema } = useAppStore();

  useEffect(() => {
    setAdvisory(null);
    api(`/advisor/${engine}`)
      .then((d) => setAdvisory(d as typeof advisory))
      .catch(() => setAdvisory(null));
  }, [engine]);

  const doExtract = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const parsed = JSON.parse(params);
      const data = await api(`/extract/${engine}`, {
        method: "POST",
        body: JSON.stringify({ params: parsed }),
      });
      setResult(data as Record<string, unknown>);
      setCurrentSchema(data as Record<string, unknown>);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [engine, params, setCurrentSchema]);

  const doUpload = useCallback(
    async (files: FileList) => {
      setLoading(true);
      setError("");
      const fd = new FormData();
      for (let i = 0; i < files.length; i++) fd.append("files", files[i]);
      try {
        const data = await apiUpload(`/extract/${engine}/file`, fd);
        setResult(data as Record<string, unknown>);
        setCurrentSchema(data as Record<string, unknown>);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [engine, setCurrentSchema]
  );

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="font-display text-2xl font-semibold mb-1">结构提取</h1>
      <p className="text-text-2 text-sm mb-6">
        从计算科学输入文件中提取数学结构
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-text-3 uppercase tracking-wider mb-2">
              计算引擎
            </label>
            <div className="flex flex-wrap gap-2">
              {ENGINES.map((e) => (
                <button
                  key={e}
                  onClick={() => setEngine(e)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    e === engine
                      ? "bg-accent-dim text-accent border border-accent/30"
                      : "bg-bg-card text-text-2 border border-border hover:border-border-light"
                  }`}
                >
                  {e.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {advisory && (
            <div className="bg-bg-card border border-accent2/20 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <Lightbulb size={16} className="text-accent2" />
                <span className="text-xs font-semibold text-accent2 uppercase tracking-wider">推荐分析</span>
              </div>
              <div className="space-y-2 text-xs text-text-2">
                <div>
                  <span className="text-text-3 font-semibold">工具链:</span>{" "}
                  {advisory.tools?.map((t) => (
                    <span key={t} className="inline-block px-1.5 py-0.5 bg-accent-dim text-accent rounded mr-1 mb-0.5">{t}</span>
                  ))}
                </div>
                <div>
                  <span className="text-text-3 font-semibold">数学领域:</span>
                  <ul className="mt-1 ml-3 list-disc space-y-0.5">
                    {advisory.math_disciplines?.map((d) => (
                      <li key={d}>{d}</li>
                    ))}
                  </ul>
                </div>
                {advisory.workflow_hint && (
                  <div>
                    <span className="text-text-3 font-semibold">推荐流程:</span>{" "}
                    <span className="text-text-2">{advisory.workflow_hint}</span>
                  </div>
                )}
              </div>
            </div>
          )}
            <label className="block text-xs font-semibold text-text-3 uppercase tracking-wider mb-2">
              参数 JSON
            </label>
            <textarea
              value={params}
              onChange={(e) => setParams(e.target.value)}
              rows={8}
              className="w-full bg-bg-card border border-border rounded-lg p-3 font-mono text-sm text-text focus:outline-none focus:border-accent/50 resize-y"
            />
          </div>

          <div
            className="border-2 border-dashed border-border rounded-lg p-6 text-center hover:border-accent/30 transition-colors cursor-pointer"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              if (e.dataTransfer.files.length) doUpload(e.dataTransfer.files);
            }}
          >
            <Upload size={24} className="mx-auto text-text-3 mb-2" />
            <p className="text-sm text-text-2">拖放文件到此处上传</p>
            <input
              type="file"
              multiple
              className="hidden"
              id="file-upload"
              onChange={(e) => e.target.files && doUpload(e.target.files)}
            />
            <label
              htmlFor="file-upload"
              className="inline-block mt-2 text-xs text-accent cursor-pointer hover:underline"
            >
              或点击选择文件
            </label>
          </div>

          <button
            onClick={doExtract}
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 px-5 py-2.5 bg-accent text-bg font-medium rounded-lg text-sm hover:bg-accent/90 disabled:opacity-50 transition-colors"
          >
            <FlaskConical size={16} />
            {loading ? "提取中..." : "提取数学结构"}
          </button>
        </div>

        <div>
          <label className="block text-xs font-semibold text-text-3 uppercase tracking-wider mb-2">
            提取结果
          </label>
          {error && (
            <div className="bg-error-dim border border-error/20 rounded-lg p-3 text-sm text-error mb-3">
              {error}
            </div>
          )}
          <div className="bg-bg-card border border-border rounded-lg p-4 min-h-[300px] max-h-[600px] overflow-auto">
            {result ? (
              <pre className="text-xs font-mono text-text-2 whitespace-pre-wrap">
                {JSON.stringify(result, null, 2)}
              </pre>
            ) : (
              <p className="text-text-3 text-sm text-center pt-20">
                提取结果将显示在这里
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
