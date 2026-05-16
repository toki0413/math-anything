import { useState } from "react";
import { CheckSquare, Target, Eye, Loader2 } from "lucide-react";
import clsx from "clsx";
import { api } from "../lib/api";

type Tab = "crossval" | "predictions" | "dual";

const CV_CYCLE = ["not_tested", "confirmed", "partially_confirmed", "unconfirmed", "contradicted"] as const;
const CV_SYMBOLS: Record<string, string> = { not_tested: "·", confirmed: "✓", partially_confirmed: "~", unconfirmed: "?", contradicted: "✗" };
const CV_CLASSES: Record<string, string> = { not_tested: "", confirmed: "text-accent3", partially_confirmed: "text-warn", unconfirmed: "text-accent", contradicted: "text-error" };

const PRED_CYCLE = ["pending", "verified", "falsified", "inconclusive"] as const;
const PRED_LABELS: Record<string, string> = { pending: "PENDING", verified: "VERIFIED", falsified: "FALSIFIED", inconclusive: "INCONCLUSIVE" };
const PRED_CLASSES: Record<string, string> = { pending: "bg-bg-card text-text-3", verified: "bg-accent3-dim text-accent3", falsified: "bg-error-dim text-error", inconclusive: "bg-warn-dim text-warn" };

export function ValidatePage() {
  const [tab, setTab] = useState<Tab>("crossval");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [cvMethods, setCvMethods] = useState("");
  const [cvConclusions, setCvConclusions] = useState("");
  const [cvCells, setCvCells] = useState<Record<string, string>>({});
  const [cvReport, setCvReport] = useState("");

  const [predictions, setPredictions] = useState<Array<{id: string; statement: string; condition: string; method: string; status: string}>>([]);
  const [predReport, setPredReport] = useState("");

  const [dualConclusion, setDualConclusion] = useState("");
  const [geoChecks, setGeoChecks] = useState<string[]>([]);
  const [anaChecks, setAnaChecks] = useState<string[]>([]);
  const [geoMarks, setGeoMarks] = useState<Record<number, string>>({});
  const [anaMarks, setAnaMarks] = useState<Record<number, string>>({});
  const [dualReport, setDualReport] = useState("");
  const [dualResult, setDualResult] = useState<Record<string, unknown> | null>(null);

  const methods = cvMethods.split(",").map((s) => s.trim()).filter(Boolean);
  const conclusions = cvConclusions.split(",").map((s) => s.trim()).filter(Boolean);

  const toggleCvCell = (m: string, c: string) => {
    const key = `${m}::${c}`;
    const current = cvCells[key] || "not_tested";
    const idx = CV_CYCLE.indexOf(current as typeof CV_CYCLE[number]);
    const next = CV_CYCLE[(idx + 1) % CV_CYCLE.length];
    setCvCells({ ...cvCells, [key]: next });
  };

  const cycleMark = (marks: Record<number, string>, setMarks: (m: Record<number, string>) => void, idx: number) => {
    const cycle = ["·", "✓", "✗", "?"];
    const current = marks[idx] || "·";
    const ci = cycle.indexOf(current);
    setMarks({ ...marks, [idx]: cycle[(ci + 1) % cycle.length] });
  };

  const submitCrossVal = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await api("/validate/crossval", {
        method: "POST",
        body: JSON.stringify({ methods, conclusions }),
      });
      setCvReport((data as Record<string, string>).report || "");
      if ((data as Record<string, Record<string, Record<string, string>>>).matrix?.cells) {
        const cells: Record<string, string> = {};
        for (const [m, row] of Object.entries((data as Record<string, Record<string, Record<string, string>>>).matrix.cells)) {
          for (const [c, status] of Object.entries(row)) {
            cells[`${m}::${c}`] = status;
          }
        }
        setCvCells(cells);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const submitPredictions = async () => {
    setLoading(true);
    setError("");
    try {
      const preds = predictions.map((p) => ({
        id: p.id,
        statement: p.statement,
        condition: p.condition,
        method: p.method,
      }));
      const data = await api("/validate/predictions", {
        method: "POST",
        body: JSON.stringify({ predictions: preds }),
      });
      setPredReport((data as Record<string, string>).report || "");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const submitDual = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await api("/validate/dual", {
        method: "POST",
        body: JSON.stringify({
          conclusion: dualConclusion,
          geometric_checks: geoChecks,
          analytic_checks: anaChecks,
        }),
      });
      setDualReport((data as Record<string, string>).report || "");
      setDualResult((data as Record<string, Record<string, unknown>>).result || null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="font-display text-2xl font-semibold mb-1">交叉验证</h1>
      <p className="text-text-2 text-sm mb-6">方法 × 结论验证网格 · 可证伪预测表 · 双视角分析</p>

      <div className="flex gap-2 mb-6">
        {[
          { id: "crossval" as Tab, icon: CheckSquare, label: "交叉验证矩阵" },
          { id: "predictions" as Tab, icon: Target, label: "可证伪预测" },
          { id: "dual" as Tab, icon: Eye, label: "双视角分析" },
        ].map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={clsx(
              "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
              tab === t.id ? "bg-accent-dim text-accent border border-accent/30" : "text-text-3 hover:text-text-2"
            )}
          >
            <t.icon size={16} /> {t.label}
          </button>
        ))}
      </div>

      {error && (
        <div className="mb-4 px-4 py-2 bg-error-dim text-error text-sm rounded-lg border border-error/20">
          {error}
        </div>
      )}

      {tab === "crossval" && (
        <div>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-xs text-text-3 mb-1">方法（逗号分隔）</label>
              <input value={cvMethods} onChange={(e) => setCvMethods(e.target.value)} placeholder="输入方法，逗号分隔" className="w-full bg-bg-card border border-border rounded-lg px-3 py-2 text-sm text-text focus:outline-none focus:border-accent/50" />
            </div>
            <div>
              <label className="block text-xs text-text-3 mb-1">结论（逗号分隔）</label>
              <input value={cvConclusions} onChange={(e) => setCvConclusions(e.target.value)} placeholder="输入结论，逗号分隔" className="w-full bg-bg-card border border-border rounded-lg px-3 py-2 text-sm text-text focus:outline-none focus:border-accent/50" />
            </div>
          </div>
          <button
            onClick={submitCrossVal}
            disabled={loading || methods.length === 0 || conclusions.length === 0}
            className="mb-4 flex items-center gap-2 px-4 py-2 bg-accent text-bg rounded-lg text-sm font-medium hover:bg-accent/90 disabled:opacity-50 transition-colors"
          >
            {loading ? <Loader2 size={14} className="animate-spin" /> : <CheckSquare size={14} />}
            提交验证
          </button>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className="text-left px-3 py-2 text-xs text-text-3 font-semibold">Method</th>
                  {conclusions.map((c) => (
                    <th key={c} className="px-3 py-2 text-xs text-text-3 font-semibold text-center">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {methods.map((m) => (
                  <tr key={m} className="border-t border-border">
                    <td className="px-3 py-2 text-sm text-text-2 font-medium">{m}</td>
                    {conclusions.map((c) => {
                      const key = `${m}::${c}`;
                      const status = cvCells[key] || "not_tested";
                      return (
                        <td key={c} className="px-3 py-2 text-center">
                          <button
                            onClick={() => toggleCvCell(m, c)}
                            className={clsx("text-lg font-bold w-8 h-8 rounded hover:bg-bg-hover transition-colors", CV_CLASSES[status])}
                          >
                            {CV_SYMBOLS[status]}
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {cvReport && (
            <div className="mt-4 p-4 bg-bg-card border border-border rounded-lg">
              <div className="text-xs text-text-3 font-semibold mb-2">验证报告</div>
              <pre className="text-sm text-text-2 whitespace-pre-wrap">{cvReport}</pre>
            </div>
          )}
        </div>
      )}

      {tab === "predictions" && (
        <div className="space-y-3">
          <button
            onClick={submitPredictions}
            disabled={loading}
            className="mb-2 flex items-center gap-2 px-4 py-2 bg-accent text-bg rounded-lg text-sm font-medium hover:bg-accent/90 disabled:opacity-50 transition-colors"
          >
            {loading ? <Loader2 size={14} className="animate-spin" /> : <Target size={14} />}
            提交预测表
          </button>
          {predictions.map((p, i) => (
            <div key={p.id} className="bg-bg-card border border-border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-mono text-xs text-accent font-semibold">{p.id}</span>
                <button
                  onClick={() => {
                    const ci = PRED_CYCLE.indexOf(p.status as typeof PRED_CYCLE[number]);
                    const next = PRED_CYCLE[(ci + 1) % PRED_CYCLE.length];
                    const updated = [...predictions];
                    updated[i] = { ...p, status: next };
                    setPredictions(updated);
                  }}
                  className={clsx("px-2 py-0.5 rounded text-xs font-semibold cursor-pointer transition-colors", PRED_CLASSES[p.status])}
                >
                  {PRED_LABELS[p.status]}
                </button>
              </div>
              <div className="text-sm font-medium mb-1">{p.statement}</div>
              <div className="text-xs text-text-3 font-mono">条件: {p.condition}</div>
              <div className="text-xs text-text-3 font-mono">检验: {p.method}</div>
            </div>
          ))}
          {predReport && (
            <div className="mt-4 p-4 bg-bg-card border border-border rounded-lg">
              <div className="text-xs text-text-3 font-semibold mb-2">预测报告</div>
              <pre className="text-sm text-text-2 whitespace-pre-wrap">{predReport}</pre>
            </div>
          )}
        </div>
      )}

      {tab === "dual" && (
        <div>
          <div className="mb-4">
            <label className="block text-xs text-text-3 mb-1">待验证结论</label>
            <input value={dualConclusion} onChange={(e) => setDualConclusion(e.target.value)} placeholder="输入待验证结论" className="w-full bg-bg-card border border-border rounded-lg px-3 py-2 text-sm text-text focus:outline-none focus:border-accent/50" />
          </div>
          <button
            onClick={submitDual}
            disabled={loading || !dualConclusion}
            className="mb-4 flex items-center gap-2 px-4 py-2 bg-accent text-bg rounded-lg text-sm font-medium hover:bg-accent/90 disabled:opacity-50 transition-colors"
          >
            {loading ? <Loader2 size={14} className="animate-spin" /> : <Eye size={14} />}
            提交双视角分析
          </button>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-bg-card border border-border rounded-lg p-4">
              <div className="text-sm font-semibold text-accent mb-1">◈ 几何视角 (微分几何)</div>
              <div className="text-xs text-text-3 italic mb-3">有什么几何结构？</div>
              {geoChecks.map((c, i) => (
                <div key={i} className="flex items-center gap-2 py-1.5 border-t border-border">
                  <button onClick={() => cycleMark(geoMarks, setGeoMarks, i)} className="w-6 h-6 rounded-full border border-border bg-bg-surface text-xs font-bold flex items-center justify-center hover:border-accent transition-colors">
                    {geoMarks[i] || "·"}
                  </button>
                  <span className="text-sm text-text-2">{c}</span>
                </div>
              ))}
              {dualResult && (
                <div className="mt-3 pt-2 border-t border-border">
                  <div className="text-xs text-accent font-semibold">
                    判定: {(dualResult as Record<string, string>).geometric_verdict || "—"}
                  </div>
                </div>
              )}
            </div>
            <div className="bg-bg-card border border-border rounded-lg p-4">
              <div className="text-sm font-semibold text-accent2 mb-1">◇ 分析视角 (概率 + 调和分析)</div>
              <div className="text-xs text-text-3 italic mb-3">统计信号是真实的吗？</div>
              {anaChecks.map((c, i) => (
                <div key={i} className="flex items-center gap-2 py-1.5 border-t border-border">
                  <button onClick={() => cycleMark(anaMarks, setAnaMarks, i)} className="w-6 h-6 rounded-full border border-border bg-bg-surface text-xs font-bold flex items-center justify-center hover:border-accent2 transition-colors">
                    {anaMarks[i] || "·"}
                  </button>
                  <span className="text-sm text-text-2">{c}</span>
                </div>
              ))}
              {dualResult && (
                <div className="mt-3 pt-2 border-t border-border">
                  <div className="text-xs text-accent2 font-semibold">
                    判定: {(dualResult as Record<string, string>).analytic_verdict || "—"}
                  </div>
                  <div className="text-xs text-text-3 mt-1">
                    一致性: {(dualResult as Record<string, string>).agreement || "—"}
                  </div>
                </div>
              )}
            </div>
          </div>
          {dualReport && (
            <div className="mt-4 p-4 bg-bg-card border border-border rounded-lg">
              <div className="text-xs text-text-3 font-semibold mb-2">双视角分析报告</div>
              <pre className="text-sm text-text-2 whitespace-pre-wrap">{dualReport}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
