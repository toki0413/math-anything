import { useState } from "react";
import { api } from "../lib/api";
import { ShieldCheck, CheckCircle2, XCircle, ChevronDown, ChevronRight } from "lucide-react";

interface LayerResult {
  passed: boolean;
  confidence: number;
  message: string;
  details?: Record<string, unknown>;
}

interface VerifyResult {
  task_id?: string;
  overall_passed?: boolean;
  overall_confidence?: number;
  results?: Record<string, LayerResult>;
  error?: string;
}

const LAYER_META: Record<string, { label: string; color: string }> = {
  symbolic: { label: "符号验证", color: "accent" },
  type_system: { label: "类型系统", color: "accent2" },
  logic: { label: "逻辑验证", color: "accent3" },
  llm_semantic: { label: "LLM 语义", color: "accent" },
  lean4_formal: { label: "Lean4 形式化", color: "accent2" },
};

function parseResult(raw: Record<string, unknown>): VerifyResult {
  if (raw.error) return { error: raw.error as string };

  const results: Record<string, LayerResult> = {};
  const rawResults = (raw.results ?? raw) as Record<string, unknown>;

  for (const [key, value] of Object.entries(rawResults)) {
    if (key === "task_id" || key === "overall_passed" || key === "overall_confidence" || key === "overall") continue;
    if (typeof value === "object" && value !== null) {
      const v = value as Record<string, unknown>;
      results[key] = {
        passed: Boolean(v.passed ?? v.valid ?? v.success),
        confidence: typeof v.confidence === "number" ? v.confidence : (v.passed || v.valid || v.success ? 1 : 0),
        message: typeof v.message === "string" ? v.message : typeof v.error === "string" ? v.error : "",
        details: (v.details ?? v) as Record<string, unknown>,
      };
    }
  }

  const overall = raw.overall as Record<string, unknown> | undefined;
  return {
    task_id: raw.task_id as string | undefined,
    overall_passed: typeof raw.overall_passed === "boolean"
      ? raw.overall_passed
      : overall?.passed as boolean | undefined,
    overall_confidence: typeof raw.overall_confidence === "number"
      ? raw.overall_confidence
      : overall?.confidence as number | undefined,
    results,
  };
}

function ConfidenceBar({ value, color }: { value: number; color: string }) {
  const pct = Math.round(value * 100);
  const barColor = pct >= 80 ? "bg-accent3" : pct >= 50 ? "bg-warn" : "bg-error";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-bg-surface rounded-full overflow-hidden">
        <div
          className={`h-full ${barColor} rounded-full transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-xs font-mono ${color}`}>{pct}%</span>
    </div>
  );
}

function LayerCard({ layerKey, result }: { layerKey: string; result: LayerResult }) {
  const [expanded, setExpanded] = useState(false);
  const meta = LAYER_META[layerKey] ?? { label: layerKey, color: "text" };

  return (
    <div className="bg-bg-card border border-border rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 p-4 text-left hover:bg-bg-hover transition-colors"
      >
        {result.passed ? (
          <CheckCircle2 size={20} className="text-accent3 shrink-0" />
        ) : (
          <XCircle size={20} className="text-error shrink-0" />
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-semibold text-text">{meta.label}</span>
            <span className="text-xs font-mono text-text-3">{layerKey}</span>
          </div>
          {result.message && (
            <p className="text-xs text-text-2 truncate">{result.message}</p>
          )}
        </div>
        <div className="w-24 shrink-0">
          <ConfidenceBar value={result.confidence} color="text-text-2" />
        </div>
        {expanded ? (
          <ChevronDown size={16} className="text-text-3 shrink-0" />
        ) : (
          <ChevronRight size={16} className="text-text-3 shrink-0" />
        )}
      </button>
      {expanded && result.details && (
        <div className="border-t border-border px-4 py-3">
          <pre className="text-xs font-mono text-text-2 whitespace-pre-wrap overflow-auto max-h-48">
            {JSON.stringify(result.details, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

export function VerifyPage() {
  const [statement, setStatement] = useState("1 + 1 = 2");
  const [proofText, setProofText] = useState("By definition of addition.");
  const [layers, setLayers] = useState<string[]>(["symbolic", "type_system", "logic"]);
  const [rawResult, setRawResult] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  const allLayers = [
    { id: "symbolic", label: "符号验证" },
    { id: "type_system", label: "类型系统" },
    { id: "logic", label: "逻辑验证" },
    { id: "llm_semantic", label: "LLM 语义" },
    { id: "lean4_formal", label: "Lean4 形式化" },
  ];

  const parsed = rawResult ? parseResult(rawResult) : null;

  const doVerify = async () => {
    setLoading(true);
    try {
      const data = await api("/verify", {
        method: "POST",
        body: JSON.stringify({
          task_id: "manual",
          task_type: "proof",
          task_name: "user_verify",
          statement,
          assumptions: [],
          goals: [statement],
          proof_text: proofText,
          engine: "generic",
          with_geometry: false,
        }),
      });
      setRawResult(data as Record<string, unknown>);
    } catch (e) {
      setRawResult({ error: e instanceof Error ? e.message : String(e) });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="font-display text-2xl font-semibold mb-1">形式化验证</h1>
      <p className="text-text-2 text-sm mb-6">5 层验证流水线：符号 → 类型 → 逻辑 → LLM → Lean4</p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-text-3 uppercase tracking-wider mb-2">验证层</label>
            <div className="flex flex-wrap gap-2">
              {allLayers.map((l) => (
                <button
                  key={l.id}
                  onClick={() =>
                    setLayers((prev) =>
                      prev.includes(l.id) ? prev.filter((x) => x !== l.id) : [...prev, l.id]
                    )
                  }
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    layers.includes(l.id)
                      ? "bg-accent3-dim text-accent3 border border-accent3/30"
                      : "bg-bg-card text-text-3 border border-border"
                  }`}
                >
                  {l.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-text-3 uppercase tracking-wider mb-2">数学陈述</label>
            <input
              value={statement}
              onChange={(e) => setStatement(e.target.value)}
              className="w-full bg-bg-card border border-border rounded-lg px-3 py-2 font-mono text-sm text-text focus:outline-none focus:border-accent/50"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-text-3 uppercase tracking-wider mb-2">证明文本</label>
            <textarea
              value={proofText}
              onChange={(e) => setProofText(e.target.value)}
              rows={4}
              className="w-full bg-bg-card border border-border rounded-lg p-3 font-mono text-sm text-text focus:outline-none focus:border-accent/50 resize-y"
            />
          </div>

          <button
            onClick={doVerify}
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 px-5 py-2.5 bg-accent text-bg font-medium rounded-lg text-sm hover:bg-accent/90 disabled:opacity-50 transition-colors"
          >
            <ShieldCheck size={16} />
            {loading ? "验证中..." : "开始验证"}
          </button>
        </div>

        <div className="space-y-4">
          <label className="block text-xs font-semibold text-text-3 uppercase tracking-wider">验证结果</label>

          {!parsed && (
            <div className="bg-bg-card border border-border rounded-lg p-4 min-h-[300px] flex items-center justify-center">
              <p className="text-text-3 text-sm">验证结果将显示在这里</p>
            </div>
          )}

          {parsed?.error && (
            <div className="bg-error-dim border border-error/20 rounded-lg p-4 text-sm text-error">
              {parsed.error}
            </div>
          )}

          {parsed && !parsed.error && (
            <>
              {/* Overall result */}
              <div className={`rounded-lg border p-4 ${
                parsed.overall_passed
                  ? "bg-accent3-dim border-accent3/20"
                  : "bg-error-dim border-error/20"
              }`}>
                <div className="flex items-center gap-3 mb-2">
                  {parsed.overall_passed ? (
                    <CheckCircle2 size={24} className="text-accent3" />
                  ) : (
                    <XCircle size={24} className="text-error" />
                  )}
                  <div>
                    <div className="text-sm font-semibold text-text">
                      {parsed.overall_passed ? "验证通过" : "验证未通过"}
                    </div>
                    {parsed.overall_confidence !== undefined && (
                      <div className="text-xs text-text-2 mt-0.5">
                        综合置信度: {Math.round(parsed.overall_confidence * 100)}%
                      </div>
                    )}
                  </div>
                </div>
                {parsed.overall_confidence !== undefined && (
                  <ConfidenceBar value={parsed.overall_confidence} color="text-text" />
                )}
              </div>

              {/* Layer cards */}
              {parsed.results && Object.entries(parsed.results).map(([key, res]) => (
                <LayerCard key={key} layerKey={key} result={res} />
              ))}

              {/* Fallback: if no structured results, show raw */}
              {(!parsed.results || Object.keys(parsed.results).length === 0) && rawResult && (
                <div className="bg-bg-card border border-border rounded-lg p-4">
                  <pre className="text-xs font-mono text-text-2 whitespace-pre-wrap overflow-auto max-h-96">
                    {JSON.stringify(rawResult, null, 2)}
                  </pre>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
