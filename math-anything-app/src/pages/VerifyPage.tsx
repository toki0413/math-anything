import { useState } from "react";
import { api } from "../lib/api";
import { ShieldCheck } from "lucide-react";

export function VerifyPage() {
  const [statement, setStatement] = useState("1 + 1 = 2");
  const [proofText, setProofText] = useState("By definition of addition.");
  const [layers, setLayers] = useState<string[]>(["symbolic", "type_system", "logic"]);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  const allLayers = [
    { id: "symbolic", label: "符号验证" },
    { id: "type_system", label: "类型系统" },
    { id: "logic", label: "逻辑验证" },
    { id: "llm_semantic", label: "LLM 语义" },
    { id: "lean4_formal", label: "Lean4 形式化" },
  ];

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
      setResult(data as Record<string, unknown>);
    } catch (e) {
      setResult({ error: e instanceof Error ? e.message : String(e) });
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

        <div>
          <label className="block text-xs font-semibold text-text-3 uppercase tracking-wider mb-2">验证结果</label>
          <div className="bg-bg-card border border-border rounded-lg p-4 min-h-[300px] max-h-[600px] overflow-auto">
            {result ? (
              <pre className="text-xs font-mono text-text-2 whitespace-pre-wrap">
                {JSON.stringify(result, null, 2)}
              </pre>
            ) : (
              <p className="text-text-3 text-sm text-center pt-20">验证结果将显示在这里</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
