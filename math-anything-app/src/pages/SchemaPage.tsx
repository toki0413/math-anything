import { useAppStore } from "../stores/appStore";
import { TreePine, AlertTriangle, CheckCircle, XCircle, Info, GitBranch } from "lucide-react";

function KatexMath({ math }: { math: string }) {
  return (
    <span
      className="katex-inline"
      dangerouslySetInnerHTML={{
        __html: tryRenderKatex(math),
      }}
    />
  );
}

function tryRenderKatex(latex: string): string {
  try {
    const katex = require("katex");
    return katex.renderToString(latex, {
      throwOnError: false,
      displayMode: false,
    });
  } catch {
    return `<code class="bg-gray-100 px-1 rounded text-xs">${escapeHtml(latex)}</code>`;
  }
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function SeverityBadge({ severity }: { severity?: string }) {
  const config: Record<string, { bg: string; text: string; icon: React.ReactNode }> = {
    critical: { bg: "bg-red-100", text: "text-red-700", icon: <XCircle size={12} /> },
    warning: { bg: "bg-yellow-100", text: "text-yellow-700", icon: <AlertTriangle size={12} /> },
    info: { bg: "bg-blue-100", text: "text-blue-700", icon: <Info size={12} /> },
  };
  const c = config[severity || "info"] || config.info;
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-medium ${c.bg} ${c.text}`}>
      {c.icon} {severity || "info"}
    </span>
  );
}

function ProvenanceTag({ prov }: { prov: Record<string, unknown> }) {
  if (!prov) return null;
  return (
    <div className="flex items-center gap-2 text-xs text-gray-500 mt-1">
      <GitBranch size={10} />
      <span>
        {prov.engine && `${prov.engine}:`}
        {prov.parameter && `${prov.parameter}`}
        {prov.line_number && `:${prov.line_number}`}
        {prov.extraction_method && ` (${prov.extraction_method})`}
      </span>
      {prov.confidence != null && (
        <span className={`px-1 rounded ${
          Number(prov.confidence) > 0.8 ? "bg-green-50 text-green-600" :
          Number(prov.confidence) > 0.5 ? "bg-yellow-50 text-yellow-600" :
          "bg-red-50 text-red-600"
        }`}>
          {(Number(prov.confidence) * 100).toFixed(0)}%
        </span>
      )}
    </div>
  );
}

export function SchemaPage() {
  const { currentSchema } = useAppStore();

  if (!currentSchema) {
    return (
      <div className="p-8 max-w-5xl mx-auto text-center pt-20">
        <TreePine size={40} className="text-text-3 mx-auto mb-4" />
        <p className="text-text-3 text-sm">尚未加载 Schema</p>
        <p className="text-text-3 text-xs mt-2">请先在"结构提取"页面提取数学结构</p>
      </div>
    );
  }

  const schema = currentSchema as Record<string, any>;
  const model = schema.schema?.mathematical_model || schema.mathematical_model;
  const equations = model?.governing_equations || [];
  const constraints = schema.schema?.symbolic_constraints || schema.symbolic_constraints || [];
  const provenance = schema.schema?.provenance || schema.provenance || [];
  const rawSymbols = schema.schema?.raw_symbols || schema.raw_symbols || {};

  const criticalCount = constraints.filter((c: any) => c.severity === "critical").length;
  const warningCount = constraints.filter((c: any) => c.severity === "warning").length;
  const failedCount = constraints.filter((c: any) => c.satisfied === false).length;

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="font-display text-2xl font-semibold mb-1">Schema 查看器</h1>
      <p className="text-text-2 text-sm mb-6">数学结构树形展示</p>

      {(criticalCount > 0 || warningCount > 0 || failedCount > 0) && (
        <div className="mb-4 flex items-center gap-3 text-sm">
          {criticalCount > 0 && (
            <span className="flex items-center gap-1 px-2 py-1 bg-red-50 text-red-700 rounded border border-red-200">
              <XCircle size={14} /> {criticalCount} critical
            </span>
          )}
          {warningCount > 0 && (
            <span className="flex items-center gap-1 px-2 py-1 bg-yellow-50 text-yellow-700 rounded border border-yellow-200">
              <AlertTriangle size={14} /> {warningCount} warning
            </span>
          )}
          {failedCount > 0 && (
            <span className="flex items-center gap-1 px-2 py-1 bg-red-50 text-red-700 rounded border border-red-200">
              <XCircle size={14} /> {failedCount} failed
            </span>
          )}
        </div>
      )}

      <div className="bg-bg-card border border-border rounded-lg p-6">
        <div className="space-y-1 font-mono text-sm">
          <div className="text-accent font-semibold">├── Mathematical Model</div>
          {equations.length > 0 ? (
            equations.map((eq: any, i: number) => (
              <div key={i} className="pl-6">
                <div className="text-text-2 flex items-center gap-2">
                  <span className="text-text-3">├──</span>
                  <span className="font-medium">{eq.name || `eq_${i}`}:</span>
                  {eq.mathematical_form ? (
                    <KatexMath math={eq.mathematical_form} />
                  ) : (
                    <span className="text-text-3">{eq.type || "—"}</span>
                  )}
                  <span className="text-text-3 text-xs">[{eq.type || "unknown"}]</span>
                </div>
                {eq.provenance && <ProvenanceTag prov={eq.provenance} />}
              </div>
            ))
          ) : (
            <div className="text-text-3 pl-6">└── (no equations)</div>
          )}

          <div className="text-accent font-semibold mt-2">├── Symbolic Constraints</div>
          {constraints.length > 0 ? (
            constraints.map((c: any, i: number) => (
              <div key={i} className="pl-6">
                <div className="flex items-center gap-2">
                  <span className="text-text-3">├──</span>
                  {c.expression ? (
                    <KatexMath math={c.expression} />
                  ) : (
                    <span className="text-text-2">{c.description || `constraint_${i}`}</span>
                  )}
                  <span className={c.satisfied !== false ? "text-accent3" : "text-error"}>
                    {c.satisfied !== false ? <CheckCircle size={14} /> : <XCircle size={14} />}
                  </span>
                  <SeverityBadge severity={c.severity} />
                </div>
                {c.description && c.expression && (
                  <div className="text-xs text-gray-500 pl-8">{c.description}</div>
                )}
                {c.physical_basis && (
                  <div className="text-xs text-gray-400 pl-8 italic">{c.physical_basis}</div>
                )}
                {c.provenance && <ProvenanceTag prov={c.provenance} />}
              </div>
            ))
          ) : (
            <div className="text-text-3 pl-6">└── (no constraints)</div>
          )}

          {Object.keys(rawSymbols).length > 0 && (
            <>
              <div className="text-accent font-semibold mt-2">├── Analysis Results</div>
              {Object.entries(rawSymbols).map(([key, val]) => (
                <div key={key} className="pl-6 text-text-2">
                  <span className="text-text-3">├──</span> {key}
                  {typeof val === "object" && val !== null && "description" in (val as Record<string, unknown>) && (
                    <span className="text-text-3 text-xs ml-2">
                      {(val as Record<string, unknown>).description as string}
                    </span>
                  )}
                </div>
              ))}
            </>
          )}

          {provenance.length > 0 && (
            <>
              <div className="text-accent font-semibold mt-2">├── Provenance</div>
              {provenance.map((p: any, i: number) => (
                <div key={i} className="pl-6">
                  <ProvenanceTag prov={p} />
                </div>
              ))}
            </>
          )}

          <div className="text-accent font-semibold mt-2">└── Raw JSON</div>
        </div>

        <details className="mt-4">
          <summary className="text-xs text-text-3 cursor-pointer hover:text-text-2">
            展开完整 JSON
          </summary>
          <pre className="mt-2 text-xs font-mono text-text-2 overflow-auto max-h-96 bg-bg-surface rounded-lg p-4">
            {JSON.stringify(currentSchema, null, 2)}
          </pre>
        </details>
      </div>
    </div>
  );
}
