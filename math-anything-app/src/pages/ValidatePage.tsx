import { useState } from "react";
import {
  FlaskConical,
  Minimize2,
  GitBranch,
  Loader2,
  Upload,
  FileText,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
} from "lucide-react";
import clsx from "clsx";
import { api, apiUpload } from "../lib/api";

type Tab = "extract" | "simplify" | "discover" | "schema";

const ENGINES = [
  { id: "vasp", label: "VASP", ext: "INCAR/POSCAR/KPOINTS" },
  { id: "lammps", label: "LAMMPS", ext: "in.lmp/data.lmp" },
  { id: "abaqus", label: "Abaqus", ext: ".inp" },
  { id: "ansys", label: "ANSYS", ext: ".dat/.cdb" },
  { id: "comsol", label: "COMSOL", ext: ".m" },
  { id: "gromacs", label: "GROMACS", ext: ".mdp/.top" },
  { id: "multiwfn", label: "Multiwfn", ext: ".wfn/.fch" },
];

export function ValidatePage() {
  const [tab, setTab] = useState<Tab>("extract");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [engine, setEngine] = useState("vasp");
  const [inputText, setInputText] = useState("");
  const [extractResult, setExtractResult] = useState<Record<
    string,
    unknown
  > | null>(null);

  const [exprInput, setExprInput] = useState("");
  const [simplifyResult, setSimplifyResult] = useState<Record<
    string,
    unknown
  > | null>(null);

  const [dataInput, setDataInput] = useState("");
  const [discoverResult, setDiscoverResult] = useState<Record<
    string,
    unknown
  > | null>(null);

  const [schemaInput, setSchemaInput] = useState("");
  const [schemaResult, setSchemaResult] = useState<Record<
    string,
    unknown
  > | null>(null);

  const [copied, setCopied] = useState(false);

  const submitExtract = async () => {
    setLoading(true);
    setError("");
    setExtractResult(null);
    try {
      const params: Record<string, unknown> = { engine };
      const lines = inputText.trim().split("\n");
      for (const line of lines) {
        const eqIdx = line.indexOf("=");
        if (eqIdx > 0) {
          const key = line.slice(0, eqIdx).trim();
          const val = line.slice(eqIdx + 1).trim();
          params[key] = isNaN(Number(val)) ? val : Number(val);
        }
      }
      const data = await api("/extract", {
        method: "POST",
        body: JSON.stringify(params),
      });
      setExtractResult(data as Record<string, unknown>);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const submitSimplify = async () => {
    setLoading(true);
    setError("");
    setSimplifyResult(null);
    try {
      const data = await api("/simplify", {
        method: "POST",
        body: JSON.stringify({ expression: exprInput }),
      });
      setSimplifyResult(data as Record<string, unknown>);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const submitDiscover = async () => {
    setLoading(true);
    setError("");
    setDiscoverResult(null);
    try {
      const rows = dataInput
        .trim()
        .split("\n")
        .map((l) =>
          l
            .split(/[\s,]+/)
            .map(Number)
            .filter((n) => !isNaN(n))
        )
        .filter((r) => r.length > 0);
      if (rows.length < 5) {
        setError("至少需要5行数据");
        setLoading(false);
        return;
      }
      const data = await api("/discover", {
        method: "POST",
        body: JSON.stringify({ data: rows }),
      });
      setDiscoverResult(data as Record<string, unknown>);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const submitSchema = async () => {
    setLoading(true);
    setError("");
    setSchemaResult(null);
    try {
      const schema = JSON.parse(schemaInput);
      const data = await api("/validate_schema", {
        method: "POST",
        body: JSON.stringify({ schema }),
      });
      setSchemaResult(data as Record<string, unknown>);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const TABS: { id: Tab; icon: typeof FlaskConical; label: string; desc: string }[] = [
    {
      id: "extract",
      icon: FlaskConical,
      label: "结构提取",
      desc: "从计算软件参数提取数学结构",
    },
    {
      id: "simplify",
      icon: Minimize2,
      label: "表达式简化",
      desc: "化简数学表达式",
    },
    {
      id: "discover",
      icon: GitBranch,
      label: "方程发现",
      desc: "从数据中发现数学方程",
    },
    {
      id: "schema",
      icon: FileText,
      label: "Schema 验证",
      desc: "验证数学结构的完整性",
    },
  ];

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="font-display text-2xl font-semibold mb-1">
        数学工具箱
      </h1>
      <p className="text-text-2 text-sm mb-6">
        结构提取 · 表达式简化 · 方程发现 · Schema验证
      </p>

      <div className="flex gap-2 mb-6">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={clsx(
              "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
              tab === t.id
                ? "bg-accent-dim text-accent border border-accent/30"
                : "text-text-3 hover:text-text-2"
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

      {tab === "extract" && (
        <div className="space-y-4">
          <div>
            <label className="block text-xs text-text-3 mb-2">
              选择计算引擎
            </label>
            <div className="flex flex-wrap gap-2">
              {ENGINES.map((e) => (
                <button
                  key={e.id}
                  onClick={() => setEngine(e.id)}
                  className={clsx(
                    "px-3 py-1.5 rounded-lg text-sm font-medium transition-colors border",
                    engine === e.id
                      ? "bg-accent-dim text-accent border-accent/30"
                      : "bg-bg-card text-text-3 border-border hover:text-text-2"
                  )}
                >
                  {e.label}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="block text-xs text-text-3 mb-1">
              输入参数（每行一个 KEY = VALUE）
            </label>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder={"ENCUT = 520\nSIGMA = 0.05\nISMEAR = 1\nEDIFF = 1e-6"}
              className="w-full bg-bg-card border border-border rounded-lg px-3 py-2 text-sm text-text font-mono h-40 focus:outline-none focus:border-accent/50"
            />
          </div>
          <button
            onClick={submitExtract}
            disabled={loading || !inputText.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-accent text-bg rounded-lg text-sm font-medium hover:bg-accent/90 disabled:opacity-50 transition-colors"
          >
            {loading ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <FlaskConical size={14} />
            )}
            提取数学结构
          </button>
          {extractResult && (
            <ResultCard
              title="提取结果"
              result={extractResult}
              onCopy={() =>
                copyToClipboard(JSON.stringify(extractResult, null, 2))
              }
              copied={copied}
            />
          )}
        </div>
      )}

      {tab === "simplify" && (
        <div className="space-y-4">
          <div>
            <label className="block text-xs text-text-3 mb-1">
              输入数学表达式
            </label>
            <input
              value={exprInput}
              onChange={(e) => setExprInput(e.target.value)}
              placeholder="sin(x)^2 + cos(x)^2"
              className="w-full bg-bg-card border border-border rounded-lg px-3 py-2 text-sm text-text font-mono focus:outline-none focus:border-accent/50"
            />
          </div>
          <button
            onClick={submitSimplify}
            disabled={loading || !exprInput.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-accent text-bg rounded-lg text-sm font-medium hover:bg-accent/90 disabled:opacity-50 transition-colors"
          >
            {loading ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Minimize2 size={14} />
            )}
            简化表达式
          </button>
          {simplifyResult && (
            <ResultCard
              title="简化结果"
              result={simplifyResult}
              onCopy={() =>
                copyToClipboard(JSON.stringify(simplifyResult, null, 2))
              }
              copied={copied}
            />
          )}
        </div>
      )}

      {tab === "discover" && (
        <div className="space-y-4">
          <div>
            <label className="block text-xs text-text-3 mb-1">
              输入数据（每行一组数值，空格或逗号分隔）
            </label>
            <textarea
              value={dataInput}
              onChange={(e) => setDataInput(e.target.value)}
              placeholder={"0.1 0.2\n0.3 0.5\n0.5 0.8\n0.7 1.1\n..."}
              className="w-full bg-bg-card border border-border rounded-lg px-3 py-2 text-sm text-text font-mono h-40 focus:outline-none focus:border-accent/50"
            />
          </div>
          <button
            onClick={submitDiscover}
            disabled={loading || !dataInput.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-accent text-bg rounded-lg text-sm font-medium hover:bg-accent/90 disabled:opacity-50 transition-colors"
          >
            {loading ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <GitBranch size={14} />
            )}
            发现方程
          </button>
          {discoverResult && (
            <ResultCard
              title="发现结果"
              result={discoverResult}
              onCopy={() =>
                copyToClipboard(JSON.stringify(discoverResult, null, 2))
              }
              copied={copied}
            />
          )}
        </div>
      )}

      {tab === "schema" && (
        <div className="space-y-4">
          <div>
            <label className="block text-xs text-text-3 mb-1">
              输入 Schema JSON
            </label>
            <textarea
              value={schemaInput}
              onChange={(e) => setSchemaInput(e.target.value)}
              placeholder={'{\n  "governing_equations": [...],\n  "boundary_conditions": [...]\n}'}
              className="w-full bg-bg-card border border-border rounded-lg px-3 py-2 text-sm text-text font-mono h-40 focus:outline-none focus:border-accent/50"
            />
          </div>
          <button
            onClick={submitSchema}
            disabled={loading || !schemaInput.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-accent text-bg rounded-lg text-sm font-medium hover:bg-accent/90 disabled:opacity-50 transition-colors"
          >
            {loading ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <FileText size={14} />
            )}
            验证 Schema
          </button>
          {schemaResult && (
            <ResultCard
              title="验证结果"
              result={schemaResult}
              onCopy={() =>
                copyToClipboard(JSON.stringify(schemaResult, null, 2))
              }
              copied={copied}
            />
          )}
        </div>
      )}
    </div>
  );
}

function ResultCard({
  title,
  result,
  onCopy,
  copied,
}: {
  title: string;
  result: Record<string, unknown>;
  onCopy: () => void;
  copied: boolean;
}) {
  const [expanded, setExpanded] = useState(true);

  const renderValue = (val: unknown, depth = 0): React.ReactNode => {
    if (val === null || val === undefined) return <span className="text-text-3">—</span>;
    if (typeof val === "boolean")
      return (
        <span className={val ? "text-accent3" : "text-error"}>
          {String(val)}
        </span>
      );
    if (typeof val === "number")
      return <span className="text-accent font-mono">{String(val)}</span>;
    if (typeof val === "string")
      return <span className="text-text font-mono">{val}</span>;
    if (Array.isArray(val)) {
      if (val.length === 0) return <span className="text-text-3">[]</span>;
      return (
        <div className="ml-4 space-y-1">
          {val.map((item, i) => (
            <div key={i} className="flex items-start gap-2">
              <span className="text-text-3 text-xs mt-0.5">[{i}]</span>
              {renderValue(item, depth + 1)}
            </div>
          ))}
        </div>
      );
    }
    if (typeof val === "object") {
      const entries = Object.entries(val as Record<string, unknown>);
      return (
        <div className={clsx("space-y-1", depth > 0 && "ml-4")}>
          {entries.map(([k, v]) => (
            <div key={k} className="flex items-start gap-2">
              <span className="text-accent2 text-xs font-mono min-w-[120px]">
                {k}:
              </span>
              {renderValue(v, depth + 1)}
            </div>
          ))}
        </div>
      );
    }
    return <span className="text-text-2">{String(val)}</span>;
  };

  return (
    <div className="bg-bg-card border border-border rounded-lg">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 text-xs text-text-3 font-semibold"
        >
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          {title}
        </button>
        <button
          onClick={onCopy}
          className="flex items-center gap-1 text-xs text-text-3 hover:text-accent transition-colors"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? "已复制" : "复制"}
        </button>
      </div>
      {expanded && (
        <div className="p-4 text-sm">{renderValue(result)}</div>
      )}
    </div>
  );
}
