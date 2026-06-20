import { useEffect, useState } from "react";
import { api, checkHealth } from "../lib/api";
import { useAppStore } from "../stores/appStore";
import {
  FlaskConical, ShieldCheck, MessageSquare, CheckSquare, Zap,
  Cpu, Activity, Clock, CheckCircle2, XCircle, Circle,
} from "lucide-react";

const ENGINES = [
  { id: "vasp", label: "VASP" },
  { id: "lammps", label: "LAMMPS" },
  { id: "abaqus", label: "Abaqus" },
  { id: "ansys", label: "ANSYS" },
  { id: "comsol", label: "COMSOL" },
  { id: "gromacs", label: "GROMACS" },
  { id: "multiwfn", label: "Multiwfn" },
];

interface RustStatus {
  available: boolean;
  module?: string;
}

function formatTime(ts: number): string {
  const d = new Date(ts);
  const now = Date.now();
  const diff = now - ts;
  if (diff < 60000) return "刚刚";
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`;
  return d.toLocaleDateString("zh-CN");
}

export function DashboardPage() {
  const { setBackendOnline, backendOnline, extractionHistory } = useAppStore();
  const [rustStatus, setRustStatus] = useState<RustStatus | null>(null);
  const [engineAvailability, setEngineAvailability] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const interval = setInterval(async () => {
      const ok = await checkHealth();
      setBackendOnline(ok);
    }, 10000);
    checkHealth().then(setBackendOnline);
    return () => clearInterval(interval);
  }, [setBackendOnline]);

  useEffect(() => {
    api("/health")
      .then((data) => {
        const d = data as Record<string, unknown>;
        setRustStatus({
          available: Boolean(d.rust_available ?? d.rust_acceleration),
          module: typeof d.rust_module === "string" ? d.rust_module : undefined,
        });
        const engines = d.engines as Record<string, boolean> | undefined;
        if (engines) setEngineAvailability(engines);
      })
      .catch(() => {
        setRustStatus({ available: false });
      });
  }, []);

  const totalExtractions = extractionHistory.length;
  const successCount = extractionHistory.filter((r) => r.success).length;
  const recentHistory = extractionHistory.slice(0, 8);

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="font-display text-3xl font-semibold mb-2">Math-Anything</h1>
      <p className="text-text-2 mb-8">计算材料科学的数学语义工作台 — 从"读数字"到"理解数学"</p>

      {/* Quick stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <div className="bg-bg-card border border-border rounded-lg p-5">
          <div className="flex items-center gap-2 mb-2">
            <Activity size={16} className="text-accent3" />
            <span className="text-xs text-text-3 uppercase tracking-wider font-semibold">后端状态</span>
          </div>
          <div className="flex items-center gap-2">
            {backendOnline ? (
              <CheckCircle2 size={18} className="text-accent3" />
            ) : (
              <XCircle size={18} className="text-error" />
            )}
            <span className={`text-lg font-semibold ${backendOnline ? "text-accent3" : "text-error"}`}>
              {backendOnline ? "在线" : "离线"}
            </span>
          </div>
        </div>

        <div className="bg-bg-card border border-border rounded-lg p-5">
          <div className="flex items-center gap-2 mb-2">
            <FlaskConical size={16} className="text-accent" />
            <span className="text-xs text-text-3 uppercase tracking-wider font-semibold">总提取次数</span>
          </div>
          <div className="text-2xl font-semibold text-text">{totalExtractions}</div>
          {totalExtractions > 0 && (
            <div className="text-xs text-text-3 mt-1">
              成功率 {Math.round((successCount / totalExtractions) * 100)}%
            </div>
          )}
        </div>

        <div className="bg-bg-card border border-border rounded-lg p-5">
          <div className="flex items-center gap-2 mb-2">
            <Cpu size={16} className="text-accent2" />
            <span className="text-xs text-text-3 uppercase tracking-wider font-semibold">可用引擎</span>
          </div>
          <div className="text-2xl font-semibold text-text">{ENGINES.length}</div>
          <div className="text-xs text-text-3 mt-1">7 个计算引擎</div>
        </div>

        <div className="bg-bg-card border border-border rounded-lg p-5">
          <div className="flex items-center gap-2 mb-2">
            <Zap size={16} className={rustStatus?.available ? "text-warn" : "text-text-3"} />
            <span className="text-xs text-text-3 uppercase tracking-wider font-semibold">Rust 加速</span>
          </div>
          <div className="flex items-center gap-2">
            {rustStatus?.available ? (
              <CheckCircle2 size={18} className="text-accent3" />
            ) : (
              <Circle size={18} className="text-text-3" />
            )}
            <span className={`text-lg font-semibold ${rustStatus?.available ? "text-accent3" : "text-text-3"}`}>
              {rustStatus?.available ? "已启用" : "未启用"}
            </span>
          </div>
          {rustStatus?.module && (
            <div className="text-xs text-text-3 mt-1">{rustStatus.module}</div>
          )}
        </div>
      </div>

      {/* Navigation cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {[
          { icon: FlaskConical, label: "结构提取", desc: "7 个引擎", to: "/extract" },
          { icon: ShieldCheck, label: "形式化验证", desc: "5 层验证", to: "/verify" },
          { icon: MessageSquare, label: "AI 对话", desc: "Agent Loop", to: "/chat" },
          { icon: CheckSquare, label: "交叉验证", desc: "3 种工具", to: "/validate" },
        ].map((card) => (
          <a
            key={card.to}
            href={card.to}
            className="group bg-bg-card border border-border rounded-lg p-5 hover:border-accent/30 transition-colors"
          >
            <card.icon size={24} className="text-accent mb-3 group-hover:scale-110 transition-transform" />
            <div className="font-medium text-sm mb-1">{card.label}</div>
            <div className="text-xs text-text-3">{card.desc}</div>
          </a>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Engine availability */}
        <div className="bg-bg-card border border-border rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <Cpu size={16} className="text-accent2" />
            <span className="text-sm font-semibold">引擎可用性</span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {ENGINES.map((eng) => {
              const checked = engineAvailability[eng.id];
              return (
                <div
                  key={eng.id}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg bg-bg-surface"
                >
                  {checked ? (
                    <CheckCircle2 size={14} className="text-accent3 shrink-0" />
                  ) : (
                    <Circle size={14} className="text-text-3 shrink-0" />
                  )}
                  <span className={`text-xs font-mono ${checked ? "text-text" : "text-text-3"}`}>
                    {eng.label}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="mt-3 text-xs text-text-3">
            {Object.values(engineAvailability).filter(Boolean).length}/{ENGINES.length} 引擎已连接
          </div>
        </div>

        {/* Recent extraction history */}
        <div className="bg-bg-card border border-border rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <Clock size={16} className="text-accent" />
            <span className="text-sm font-semibold">最近提取</span>
          </div>
          {recentHistory.length > 0 ? (
            <div className="space-y-2">
              {recentHistory.map((rec, i) => (
                <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-lg bg-bg-surface">
                  {rec.success ? (
                    <CheckCircle2 size={14} className="text-accent3 shrink-0" />
                  ) : (
                    <XCircle size={14} className="text-error shrink-0" />
                  )}
                  <span className="text-xs font-mono text-text flex-1">{rec.engine.toUpperCase()}</span>
                  <span className="text-xs text-text-3">{formatTime(rec.timestamp)}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Clock size={32} className="text-text-3 mx-auto mb-3" />
              <p className="text-text-3 text-xs">暂无提取记录</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
