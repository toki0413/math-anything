import { useEffect } from "react";
import { api, checkHealth } from "../lib/api";
import { useAppStore } from "../stores/appStore";
import { FlaskConical, ShieldCheck, MessageSquare, CheckSquare, Zap } from "lucide-react";

export function DashboardPage() {
  const { setBackendOnline, currentSchema } = useAppStore();

  useEffect(() => {
    const interval = setInterval(async () => {
      const ok = await checkHealth();
      setBackendOnline(ok);
    }, 10000);
    checkHealth().then(setBackendOnline);
    return () => clearInterval(interval);
  }, [setBackendOnline]);

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="font-display text-3xl font-semibold mb-2">
        Math-Anything
      </h1>
      <p className="text-text-2 mb-8">
        计算材料科学的数学语义工作台 — 从"读数字"到"理解数学"
      </p>

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
            <card.icon
              size={24}
              className="text-accent mb-3 group-hover:scale-110 transition-transform"
            />
            <div className="font-medium text-sm mb-1">{card.label}</div>
            <div className="text-xs text-text-3">{card.desc}</div>
          </a>
        ))}
      </div>

      {currentSchema && (
        <div className="bg-bg-card border border-border rounded-lg p-6">
          <h2 className="font-display text-lg font-semibold mb-3">
            当前 Schema
          </h2>
          <pre className="text-xs font-mono text-text-2 overflow-auto max-h-64">
            {JSON.stringify(currentSchema, null, 2)}
          </pre>
        </div>
      )}

      {!currentSchema && (
        <div className="bg-bg-card border border-border rounded-lg p-12 text-center">
          <Zap size={40} className="text-text-3 mx-auto mb-4" />
          <p className="text-text-2 text-sm">
            上传计算科学输入文件开始分析
          </p>
          <a
            href="/extract"
            className="inline-block mt-4 px-5 py-2 bg-accent text-bg font-medium rounded-lg text-sm hover:bg-accent/90 transition-colors"
          >
            开始提取
          </a>
        </div>
      )}
    </div>
  );
}
