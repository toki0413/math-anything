import { useState, useEffect } from "react";
import { History, Loader2, BarChart3, AlertTriangle } from "lucide-react";
import { api } from "../lib/api";

interface FlywheelStats {
  total_operations: number;
  success_rate: number;
  engine_performance: Record<string, { count: number; success_rate: number }>;
}

export function HistoryPage() {
  const [stats, setStats] = useState<FlywheelStats | null>(null);
  const [degraded, setDegraded] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const data = await api("/flywheel/stats") as Record<string, unknown>;
        setStats(((data as Record<string, unknown>).stats || data) as unknown as FlywheelStats);
        setDegraded((data as Record<string, Record<string, boolean>>).degraded_engines || {});
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center h-64">
        <Loader2 size={24} className="animate-spin text-text-3" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 max-w-5xl mx-auto">
        <div className="px-4 py-2 bg-error-dim text-error text-sm rounded-lg border border-error/20">
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="font-display text-2xl font-semibold mb-1">历史记录</h1>
      <p className="text-text-2 text-sm mb-6">数据飞轮统计与引擎性能监控</p>

      {stats && (
        <div className="space-y-6">
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-bg-card border border-border rounded-lg p-4">
              <div className="text-xs text-text-3 mb-1">总操作数</div>
              <div className="text-2xl font-semibold text-text">{stats.total_operations}</div>
            </div>
            <div className="bg-bg-card border border-border rounded-lg p-4">
              <div className="text-xs text-text-3 mb-1">成功率</div>
              <div className="text-2xl font-semibold text-accent3">
                {(stats.success_rate * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-bg-card border border-border rounded-lg p-4">
              <div className="text-xs text-text-3 mb-1">引擎数</div>
              <div className="text-2xl font-semibold text-text">
                {Object.keys(stats.engine_performance).length}
              </div>
            </div>
          </div>

          <div className="bg-bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 size={16} className="text-accent" />
              <span className="text-sm font-semibold">引擎性能</span>
            </div>
            <div className="space-y-3">
              {Object.entries(stats.engine_performance).map(([engine, perf]) => (
                <div key={engine} className="flex items-center gap-4">
                  <div className="w-24 text-sm font-mono text-text-2">{engine}</div>
                  <div className="flex-1 h-2 bg-bg-surface rounded-full overflow-hidden">
                    <div
                      className="h-full bg-accent3 rounded-full transition-all"
                      style={{ width: `${perf.success_rate * 100}%` }}
                    />
                  </div>
                  <div className="w-16 text-xs text-text-3 text-right">
                    {(perf.success_rate * 100).toFixed(0)}% ({perf.count})
                  </div>
                  {degraded[engine] && (
                    <AlertTriangle size={14} className="text-warn" />
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {!stats && (
        <div className="text-center pt-20">
          <History size={40} className="text-text-3 mx-auto mb-4" />
          <p className="text-text-3 text-sm">暂无历史数据</p>
        </div>
      )}
    </div>
  );
}
