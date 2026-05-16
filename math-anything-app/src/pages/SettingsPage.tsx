import { useState, useEffect } from "react";
import { api } from "../lib/api";
import { useAppStore } from "../stores/appStore";
import { Settings as SettingsIcon, Key, Server, Shield } from "lucide-react";

export function SettingsPage() {
  const { llmConfig, setLlmConfig } = useAppStore();
  const [lean4Available, setLean4Available] = useState<boolean | null>(null);
  const [firewallEnabled, setFirewallEnabled] = useState<boolean | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    api<{ enabled: boolean }>("/firewall/status")
      .then((d) => setFirewallEnabled(d.enabled))
      .catch(() => setFirewallEnabled(null));
  }, []);

  const toggleFirewall = async () => {
    try {
      const d = await api<{ enabled: boolean }>("/firewall/toggle", { method: "POST" });
      setFirewallEnabled(d.enabled);
    } catch {}
  };

  const checkLean4 = async () => {
    try {
      const data = await api<{ available: boolean }>("/lean4/status");
      setLean4Available(data.available);
    } catch {
      setLean4Available(false);
    }
  };

  const saveConfig = async () => {
    setSaving(true);
    try {
      await api("/config", {
        method: "POST",
        body: JSON.stringify(llmConfig),
      });
    } catch {}
    setSaving(false);
  };

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <h1 className="font-display text-2xl font-semibold mb-1">设置</h1>
      <p className="text-text-2 text-sm mb-6">LLM 配置 · Lean4 · 引擎管理</p>

      <div className="space-y-6">
        <div className="bg-bg-card border border-border rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <Key size={18} className="text-accent" />
            <h2 className="font-semibold">LLM API 配置</h2>
          </div>
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-text-3 mb-1">Provider</label>
              <select
                value={llmConfig.provider}
                onChange={(e) => setLlmConfig({ provider: e.target.value })}
                className="w-full bg-bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text focus:outline-none focus:border-accent/50"
              >
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="custom">Custom</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-text-3 mb-1">API Key</label>
              <input
                type="password"
                value={llmConfig.apiKey}
                onChange={(e) => setLlmConfig({ apiKey: e.target.value })}
                className="w-full bg-bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text focus:outline-none focus:border-accent/50"
              />
            </div>
            <div>
              <label className="block text-xs text-text-3 mb-1">Base URL (可选)</label>
              <input
                value={llmConfig.baseUrl}
                onChange={(e) => setLlmConfig({ baseUrl: e.target.value })}
                placeholder="https://api.openai.com/v1"
                className="w-full bg-bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text focus:outline-none focus:border-accent/50"
              />
            </div>
            <div>
              <label className="block text-xs text-text-3 mb-1">Model</label>
              <input
                value={llmConfig.model}
                onChange={(e) => setLlmConfig({ model: e.target.value })}
                placeholder="gpt-4o / claude-sonnet-4-20250514"
                className="w-full bg-bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text focus:outline-none focus:border-accent/50"
              />
            </div>
            <button
              onClick={saveConfig}
              disabled={saving}
              className="px-4 py-2 bg-accent text-bg rounded-lg text-sm font-medium hover:bg-accent/90 disabled:opacity-50 transition-colors"
            >
              {saving ? "保存中..." : "保存配置"}
            </button>
          </div>
        </div>

        <div className="bg-bg-card border border-border rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <Server size={18} className="text-accent2" />
            <h2 className="font-semibold">Lean4 状态</h2>
          </div>
          <button
            onClick={checkLean4}
            className="px-4 py-2 bg-bg-surface border border-border rounded-lg text-sm text-text-2 hover:border-accent/30 transition-colors"
          >
            检查 Lean4 可用性
          </button>
          {lean4Available !== null && (
            <p className={`mt-3 text-sm ${lean4Available ? "text-accent3" : "text-error"}`}>
              Lean4 {lean4Available ? "可用 ✓" : "不可用 ✗"}
            </p>
          )}
        </div>

        <div className="bg-bg-card border border-border rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <Shield size={18} className="text-accent" />
            <h2 className="font-semibold">数据安全防火墙</h2>
          </div>
          <p className="text-sm text-text-2 mb-4">
            启用后，发送至 LLM 的数值参数将被替换为占位符，防止数据泄露。
          </p>
          <div className="flex items-center gap-4">
            <button
              onClick={toggleFirewall}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                firewallEnabled
                  ? "bg-accent3-dim text-accent3 border border-accent3/30"
                  : "bg-bg-surface border border-border text-text-2"
              }`}
            >
              {firewallEnabled ? "🛡️ 已开启" : "🔓 已关闭"}
            </button>
            {firewallEnabled === null && (
              <span className="text-xs text-text-3">加载中...</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
