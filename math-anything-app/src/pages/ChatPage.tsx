import { useState, useRef, useCallback, useEffect } from "react";
import { MessageSquare, Send, Loader2, ChevronDown, ChevronRight, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { api, wsUrl } from "../lib/api";
import { useAppStore } from "../stores/appStore";
import clsx from "clsx";

interface ToolCallInfo {
  toolName: string;
  toolUseId: string;
  args: Record<string, unknown>;
  result?: string;
  display?: string;
  success?: boolean;
  expanded: boolean;
}

interface Message {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  toolCalls?: ToolCallInfo[];
}

let msgCounter = 0;
function nextId() { return `msg-${++msgCounter}`; }

export function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showSchema, setShowSchema] = useState(false);
  const [firewallOn, setFirewallOn] = useState<boolean | null>(null);
  const currentSchema = useAppStore((s) => s.currentSchema);
  const scrollRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    api<{ enabled: boolean }>("/firewall/status")
      .then((d) => setFirewallOn(d.enabled))
      .catch(() => setFirewallOn(null));
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const appendToLastAssistant = useCallback((text: string) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last?.role === "assistant") {
        return [...prev.slice(0, -1), { ...last, content: last.content + text }];
      }
      return [...prev, { id: nextId(), role: "assistant", content: text }];
    });
  }, []);

  const addToolCall = useCallback((toolName: string, toolUseId: string, args: Record<string, unknown>) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      const tc: ToolCallInfo = { toolName, toolUseId, args, expanded: false };
      if (last?.role === "assistant") {
        return [...prev.slice(0, -1), { ...last, toolCalls: [...(last.toolCalls || []), tc] }];
      }
      return [...prev, { id: nextId(), role: "assistant", content: "", toolCalls: [tc] }];
    });
  }, []);

  const updateToolResult = useCallback((toolUseId: string, display: string, success: boolean, result?: string) => {
    setMessages((prev) =>
      prev.map((msg) => {
        if (!msg.toolCalls) return msg;
        return {
          ...msg,
          toolCalls: msg.toolCalls.map((tc) =>
            tc.toolUseId === toolUseId ? { ...tc, display, success, result, expanded: false } : tc
          ),
        };
      })
    );
  }, []);

  const toggleToolExpand = useCallback((msgId: string, toolUseId: string) => {
    setMessages((prev) =>
      prev.map((msg) => {
        if (msg.id !== msgId || !msg.toolCalls) return msg;
        return {
          ...msg,
          toolCalls: msg.toolCalls.map((tc) =>
            tc.toolUseId === toolUseId ? { ...tc, expanded: !tc.expanded } : tc
          ),
        };
      })
    );
  }, []);

  const send = async () => {
    if (!input.trim() || loading) return;
    const userMsg: Message = { id: nextId(), role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    const userInput = input;
    setInput("");
    setLoading(true);

    try {
      const url = await wsUrl("/ws/agent");
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({
          type: "user_input",
          content: userInput,
          context: { schema_context: currentSchema },
        }));
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch (data.type) {
          case "text_delta":
            appendToLastAssistant(data.text);
            break;
          case "tool_call_start":
            addToolCall(data.tool_name, data.tool_use_id, data.args || {});
            break;
          case "tool_result":
            updateToolResult(
              data.tool_use_id,
              data.display || "",
              data.success ?? true,
              typeof data.data === "string" ? data.data : JSON.stringify(data.data, null, 2)
            );
            break;
          case "error":
            appendToLastAssistant(`\n\n⚠️ Error: ${data.error}`);
            break;
          case "done":
            ws.close();
            wsRef.current = null;
            setLoading(false);
            break;
        }
      };

      ws.onerror = () => {
        setMessages((prev) => [
          ...prev,
          { id: nextId(), role: "assistant", content: "WebSocket 连接失败。请确保后端正在运行。" },
        ]);
        setLoading(false);
        wsRef.current = null;
      };

      setTimeout(() => {
        if (ws.readyState !== WebSocket.CLOSED) {
          ws.close();
          wsRef.current = null;
          setLoading(false);
        }
      }, 120000);
    } catch {
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "assistant", content: "无法连接到后端。请检查服务器是否运行。" },
      ]);
      setLoading(false);
    }
  };

  const cancelRequest = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "cancel" }));
      wsRef.current.close();
    }
    wsRef.current = null;
    setLoading(false);
  };

  return (
    <div className="flex h-full">
      <div className="flex-1 flex flex-col min-w-0">
        <div className="px-8 pt-6 pb-4 border-b border-border">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="font-display text-2xl font-semibold">AI 对话</h1>
              <p className="text-text-2 text-sm">与 Math-Anything Agent 对话，自动调用数学工具</p>
            </div>
            {firewallOn !== null && (
              <span className="text-sm" title={firewallOn ? "数据防火墙已开启" : "数据防火墙已关闭"}>
                {firewallOn ? "🛡️" : "🔓"}
              </span>
            )}
          </div>
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto px-8 py-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-center pt-20">
              <MessageSquare size={40} className="text-text-3 mx-auto mb-4" />
              <p className="text-text-3 text-sm">输入问题开始对话</p>
              <p className="text-text-3 text-xs mt-2">
                例如: "分析这个 VASP INCAR 文件的数学结构"
              </p>
              <div className="mt-6 flex flex-wrap gap-2 justify-center max-w-lg mx-auto">
                {[
                  "提取 VASP 的数学结构",
                  "验证 LAMMPS 参数约束",
                  "对比两个 Schema",
                  "生成数学命题",
                ].map((hint) => (
                  <button
                    key={hint}
                    onClick={() => setInput(hint)}
                    className="px-3 py-1.5 bg-bg-card border border-border rounded-lg text-xs text-text-2 hover:border-accent/40 hover:text-accent transition-colors"
                  >
                    {hint}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={clsx("flex", msg.role === "user" ? "justify-end" : "justify-start")}
            >
              <div
                className={clsx(
                  "max-w-[75%] rounded-lg px-4 py-3 text-sm",
                  msg.role === "user"
                    ? "bg-accent-dim text-accent"
                    : "bg-bg-card border border-border text-text"
                )}
              >
                {msg.content && (
                  <div className="prose prose-sm prose-invert max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                      {msg.content}
                    </ReactMarkdown>
                  </div>
                )}
                {msg.toolCalls && msg.toolCalls.length > 0 && (
                  <div className="mt-2 space-y-2">
                    {msg.toolCalls.map((tc) => (
                      <div
                        key={tc.toolUseId}
                        className={clsx(
                          "rounded-lg border px-3 py-2 text-xs",
                          tc.success === undefined
                            ? "border-accent2/30 bg-accent2-dim"
                            : tc.success
                            ? "border-accent3/30 bg-accent3-dim"
                            : "border-error/30 bg-error-dim"
                        )}
                      >
                        <button
                          onClick={() => toggleToolExpand(msg.id, tc.toolUseId)}
                          className="flex items-center gap-1 w-full text-left"
                        >
                          {tc.expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                          <span className="font-mono font-semibold uppercase tracking-wider">
                            {tc.toolName}
                          </span>
                          {tc.display && (
                            <span className="ml-2 text-text-3 truncate">{tc.display}</span>
                          )}
                          {tc.success !== undefined && (
                            <span className={clsx("ml-auto", tc.success ? "text-accent3" : "text-error")}>
                              {tc.success ? "✓" : "✗"}
                            </span>
                          )}
                        </button>
                        {tc.expanded && (
                          <div className="mt-2 space-y-1">
                            <div>
                              <span className="text-text-3 font-semibold">Input:</span>
                              <pre className="mt-1 p-2 bg-bg-surface rounded text-[10px] overflow-x-auto">
                                {JSON.stringify(tc.args, null, 2)}
                              </pre>
                            </div>
                            {tc.result && (
                              <div>
                                <span className="text-text-3 font-semibold">Output:</span>
                                <pre className="mt-1 p-2 bg-bg-surface rounded text-[10px] overflow-x-auto max-h-48">
                                  {tc.result.length > 2000 ? tc.result.slice(0, 2000) + "..." : tc.result}
                                </pre>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && !messages[messages.length - 1]?.content && messages[messages.length - 1]?.role !== "assistant" && (
            <div className="text-text-3 text-sm animate-pulse flex items-center gap-2">
              <Loader2 size={14} className="animate-spin" /> 思考中...
            </div>
          )}
        </div>

        <div className="px-8 py-4 border-t border-border">
          <div className="flex gap-3">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
              placeholder="输入问题... (支持 LaTeX: $E=mc^2$)"
              className="flex-1 bg-bg-card border border-border rounded-lg px-4 py-2.5 text-sm text-text focus:outline-none focus:border-accent/50"
            />
            {loading ? (
              <button
                onClick={cancelRequest}
                className="px-4 py-2.5 bg-error text-bg rounded-lg hover:bg-error/90 transition-colors"
              >
                <X size={16} />
              </button>
            ) : (
              <button
                onClick={send}
                disabled={!input.trim()}
                className="px-4 py-2.5 bg-accent text-bg rounded-lg hover:bg-accent/90 disabled:opacity-50 transition-colors"
              >
                <Send size={16} />
              </button>
            )}
          </div>
        </div>
      </div>

      {currentSchema && (
        <div className={clsx(
          "border-l border-border bg-bg-surface transition-all duration-200 overflow-y-auto",
          showSchema ? "w-72" : "w-10"
        )}>
          <button
            onClick={() => setShowSchema(!showSchema)}
            className="w-full px-3 py-2 text-xs text-text-3 hover:text-text-2 border-b border-border flex items-center gap-1"
          >
            {showSchema ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            Schema
          </button>
          {showSchema && (
            <div className="p-3">
              <pre className="text-[10px] text-text-2 whitespace-pre-wrap overflow-x-auto">
                {JSON.stringify(currentSchema, null, 2).slice(0, 3000)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
