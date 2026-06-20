import { Outlet, NavLink } from "react-router-dom";
import {
  Home,
  FlaskConical,
  ShieldCheck,
  MessageSquare,
  CheckSquare,
  Waves,
  Triangle,
  TreePine,
  Settings,
  History,
  ChevronLeft,
  ChevronRight,
  Circle,
  Sigma,
  Terminal,
} from "lucide-react";
import { useAppStore } from "../../stores/appStore";
import clsx from "clsx";

const NAV_ITEMS = [
  { to: "/", icon: Home, label: "Dashboard", group: "Observe" },
  { to: "/extract", icon: FlaskConical, label: "结构提取", group: "Extract" },
  { to: "/emergence", icon: Waves, label: "涌现分析", group: "Extract" },
  { to: "/geometry", icon: Triangle, label: "几何提取", group: "Extract" },
  { to: "/analysis", icon: Sigma, label: "数学分析", group: "Extract" },
  { to: "/sandbox", icon: Terminal, label: "沙箱", group: "Extract" },
  { to: "/verify", icon: ShieldCheck, label: "形式化验证", group: "Reason" },
  { to: "/chat", icon: MessageSquare, label: "AI 对话", group: "Reason" },
  { to: "/validate", icon: CheckSquare, label: "交叉验证", group: "Validate" },
  { to: "/schema/current", icon: TreePine, label: "Schema 查看器", group: "Inspect" },
  { to: "/history", icon: History, label: "历史记录", group: "Observe" },
  { to: "/settings", icon: Settings, label: "设置", group: "System" },
];

export function AppLayout() {
  const { sidebarCollapsed, toggleSidebar, backendOnline } = useAppStore();

  let currentGroup = "";
  return (
    <div className="flex h-screen overflow-hidden bg-bg">
      <aside
        className={clsx(
          "flex flex-col border-r border-border bg-bg-surface transition-all duration-200",
          sidebarCollapsed ? "w-16" : "w-56"
        )}
      >
        <div className="flex items-center gap-2 px-4 h-14 border-b border-border">
          <span className="text-accent font-display text-xl font-semibold">
            ∞
          </span>
          {!sidebarCollapsed && (
            <span className="font-display text-lg font-semibold text-text">
              Math-Anything
            </span>
          )}
        </div>

        <nav className="flex-1 overflow-y-auto py-2">
          {NAV_ITEMS.map((item) => {
            const showGroup = item.group !== currentGroup;
            currentGroup = item.group;
            return (
              <div key={item.to}>
                {showGroup && !sidebarCollapsed && (
                  <div className="px-4 pt-4 pb-1 text-[10px] font-semibold uppercase tracking-widest text-text-3">
                    {item.group}
                  </div>
                )}
                <NavLink
                  to={item.to}
                  end={item.to === "/"}
                  className={({ isActive }) =>
                    clsx(
                      "flex items-center gap-3 mx-2 px-3 py-2 rounded-lg text-sm transition-colors",
                      isActive
                        ? "bg-accent-dim text-accent font-medium"
                        : "text-text-2 hover:bg-bg-hover hover:text-text"
                    )
                  }
                >
                  <item.icon size={18} />
                  {!sidebarCollapsed && <span>{item.label}</span>}
                </NavLink>
              </div>
            );
          })}
        </nav>

        <div className="border-t border-border p-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Circle
              size={8}
              className={backendOnline ? "fill-accent3 text-accent3" : "fill-error text-error"}
            />
            {!sidebarCollapsed && (
              <span className="text-xs text-text-3">
                {backendOnline ? "Backend Online" : "Backend Offline"}
              </span>
            )}
          </div>
          <button
            onClick={toggleSidebar}
            className="p-1 rounded hover:bg-bg-hover text-text-3"
          >
            {sidebarCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
          </button>
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
