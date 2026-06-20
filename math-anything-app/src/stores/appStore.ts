import { create } from "zustand";
import { persist } from "zustand/middleware";

interface AppState {
  backendOnline: boolean;
  setBackendOnline: (v: boolean) => void;
  currentSchema: Record<string, unknown> | null;
  setCurrentSchema: (s: Record<string, unknown> | null) => void;
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
  llmConfig: { provider: string; apiKey: string; baseUrl: string; model: string };
  setLlmConfig: (c: Partial<AppState["llmConfig"]>) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      backendOnline: false,
      setBackendOnline: (v) => set({ backendOnline: v }),
      currentSchema: null,
      setCurrentSchema: (s) => set({ currentSchema: s }),
      sidebarCollapsed: false,
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      llmConfig: {
        provider: "openai",
        apiKey: "",
        baseUrl: "",
        model: "",
      },
      setLlmConfig: (c) =>
        set((s) => ({ llmConfig: { ...s.llmConfig, ...c } })),
    }),
    {
      name: "math-anything-store",
      partialize: (state) => ({
        sidebarCollapsed: state.sidebarCollapsed,
        llmConfig: state.llmConfig,
        currentSchema: state.currentSchema,
      }),
    }
  )
);
