import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface ExtractionRecord {
  engine: string;
  timestamp: number;
  success: boolean;
}

interface AppState {
  backendOnline: boolean;
  setBackendOnline: (v: boolean) => void;
  currentSchema: Record<string, unknown> | null;
  setCurrentSchema: (s: Record<string, unknown> | null) => void;
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
  hasApiKey: boolean;
  llmConfig: { provider: string; apiKey: string; baseUrl: string; model: string };
  setLlmConfig: (c: Partial<AppState["llmConfig"]>) => void;
  extractionHistory: ExtractionRecord[];
  addExtractionRecord: (r: ExtractionRecord) => void;
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
      hasApiKey: false,
      llmConfig: {
        provider: "openai",
        apiKey: "",
        baseUrl: "",
        model: "",
      },
      setLlmConfig: (c) =>
        set((s) => {
          const newConfig = { ...s.llmConfig, ...c };
          return {
            llmConfig: newConfig,
            hasApiKey: newConfig.apiKey !== "",
          };
        }),
      extractionHistory: [],
      addExtractionRecord: (r) =>
        set((s) => ({
          extractionHistory: [r, ...s.extractionHistory].slice(0, 50),
        })),
    }),
    {
      name: "math-anything-store",
      partialize: (state) => ({
        sidebarCollapsed: state.sidebarCollapsed,
        hasApiKey: state.hasApiKey,
        llmConfig: {
          provider: state.llmConfig.provider,
          baseUrl: state.llmConfig.baseUrl,
          model: state.llmConfig.model,
        },
        extractionHistory: state.extractionHistory,
      }),
    }
  )
);
