import { Routes, Route, Navigate } from "react-router-dom";
import { AppLayout } from "./components/layout/AppLayout";
import { DashboardPage } from "./pages/DashboardPage";
import { ExtractPage } from "./pages/ExtractPage";
import { VerifyPage } from "./pages/VerifyPage";
import { ChatPage } from "./pages/ChatPage";
import { ValidatePage } from "./pages/ValidatePage";
import { SchemaPage } from "./pages/SchemaPage";
import { SettingsPage } from "./pages/SettingsPage";
import { HistoryPage } from "./pages/HistoryPage";
import { EmergencePage } from "./pages/EmergencePage";
import { GeometryPage } from "./pages/GeometryPage";
import { AnalysisPage } from "./pages/AnalysisPage";
import SandboxPage from "./pages/SandboxPage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<DashboardPage />} />
        <Route path="extract" element={<ExtractPage />} />
        <Route path="verify" element={<VerifyPage />} />
        <Route path="chat" element={<ChatPage />} />
        <Route path="validate" element={<ValidatePage />} />
        <Route path="emergence" element={<EmergencePage />} />
        <Route path="geometry" element={<GeometryPage />} />
        <Route path="analysis" element={<AnalysisPage />} />
        <Route path="sandbox" element={<SandboxPage />} />
        <Route path="schema/:id" element={<SchemaPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
