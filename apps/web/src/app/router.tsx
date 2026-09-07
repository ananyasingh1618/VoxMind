import { Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "@/components/layout/AppShell";
import { ProtectedRoute, PublicOnlyRoute } from "@/app/ProtectedRoute";
import { AnalyticsDashboardPage } from "@/app/routes/app/AnalyticsDashboardPage";
import { ConversationInsightsPage } from "@/app/routes/app/ConversationInsightsPage";
import { ConversationsListPage } from "@/app/routes/app/ConversationsListPage";
import { ConversationWorkspacePage } from "@/app/routes/app/ConversationWorkspacePage";
import { EvaluationDashboardPage } from "@/app/routes/app/EvaluationDashboardPage";
import { SettingsPage } from "@/app/routes/app/SettingsPage";
import { Landing } from "@/app/routes/Landing";
import { Login } from "@/app/routes/Login";
import { Register } from "@/app/routes/Register";
import { useBootstrapSession } from "@/features/auth/hooks";

export function AppRoutes() {
  useBootstrapSession();

  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route
        path="/login"
        element={
          <PublicOnlyRoute>
            <Login />
          </PublicOnlyRoute>
        }
      />
      <Route
        path="/register"
        element={
          <PublicOnlyRoute>
            <Register />
          </PublicOnlyRoute>
        }
      />
      <Route
        path="/app"
        element={
          <ProtectedRoute>
            <AppShell />
          </ProtectedRoute>
        }
      >
        <Route index element={<Navigate to="conversations" replace />} />
        <Route path="conversations" element={<ConversationsListPage />} />
        <Route path="conversations/:id" element={<ConversationWorkspacePage />} />
        <Route path="conversations/:id/insights" element={<ConversationInsightsPage />} />
        <Route path="analytics" element={<AnalyticsDashboardPage />} />
        <Route path="evaluation" element={<EvaluationDashboardPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
