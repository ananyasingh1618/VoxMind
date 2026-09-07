import type { ReactElement } from "react";
import { Navigate, useLocation } from "react-router-dom";

import { LoadingPanel } from "@/components/ui/StatePanel";
import { useAuthStore } from "@/stores/authStore";

export function ProtectedRoute({ children }: { children: ReactElement }) {
  const status = useAuthStore((s) => s.status);
  const location = useLocation();

  if (status === "checking") {
    return <LoadingPanel label="Checking your session…" />;
  }

  if (status === "unauthenticated") {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  return children;
}

export function PublicOnlyRoute({ children }: { children: ReactElement }) {
  const status = useAuthStore((s) => s.status);

  if (status === "authenticated") {
    return <Navigate to="/app/conversations" replace />;
  }

  return children;
}
