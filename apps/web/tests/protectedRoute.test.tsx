import { screen } from "@testing-library/react";
import { Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it } from "vitest";

import { ProtectedRoute } from "@/app/ProtectedRoute";
import { useAuthStore } from "@/stores/authStore";

import { renderWithProviders } from "./testUtils";

function ProtectedContent() {
  return <p>Secret conversation content</p>;
}

function LoginStub() {
  return <p>Login page</p>;
}

function renderProtected() {
  return renderWithProviders(
    <Routes>
      <Route path="/login" element={<LoginStub />} />
      <Route
        path="/app"
        element={
          <ProtectedRoute>
            <ProtectedContent />
          </ProtectedRoute>
        }
      />
    </Routes>,
    { route: "/app" },
  );
}

describe("ProtectedRoute", () => {
  beforeEach(() => {
    useAuthStore.setState({ user: null, accessToken: null, status: "checking" });
  });

  it("shows a loading state while the session is being checked", () => {
    renderProtected();
    expect(screen.getByText(/checking your session/i)).toBeInTheDocument();
  });

  it("redirects to /login when unauthenticated", () => {
    useAuthStore.setState({ status: "unauthenticated" });
    renderProtected();
    expect(screen.getByText(/login page/i)).toBeInTheDocument();
  });

  it("renders the protected content when authenticated", () => {
    useAuthStore.setState({
      status: "authenticated",
      user: { id: "1", email: "user@voxmind.dev", created_at: "2026-01-01T00:00:00Z" },
      accessToken: "token",
    });
    renderProtected();
    expect(screen.getByText(/secret conversation content/i)).toBeInTheDocument();
  });
});
