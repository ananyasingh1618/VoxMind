import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Login } from "@/app/routes/Login";
import { useAuthStore } from "@/stores/authStore";

import { renderWithProviders } from "./testUtils";

describe("Login page", () => {
  beforeEach(() => {
    useAuthStore.setState({ user: null, accessToken: null, status: "unauthenticated" });
    vi.restoreAllMocks();
  });

  it("shows an error message on invalid credentials without crashing", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          data: null,
          error: { code: "invalid_credentials", message: "Incorrect email or password." },
          request_id: "test",
        }),
        { status: 401, headers: { "Content-Type": "application/json" } },
      ),
    );

    renderWithProviders(<Login />);

    await userEvent.type(screen.getByLabelText(/email/i), "user@voxmind.dev");
    await userEvent.type(screen.getByLabelText(/password/i), "wrong-password");
    await userEvent.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/incorrect email or password/i);
    expect(useAuthStore.getState().status).toBe("unauthenticated");
  });

  it("stores the session and clears the form error on successful login", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          access_token: "fake-access-token",
          token_type: "bearer",
          expires_in_seconds: 900,
          user: { id: "1", email: "user@voxmind.dev", created_at: "2026-01-01T00:00:00Z" },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );

    renderWithProviders(<Login />);

    await userEvent.type(screen.getByLabelText(/email/i), "user@voxmind.dev");
    await userEvent.type(screen.getByLabelText(/password/i), "correct-password");
    await userEvent.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => expect(useAuthStore.getState().status).toBe("authenticated"));
    expect(useAuthStore.getState().user?.email).toBe("user@voxmind.dev");
  });
});
