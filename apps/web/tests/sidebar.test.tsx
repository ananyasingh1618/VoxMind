import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import * as conversationsApi from "@/api/conversations";
import { Sidebar } from "@/components/layout/Sidebar";

vi.mock("@/api/conversations");
vi.mock("@/components/layout/UserMenu", () => ({ UserMenu: () => null }));

function renderSidebar() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/app/conversations"]}>
        <Routes>
          <Route path="/app/conversations" element={<Sidebar />} />
          <Route
            path="/app/conversations/:id"
            element={<div data-testid="navigated-to-conversation" />}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Sidebar", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.mocked(conversationsApi.listConversations).mockResolvedValue([]);
  });

  it("navigates to the newly created conversation after clicking 'New conversation'", async () => {
    // Regression test for a real bug found via a real browser walkthrough
    // (Playwright): this button used to call the create mutation with
    // fire-and-forget `.mutate()` and never navigated anywhere, unlike the
    // identical action on ConversationsListPage.tsx - a real conversation
    // was always created on the backend, the user was just never taken to
    // it. Must behave the same way ConversationsListPage's own "New
    // conversation" button already correctly does.
    vi.mocked(conversationsApi.createConversation).mockResolvedValue({
      id: "new-conv-id",
      title: null,
      status: "active",
      created_at: "2026-01-01T00:00:00Z",
      ended_at: null,
    });

    renderSidebar();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /new conversation/i }));

    await waitFor(() => expect(conversationsApi.createConversation).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByTestId("navigated-to-conversation")).toBeInTheDocument());
  });
});
