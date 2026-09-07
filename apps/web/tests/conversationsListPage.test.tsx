import { screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import * as conversationsApi from "@/api/conversations";
import { ConversationsListPage } from "@/app/routes/app/ConversationsListPage";

import { renderWithProviders } from "./testUtils";

vi.mock("@/api/conversations");

describe("ConversationsListPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows an empty state when there are no conversations", async () => {
    vi.mocked(conversationsApi.listConversations).mockResolvedValue([]);

    renderWithProviders(<ConversationsListPage />);

    expect(await screen.findByText(/no conversations yet/i)).toBeInTheDocument();
  });

  it("shows an error state when the request fails", async () => {
    vi.mocked(conversationsApi.listConversations).mockRejectedValue(new Error("network down"));

    renderWithProviders(<ConversationsListPage />);

    expect(await screen.findByText(/couldn't load your conversations/i)).toBeInTheDocument();
  });

  it("renders the list when conversations exist", async () => {
    vi.mocked(conversationsApi.listConversations).mockResolvedValue([
      {
        id: "conv-1",
        title: "My first session",
        status: "active",
        created_at: "2026-01-01T00:00:00Z",
        ended_at: null,
      },
    ]);

    renderWithProviders(<ConversationsListPage />);

    await waitFor(() => expect(screen.getByText("My first session")).toBeInTheDocument());
  });
});
