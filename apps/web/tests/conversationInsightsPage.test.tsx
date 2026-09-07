import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ConversationInsightsPage } from "@/app/routes/app/ConversationInsightsPage";
import * as insightsHooks from "@/features/insights/hooks";
import type { ConversationInsights } from "@/types/api";

import { renderWithProviders } from "./testUtils";

vi.mock("@/features/insights/hooks");

function mockInsights(data: ConversationInsights) {
  vi.mocked(insightsHooks.useConversationInsights).mockReturnValue({
    data,
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof insightsHooks.useConversationInsights>);
}

describe("ConversationInsightsPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows a genuine empty state for a conversation with no messages", () => {
    mockInsights({
      conversation_id: "conv-1",
      message_count: 0,
      timeline: [],
      observations: [],
      interpretations: [],
    });
    renderWithProviders(<ConversationInsightsPage />);

    expect(screen.getByText(/nothing to show yet/i)).toBeInTheDocument();
  });

  it("renders real observations with their basis, interpretations with their caveat, and the timeline", () => {
    mockInsights({
      conversation_id: "conv-1",
      message_count: 1,
      timeline: [
        {
          message_id: "msg-1",
          role: "user",
          content: "I love how well this works!",
          created_at: "2026-01-01T00:00:00Z",
          sentiment_label: "positive",
          sentiment_score: 0.95,
          intent_label: "statement",
          topics: ["works"],
          emotion: [],
          incongruence: [],
        },
      ],
      observations: [
        {
          text: "Sentiment was classified across 1 message(s) - positive: 1.",
          basis: "cardiffnlp/twitter-roberta-base-sentiment-latest (real sentiment model)",
        },
      ],
      interpretations: [
        {
          text: "Most messages carried positive sentiment, which may suggest the conversation was generally constructive.",
          caveat: "This is a descriptive, pattern-based observation - not a psychological or medical assessment.",
        },
      ],
    });

    renderWithProviders(<ConversationInsightsPage />);

    expect(screen.getByText(/sentiment was classified across 1 message/i)).toBeInTheDocument();
    expect(screen.getByText(/real sentiment model/i)).toBeInTheDocument();
    expect(screen.getByText(/not a psychological or medical assessment/i)).toBeInTheDocument();
    expect(screen.getByText(/i love how well this works/i)).toBeInTheDocument();
    expect(screen.getByText(/sentiment: positive/i)).toBeInTheDocument();
  });
});
