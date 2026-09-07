import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import * as incongruenceApi from "@/api/incongruence";
import { IncongruencePanel } from "@/components/conversation/IncongruencePanel";

import { renderWithProviders } from "./testUtils";

vi.mock("@/api/incongruence");

describe("IncongruencePanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("never frames the signal as a lie detector, even in the empty-selection state", () => {
    renderWithProviders(<IncongruencePanel conversationId="conv-1" messageId={null} />);

    expect(screen.getByText(/tone & word alignment/i)).toBeInTheDocument();
    expect(screen.getByText(/not a lie detector/i)).toBeInTheDocument();
  });

  it("renders a genuinely returned incongruence signal", async () => {
    vi.mocked(incongruenceApi.listIncongruence).mockResolvedValue([
      {
        id: "sig-1",
        aligned_turn_id: "turn-1",
        incongruence_score: 0.82,
        semantic_signal: { sentiment_label: "positive", sentiment_score: 0.9 },
        vocal_signal: { predicted_label: "sad", confidence: 0.7 },
        confidence: 0.7,
        explanation:
          "The words carried positive sentiment while the vocal tone was sad, showing a notable " +
          "divergence. This is an analytical signal only - it does not indicate deception.",
        signal_category: "analytical_not_diagnostic",
        created_at: "2026-01-01T00:00:00Z",
      },
    ]);

    renderWithProviders(<IncongruencePanel conversationId="conv-1" messageId="msg-1" />);

    expect(await screen.findByText(/incongruence: 82%/i)).toBeInTheDocument();
    expect(screen.getByText(/does not indicate deception/i)).toBeInTheDocument();
  });
});
