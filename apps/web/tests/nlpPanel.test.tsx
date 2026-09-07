import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import * as nlpApi from "@/api/nlp";
import { NlpPanel } from "@/components/conversation/NlpPanel";

import { renderWithProviders } from "./testUtils";

vi.mock("@/api/nlp");

describe("NlpPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("prompts to select a message when none is selected", () => {
    renderWithProviders(<NlpPanel conversationId="conv-1" messageId={null} />);

    expect(screen.getByText(/language understanding/i)).toBeInTheDocument();
    expect(screen.getAllByText(/select a message/i).length).toBeGreaterThan(0);
  });

  it("shows 'not yet analyzed' before any annotation exists", async () => {
    vi.mocked(nlpApi.getNlp).mockResolvedValue(null);

    renderWithProviders(<NlpPanel conversationId="conv-1" messageId="msg-1" />);

    expect(await screen.findByText(/not yet analyzed/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^analyze$/i })).toBeInTheDocument();
  });

  it("renders the genuinely returned sentiment/intent/topics/entities, never hardcoded", async () => {
    vi.mocked(nlpApi.getNlp).mockResolvedValue({
      id: "ann-1",
      message_id: "msg-1",
      sentiment_label: "positive",
      sentiment_score: 0.91,
      intent_label: "question",
      intent_confidence: 1.0,
      topics: ["retrieval", "pipeline"],
      entities: [{ text: "VoxMind", label: "ORG", start_char: 0, end_char: 7 }],
      model_versions: { sentiment: "cardiffnlp/twitter-roberta-base-sentiment-latest", ner: "dslim/bert-base-NER" },
      created_at: "2026-01-01T00:00:00Z",
    });

    renderWithProviders(<NlpPanel conversationId="conv-1" messageId="msg-1" />);

    expect(await screen.findByText(/positive/i)).toBeInTheDocument();
    // The real 91% score is now shown twice by design (inline text + the
    // Phase 8 ScoreBar visualization) - both must reflect the same real value.
    expect(screen.getAllByText(/91%/).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/question/i)).toBeInTheDocument();
    expect(screen.getByText(/retrieval, pipeline/i)).toBeInTheDocument();
    expect(screen.getByText(/VoxMind \(ORG\)/i)).toBeInTheDocument();
  });
});
