import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import * as audioApi from "@/api/audio";
import * as emotionApi from "@/api/emotion";
import { EmotionPanel } from "@/components/conversation/EmotionPanel";

import { renderWithProviders } from "./testUtils";

vi.mock("@/api/audio");
vi.mock("@/api/emotion");

describe("EmotionPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("prompts to select a transcript when no message is selected", () => {
    renderWithProviders(<EmotionPanel conversationId="conv-1" messageId={null} />);

    expect(screen.getByText(/speaker & emotion/i)).toBeInTheDocument();
    expect(screen.getByText(/process an audio upload/i)).toBeInTheDocument();
  });

  it("shows aligned turns with an 'analyze emotion' action once a message is selected", async () => {
    vi.mocked(audioApi.getTranscript).mockResolvedValue({
      message_id: "msg-1",
      transcript_segments: [],
      speaker_segments: [],
      aligned_turns: [
        { id: "turn-1", speaker_label: "speaker_0", start_ms: 0, end_ms: 2000, text: "I am happy about this." },
      ],
    });
    vi.mocked(emotionApi.listEmotionPredictions).mockResolvedValue([]);

    renderWithProviders(<EmotionPanel conversationId="conv-1" messageId="msg-1" />);

    expect(await screen.findByText(/i am happy about this/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /analyze emotion/i })).toBeInTheDocument();
    expect(screen.getByText(/not yet analyzed/i)).toBeInTheDocument();
  });

  it("renders genuine prediction output once available, never a hardcoded value", async () => {
    vi.mocked(audioApi.getTranscript).mockResolvedValue({
      message_id: "msg-1",
      transcript_segments: [],
      speaker_segments: [],
      aligned_turns: [
        { id: "turn-1", speaker_label: "speaker_0", start_ms: 0, end_ms: 2000, text: "I am happy about this." },
      ],
    });
    vi.mocked(emotionApi.listEmotionPredictions).mockResolvedValue([
      {
        id: "pred-1",
        aligned_turn_id: "turn-1",
        model_version_id: "model-1",
        predicted_label: "happy",
        confidence: 0.82,
        probabilities: { happy: 0.82, sad: 0.18 },
        created_at: "2026-01-01T00:00:00Z",
      },
    ]);

    renderWithProviders(<EmotionPanel conversationId="conv-1" messageId="msg-1" />);

    expect(await screen.findByText(/happy · 82%/i)).toBeInTheDocument();
    expect(screen.getByText(/not a verified fact/i)).toBeInTheDocument();
  });
});
