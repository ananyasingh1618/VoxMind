import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { VoiceSessionControl } from "@/components/voice/VoiceSessionControl";
import * as useVoiceTurnModule from "@/features/voice/useVoiceTurn";

import { renderWithProviders } from "./testUtils";

vi.mock("@/features/voice/useVoiceTurn");

function mockVoiceTurn(overrides: Partial<ReturnType<typeof useVoiceTurnModule.useVoiceTurn>> = {}) {
  const base: ReturnType<typeof useVoiceTurnModule.useVoiceTurn> = {
    state: "idle",
    transcript: null,
    answer: null,
    groundingStatus: null,
    errorMessage: null,
    stageLatenciesMs: null,
    audioObjectUrl: null,
    recorderStatus: "idle",
    level: 0,
    startListening: vi.fn(),
    stopAndSend: vi.fn(),
    interrupt: vi.fn(),
  };
  vi.mocked(useVoiceTurnModule.useVoiceTurn).mockReturnValue({ ...base, ...overrides });
  return { ...base, ...overrides };
}

describe("VoiceSessionControl", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("prompts to tap the microphone in the idle state, with no interrupt button", () => {
    mockVoiceTurn({ state: "idle" });
    renderWithProviders(<VoiceSessionControl conversationId="conv-1" />);

    expect(screen.getByText(/tap the microphone to speak/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /interrupt/i })).not.toBeInTheDocument();
  });

  it("starts listening when the mic button is pressed", async () => {
    const voice = mockVoiceTurn({ state: "idle" });
    renderWithProviders(<VoiceSessionControl conversationId="conv-1" />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /start speaking/i }));
    expect(voice.startListening).toHaveBeenCalledOnce();
  });

  it("shows an interrupt control and the real backend stage label while busy", () => {
    mockVoiceTurn({ state: "thinking" });
    renderWithProviders(<VoiceSessionControl conversationId="conv-1" />);

    expect(screen.getByText(/analyzing emotion, sentiment, and memory/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /interrupt/i })).toBeInTheDocument();
  });

  it("calls interrupt() when the stop control is clicked while speaking", async () => {
    const voice = mockVoiceTurn({ state: "speaking" });
    renderWithProviders(<VoiceSessionControl conversationId="conv-1" />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /interrupt/i }));
    expect(voice.interrupt).toHaveBeenCalledOnce();
  });

  it("renders the real transcript once available, never a placeholder", () => {
    mockVoiceTurn({ state: "retrieving", transcript: "what does voxmind combine?" });
    renderWithProviders(<VoiceSessionControl conversationId="conv-1" />);

    expect(screen.getByText(/what does voxmind combine\?/i)).toBeInTheDocument();
  });

  it("renders the genuine per-stage latency breakdown from the backend", () => {
    mockVoiceTurn({
      state: "idle",
      stageLatenciesMs: { stt_ms: 1055, analysis_ms: 6850, retrieval_ms: 23, total_ms: 7944 },
    });
    renderWithProviders(<VoiceSessionControl conversationId="conv-1" />);

    expect(screen.getByText(/timing breakdown/i)).toBeInTheDocument();
    expect(screen.getByText("1055 ms")).toBeInTheDocument();
    expect(screen.getByText("7944 ms")).toBeInTheDocument();
  });

  it("surfaces a real error message rather than failing silently", () => {
    mockVoiceTurn({ state: "error", errorMessage: "No LLM provider is configured." });
    renderWithProviders(<VoiceSessionControl conversationId="conv-1" />);

    expect(screen.getByRole("alert")).toHaveTextContent(/no llm provider is configured/i);
  });
});
