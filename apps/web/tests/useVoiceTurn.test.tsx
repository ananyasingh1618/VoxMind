import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import * as voiceApi from "@/api/voice";
import { useVoiceTurn } from "@/features/voice/useVoiceTurn";
import * as useVoiceRecorderModule from "@/features/voice/useVoiceRecorder";

vi.mock("@/api/voice");
vi.mock("@/features/voice/useVoiceRecorder");

function wrapper({ children }: { children: ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

describe("useVoiceTurn", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.mocked(useVoiceRecorderModule.useVoiceRecorder).mockReturnValue({
      status: "idle",
      level: 0.3,
      start: vi.fn().mockResolvedValue(undefined),
      stop: vi.fn().mockResolvedValue(new Blob(["audio bytes"], { type: "audio/webm" })),
      cancel: vi.fn(),
    });
  });

  it("walks through real backend stage events and lands on the honest unavailable outcome", async () => {
    vi.mocked(voiceApi.streamVoiceTurn).mockImplementation(async (_conv, _blob, onEvent) => {
      onEvent({ stage: "processing", status: "start", voice_turn_id: "turn-1" });
      onEvent({ stage: "processing", status: "completed", transcript: "the quick brown fox", latency_ms: 500 });
      onEvent({ stage: "thinking", status: "start" });
      onEvent({ stage: "thinking", status: "completed", latency_ms: 900 });
      onEvent({ stage: "retrieving", status: "start" });
      onEvent({ stage: "retrieving", status: "completed", retrieved_chunk_count: 0, latency_ms: 5 });
      onEvent({ stage: "generating", status: "unavailable", answer: "", grounding_status: "unavailable" });
      onEvent({ stage: "speaking", status: "unavailable", reason: "No answer was generated." });
      onEvent({
        stage: "done",
        status: "partial",
        voice_turn_id: "turn-1",
        user_message_id: "msg-1",
        assistant_message_id: null,
        answer: "",
        citations: [],
        grounding_status: "unavailable",
        audio_url: null,
        stage_latencies_ms: { stt_ms: 500, analysis_ms: 900, retrieval_ms: 5, llm_ms: 0, tts_ms: 0, total_ms: 1405 },
      });
    });

    const { result } = renderHook(() => useVoiceTurn("conv-1"), { wrapper });

    await act(async () => {
      await result.current.startListening();
    });
    expect(result.current.state).toBe("listening");

    await act(async () => {
      await result.current.stopAndSend();
    });

    await waitFor(() => expect(result.current.state).toBe("idle"));
    expect(result.current.transcript).toBe("the quick brown fox");
    expect(result.current.stageLatenciesMs?.total_ms).toBe(1405);
    expect(result.current.audioObjectUrl).toBeNull();
  });

  it("fetches and plays real synthesized audio when the backend produces one", async () => {
    vi.mocked(voiceApi.fetchVoiceTurnAudio).mockResolvedValue("blob:mock-audio-url");
    vi.mocked(voiceApi.streamVoiceTurn).mockImplementation(async (_conv, _blob, onEvent) => {
      onEvent({
        stage: "done",
        status: "completed",
        voice_turn_id: "turn-2",
        user_message_id: "msg-2",
        assistant_message_id: "msg-3",
        answer: "Here is the answer.",
        citations: [],
        grounding_status: "grounded",
        audio_url: "/api/v1/conversations/conv-1/voice-turns/turn-2/audio",
        stage_latencies_ms: { stt_ms: 1, analysis_ms: 1, retrieval_ms: 1, llm_ms: 1, tts_ms: 100, total_ms: 104 },
      });
    });
    // jsdom has no real audio decoding - stub play() so it doesn't throw.
    window.HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
    window.HTMLMediaElement.prototype.pause = vi.fn();

    const { result } = renderHook(() => useVoiceTurn("conv-1"), { wrapper });
    await act(async () => {
      await result.current.startListening();
    });
    await act(async () => {
      await result.current.stopAndSend();
    });

    await waitFor(() => expect(result.current.audioObjectUrl).toBe("blob:mock-audio-url"));
    expect(result.current.answer).toBe("Here is the answer.");
    expect(result.current.state).toBe("speaking");
  });

  it("marks the turn interrupted when the in-flight request is aborted", async () => {
    vi.mocked(voiceApi.streamVoiceTurn).mockImplementation(async (_conv, _blob, _onEvent, signal) => {
      await new Promise((resolve) => {
        signal.addEventListener("abort", resolve);
      });
      const abortError = new Error("aborted");
      abortError.name = "AbortError";
      throw abortError;
    });

    const { result } = renderHook(() => useVoiceTurn("conv-1"), { wrapper });
    await act(async () => {
      await result.current.startListening();
    });

    let sendPromise: Promise<void>;
    act(() => {
      sendPromise = result.current.stopAndSend();
    });
    await waitFor(() => expect(result.current.state).toBe("processing"));

    act(() => {
      result.current.interrupt();
    });
    await act(async () => {
      await sendPromise;
    });

    expect(result.current.state).toBe("interrupted");
  });
});
