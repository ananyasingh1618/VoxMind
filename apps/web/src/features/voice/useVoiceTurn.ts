import { useCallback, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { fetchVoiceTurnAudio, streamVoiceTurn } from "@/api/voice";
import { conversationKeys } from "@/features/conversations/hooks";
import { useVoiceRecorder } from "@/features/voice/useVoiceRecorder";
import type { VoiceStageEvent, VoiceTurnDoneEvent, VoiceUIState } from "@/types/api";

interface VoiceTurnUiState {
  state: VoiceUIState;
  transcript: string | null;
  answer: string | null;
  groundingStatus: string | null;
  errorMessage: string | null;
  stageLatenciesMs: Record<string, number> | null;
  audioObjectUrl: string | null;
}

const INITIAL_STATE: VoiceTurnUiState = {
  state: "idle",
  transcript: null,
  answer: null,
  groundingStatus: null,
  errorMessage: null,
  stageLatenciesMs: null,
  audioObjectUrl: null,
};

const STAGE_TO_UI_STATE: Record<string, VoiceUIState> = {
  processing: "processing",
  thinking: "thinking",
  retrieving: "retrieving",
  generating: "generating",
  speaking: "speaking",
  error: "error",
};

/**
 * Orchestrates the full real-time voice loop: record -> stream real backend
 * progress over SSE -> play back the real synthesized response -> allow
 * interruption at any point. This hook owns the single `<audio>` element
 * used for playback so two responses can never overlap - starting a new
 * turn (or interrupting) always stops whatever is currently playing first.
 */
export function useVoiceTurn(conversationId: string) {
  const [ui, setUi] = useState<VoiceTurnUiState>(INITIAL_STATE);
  const recorder = useVoiceRecorder();
  const abortControllerRef = useRef<AbortController | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const queryClient = useQueryClient();

  const stopPlayback = useCallback(() => {
    const audio = audioElementRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }
    setUi((prev) => {
      if (prev.audioObjectUrl) URL.revokeObjectURL(prev.audioObjectUrl);
      return prev;
    });
  }, []);

  const playAudio = useCallback((objectUrl: string) => {
    stopPlayback(); // never overlap two responses
    const audio = new Audio(objectUrl);
    audioElementRef.current = audio;
    audio.onended = () => setUi((prev) => ({ ...prev, state: "idle" }));
    audio.play().catch(() => undefined);
  }, [stopPlayback]);

  const handleEvent = useCallback(
    async (event: VoiceStageEvent) => {
      const mapped = STAGE_TO_UI_STATE[event.stage];

      if (event.stage === "done") {
        const done = event as unknown as VoiceTurnDoneEvent;
        setUi((prev) => ({
          ...prev,
          answer: done.answer || null,
          groundingStatus: done.grounding_status,
          stageLatenciesMs: done.stage_latencies_ms,
          state: done.audio_url ? "speaking" : "idle",
        }));
        queryClient.invalidateQueries({ queryKey: conversationKeys.messages(conversationId) });
        if (done.audio_url) {
          try {
            const objectUrl = await fetchVoiceTurnAudio(done.audio_url);
            setUi((prev) => ({ ...prev, audioObjectUrl: objectUrl }));
            playAudio(objectUrl);
          } catch {
            setUi((prev) => ({ ...prev, state: "error", errorMessage: "Could not load the spoken response." }));
          }
        }
        return;
      }

      if (event.stage === "error") {
        setUi((prev) => ({
          ...prev,
          state: "error",
          errorMessage: (event.error_message as string) || "The voice turn failed.",
        }));
        return;
      }

      setUi((prev) => ({
        ...prev,
        state: mapped ?? prev.state,
        transcript: event.stage === "processing" && event.status === "completed"
          ? (event.transcript as string)
          : prev.transcript,
      }));
    },
    [conversationId, playAudio, queryClient],
  );

  const startListening = useCallback(async () => {
    stopPlayback();
    setUi({ ...INITIAL_STATE, state: "listening" });
    await recorder.start();
  }, [recorder, stopPlayback]);

  const stopAndSend = useCallback(async () => {
    const blob = await recorder.stop();
    if (!blob || blob.size === 0) {
      setUi((prev) => ({ ...prev, state: "idle" }));
      return;
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;
    setUi((prev) => ({ ...prev, state: "processing" }));
    try {
      await streamVoiceTurn(conversationId, blob, handleEvent, controller.signal);
    } catch (error) {
      if (controller.signal.aborted) {
        setUi((prev) => ({ ...prev, state: "interrupted" }));
      } else {
        setUi((prev) => ({
          ...prev,
          state: "error",
          errorMessage: error instanceof Error ? error.message : "The voice turn failed.",
        }));
      }
    } finally {
      abortControllerRef.current = null;
    }
  }, [conversationId, handleEvent, recorder]);

  /** Stops whatever is happening right now - an in-flight request (a real
   * client disconnect, which the server detects and records as a genuine
   * interruption) or already-playing audio (stopped locally; nothing to
   * tell the server, the turn already completed). Never both do nothing. */
  const interrupt = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setUi((prev) => ({ ...prev, state: "interrupted" }));
      return;
    }
    if (audioElementRef.current && !audioElementRef.current.paused) {
      stopPlayback();
      setUi((prev) => ({ ...prev, state: "interrupted" }));
    }
  }, [stopPlayback]);

  return {
    ...ui,
    recorderStatus: recorder.status,
    level: recorder.level,
    startListening,
    stopAndSend,
    interrupt,
  };
}
