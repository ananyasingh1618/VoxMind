import { useAuthStore } from "@/stores/authStore";
import type { VoiceStageEvent } from "@/types/api";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";

/**
 * Real Server-Sent-Events consumption over a POST body - the browser's
 * native `EventSource` only supports GET, so this reads
 * `response.body`'s stream directly and parses the `data: {...}\n\n` frames
 * by hand. Every event handed to `onEvent` corresponds to a real backend
 * stage genuinely completing (see `VoiceTurnService.run()`), not simulated
 * client-side progress.
 */
export async function streamVoiceTurn(
  conversationId: string,
  audioBlob: Blob,
  onEvent: (event: VoiceStageEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const accessToken = useAuthStore.getState().accessToken;
  const formData = new FormData();
  formData.append("file", audioBlob, "voice-turn.webm");

  const response = await fetch(`${API_BASE}/conversations/${conversationId}/voice-turns`, {
    method: "POST",
    credentials: "include",
    headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : {},
    body: formData,
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Voice turn request failed with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith("data: ")) continue;
      try {
        onEvent(JSON.parse(line.slice("data: ".length)) as VoiceStageEvent);
      } catch {
        // A frame split across two read() chunks in a way our buffering
        // didn't fully reassemble - drop it rather than crash the stream;
        // the next well-formed frame still arrives.
      }
    }
  }
}

/** Fetches synthesized audio bytes with the required auth header (a plain
 * `<audio src>` can't send one) and returns a blob object URL for playback.
 * Callers must `URL.revokeObjectURL()` it when no longer needed. */
export async function fetchVoiceTurnAudio(audioUrl: string): Promise<string> {
  const accessToken = useAuthStore.getState().accessToken;
  const response = await fetch(audioUrl, {
    credentials: "include",
    headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : {},
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch synthesized audio (status ${response.status})`);
  }
  const blob = await response.blob();
  return URL.createObjectURL(blob);
}
