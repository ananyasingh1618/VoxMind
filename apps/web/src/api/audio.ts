import { apiFetch } from "@/api/client";
import type { AudioAsset, AudioProcessingJob, Transcript } from "@/types/api";

export function uploadAudio(conversationId: string, file: File): Promise<AudioAsset> {
  const formData = new FormData();
  formData.append("file", file);
  return apiFetch<AudioAsset>(`/conversations/${conversationId}/audio`, {
    method: "POST",
    formData,
  });
}

export function listAudio(conversationId: string): Promise<AudioAsset[]> {
  return apiFetch<AudioAsset[]>(`/conversations/${conversationId}/audio`);
}

export function processAudio(
  conversationId: string,
  audioAssetId: string,
  language?: string,
): Promise<AudioProcessingJob> {
  return apiFetch<AudioProcessingJob>(
    `/conversations/${conversationId}/audio/${audioAssetId}/process`,
    { method: "POST", body: { language: language ?? null } },
  );
}

export function listProcessingJobs(
  conversationId: string,
  audioAssetId: string,
): Promise<AudioProcessingJob[]> {
  return apiFetch<AudioProcessingJob[]>(
    `/conversations/${conversationId}/audio/${audioAssetId}/processing`,
  );
}

export function getTranscript(conversationId: string, messageId: string): Promise<Transcript> {
  return apiFetch<Transcript>(`/conversations/${conversationId}/messages/${messageId}/transcript`);
}
