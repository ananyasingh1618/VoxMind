import { apiFetch } from "@/api/client";
import type { EmotionPrediction, EmotionProcessingJob, ModelVersion } from "@/types/api";

export function processEmotion(conversationId: string, messageId: string): Promise<EmotionProcessingJob> {
  return apiFetch<EmotionProcessingJob>(
    `/conversations/${conversationId}/messages/${messageId}/emotion/process`,
    { method: "POST" },
  );
}

export function listEmotionPredictions(
  conversationId: string,
  messageId: string,
): Promise<EmotionPrediction[]> {
  return apiFetch<EmotionPrediction[]>(`/conversations/${conversationId}/messages/${messageId}/emotion`);
}

export function listModelVersions(component: string): Promise<ModelVersion[]> {
  return apiFetch<ModelVersion[]>(`/models/${component}`);
}
