import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import * as emotionApi from "@/api/emotion";

const emotionKeys = {
  predictions: (conversationId: string, messageId: string) =>
    ["emotion", conversationId, messageId] as const,
  models: (component: string) => ["models", component] as const,
};

export function useEmotionPredictions(conversationId: string, messageId: string | null) {
  return useQuery({
    queryKey: emotionKeys.predictions(conversationId, messageId ?? ""),
    queryFn: () => emotionApi.listEmotionPredictions(conversationId, messageId as string),
    enabled: Boolean(messageId),
  });
}

export function useProcessEmotion(conversationId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (messageId: string) => emotionApi.processEmotion(conversationId, messageId),
    onSuccess: (_job, messageId) => {
      queryClient.invalidateQueries({ queryKey: emotionKeys.predictions(conversationId, messageId) });
    },
  });
}

export function useModelVersions(component: string) {
  return useQuery({
    queryKey: emotionKeys.models(component),
    queryFn: () => emotionApi.listModelVersions(component),
  });
}
