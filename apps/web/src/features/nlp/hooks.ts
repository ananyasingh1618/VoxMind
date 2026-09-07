import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import * as incongruenceApi from "@/api/incongruence";
import * as nlpApi from "@/api/nlp";

const nlpKeys = {
  annotation: (conversationId: string, messageId: string) =>
    ["nlp", conversationId, messageId] as const,
};

const incongruenceKeys = {
  signals: (conversationId: string, messageId: string) =>
    ["incongruence", conversationId, messageId] as const,
};

export function useNlpAnnotation(conversationId: string, messageId: string | null) {
  return useQuery({
    queryKey: nlpKeys.annotation(conversationId, messageId ?? ""),
    queryFn: () => nlpApi.getNlp(conversationId, messageId as string),
    enabled: Boolean(messageId),
  });
}

export function useProcessNlp(conversationId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (messageId: string) => nlpApi.processNlp(conversationId, messageId),
    onSuccess: (_annotation, messageId) => {
      queryClient.invalidateQueries({ queryKey: nlpKeys.annotation(conversationId, messageId) });
    },
  });
}

export function useIncongruenceSignals(conversationId: string, messageId: string | null) {
  return useQuery({
    queryKey: incongruenceKeys.signals(conversationId, messageId ?? ""),
    queryFn: () => incongruenceApi.listIncongruence(conversationId, messageId as string),
    enabled: Boolean(messageId),
  });
}

export function useAnalyzeIncongruence(conversationId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (messageId: string) => incongruenceApi.analyzeIncongruence(conversationId, messageId),
    onSuccess: (_signals, messageId) => {
      queryClient.invalidateQueries({ queryKey: incongruenceKeys.signals(conversationId, messageId) });
    },
  });
}
