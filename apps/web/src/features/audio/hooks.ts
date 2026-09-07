import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import * as audioApi from "@/api/audio";
import { conversationKeys } from "@/features/conversations/hooks";

const audioKeys = {
  list: (conversationId: string) => ["audio", conversationId] as const,
  transcript: (conversationId: string, messageId: string) =>
    ["audio", conversationId, "transcript", messageId] as const,
};

export function useAudioAssets(conversationId: string) {
  return useQuery({
    queryKey: audioKeys.list(conversationId),
    queryFn: () => audioApi.listAudio(conversationId),
  });
}

export function useUploadAudio(conversationId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => audioApi.uploadAudio(conversationId, file),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: audioKeys.list(conversationId) }),
  });
}

export function useProcessAudio(conversationId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (audioAssetId: string) => audioApi.processAudio(conversationId, audioAssetId),
    onSuccess: () => {
      // A completed job creates a new Message with the transcript text, and
      // the transcript itself is keyed by that message id.
      queryClient.invalidateQueries({ queryKey: conversationKeys.messages(conversationId) });
    },
  });
}

export function useTranscript(conversationId: string, messageId: string | null) {
  return useQuery({
    queryKey: audioKeys.transcript(conversationId, messageId ?? ""),
    queryFn: () => audioApi.getTranscript(conversationId, messageId as string),
    enabled: Boolean(messageId),
  });
}
