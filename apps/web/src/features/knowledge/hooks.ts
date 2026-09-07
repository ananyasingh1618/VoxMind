import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import * as knowledgeApi from "@/api/knowledge";

const knowledgeKeys = {
  list: (conversationId: string) => ["knowledge-documents", conversationId] as const,
};

export function useKnowledgeDocuments(conversationId: string) {
  return useQuery({
    queryKey: knowledgeKeys.list(conversationId),
    queryFn: () => knowledgeApi.listDocuments(conversationId),
  });
}

export function useUploadDocument(conversationId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => knowledgeApi.uploadDocument(conversationId, file),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: knowledgeKeys.list(conversationId) }),
  });
}

export function useProcessDocument(conversationId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (documentId: string) => knowledgeApi.processDocument(conversationId, documentId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: knowledgeKeys.list(conversationId) }),
  });
}
