import { apiFetch } from "@/api/client";
import type { KnowledgeDocument } from "@/types/api";

export function listDocuments(conversationId: string): Promise<KnowledgeDocument[]> {
  return apiFetch<KnowledgeDocument[]>(`/conversations/${conversationId}/documents`);
}

export function uploadDocument(conversationId: string, file: File): Promise<KnowledgeDocument> {
  const formData = new FormData();
  formData.append("file", file);
  return apiFetch<KnowledgeDocument>(`/conversations/${conversationId}/documents`, {
    method: "POST",
    formData,
  });
}

export function processDocument(conversationId: string, documentId: string): Promise<KnowledgeDocument> {
  return apiFetch<KnowledgeDocument>(
    `/conversations/${conversationId}/documents/${documentId}/process`,
    { method: "POST" },
  );
}
