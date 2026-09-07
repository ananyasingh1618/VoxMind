import { AlertCircle, BookOpen, FileText, Search, Upload } from "lucide-react";
import { useRef, useState } from "react";

import { ApiError } from "@/api/client";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";
import { useKnowledgeDocuments, useProcessDocument, useUploadDocument } from "@/features/knowledge/hooks";
import { useAsk } from "@/features/rag/hooks";
import { cn } from "@/lib/cn";
import type { AskResponse, KnowledgeDocument } from "@/types/api";

function uploadErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.code === "unsupported_document_format") {
      return "That file type isn't supported yet (.txt, .md, .pdf only).";
    }
    if (error.code === "document_too_large") {
      return "That document is too large.";
    }
    return error.message;
  }
  return "Upload failed. Please try again.";
}

function groundingLabel(status: AskResponse["generation"]["grounding_status"]): string {
  switch (status) {
    case "grounded":
      return "Grounded in retrieved sources";
    case "partially_grounded":
      return "Partially grounded";
    case "ungrounded":
      return "Not grounded in any retrieved source";
    case "unavailable":
      return "No LLM provider configured";
  }
}

function groundingColor(status: AskResponse["generation"]["grounding_status"]): string {
  switch (status) {
    case "grounded":
      return "text-[var(--color-signal-positive)]";
    case "partially_grounded":
      return "text-[var(--color-signal-warning,#d9a441)]";
    default:
      return "text-[var(--color-text-tertiary)]";
  }
}

export function KnowledgePanel({ conversationId }: { conversationId: string }) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { data: documents, isLoading, isError } = useKnowledgeDocuments(conversationId);
  const upload = useUploadDocument(conversationId);
  const process = useProcessDocument(conversationId);
  const ask = useAsk(conversationId);
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [lastAnswer, setLastAnswer] = useState<AskResponse | null>(null);

  async function handleFileSelected(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    await upload.mutateAsync(file);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function handleProcess(document: KnowledgeDocument) {
    setProcessingId(document.id);
    try {
      await process.mutateAsync(document.id);
    } finally {
      setProcessingId(null);
    }
  }

  async function handleAsk() {
    const trimmed = question.trim();
    if (!trimmed) return;
    const result = await ask.mutateAsync(trimmed);
    setLastAnswer(result);
    setQuestion("");
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex items-center justify-between gap-2 border-b border-[var(--color-border)] p-2">
        <span className="flex items-center gap-1.5 text-xs font-medium text-[var(--color-text-secondary)]">
          <BookOpen size={13} /> Knowledge
        </span>
      </div>

      <div className="border-b border-[var(--color-border)] p-2">
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.md,.pdf,text/plain,text/markdown,application/pdf"
          onChange={handleFileSelected}
          className="hidden"
          id="document-upload-input"
        />
        <Button
          variant="secondary"
          className="w-full justify-center py-1 text-xs"
          isLoading={upload.isPending}
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload size={13} /> Upload document
        </Button>
        {upload.isError && (
          <p role="alert" className="mt-1 flex items-center gap-1 text-xs text-[var(--color-signal-negative)]">
            <AlertCircle size={11} /> {uploadErrorMessage(upload.error)}
          </p>
        )}
      </div>

      <div className="max-h-32 overflow-y-auto border-b border-[var(--color-border)]">
        {isLoading && (
          <div className="flex justify-center py-3">
            <Spinner />
          </div>
        )}
        {isError && <p className="p-2 text-xs text-[var(--color-signal-negative)]">Couldn't load documents.</p>}
        {!isLoading && !isError && (documents ?? []).length === 0 && (
          <p className="p-2 text-xs text-[var(--color-text-tertiary)]">No documents uploaded yet.</p>
        )}
        {(documents ?? []).map((document) => (
          <div
            key={document.id}
            className="flex items-center justify-between gap-2 px-2 py-1.5 text-xs last:border-b-0"
          >
            <span className="flex min-w-0 flex-1 items-center gap-1.5">
              <FileText size={12} className="shrink-0 text-[var(--color-text-tertiary)]" />
              <span className="truncate">{document.title}</span>
            </span>
            {document.status === "pending" && (
              <Button
                variant="ghost"
                className="shrink-0 px-1.5 py-0.5 text-[11px]"
                isLoading={processingId === document.id && process.isPending}
                onClick={() => handleProcess(document)}
              >
                Ingest
              </Button>
            )}
            {document.status === "processing" && (
              <span className="shrink-0 text-[var(--color-text-tertiary)]">Processing…</span>
            )}
            {document.status === "completed" && (
              <span className="shrink-0 text-[var(--color-signal-positive)]">
                {document.chunk_count} chunk{document.chunk_count === 1 ? "" : "s"}
              </span>
            )}
            {document.status === "failed" && (
              <span className="shrink-0 text-[var(--color-signal-negative)]" title={document.error_message ?? ""}>
                Failed
              </span>
            )}
          </div>
        ))}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2 text-xs">
        {!lastAnswer && (
          <p className="p-2 text-[var(--color-text-tertiary)]">
            Ask a question below to retrieve evidence from your documents and generate a grounded answer.
          </p>
        )}
        {lastAnswer && (
          <div className="flex flex-col gap-2">
            <p className="font-medium text-[var(--color-text-primary)]">
              &ldquo;{lastAnswer.user_message.content}&rdquo;
            </p>

            <div>
              <p className="mb-1 flex items-center gap-1 font-medium text-[var(--color-text-secondary)]">
                <Search size={11} /> Retrieved sources ({lastAnswer.retrieved_chunks.length})
              </p>
              {lastAnswer.retrieved_chunks.length === 0 && (
                <p className="text-[var(--color-text-tertiary)]">No matching sources were retrieved.</p>
              )}
              <ol className="flex flex-col gap-1.5">
                {lastAnswer.retrieved_chunks.map((chunk) => (
                  <li key={chunk.chunk_id} className="rounded-lg border border-[var(--color-border)] p-1.5">
                    <p className="mb-0.5 text-[11px] font-medium text-[var(--color-accent-strong)]">
                      {chunk.citation}
                      {chunk.rerank_score !== null && (
                        <span className="ml-1 font-normal text-[var(--color-text-tertiary)]">reranked</span>
                      )}
                    </p>
                    <p className="line-clamp-3 text-[var(--color-text-secondary)]">{chunk.text}</p>
                  </li>
                ))}
              </ol>
            </div>

            <div>
              <p className="mb-1 font-medium text-[var(--color-text-secondary)]">Answer</p>
              {lastAnswer.generation.grounding_status === "unavailable" ? (
                <p className="text-[var(--color-text-tertiary)]">{lastAnswer.generation.error_message}</p>
              ) : (
                <p className="text-[var(--color-text-primary)]">{lastAnswer.generation.answer}</p>
              )}
            </div>

            {lastAnswer.generation.citations.length > 0 && (
              <div>
                <p className="mb-1 font-medium text-[var(--color-text-secondary)]">Citations</p>
                <ul className="flex flex-col gap-1.5">
                  {lastAnswer.generation.citations.map((citation, i) => (
                    <li key={i} className="rounded-lg border border-[var(--color-border)] p-1.5">
                      <p className="mb-0.5 text-[11px] font-medium text-[var(--color-accent-strong)]">
                        [{i + 1}] {citation.document_title}
                      </p>
                      <p className="italic text-[var(--color-text-tertiary)]">&ldquo;{citation.excerpt}&rdquo;</p>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <p className={cn("font-medium", groundingColor(lastAnswer.generation.grounding_status))}>
              {groundingLabel(lastAnswer.generation.grounding_status)}
            </p>

            {lastAnswer.guardrail && lastAnswer.guardrail.decision !== "approved" && (
              <p
                className={cn(
                  "font-medium",
                  lastAnswer.guardrail.decision === "blocked"
                    ? "text-[var(--color-signal-negative)]"
                    : "text-[var(--color-text-tertiary)]",
                )}
              >
                Guardrail: {lastAnswer.guardrail.decision}
                {lastAnswer.guardrail.reasons.length > 0 && ` — ${lastAnswer.guardrail.reasons[0]}`}
              </p>
            )}
            {lastAnswer.guardrail && lastAnswer.guardrail.filtered_chunk_ids.length > 0 && (
              <p className="text-[var(--color-text-tertiary)]">
                {lastAnswer.guardrail.filtered_chunk_ids.length} retrieved chunk(s) were filtered out for
                containing suspicious instructions before reaching the model.
              </p>
            )}
          </div>
        )}
        {ask.isError && (
          <p role="alert" className="mt-2 text-[var(--color-signal-negative)]">
            {ask.error instanceof ApiError ? ask.error.message : "Failed to get an answer."}
          </p>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleAsk();
        }}
        className="flex items-center gap-2 border-t border-[var(--color-border)] p-2"
      >
        <label htmlFor="ask-input" className="sr-only">
          Ask a knowledge-grounded question
        </label>
        <input
          id="ask-input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about your documents…"
          className="flex-1 rounded-lg border border-[var(--color-border-strong)] bg-[var(--color-surface)] px-2 py-1.5 text-xs outline-none focus:border-[var(--color-accent)]"
        />
        <Button type="submit" className="px-2 py-1 text-xs" isLoading={ask.isPending} disabled={!question.trim()}>
          Ask
        </Button>
      </form>
    </div>
  );
}
