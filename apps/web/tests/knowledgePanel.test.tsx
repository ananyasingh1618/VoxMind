import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import * as knowledgeApi from "@/api/knowledge";
import * as ragApi from "@/api/rag";
import { KnowledgePanel } from "@/components/conversation/KnowledgePanel";

import { renderWithProviders } from "./testUtils";

vi.mock("@/api/knowledge");
vi.mock("@/api/rag");

describe("KnowledgePanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.mocked(knowledgeApi.listDocuments).mockResolvedValue([]);
  });

  it("shows an empty state before any document is uploaded", async () => {
    renderWithProviders(<KnowledgePanel conversationId="conv-1" />);
    expect(await screen.findByText(/no documents uploaded yet/i)).toBeInTheDocument();
  });

  it("lists documents with their real ingestion status", async () => {
    vi.mocked(knowledgeApi.listDocuments).mockResolvedValue([
      {
        id: "doc-1",
        session_id: "conv-1",
        title: "notes.txt",
        original_filename: "notes.txt",
        content_type: "text/plain",
        size_bytes: 123,
        status: "completed",
        error_message: null,
        chunk_count: 3,
        created_at: "2026-01-01T00:00:00Z",
        processed_at: "2026-01-01T00:00:01Z",
      },
    ]);

    renderWithProviders(<KnowledgePanel conversationId="conv-1" />);

    expect(await screen.findByText("notes.txt")).toBeInTheDocument();
    expect(screen.getByText(/3 chunks/i)).toBeInTheDocument();
  });

  it("renders retrieved sources, the answer, citations, and grounding status from a real ask() response", async () => {
    vi.mocked(ragApi.ask).mockResolvedValue({
      user_message: {
        id: "msg-user",
        session_id: "conv-1",
        role: "user",
        content: "What does the pipeline combine?",
        audio_asset_id: null,
        created_at: "2026-01-01T00:00:00Z",
      },
      assistant_message: {
        id: "msg-assistant",
        session_id: "conv-1",
        role: "assistant",
        content: "It combines vector and lexical search.",
        audio_asset_id: null,
        created_at: "2026-01-01T00:00:01Z",
      },
      retrieval_result_id: "ret-1",
      retrieved_chunks: [
        {
          chunk_id: "chunk-1",
          document_id: "doc-1",
          text: "The pipeline combines vector and lexical search via RRF.",
          citation: "notes.txt (chunk 1)",
          vector_score: 0.9,
          lexical_score: 1.2,
          rerank_score: null,
        },
      ],
      generation: {
        id: "gen-1",
        provider: "mock",
        model_name: "mock-llm-v1",
        context_token_estimate: 42,
        answer: "It combines vector and lexical search.",
        raw_model_answer: "It combines vector and lexical search.",
        citations: [{ chunk_id: "chunk-1", document_title: "notes.txt (chunk 1)", excerpt: "combines vector" }],
        confidence: 0.5,
        evidence_summary: "Used chunk-1.",
        grounding_status: "grounded",
        grounding_details: {},
        error_message: null,
        latency_ms: 120,
        created_at: "2026-01-01T00:00:01Z",
      },
      guardrail: {
        decision: "approved",
        reasons: [],
        filtered_chunk_ids: [],
        safety_flagged: false,
        safety_categories: [],
      },
    });

    renderWithProviders(<KnowledgePanel conversationId="conv-1" />);
    const user = userEvent.setup();

    const input = await screen.findByPlaceholderText(/ask about your documents/i);
    await user.type(input, "What does the pipeline combine?");
    await user.click(screen.getByRole("button", { name: /^ask$/i }));

    expect(await screen.findByText(/retrieved sources \(1\)/i)).toBeInTheDocument();
    expect(screen.getByText(/it combines vector and lexical search/i)).toBeInTheDocument();
    expect(screen.getByText(/grounded in retrieved sources/i)).toBeInTheDocument();
  });

  it("surfaces a real guardrail 'blocked' decision instead of showing it as a normal answer", async () => {
    vi.mocked(ragApi.ask).mockResolvedValue({
      user_message: {
        id: "msg-user",
        session_id: "conv-1",
        role: "user",
        content: "(guardrail test)",
        audio_asset_id: null,
        created_at: "2026-01-01T00:00:00Z",
      },
      assistant_message: {
        id: "msg-assistant",
        session_id: "conv-1",
        role: "assistant",
        content: "I'm not able to provide a response to that request.",
        audio_asset_id: null,
        created_at: "2026-01-01T00:00:01Z",
      },
      retrieval_result_id: "ret-1",
      retrieved_chunks: [],
      generation: {
        id: "gen-1",
        provider: "mock",
        model_name: "mock-llm-v1",
        context_token_estimate: 42,
        answer: "I'm not able to provide a response to that request.",
        raw_model_answer: "the raw unsafe content the model produced",
        citations: [],
        confidence: 0.5,
        evidence_summary: "",
        grounding_status: "ungrounded",
        grounding_details: {},
        error_message: null,
        latency_ms: 120,
        created_at: "2026-01-01T00:00:01Z",
      },
      guardrail: {
        decision: "blocked",
        reasons: ["Safety check (keyword_fallback) flagged: self_harm"],
        filtered_chunk_ids: [],
        safety_flagged: true,
        safety_categories: ["self_harm"],
      },
    });

    renderWithProviders(<KnowledgePanel conversationId="conv-1" />);
    const user = userEvent.setup();

    const input = await screen.findByPlaceholderText(/ask about your documents/i);
    await user.type(input, "(guardrail test)");
    await user.click(screen.getByRole("button", { name: /^ask$/i }));

    expect(await screen.findByText(/guardrail: blocked/i)).toBeInTheDocument();
    expect(screen.getByText(/not able to provide a response/i)).toBeInTheDocument();
    expect(screen.queryByText(/raw unsafe content/i)).not.toBeInTheDocument();
  });
});
