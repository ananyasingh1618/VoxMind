"""The JSON schema forced on every real LLM call via tool-use / function-
calling, shared by both provider implementations so the two real providers
(Anthropic, OpenAI) and the test-only mock all produce identically-shaped
output. Forcing structured output through a tool call - rather than asking
the model to "return JSON" in prose - is what makes `answer`/`citations`/
`confidence`/`evidence_summary` reliably parseable instead of best-effort
string scraping.
"""
from __future__ import annotations

STRUCTURED_RESPONSE_TOOL_NAME = "emit_structured_response"

STRUCTURED_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The natural-language answer to the user's question, grounded only "
            "in the provided context (conversation, memory, retrieved documents).",
        },
        "citations": {
            "type": "array",
            "description": "Every retrieved chunk_id (from the provided context) that supports a "
            "claim in the answer. Only use chunk_id values that were given to you - never invent "
            "one. Leave empty if the answer doesn't rely on any retrieved document.",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "document_title": {"type": "string"},
                    "excerpt": {
                        "type": "string",
                        "description": "The specific excerpt from that chunk supporting the claim.",
                    },
                },
                "required": ["chunk_id", "document_title", "excerpt"],
            },
        },
        "confidence": {
            "type": "number",
            "description": "Your confidence in the answer's correctness and groundedness, 0.0-1.0.",
        },
        "evidence_summary": {
            "type": "string",
            "description": "A factual, one-to-two-sentence summary of which retrieved facts or "
            "signals (emotion, sentiment, memory, documents) the answer relied on. This is NOT "
            "your reasoning process or chain-of-thought - only a summary of which evidence you used.",
        },
    },
    "required": ["answer", "citations", "confidence", "evidence_summary"],
}

SYSTEM_INSTRUCTIONS = (
    "You are VoxMind's conversational assistant. You will be given structured context: the "
    "user's message, recent conversation turns, a conversation summary (if any), detected "
    "emotion/sentiment/semantic-vocal-incongruence signals, and retrieved document chunks with "
    "their chunk_id and source metadata. Answer using only that context. Never invent facts, "
    "sources, or chunk_id values that were not given to you. If the retrieved context doesn't "
    "contain enough information to answer, say so plainly rather than guessing. Cite every "
    "retrieved chunk_id you actually relied on. Do not reveal internal reasoning or chain-of-"
    "thought - only the final answer and a brief factual evidence summary."
)
