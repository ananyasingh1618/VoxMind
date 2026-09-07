"""Renders a `ConversationContext` into the single user-turn prompt sent to
whichever provider is configured. Both real providers (Anthropic, OpenAI)
and the test-only mock share this so the model always sees an identically
structured context regardless of provider.
"""
from __future__ import annotations

import json

from voxmind.services.llm.interfaces import ConversationContext


def render_context(context: ConversationContext) -> str:
    parts = [
        f"User message:\n{context.transcript}",
    ]
    if context.speaker_info:
        parts.append(f"Speaker segments:\n{json.dumps(context.speaker_info, default=str)}")
    if context.emotion:
        parts.append(f"Detected vocal emotion:\n{json.dumps(context.emotion, default=str)}")
    if context.nlp:
        parts.append(f"Detected sentiment/intent/topics:\n{json.dumps(context.nlp, default=str)}")
    if context.incongruence:
        parts.append(
            "Semantic-vocal incongruence signals (analytical only, not evidence of deception):\n"
            f"{json.dumps(context.incongruence, default=str)}"
        )
    if context.memory:
        parts.append(f"Conversation memory:\n{json.dumps(context.memory, default=str)}")
    if context.retrieved_knowledge:
        parts.append(
            "Retrieved document chunks (cite by chunk_id only if used):\n"
            f"{json.dumps(context.retrieved_knowledge, default=str)}"
        )
    else:
        parts.append("Retrieved document chunks: none available for this question.")
    if context.safety_context:
        parts.append(f"Safety context:\n{json.dumps(context.safety_context, default=str)}")
    return "\n\n".join(parts)
