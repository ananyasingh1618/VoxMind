"""Real Anthropic Messages API provider. Forces structured output via a
single tool definition + `tool_choice` - the model cannot reply in free
prose, only by calling `emit_structured_response` with the schema in
schema.py, so parsing is a JSON `.loads` on the tool call's `input`, never
regex/string scraping of a free-form reply.

Requires `ANTHROPIC_API_KEY` (a real developer-console API key - a Claude
Pro/Max chat subscription does not grant API access, see docs/architecture.md
and the Phase 0 correction that established this distinction). The
`anthropic` package is imported at module load time since it's a lightweight,
pure-network SDK (unlike MLflow/torch, it doesn't justify a lazy-import/
optional-extra treatment).
"""
from __future__ import annotations

import asyncio

import structlog
from anthropic import Anthropic

from voxmind.core.config import Settings
from voxmind.core.exceptions import PipelineProcessingError
from voxmind.services.llm.interfaces import ConversationContext, LlmResponse
from voxmind.services.llm.parsing import parse_structured_response
from voxmind.services.llm.prompt import render_context
from voxmind.services.llm.schema import (
    STRUCTURED_RESPONSE_SCHEMA,
    STRUCTURED_RESPONSE_TOOL_NAME,
    SYSTEM_INSTRUCTIONS,
)

logger = structlog.get_logger(__name__)


class AnthropicLlmProvider:
    provider_name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    @property
    def model_name(self) -> str:
        return self._settings.LLM_MODEL

    async def generate(self, context: ConversationContext) -> LlmResponse:
        try:
            return await asyncio.to_thread(self._call, context)
        except PipelineProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001 - the SDK raises many exception types
            logger.warning("anthropic_generation_failed", error=str(exc))
            raise PipelineProcessingError("LLM generation failed (Anthropic).") from exc

    def _call(self, context: ConversationContext) -> LlmResponse:
        response = self._client.messages.create(
            model=self._settings.LLM_MODEL,
            max_tokens=self._settings.LLM_MAX_OUTPUT_TOKENS,
            system=context.system_instructions or SYSTEM_INSTRUCTIONS,
            messages=[{"role": "user", "content": render_context(context)}],
            tools=[
                {
                    "name": STRUCTURED_RESPONSE_TOOL_NAME,
                    "description": "Emit the final structured response.",
                    "input_schema": STRUCTURED_RESPONSE_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": STRUCTURED_RESPONSE_TOOL_NAME},
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == STRUCTURED_RESPONSE_TOOL_NAME:
                return parse_structured_response(block.input)
        raise PipelineProcessingError("Anthropic response did not include the expected tool call.")
