"""Real OpenAI Chat Completions provider. Structured output is forced via
function calling with `tool_choice` pinned to the one function - the same
approach and the same schema (schema.py) as `AnthropicLlmProvider`, so
switching `LLM_PROVIDER` never changes what shape of response the rest of
the app receives.

Requires `OPENAI_API_KEY` (a developer-console API key - a ChatGPT Plus
subscription does not grant API access).

`OPENAI_BASE_URL` (optional) points the same client at any other
OpenAI-Chat-Completions-compatible endpoint - e.g. Groq
(`https://api.groq.com/openai/v1`) - without a separate provider class.
Nothing about the request shape below changes: Groq's `tool_choice`
(including forcing one specific function by name) and `tools` schema are
documented as matching OpenAI's exactly, which is what makes reusing this
provider correct rather than coincidental.
"""
from __future__ import annotations

import asyncio
import json

import structlog
from openai import OpenAI

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


class OpenAiLlmProvider:
    provider_name = "openai"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # base_url=None leaves the openai-python SDK's own default endpoint
        # untouched for real OpenAI users - only set when this deployment is
        # deliberately pointed at a compatible alternative (see module docstring).
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)

    @property
    def model_name(self) -> str:
        return self._settings.OPENAI_MODEL

    async def generate(self, context: ConversationContext) -> LlmResponse:
        try:
            return await asyncio.to_thread(self._call, context)
        except PipelineProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Never log str(exc) here: proven live (during real Groq
            # verification) that this SDK's own AuthenticationError message
            # includes a masked-but-partial echo of the real API key (first/
            # last few characters visible around asterisks) - the same
            # exception class this generation call can also raise. Only the
            # exception type is safe to log.
            logger.warning("openai_generation_failed", error_type=type(exc).__name__)
            raise PipelineProcessingError("LLM generation failed (OpenAI).") from exc

    def _call(self, context: ConversationContext) -> LlmResponse:
        response = self._client.chat.completions.create(
            model=self._settings.OPENAI_MODEL,
            max_tokens=self._settings.LLM_MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "system", "content": context.system_instructions or SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": render_context(context)},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": STRUCTURED_RESPONSE_TOOL_NAME,
                        "description": "Emit the final structured response.",
                        "parameters": STRUCTURED_RESPONSE_SCHEMA,
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": STRUCTURED_RESPONSE_TOOL_NAME}},
        )
        message = response.choices[0].message
        if not message.tool_calls:
            raise PipelineProcessingError("OpenAI response did not include the expected tool call.")
        call = message.tool_calls[0]
        if call.type != "function":
            raise PipelineProcessingError("OpenAI response returned an unexpected tool call type.")
        data = json.loads(call.function.arguments)
        return parse_structured_response(data)
