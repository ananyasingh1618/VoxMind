from __future__ import annotations

from pydantic import BaseModel

from voxmind.services.tts.interfaces import TextToSpeechProvider, TtsResult


class TtsSynthesisInput(BaseModel):
    text: str
    voice_profile: str | None = None


class TtsSynthesisOutput(BaseModel):
    result: TtsResult


class TtsSynthesisStage:
    name = "tts_synthesis"

    def __init__(self, provider: TextToSpeechProvider) -> None:
        self._provider = provider

    async def run(self, input: TtsSynthesisInput) -> TtsSynthesisOutput:
        result = await self._provider.synthesize(input.text, voice_profile=input.voice_profile)
        return TtsSynthesisOutput(result=result)
