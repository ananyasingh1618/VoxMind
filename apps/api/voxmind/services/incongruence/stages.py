from __future__ import annotations

from pydantic import BaseModel

from voxmind.services.incongruence.analyzer import DeterministicIncongruenceAnalyzer
from voxmind.services.incongruence.interfaces import IncongruenceSignal


class IncongruenceAnalysisInput(BaseModel):
    semantic_signal: dict
    vocal_signal: dict


class IncongruenceAnalysisOutput(BaseModel):
    signal: IncongruenceSignal


class IncongruenceAnalysisStage:
    name = "incongruence_analysis"

    def __init__(self, analyzer: DeterministicIncongruenceAnalyzer) -> None:
        self._analyzer = analyzer

    async def run(self, input: IncongruenceAnalysisInput) -> IncongruenceAnalysisOutput:
        signal = await self._analyzer.analyze(input.semantic_signal, input.vocal_signal)
        return IncongruenceAnalysisOutput(signal=signal)
