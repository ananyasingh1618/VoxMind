"""`NlpAnalysisStage` implements the approved, unmodified `PipelineStage`
contract - same pattern as Phase 2/3's stage modules. Plain, JSON-serializable
Pydantic input/output only.
"""
from __future__ import annotations

from pydantic import BaseModel

from voxmind.services.nlp.analyzer import RealNlpAnalyzer
from voxmind.services.nlp.interfaces import NlpAnnotation


class NlpAnalysisInput(BaseModel):
    text: str


class NlpAnalysisOutput(BaseModel):
    annotation: NlpAnnotation


class NlpAnalysisStage:
    name = "nlp_analysis"

    def __init__(self, analyzer: RealNlpAnalyzer) -> None:
        self._analyzer = analyzer

    async def run(self, input: NlpAnalysisInput) -> NlpAnalysisOutput:
        annotation = await self._analyzer.analyze(input.text)
        return NlpAnalysisOutput(annotation=annotation)
