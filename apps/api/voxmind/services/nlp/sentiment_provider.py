"""Real 3-class sentiment classification via a Hugging Face Transformers
pipeline. `cardiffnlp/twitter-roberta-base-sentiment-latest` is not gated -
freely downloadable, same "real, no credentials" bar as Phase 3's
facebook/wav2vec2-base. Runs on CPU; a single short message is cheap enough
that no batching is needed (mirrors the one-clip-at-a-time reasoning in
services/emotion/wav2vec_provider.py).
"""
from __future__ import annotations

import asyncio

import structlog
from transformers import pipeline

from voxmind.core.config import Settings
from voxmind.core.exceptions import PipelineProcessingError

logger = structlog.get_logger(__name__)


class SentimentResult:
    __slots__ = ("label", "score")

    def __init__(self, label: str, score: float) -> None:
        self.label = label
        self.score = score


class HuggingFaceSentimentProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline = None

    @property
    def model_id(self) -> str:
        return self._settings.SENTIMENT_MODEL

    def _load(self):
        if self._pipeline is None:
            try:
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self._settings.SENTIMENT_MODEL,
                    device=-1 if self._settings.NLP_DEVICE == "cpu" else 0,
                )
            except Exception as exc:  # noqa: BLE001 - huggingface_hub raises many exception types
                logger.warning("sentiment_model_load_failed", model=self._settings.SENTIMENT_MODEL)
                raise PipelineProcessingError("Sentiment model could not be loaded.") from exc
        return self._pipeline

    async def analyze(self, text: str) -> SentimentResult:
        try:
            result = await asyncio.to_thread(self._run, text)
        except PipelineProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("sentiment_inference_failed", error=str(exc))
            raise PipelineProcessingError("Sentiment analysis failed during inference.") from exc
        return result

    def _run(self, text: str) -> SentimentResult:
        clf = self._load()
        # Most transformer encoders cap at 512 tokens; a hard character
        # truncation keeps this simple and avoids a second tokenizer pass
        # just to count tokens for a short chat message.
        output = clf(text[:2000], truncation=True)[0]
        return SentimentResult(label=output["label"].lower(), score=float(output["score"]))
