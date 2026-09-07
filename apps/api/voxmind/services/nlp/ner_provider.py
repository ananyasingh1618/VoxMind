"""Real named-entity recognition via a Hugging Face Transformers pipeline.
`dslim/bert-base-NER` is not gated - freely downloadable. `aggregation_strategy="simple"`
merges sub-word BIO tags (B-PER/I-PER/...) into whole-entity spans with
plain labels (PER/ORG/LOC/MISC) before they ever leave this module.
"""
from __future__ import annotations

import asyncio

import structlog
from transformers import pipeline

from voxmind.core.config import Settings
from voxmind.core.exceptions import PipelineProcessingError
from voxmind.services.nlp.interfaces import Entity

logger = structlog.get_logger(__name__)


class HuggingFaceNerProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline = None

    @property
    def model_id(self) -> str:
        return self._settings.NER_MODEL

    def _load(self):
        if self._pipeline is None:
            try:
                self._pipeline = pipeline(
                    "ner",
                    model=self._settings.NER_MODEL,
                    aggregation_strategy="simple",
                    device=-1 if self._settings.NLP_DEVICE == "cpu" else 0,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ner_model_load_failed", model=self._settings.NER_MODEL)
                raise PipelineProcessingError("NER model could not be loaded.") from exc
        return self._pipeline

    async def extract(self, text: str) -> list[Entity]:
        try:
            return await asyncio.to_thread(self._run, text)
        except PipelineProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("ner_inference_failed", error=str(exc))
            raise PipelineProcessingError("Entity extraction failed during inference.") from exc

    def _run(self, text: str) -> list[Entity]:
        clf = self._load()
        truncated = text[:2000]
        results = clf(truncated)
        return [
            Entity(
                text=item["word"],
                label=item["entity_group"],
                start_char=int(item["start"]),
                end_char=int(item["end"]),
            )
            for item in results
        ]
