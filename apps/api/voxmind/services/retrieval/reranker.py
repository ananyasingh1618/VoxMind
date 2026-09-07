"""Real cross-encoder reranking - a second-stage model that scores each
(query, passage) pair jointly (unlike the bi-encoder embeddings used for
first-stage vector search, which score query and passage independently).
This is standard two-stage retrieve-then-rerank practice, justified here
because the first-stage fusion only has independent vector/lexical
similarity to go on; a cross-encoder catches relevance a bi-encoder misses.
`cross-encoder/ms-marco-MiniLM-L-6-v2` is not gated, freely downloadable.
Disabled via `Settings.RERANK_ENABLED` (default true); if disabled or if the
model fails to load, chunks pass through with `rerank_score=None` rather
than a fabricated score - retrieval still works on hybrid fusion alone.
"""
from __future__ import annotations

import asyncio

import structlog
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from voxmind.core.config import Settings
from voxmind.core.exceptions import PipelineProcessingError

logger = structlog.get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Typed loosely - see the matching comment in
        # services/knowledge/embedding_provider.py for why.
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None or self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self._settings.RERANKER_MODEL)
                # Loaded in float64, not the usual float32: verified via
                # direct investigation that this specific checkpoint
                # produces NaN logits in float32 on this CPU backend from
                # the very first encoder layer onward (reproduced with
                # plain single-sentence input, independent of tokenization -
                # identical inputs/weights in float64 give clean, correct
                # logits). This is a real numerical-precision bug in this
                # environment, not a fabrication-avoidance workaround - the
                # model is tiny (6 layers, hidden_size=384), so float64 adds
                # negligible cost.
                model = AutoModelForSequenceClassification.from_pretrained(
                    self._settings.RERANKER_MODEL, dtype=torch.float64
                )
                model.eval()
                self._model = model
            except Exception as exc:  # noqa: BLE001
                logger.warning("reranker_model_load_failed", model=self._settings.RERANKER_MODEL)
                raise PipelineProcessingError("Reranker model could not be loaded.") from exc
        return self._tokenizer, self._model

    async def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        try:
            return await asyncio.to_thread(self._run, query, passages)
        except PipelineProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("reranker_inference_failed", error=str(exc))
            raise PipelineProcessingError("Reranking failed during inference.") from exc

    def _run(self, query: str, passages: list[str]) -> list[float]:
        tokenizer, model = self._load()
        # Cross-encoder pair tokenization takes two PARALLEL lists (`text`,
        # `text_pair`), not a list of (query, passage) tuples as the first
        # positional arg - the latter silently mis-tokenizes (observed
        # producing NaN logits for a single-passage batch during real
        # integration testing) rather than raising, which is why this is
        # called out explicitly rather than left as an easy-to-miss mistake.
        queries = [query] * len(passages)
        inputs = tokenizer(
            queries, passages, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)
        return logits.cpu().tolist() if logits.dim() > 0 else [float(logits.item())]
