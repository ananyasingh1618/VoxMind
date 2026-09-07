"""Real sentence embeddings via plain Hugging Face Transformers - no
`sentence-transformers` package dependency, same pattern as
`HuggingFaceWav2Vec2Provider` (Phase 3): load a base encoder
(`AutoModel`/`AutoTokenizer`), run it, and pool the token-level output into
one fixed-size vector.

Pooling: attention-mask-aware mean pooling over `last_hidden_state`, then
L2-normalization - the standard recipe `sentence-transformers/all-MiniLM-L6-v2`
was itself trained/evaluated with, so cosine similarity between normalized
vectors is meaningful (this is also why `KnowledgeChunkRepository.search_vector`
uses cosine distance, not L2).
"""
from __future__ import annotations

import asyncio

import numpy as np
import structlog
import torch
from transformers import AutoModel, AutoTokenizer

from voxmind.core.config import Settings
from voxmind.core.exceptions import PipelineProcessingError

logger = structlog.get_logger(__name__)


class HuggingFaceEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Typed loosely (not `AutoModel`/`AutoTokenizer`) - those "Auto*"
        # classes are factories whose `.from_pretrained()` returns a
        # concrete-but-stub-untyped instance; annotating with the factory
        # type itself makes mypy think the *class* is being called later.
        self._model = None
        self._tokenizer = None

    @property
    def model_id(self) -> str:
        return self._settings.EMBEDDING_MODEL

    def _load(self):
        if self._model is None or self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self._settings.EMBEDDING_MODEL)
                model = AutoModel.from_pretrained(self._settings.EMBEDDING_MODEL)
                model.eval()
                model.to(self._settings.EMBEDDING_DEVICE)
                self._model = model
            except Exception as exc:  # noqa: BLE001
                logger.warning("embedding_model_load_failed", model=self._settings.EMBEDDING_MODEL)
                raise PipelineProcessingError("Embedding model could not be loaded.") from exc
        return self._tokenizer, self._model

    async def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise PipelineProcessingError("Cannot embed empty text.")
        try:
            return await asyncio.to_thread(self._run, text)
        except PipelineProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding_inference_failed", error=str(exc))
            raise PipelineProcessingError("Embedding extraction failed during inference.") from exc

    def _run(self, text: str) -> list[float]:
        tokenizer, model = self._load()
        inputs = tokenizer(
            text, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        inputs = {k: v.to(self._settings.EMBEDDING_DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # [1, T, H]

        mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vector = normalized.squeeze(0).cpu().numpy()
        return np.asarray(vector, dtype=np.float32).tolist()
