"""Loads the currently-registered active emotion classifier - extracted out
of `EmotionService` (Phase 9) so the exact same logic can be reused by the
Celery worker's stage registry (`workers/stage_registry.py`) without
duplicating it. Behavior is unchanged from the original
`EmotionService._load_active_classifier`: returns `None` (never a
fabricated classifier) when no trained+active model is registered, per
docs/DECISIONS/0005-emotion-requires-trained-model.md.

Extended for Emotion Model v2 (docs/emotion.md): a `ModelVersion` row is
tagged `training_config["architecture"] == "v2-wav2vec2-attention"` by
`ml/training/train_emotion_model_v2.py` when it registers a v2 model; a
missing key (every v1 row, including the frozen `ravdess-v1`) defaults to
v1's architecture, so nothing about existing rows changes meaning. This is
an additive `training_config` convention, not a schema change - no
migration was needed. `load_active_classifier` now returns the
embedding-provider paired with whichever classifier it loaded (`None` for
v1, since v1's provider is the fixed, shared
`deps.get_wav2vec2_provider()` singleton every caller already has) - see
`v2/inference.py`'s module docstring for exactly why v2 needs a *paired*,
model-version-specific provider instead of that fixed singleton.

Extended again for the emotion2vec+ production candidate
(`training_config["architecture"] == "emotion2vec-plus-base"`,
`emotion2vec/classifier_provider.py`/`encoder.py`) - the same paired-
provider pattern v2 established, since emotion2vec+'s frozen encoder is
also not the fixed Wav2Vec2 singleton every v1 caller already has.
"""
from __future__ import annotations

from typing import Union

from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.config import Settings
from voxmind.models.model_version import ModelVersion
from voxmind.repositories.model_version_repository import ModelVersionRepository
from voxmind.services.emotion.classifier_provider import TorchEmotionClassifier
from voxmind.services.emotion.emotion2vec.classifier_provider import Emotion2VecClassifier
from voxmind.services.emotion.emotion2vec.encoder import Emotion2VecEncoder
from voxmind.services.emotion.emotion2vec.model import Emotion2VecCheckpoint
from voxmind.services.emotion.interfaces import Wav2Vec2EmbeddingProvider
from voxmind.services.emotion.model import EmotionCheckpoint
from voxmind.services.emotion.v2.inference import TorchEmotionClassifierV2, build_inference_components
from voxmind.services.emotion.v2.model import EmotionCheckpointV2
from voxmind.services.storage.interfaces import StorageBackend

EMOTION_COMPONENT = "emotion_classifier"
V2_ARCHITECTURE_TAG = "v2-wav2vec2-attention"
EMOTION2VEC_ARCHITECTURE_TAG = "emotion2vec-plus-base"

AnyEmotionClassifier = Union[TorchEmotionClassifier, TorchEmotionClassifierV2, Emotion2VecClassifier]


async def load_active_classifier(
    session: AsyncSession, storage: StorageBackend, *, settings: Settings | None = None
) -> tuple[AnyEmotionClassifier, ModelVersion, Wav2Vec2EmbeddingProvider | None] | None:
    """Returns `(classifier, model_version, embedding_provider_override)`.
    `embedding_provider_override` is `None` for a v1 model (callers keep
    using their own fixed, shared embedding provider) and a real,
    model-version-specific provider for a v2 or emotion2vec+ model - see
    module docstring."""
    active = await ModelVersionRepository(session).get_active_for_component(EMOTION_COMPONENT)
    if active is None or not active.artifact_storage_key:
        return None

    architecture = (active.training_config or {}).get("architecture")
    checkpoint_bytes = await storage.download(active.artifact_storage_key)

    resolved_settings = settings
    if resolved_settings is None:
        from voxmind.core.config import get_settings

        resolved_settings = get_settings()

    if architecture == V2_ARCHITECTURE_TAG:
        checkpoint_v2 = EmotionCheckpointV2.from_bytes(checkpoint_bytes)
        provider, classifier_v2 = build_inference_components(
            checkpoint_v2, model_version_id=str(active.id), trained=True, settings=resolved_settings
        )
        return classifier_v2, active, provider

    if architecture == EMOTION2VEC_ARCHITECTURE_TAG:
        checkpoint_e2v = Emotion2VecCheckpoint.from_bytes(checkpoint_bytes)
        classifier_e2v = Emotion2VecClassifier.from_checkpoint(
            checkpoint_e2v, model_version_id=str(active.id), trained=True
        )
        provider_e2v = Emotion2VecEncoder(resolved_settings)
        return classifier_e2v, active, provider_e2v

    checkpoint = EmotionCheckpoint.from_bytes(checkpoint_bytes)
    classifier = TorchEmotionClassifier.from_checkpoint(
        checkpoint, model_version_id=str(active.id), trained=True
    )
    return classifier, active, None
