#!/usr/bin/env python3
"""Secondary, restricted "fair comparison" between the frozen v1
(`ravdess-v1`, 8-class RAVDESS) and Emotion Model v2 (6-class RAVDESS+
CREMA-D) - docs/emotion.md, "V1 vs V2" section.

**v1 is never modified, retrained, or re-evaluated with different
weights.** This script re-runs v1's *existing, frozen, already-registered*
checkpoint through its *existing, unmodified* inference path
(acoustic features + frozen Wav2Vec2 embedding -> MLP), then restricts the
scoring - not the model - to the six classes v1 and v2 share, and only
over the *same* 3 held-out RAVDESS test speakers (verified identical
between the two manifests: v1's `ravdess_manifest.jsonl` and v2's combined
manifest both shuffle the same 24 RAVDESS actor ids with the same seed=42,
so their test-speaker assignment is provably identical - checked, not
assumed, before this script was written).

"fearful" (v1's label) and "fear" (v2's label) are treated as the same
class for this comparison - a naming normalization only (see
ml/datasets/emotion_v2_labels.py's identical note for the RAVDESS->v2
mapping), never a reinterpretation of the emotion itself. RAVDESS clips
whose true label is "calm" or "surprised" (outside v2's six-class task
entirely) are excluded from this comparison's ground truth - v1's
predictions on them are not scored here, since v2 was never asked to
recognize those classes at all.

This produces a **secondary** comparison only. The **primary** statement
of each model's performance is its own native task (v1: 8-class RAVDESS;
v2: 6-class RAVDESS+CREMA-D) - see docs/emotion.md for both.

Usage:
    python -m ml.evaluation.compare_v1_v2_ravdess_six_class \\
        --v1-manifest ml/datasets/manifests/ravdess_manifest.jsonl \\
        --v2-manifest ml/datasets/manifests/emotion_v2_manifest.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from ml.datasets.schema import DatasetSplit, load_manifest
from ml.evaluation.metrics import compute_classification_metrics
from voxmind.core.config import get_settings
from voxmind.services.emotion.acoustic_features import LibrosaAcousticFeatureExtractor
from voxmind.services.emotion.classifier_provider import TorchEmotionClassifier
from voxmind.services.emotion.model import EmotionCheckpoint
from voxmind.services.emotion.wav2vec_provider import HuggingFaceWav2Vec2Provider
from voxmind.services.speech.preprocessing import preprocess_audio
from voxmind.services.storage.factory import build_storage_backend

V1_TO_V2_LABEL = {
    "angry": "angry", "disgust": "disgust", "fearful": "fear",
    "happy": "happy", "neutral": "neutral", "sad": "sad",
    # "calm" and "surprised" deliberately have no entry - excluded, not remapped.
}
SIX_CLASS_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


async def main_async(args: argparse.Namespace) -> None:
    settings = get_settings()

    v1_samples = load_manifest(args.v1_manifest)
    v1_test = [s for s in v1_samples if s.split == DatasetSplit.TEST]
    v1_test_speakers = sorted({s.speaker_id for s in v1_test})

    v2_samples = load_manifest(args.v2_manifest)
    v2_ravdess_test_speakers = sorted(
        {s.speaker_id.split(":", 1)[1] for s in v2_samples if s.split == DatasetSplit.TEST and s.dataset_source == "ravdess"}
    )
    if v1_test_speakers != v2_ravdess_test_speakers:
        print(
            f"error: v1 test speakers {v1_test_speakers} != v2 RAVDESS test speakers "
            f"{v2_ravdess_test_speakers} - this comparison requires them to be identical "
            "(they were verified identical when this script was written; if a manifest was "
            "regenerated with a different seed/fraction since, this check will correctly refuse "
            "to produce a misleading comparison).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Confirmed: v1 and v2 share the same {len(v1_test_speakers)} held-out RAVDESS test speakers: {v1_test_speakers}")

    six_class_v1_test = [s for s in v1_test if s.label in V1_TO_V2_LABEL]
    excluded = len(v1_test) - len(six_class_v1_test)
    print(f"v1 test set: {len(v1_test)} total, {len(six_class_v1_test)} within the six shared classes "
          f"({excluded} calm/surprised clips excluded from this comparison).")

    storage = build_storage_backend(settings)
    checkpoint_bytes = await storage.download(f"{settings.EMOTION_MODEL_ARTIFACT_ROOT}/ravdess-v1/classifier.pt")
    checkpoint = EmotionCheckpoint.from_bytes(checkpoint_bytes)
    classifier = TorchEmotionClassifier.from_checkpoint(checkpoint, model_version_id="ravdess-v1", trained=True)
    extractor = LibrosaAcousticFeatureExtractor()
    embedder = HuggingFaceWav2Vec2Provider(settings)

    y_true: list[str] = []
    y_pred: list[str] = []
    for sample in six_class_v1_test:
        raw_bytes = Path(sample.audio_path).read_bytes()
        _metadata, wav_bytes = await preprocess_audio(raw_bytes, settings=settings)
        features = await extractor.extract(wav_bytes, sample_rate=16000)
        embedding = await embedder.embed(wav_bytes, sample_rate=16000)
        prediction = await classifier.predict(features, embedding)
        y_true.append(V1_TO_V2_LABEL[sample.label])
        # v1's raw prediction is used as-is (never modified) - if it predicts
        # "calm"/"surprised" for a six-class sample, that's scored as
        # simply wrong against the six-class label set, exactly like any
        # other real classification error would be.
        predicted = V1_TO_V2_LABEL.get(prediction.label, prediction.label)
        y_pred.append(predicted)

    v1_six_class_metrics = compute_classification_metrics(y_true, y_pred, SIX_CLASS_LABELS)

    print("\n=== v1 (ravdess-v1), restricted to the six shared classes, same 3 held-out speakers ===")
    print(f"n={v1_six_class_metrics.n_samples} accuracy={v1_six_class_metrics.accuracy:.4f} "
          f"macro_f1={v1_six_class_metrics.macro_f1:.4f} weighted_f1={v1_six_class_metrics.weighted_f1:.4f} "
          f"balanced_accuracy={v1_six_class_metrics.balanced_accuracy:.4f}")
    print("per-class:")
    for label, m in v1_six_class_metrics.per_class.items():
        print(f"  {label}: precision={m.precision:.3f} recall={m.recall:.3f} f1={m.f1:.3f} support={m.support}")

    print(
        "\nNOTE: this is a SECONDARY, restricted comparison (six shared classes, "
        "same 3 RAVDESS speakers only) - it does not represent v1's native 8-class "
        "performance (see docs/emotion.md's 'Genuine trained model: ravdess-v1' section "
        "for that) nor v2's native cross-corpus performance (see this file's own RAVDESS "
        "test-subset metric, already computed by train_emotion_model_v2.py and stored in "
        "the emotion-v2 ModelVersion row's metrics.test_by_dataset_source.ravdess)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v1-manifest", default="ml/datasets/manifests/ravdess_manifest.jsonl")
    parser.add_argument("--v2-manifest", default="ml/datasets/manifests/emotion_v2_manifest.jsonl")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
