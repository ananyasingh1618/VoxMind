"""The training-data abstraction every emotion dataset (RAVDESS, or any
future one) is normalized into. Pure dataclasses with no dependency on the
`voxmind` application package - this module is usable standalone for
dataset preparation, independent of the model/training code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass
class EmotionSample:
    sample_id: str
    audio_path: str
    label: str
    dataset_source: str
    speaker_id: str | None = None
    split: DatasetSplit | None = None
    # Added for Emotion Model v2's data-integrity manifest requirement
    # (docs/emotion.md, "Dataset manifest"): both fields are optional and
    # default to None so the existing, frozen ravdess-v1 manifest
    # (ml/datasets/manifests/ravdess_manifest.jsonl) and its own loader
    # remain byte-for-byte compatible - v1 never populates these, v2 always
    # does. Duration/sample-rate are measured directly from each audio
    # file (never assumed) by the v2 manifest builder.
    duration_seconds: float | None = None
    sample_rate: int | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["split"] = self.split.value if self.split else None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionSample":
        split = DatasetSplit(data["split"]) if data.get("split") else None
        return cls(
            sample_id=data["sample_id"],
            audio_path=data["audio_path"],
            label=data["label"],
            dataset_source=data["dataset_source"],
            speaker_id=data.get("speaker_id"),
            split=split,
            duration_seconds=data.get("duration_seconds"),
            sample_rate=data.get("sample_rate"),
        )


def save_manifest(samples: list[EmotionSample], path: str | Path) -> None:
    """One JSON object per line (JSONL) - simple, diffable, streamable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")


def load_manifest(path: str | Path) -> list[EmotionSample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found at {path}. Run "
            "ml/datasets/prepare_emotion_dataset.py first - see ml/datasets/README.md."
        )
    samples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(EmotionSample.from_dict(json.loads(line)))
    return samples


def label_distribution(samples: list[EmotionSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.label] = counts.get(sample.label, 0) + 1
    return dict(sorted(counts.items()))
