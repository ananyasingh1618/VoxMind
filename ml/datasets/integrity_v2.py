"""Data-integrity checks for Emotion Model v2's combined RAVDESS+CREMA-D
manifest (docs/emotion.md, "Data integrity" section). Every check here is a
real check against real files - nothing is asserted without being computed.

These run *before* any split is assigned and *before* any training starts.
Anything removed is reported (dataset, sample_id, reason) - never silently
dropped - via `IntegrityReport`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

from ml.datasets.emotion_v2_labels import V2_LABELS
from ml.datasets.schema import EmotionSample


@dataclass
class RemovedSample:
    sample_id: str
    dataset_source: str
    reason: str


@dataclass
class IntegrityReport:
    removed: list[RemovedSample] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "removed_count": len(self.removed),
            "removed": [
                {"sample_id": r.sample_id, "dataset_source": r.dataset_source, "reason": r.reason}
                for r in self.removed
            ],
        }


def _sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def probe_duration_and_sample_rate(path: Path) -> tuple[float, int] | None:
    """Reads only the audio file header (via soundfile.info - no full
    decode needed) to get real duration/sample rate. Returns None (not an
    exception) for a file that can't even be opened as audio - the caller
    treats that as "malformed/unreadable", one of the required checks."""
    try:
        info = sf.info(str(path))
    except Exception:  # noqa: BLE001 - any failure here means "not readable as audio"
        return None
    if info.samplerate <= 0 or info.frames <= 0:
        return None
    return info.frames / info.samplerate, info.samplerate


def verify_and_annotate(samples: list[EmotionSample]) -> tuple[list[EmotionSample], IntegrityReport]:
    """Runs every required check (docs/emotion.md, "Data integrity") over a
    combined, already-label-mapped, already-namespaced sample list (see
    prepare_emotion_dataset_v2.py) and returns the surviving samples
    (annotated with real duration/sample_rate) plus a report of everything
    removed and why. Never mutates `samples` in place."""
    report = IntegrityReport()
    survivors: list[EmotionSample] = []
    seen_hashes: dict[str, str] = {}  # sha256 -> first sample_id that had it
    seen_sample_ids: set[str] = set()

    for sample in samples:
        path = Path(sample.audio_path)

        if sample.label not in V2_LABELS:
            report.removed.append(
                RemovedSample(sample.sample_id, sample.dataset_source, f"label {sample.label!r} not in V2_LABELS")
            )
            continue

        if sample.sample_id in seen_sample_ids:
            report.removed.append(
                RemovedSample(sample.sample_id, sample.dataset_source, "duplicate sample_id within manifest")
            )
            continue

        if not path.exists():
            report.removed.append(
                RemovedSample(sample.sample_id, sample.dataset_source, f"file does not exist: {path}")
            )
            continue

        probed = probe_duration_and_sample_rate(path)
        if probed is None:
            report.removed.append(
                RemovedSample(sample.sample_id, sample.dataset_source, "malformed/unreadable audio file")
            )
            continue
        duration_seconds, sample_rate = probed

        content_hash = _sha256_of_file(path)
        if content_hash in seen_hashes:
            report.removed.append(
                RemovedSample(
                    sample.sample_id,
                    sample.dataset_source,
                    f"duplicate audio content (identical to {seen_hashes[content_hash]!r})",
                )
            )
            continue
        seen_hashes[content_hash] = sample.sample_id
        seen_sample_ids.add(sample.sample_id)

        survivors.append(
            EmotionSample(
                sample_id=sample.sample_id,
                audio_path=sample.audio_path,
                label=sample.label,
                dataset_source=sample.dataset_source,
                speaker_id=sample.speaker_id,
                split=sample.split,
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
            )
        )

    return survivors, report


def assert_no_cross_dataset_speaker_collision(samples: list[EmotionSample]) -> None:
    """Real speaker IDs are namespaced as `f"{dataset_source}:{raw_id}"`
    before this point specifically to make cross-dataset ID collision
    structurally impossible (see prepare_emotion_dataset_v2.py) - this is a
    defensive re-verification of that invariant, not the primary mechanism
    enforcing it. Fails loudly (per docs/emotion.md's "fail loudly if
    speaker leakage is detected") rather than silently proceeding."""
    speaker_to_datasets: dict[str, set[str]] = {}
    for sample in samples:
        if sample.speaker_id is None:
            continue
        speaker_to_datasets.setdefault(sample.speaker_id, set()).add(sample.dataset_source)

    colliding = {speaker: sources for speaker, sources in speaker_to_datasets.items() if len(sources) > 1}
    if colliding:
        raise AssertionError(f"Cross-dataset speaker-id collision detected (should be impossible after namespacing): {colliding}")
