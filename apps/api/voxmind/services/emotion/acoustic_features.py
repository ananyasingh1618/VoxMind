"""Real acoustic feature extraction via librosa - every number here is
computed from the actual decoded audio signal, never invented. See
`interfaces.AcousticFeatures` for what's extracted and why a plausible but
musically-oriented feature (tempo/"speech rate") was deliberately left out.

Consumes the canonical mono/16kHz WAV bytes that Phase 2's preprocessing
stage already produces - no separate resampling/channel-mixing needed here.
"""
from __future__ import annotations

import io

import librosa
import numpy as np
import soundfile as sf
import structlog

from voxmind.core.exceptions import AudioProcessingError
from voxmind.services.emotion.interfaces import AcousticFeatures, AcousticStat

logger = structlog.get_logger(__name__)

N_MFCC = 13
FRAME_LENGTH = 2048
HOP_LENGTH = 512
FMIN_HZ = 65.0  # below typical adult male voice floor
FMAX_HZ = 400.0  # above typical adult female voice ceiling


def _stat(values: np.ndarray) -> AcousticStat:
    if values.size == 0:
        return AcousticStat(mean=0.0, std=0.0, min=0.0, max=0.0, p25=0.0, p50=0.0, p75=0.0)
    return AcousticStat(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        p25=float(np.percentile(values, 25)),
        p50=float(np.percentile(values, 50)),
        p75=float(np.percentile(values, 75)),
    )


class LibrosaAcousticFeatureExtractor:
    async def extract(self, audio_bytes: bytes, *, sample_rate: int) -> AcousticFeatures:
        # librosa's analysis itself is CPU-bound and synchronous; this
        # provider is small enough that running it inline (no thread
        # offload) is acceptable for Phase 3's segment-length clips - the
        # PipelineStage wrapping it is still async-shaped for TaskRunner.
        try:
            samples, actual_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:  # noqa: BLE001
            raise AudioProcessingError("Acoustic feature extraction received unreadable audio.") from exc

        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        samples = np.asarray(samples, dtype=np.float32)

        if samples.size == 0:
            raise AudioProcessingError("Acoustic feature extraction received empty audio.")

        duration_seconds = len(samples) / actual_rate

        rms = librosa.feature.rms(y=samples, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        zcr = librosa.feature.zero_crossing_rate(
            y=samples, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
        )[0]
        centroid = librosa.feature.spectral_centroid(
            y=samples, sr=actual_rate, hop_length=HOP_LENGTH
        )[0]
        bandwidth = librosa.feature.spectral_bandwidth(
            y=samples, sr=actual_rate, hop_length=HOP_LENGTH
        )[0]
        rolloff = librosa.feature.spectral_rolloff(y=samples, sr=actual_rate, hop_length=HOP_LENGTH)[0]
        mfcc = librosa.feature.mfcc(
            y=samples, sr=actual_rate, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
        )

        # pyin needs at least one full frame; shorter clips genuinely have
        # no computable pitch track rather than a fabricated one.
        if len(samples) >= FRAME_LENGTH:
            f0, voiced_flag, _voiced_prob = librosa.pyin(
                samples,
                fmin=FMIN_HZ,
                fmax=FMAX_HZ,
                sr=actual_rate,
                frame_length=FRAME_LENGTH,
                hop_length=HOP_LENGTH,
            )
            voiced_flag = np.nan_to_num(voiced_flag.astype(np.float32), nan=0.0).astype(bool)
            voiced_f0 = f0[voiced_flag]
            voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
            voiced_fraction = float(np.mean(voiced_flag)) if voiced_flag.size else 0.0
        else:
            voiced_f0 = np.array([], dtype=np.float32)
            voiced_fraction = 0.0

        return AcousticFeatures(
            duration_seconds=duration_seconds,
            voiced_fraction=voiced_fraction,
            pitch_hz=_stat(voiced_f0),
            rms_energy=_stat(rms),
            zero_crossing_rate=_stat(zcr),
            spectral_centroid=_stat(centroid),
            spectral_bandwidth=_stat(bandwidth),
            spectral_rolloff=_stat(rolloff),
            mfcc=[_stat(mfcc[i]) for i in range(N_MFCC)],
        )
