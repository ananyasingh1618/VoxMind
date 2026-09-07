"""Extracts a [start_ms, end_ms) sub-segment from a canonical WAV byte
string - used to isolate one aligned turn's audio before feature/embedding
extraction, since Phase 3 predicts emotion per speaker turn, not per whole
recording (see docs/emotion.md for the granularity decision).
"""
from __future__ import annotations

import io

import soundfile as sf

from voxmind.core.exceptions import InvalidAudioError


def slice_wav_segment(wav_bytes: bytes, *, start_ms: int, end_ms: int) -> tuple[bytes, int]:
    """Returns (segment_wav_bytes, sample_rate)."""
    with sf.SoundFile(io.BytesIO(wav_bytes)) as f:
        sample_rate = f.samplerate
        total_frames = len(f)
        start_frame = max(0, int(start_ms / 1000 * sample_rate))
        end_frame = min(total_frames, int(end_ms / 1000 * sample_rate))
        frame_count = end_frame - start_frame
        if frame_count <= 0:
            raise InvalidAudioError(
                f"Requested segment [{start_ms}ms, {end_ms}ms) is empty or out of range."
            )
        f.seek(start_frame)
        data = f.read(frame_count, dtype="float32")

    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), sample_rate
