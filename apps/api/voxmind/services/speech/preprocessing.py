"""Real audio preprocessing: decodes any supported source container into a
canonical mono, 16kHz, 16-bit PCM WAV using ffmpeg, then validates the
result with soundfile (independent of whatever ffmpeg itself reported).

Deliberately uses ffmpeg via stdin/stdout pipes rather than temp files -
the raw upload bytes and the decoded WAV bytes both stay in memory for the
lifetime of this call. This is a stronger security/cleanup posture than
writing-then-deleting temp files: there is nothing on disk to leak or to
forget to clean up in the first place. ffmpeg's own demuxer probing (reading
and buffering the start of the input stream) works the same over a pipe as
over a file, so this doesn't sacrifice format detection.

The original upload is never overwritten - callers persist this function's
output as a *separate* AudioAsset (kind="processed"), linked back to the
original via `source_asset_id`.
"""
from __future__ import annotations

import asyncio
import io

import soundfile as sf
import structlog

from voxmind.core.config import Settings
from voxmind.core.exceptions import AudioProcessingError, InvalidAudioError
from voxmind.services.speech.interfaces import PreprocessedAudio

logger = structlog.get_logger(__name__)

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


async def _run_ffmpeg(raw_bytes: bytes) -> bytes:
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-ac",
            str(TARGET_CHANNELS),
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            "-f",
            "wav",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        # ffmpeg isn't installed on this host - a deployment/config problem,
        # not something caused by the uploaded file.
        raise AudioProcessingError("Audio preprocessing is unavailable on this server.") from exc

    stdout, stderr = await process.communicate(input=raw_bytes)

    if process.returncode != 0 or not stdout:
        logger.warning(
            "audio_preprocessing_ffmpeg_failed",
            returncode=process.returncode,
            stderr_tail=stderr[-500:].decode("utf-8", errors="replace") if stderr else "",
        )
        raise InvalidAudioError(
            "Audio could not be decoded - it may be corrupt, empty, or use an unsupported encoding."
        )
    return stdout


async def preprocess_audio(raw_bytes: bytes, *, settings: Settings) -> tuple[PreprocessedAudio, bytes]:
    """Returns (metadata, canonical_wav_bytes). Raises InvalidAudioError for
    any audio that fails real decoding or falls outside configured duration
    bounds; raises AudioProcessingError for environment/deployment failures
    (e.g. ffmpeg missing) that aren't the caller's fault.
    """
    wav_bytes = await _run_ffmpeg(raw_bytes)

    try:
        with sf.SoundFile(io.BytesIO(wav_bytes)) as f:
            frames = len(f)
            sample_rate = f.samplerate
            channels = f.channels
    except Exception as exc:  # noqa: BLE001 - any libsndfile failure means genuinely invalid audio
        raise InvalidAudioError("Decoded audio failed validation and cannot be processed.") from exc

    if frames == 0 or sample_rate == 0:
        raise InvalidAudioError("Audio contains no playable samples.")

    duration_seconds = frames / sample_rate

    if duration_seconds > settings.MAX_AUDIO_DURATION_SECONDS:
        raise InvalidAudioError(
            f"Audio duration ({duration_seconds:.1f}s) exceeds the "
            f"{settings.MAX_AUDIO_DURATION_SECONDS}s limit."
        )

    metadata = PreprocessedAudio(
        storage_key="",  # filled in by the caller once uploaded to storage
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        format="wav",
    )
    return metadata, wav_bytes
