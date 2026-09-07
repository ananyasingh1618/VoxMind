"""Ingestion-time audio validation: format sniffing, size limits, and safe
filename handling. Deliberately does NOT decode the audio - that is real
preprocessing's job (preprocessing.py) and requires ffmpeg. This module only
answers "is this plausibly one of our supported containers, and is it a
sane size/name" cheaply, without trusting the client.

Sniffing is based on magic bytes at the start of the file, never on the
browser/client-supplied Content-Type header - a renamed or mislabeled file
is caught here; a same-named-but-corrupt file is caught later by
preprocessing's real ffmpeg decode.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from voxmind.core.config import Settings
from voxmind.core.exceptions import AudioTooLargeError, UnsupportedAudioFormatError

_MAGIC_SNIFFERS: list[tuple[str, Callable[[bytes], bool]]] = [
    ("wav", lambda h: h[:4] == b"RIFF" and h[8:12] == b"WAVE"),
    ("ogg", lambda h: h[:4] == b"OggS"),
    ("webm", lambda h: h[:4] == b"\x1a\x45\xdf\xa3"),
    ("m4a", lambda h: h[4:8] == b"ftyp"),
    ("mp3", lambda h: h[:3] == b"ID3" or (len(h) >= 2 and h[0] == 0xFF and (h[1] & 0xE0) == 0xE0)),
]

_SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sniff_audio_format(header: bytes) -> str | None:
    """Returns the detected format ('wav', 'mp3', 'm4a', 'ogg', 'webm') from
    the file's magic bytes, or None if nothing recognized matched."""
    for fmt, matcher in _MAGIC_SNIFFERS:
        if matcher(header):
            return fmt
    return None


def validate_audio_upload(
    *, header: bytes, size_bytes: int, settings: Settings
) -> str:
    """Validates a candidate audio upload and returns its sniffed format.
    Raises UnsupportedAudioFormatError / AudioTooLargeError - never trusts
    client-declared metadata."""
    if size_bytes > settings.MAX_AUDIO_UPLOAD_BYTES:
        raise AudioTooLargeError(
            f"Audio upload exceeds the {settings.MAX_AUDIO_UPLOAD_BYTES} byte limit."
        )
    if size_bytes == 0:
        raise UnsupportedAudioFormatError("Uploaded file is empty.")

    detected_format = sniff_audio_format(header)
    if detected_format is None or detected_format not in settings.SUPPORTED_AUDIO_EXTENSIONS:
        raise UnsupportedAudioFormatError(
            "Unrecognized or unsupported audio format. Supported formats: "
            + ", ".join(settings.SUPPORTED_AUDIO_EXTENSIONS)
        )
    return detected_format


def safe_filename(original_filename: str | None, *, fallback_extension: str) -> str:
    """Strips any directory components and unsafe characters from a
    client-supplied filename, guaranteeing the result can never escape the
    intended storage prefix (defends against path traversal via a crafted
    filename like '../../etc/passwd')."""
    if not original_filename:
        return f"audio.{fallback_extension}"
    basename = original_filename.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = _SAFE_FILENAME_CHARS.sub("_", basename).strip("._") or f"audio.{fallback_extension}"
    return cleaned[:255]
