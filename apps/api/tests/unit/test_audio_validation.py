from __future__ import annotations

import pytest

from voxmind.core.config import get_settings
from voxmind.core.exceptions import AudioTooLargeError, UnsupportedAudioFormatError
from voxmind.services.speech.audio_validation import (
    safe_filename,
    sniff_audio_format,
    validate_audio_upload,
)

WAV_HEADER = b"RIFF\x24\x00\x00\x00WAVEfmt "
MP3_ID3_HEADER = b"ID3\x03\x00\x00\x00\x00\x00\x00"
MP3_FRAME_HEADER = bytes([0xFF, 0xFB, 0x90, 0x00])
OGG_HEADER = b"OggS\x00\x02\x00\x00"
WEBM_HEADER = b"\x1a\x45\xdf\xa3\x01\x00\x00\x00"
M4A_HEADER = b"\x00\x00\x00\x18ftypM4A "


@pytest.mark.parametrize(
    "header,expected",
    [
        (WAV_HEADER, "wav"),
        (MP3_ID3_HEADER, "mp3"),
        (MP3_FRAME_HEADER, "mp3"),
        (OGG_HEADER, "ogg"),
        (WEBM_HEADER, "webm"),
        (M4A_HEADER, "m4a"),
        (b"not an audio file at all", None),
        (b"", None),
    ],
)
def test_sniff_audio_format(header: bytes, expected: str | None) -> None:
    assert sniff_audio_format(header) == expected


def test_sniff_never_trusts_extension_only_content() -> None:
    """A file that merely has a .wav-looking name but PNG magic bytes must
    not be classified as wav - sniffing is content-based only."""
    png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    assert sniff_audio_format(png_header) is None


def test_validate_audio_upload_accepts_supported_format() -> None:
    settings = get_settings()
    detected = validate_audio_upload(header=WAV_HEADER, size_bytes=1000, settings=settings)
    assert detected == "wav"


def test_validate_audio_upload_rejects_unsupported_format() -> None:
    settings = get_settings()
    with pytest.raises(UnsupportedAudioFormatError):
        validate_audio_upload(header=b"not audio", size_bytes=1000, settings=settings)


def test_validate_audio_upload_rejects_empty_file() -> None:
    settings = get_settings()
    with pytest.raises(UnsupportedAudioFormatError):
        validate_audio_upload(header=WAV_HEADER, size_bytes=0, settings=settings)


def test_validate_audio_upload_rejects_oversized_file() -> None:
    settings = get_settings()
    with pytest.raises(AudioTooLargeError):
        validate_audio_upload(
            header=WAV_HEADER, size_bytes=settings.MAX_AUDIO_UPLOAD_BYTES + 1, settings=settings
        )


@pytest.mark.parametrize(
    "original,expected",
    [
        ("song.wav", "song.wav"),
        ("../../etc/passwd.wav", "passwd.wav"),
        ("..\\..\\windows\\system32\\evil.wav", "evil.wav"),
        ("my recording (final) v2.mp3", "my_recording_final_v2.mp3"),
        (None, "audio.wav"),
        ("", "audio.wav"),
        ("....", "audio.wav"),
    ],
)
def test_safe_filename(original: str | None, expected: str) -> None:
    assert safe_filename(original, fallback_extension="wav") == expected


def test_safe_filename_truncates_extremely_long_names() -> None:
    long_name = ("a" * 500) + ".wav"
    result = safe_filename(long_name, fallback_extension="wav")
    assert len(result) <= 255
