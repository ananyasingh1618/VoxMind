from __future__ import annotations

import pytest

from ml.datasets.tess import build_manifest_from_directory, parse_tess_filename


def test_parses_real_tess_filename_convention():
    fields = parse_tess_filename("OAF_back_angry.wav")

    assert fields.speaker == "OAF"
    assert fields.word == "back"
    assert fields.emotion == "angry"


def test_rejects_malformed_filename():
    with pytest.raises(ValueError, match="Not a valid TESS filename"):
        parse_tess_filename("not-a-tess-file.wav")


def test_rejects_unknown_speaker():
    with pytest.raises(ValueError, match="Unknown TESS speaker"):
        parse_tess_filename("XXX_back_angry.wav")


def test_rejects_unknown_emotion():
    with pytest.raises(ValueError, match="Unknown TESS emotion"):
        parse_tess_filename("OAF_back_ecstatic.wav")


def test_build_manifest_skips_pleasant_surprise_and_non_matching_files(tmp_path):
    (tmp_path / "OAF_back_angry.wav").write_bytes(b"fake-audio")
    (tmp_path / "OAF_back_ps.wav").write_bytes(b"fake-audio")  # pleasant surprise - excluded
    (tmp_path / "readme.txt").write_bytes(b"not audio")

    samples = build_manifest_from_directory(tmp_path)

    assert len(samples) == 1
    assert samples[0].label == "angry"
    assert samples[0].speaker_id == "OAF"
    assert samples[0].dataset_source == "tess"


def test_build_manifest_raises_when_directory_has_no_matching_files(tmp_path):
    (tmp_path / "irrelevant.wav").write_bytes(b"x")

    with pytest.raises(FileNotFoundError, match="No TESS-named"):
        build_manifest_from_directory(tmp_path)
