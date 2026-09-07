from __future__ import annotations

import pytest

from ml.datasets.ravdess import build_manifest_from_directory, parse_ravdess_filename


def test_parses_real_ravdess_filename_convention():
    fields = parse_ravdess_filename("03-01-06-01-02-01-12.wav")

    assert fields.modality == "03"
    assert fields.vocal_channel == "01"
    assert fields.emotion_label == "fearful"
    assert fields.actor == "12"


@pytest.mark.parametrize(
    "code,label",
    [
        ("01", "neutral"),
        ("02", "calm"),
        ("03", "happy"),
        ("04", "sad"),
        ("05", "angry"),
        ("06", "fearful"),
        ("07", "disgust"),
        ("08", "surprised"),
    ],
)
def test_all_official_emotion_codes_map_correctly(code, label):
    filename = f"03-01-{code}-01-01-01-01.wav"
    assert parse_ravdess_filename(filename).emotion_label == label


def test_rejects_malformed_filename():
    with pytest.raises(ValueError, match="Not a valid RAVDESS filename"):
        parse_ravdess_filename("not-a-ravdess-file.wav")


def test_rejects_unknown_emotion_code():
    with pytest.raises(ValueError, match="Unknown RAVDESS emotion code"):
        parse_ravdess_filename("03-01-99-01-01-01-01.wav")


def test_build_manifest_from_directory_skips_song_and_non_matching_files(tmp_path):
    (tmp_path / "03-01-03-01-01-01-01.wav").write_bytes(b"fake-speech-audio")  # speech, happy, actor 1
    (tmp_path / "03-02-03-01-01-01-01.wav").write_bytes(b"fake-song-audio")  # song - should be skipped
    (tmp_path / "readme.txt").write_bytes(b"not audio at all")

    samples = build_manifest_from_directory(tmp_path)

    assert len(samples) == 1
    assert samples[0].label == "happy"
    assert samples[0].speaker_id == "01"
    assert samples[0].dataset_source == "ravdess"


def test_build_manifest_raises_when_directory_has_no_matching_files(tmp_path):
    (tmp_path / "irrelevant.wav").write_bytes(b"x")

    with pytest.raises(FileNotFoundError, match="No RAVDESS-named"):
        build_manifest_from_directory(tmp_path)
