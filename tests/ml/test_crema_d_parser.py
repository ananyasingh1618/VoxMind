from __future__ import annotations

import pytest

from ml.datasets.crema_d import build_manifest_from_directory, parse_crema_d_filename


def test_parses_real_crema_d_filename_convention():
    fields = parse_crema_d_filename("1001_DFA_ANG_XX.wav")

    assert fields.actor_id == "1001"
    assert fields.sentence_code == "DFA"
    assert fields.emotion_label == "angry"
    assert fields.level_code == "XX"


@pytest.mark.parametrize(
    "code,label",
    [
        ("ANG", "angry"),
        ("DIS", "disgust"),
        ("FEA", "fear"),
        ("HAP", "happy"),
        ("NEU", "neutral"),
        ("SAD", "sad"),
    ],
)
def test_all_official_emotion_codes_map_correctly(code, label):
    filename = f"1001_DFA_{code}_XX.wav"
    assert parse_crema_d_filename(filename).emotion_label == label


def test_rejects_malformed_filename():
    with pytest.raises(ValueError, match="Not a valid CREMA-D filename"):
        parse_crema_d_filename("not-a-crema-d-file.wav")


def test_rejects_unknown_emotion_code():
    with pytest.raises(ValueError, match="Unknown CREMA-D emotion code"):
        parse_crema_d_filename("1001_DFA_ZZZ_XX.wav")


def test_rejects_unknown_level_code():
    with pytest.raises(ValueError, match="Unknown CREMA-D level code"):
        parse_crema_d_filename("1001_DFA_ANG_ZZ.wav")


def test_rejects_non_four_digit_actor_id():
    with pytest.raises(ValueError, match="Unexpected CREMA-D actor id"):
        parse_crema_d_filename("101_DFA_ANG_XX.wav")


def test_build_manifest_from_directory_skips_non_matching_files(tmp_path):
    (tmp_path / "1001_DFA_ANG_XX.wav").write_bytes(b"fake-audio")
    (tmp_path / "readme.txt").write_bytes(b"not audio at all")

    samples = build_manifest_from_directory(tmp_path)

    assert len(samples) == 1
    assert samples[0].label == "angry"
    assert samples[0].speaker_id == "1001"
    assert samples[0].dataset_source == "crema_d"


def test_build_manifest_raises_when_directory_has_no_matching_files(tmp_path):
    (tmp_path / "irrelevant.wav").write_bytes(b"x")

    with pytest.raises(FileNotFoundError, match="No CREMA-D-named"):
        build_manifest_from_directory(tmp_path)
