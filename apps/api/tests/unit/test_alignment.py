from __future__ import annotations

from voxmind.services.speech.alignment import align_transcript_with_speakers
from voxmind.services.speech.interfaces import SpeakerSegment, TranscriptSegment


def _ts(start_ms: int, end_ms: int, text: str) -> TranscriptSegment:
    return TranscriptSegment(start_ms=start_ms, end_ms=end_ms, text=text, confidence=0.9)


def _ss(label: str, start_ms: int, end_ms: int) -> SpeakerSegment:
    return SpeakerSegment(speaker_label=label, start_ms=start_ms, end_ms=end_ms, confidence=None)


def test_empty_transcript_returns_no_turns() -> None:
    assert align_transcript_with_speakers([], []) == []


def test_single_segment_with_full_speaker_overlap() -> None:
    turns = align_transcript_with_speakers(
        [_ts(0, 2000, "hello there")], [_ss("speaker_0", 0, 2000)]
    )
    assert len(turns) == 1
    assert turns[0].speaker_label == "speaker_0"
    assert turns[0].text == "hello there"


def test_no_overlap_leaves_speaker_unknown() -> None:
    """No diarization segment overlaps the transcript segment at all - the
    turn's speaker must be None, never guessed."""
    turns = align_transcript_with_speakers(
        [_ts(5000, 6000, "unattributed speech")], [_ss("speaker_0", 0, 1000)]
    )
    assert len(turns) == 1
    assert turns[0].speaker_label is None


def test_partial_overlap_assigns_the_overlapping_speaker() -> None:
    # Speaker segment covers only the tail of the transcript segment.
    turns = align_transcript_with_speakers(
        [_ts(0, 4000, "partially overlapping text")], [_ss("speaker_1", 3000, 5000)]
    )
    assert turns[0].speaker_label == "speaker_1"


def test_dominant_speaker_wins_when_multiple_speakers_overlap_one_segment() -> None:
    """speaker_0 overlaps 3000ms (0-3000), speaker_1 overlaps only 500ms
    (3000-3500) of a 0-3500ms transcript segment - speaker_0 must win."""
    turns = align_transcript_with_speakers(
        [_ts(0, 3500, "a long segment spanning two speakers")],
        [_ss("speaker_0", 0, 3000), _ss("speaker_1", 3000, 4000)],
    )
    assert len(turns) == 1
    assert turns[0].speaker_label == "speaker_0"


def test_tie_broken_by_earliest_then_label() -> None:
    # Equal overlap (1000ms each) - tie-break must be deterministic.
    turns = align_transcript_with_speakers(
        [_ts(0, 2000, "tied segment")],
        [_ss("speaker_1", 0, 1000), _ss("speaker_0", 1000, 2000)],
    )
    # speaker_1's overlap starts earlier (0 vs 1000), so it wins the tie
    # despite "speaker_0" being lexicographically smaller.
    assert turns[0].speaker_label == "speaker_1"


def test_very_short_speaker_segment_still_counts() -> None:
    turns = align_transcript_with_speakers(
        [_ts(0, 5000, "mostly silence with a brief interjection")],
        [_ss("speaker_0", 2000, 2050)],  # 50ms segment
    )
    assert turns[0].speaker_label == "speaker_0"


def test_consecutive_same_speaker_segments_merge_into_one_turn() -> None:
    turns = align_transcript_with_speakers(
        [_ts(0, 1000, "first part"), _ts(1000, 2000, "second part")],
        [_ss("speaker_0", 0, 2000)],
    )
    assert len(turns) == 1
    assert turns[0].text == "first part second part"
    assert turns[0].start_ms == 0
    assert turns[0].end_ms == 2000


def test_adjacent_different_speaker_turns_stay_separate() -> None:
    turns = align_transcript_with_speakers(
        [_ts(0, 1000, "speaker A talking"), _ts(1000, 2000, "speaker B talking")],
        [_ss("speaker_0", 0, 1000), _ss("speaker_1", 1000, 2000)],
    )
    assert len(turns) == 2
    assert [t.speaker_label for t in turns] == ["speaker_0", "speaker_1"]


def test_large_gap_between_same_speaker_segments_starts_a_new_turn() -> None:
    turns = align_transcript_with_speakers(
        [_ts(0, 1000, "before the pause"), _ts(10_000, 11_000, "after a long pause")],
        [_ss("speaker_0", 0, 1000), _ss("speaker_0", 10_000, 11_000)],
        max_gap_ms=2000,
    )
    assert len(turns) == 2


def test_small_gap_between_same_speaker_segments_merges() -> None:
    turns = align_transcript_with_speakers(
        [_ts(0, 1000, "part one"), _ts(1500, 2500, "part two")],
        [_ss("speaker_0", 0, 1000), _ss("speaker_0", 1500, 2500)],
        max_gap_ms=2000,
    )
    assert len(turns) == 1
    assert turns[0].text == "part one part two"


def test_alternating_none_and_known_speaker_do_not_incorrectly_merge() -> None:
    turns = align_transcript_with_speakers(
        [_ts(0, 1000, "known"), _ts(1000, 2000, "unknown"), _ts(2000, 3000, "known again")],
        [_ss("speaker_0", 0, 1000), _ss("speaker_0", 2000, 3000)],
    )
    assert len(turns) == 3
    assert [t.speaker_label for t in turns] == ["speaker_0", None, "speaker_0"]
