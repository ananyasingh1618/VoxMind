"""Deterministic transcript/diarization alignment - pure computation, no ML,
fully unit-testable without any model.

Algorithm
---------
1. For each transcript segment, find every diarization speaker segment that
   temporally overlaps it (`overlap_ms = max(0, min(ends) - max(starts))`).
2. Assign the transcript segment to the speaker with the LARGEST total
   overlap duration. Ties are broken deterministically: earliest-starting
   overlapping speaker segment first, then lexicographically smallest
   speaker label.
3. If no speaker segment overlaps at all, the transcript segment's speaker
   is `None` - explicitly "unknown", never guessed or defaulted to a
   fabricated label.
4. Consecutive transcript segments assigned to the same speaker (including
   consecutive `None`s) are merged into one `AlignedTurn`, as long as the
   gap between them does not exceed `max_gap_ms` - a large gap starts a new
   turn even for the same speaker, since it plausibly represents the
   conversation moving on and back rather than one continuous turn.

What this deliberately does NOT do: split a single transcript segment's text
across multiple speakers when more than one speaker overlaps it.
faster-whisper's default segments have no word-level timestamps to split on
reliably, and guessing a word boundary would be exactly the kind of
fabrication this project forbids - the segment goes to its single dominant
(highest-overlap) speaker instead. This is a real, documented limitation of
segment-level (not word-level) alignment, not a hidden shortcut.
"""
from __future__ import annotations

from voxmind.services.speech.interfaces import AlignedTurn, SpeakerSegment, TranscriptSegment

DEFAULT_MAX_GAP_MS = 2000


def _overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _dominant_speaker(
    segment: TranscriptSegment, speaker_segments: list[SpeakerSegment]
) -> str | None:
    overlaps: dict[str, int] = {}
    first_overlap_start: dict[str, int] = {}

    for speaker_segment in speaker_segments:
        overlap = _overlap_ms(
            segment.start_ms, segment.end_ms, speaker_segment.start_ms, speaker_segment.end_ms
        )
        if overlap <= 0:
            continue
        overlaps[speaker_segment.speaker_label] = overlaps.get(speaker_segment.speaker_label, 0) + overlap
        if speaker_segment.speaker_label not in first_overlap_start:
            first_overlap_start[speaker_segment.speaker_label] = speaker_segment.start_ms
        else:
            first_overlap_start[speaker_segment.speaker_label] = min(
                first_overlap_start[speaker_segment.speaker_label], speaker_segment.start_ms
            )

    if not overlaps:
        return None

    # Sort by: largest overlap first, then earliest first-overlap start,
    # then lexicographically smallest label - a total order, so ties never
    # depend on input/iteration order.
    ranked = sorted(
        overlaps.keys(),
        key=lambda label: (-overlaps[label], first_overlap_start[label], label),
    )
    return ranked[0]


def align_transcript_with_speakers(
    transcript_segments: list[TranscriptSegment],
    speaker_segments: list[SpeakerSegment],
    *,
    max_gap_ms: int = DEFAULT_MAX_GAP_MS,
) -> list[AlignedTurn]:
    if not transcript_segments:
        return []

    assigned: list[tuple[TranscriptSegment, str | None]] = [
        (segment, _dominant_speaker(segment, speaker_segments)) for segment in transcript_segments
    ]

    turns: list[AlignedTurn] = []
    for segment, speaker_label in assigned:
        if (
            turns
            and turns[-1].speaker_label == speaker_label
            and (segment.start_ms - turns[-1].end_ms) <= max_gap_ms
        ):
            previous = turns[-1]
            turns[-1] = AlignedTurn(
                start_ms=previous.start_ms,
                end_ms=segment.end_ms,
                text=f"{previous.text} {segment.text}".strip(),
                speaker_label=speaker_label,
            )
        else:
            turns.append(
                AlignedTurn(
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    text=segment.text,
                    speaker_label=speaker_label,
                )
            )

    return turns
