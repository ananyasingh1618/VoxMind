# ADR 0003: Tone/word alignment signal, not deception detection

## Decision

The feature comparing spoken words (semantic signal) to vocal tone (vocal signal) is named and framed throughout the codebase, API, and UI as an **analytical signal** - divergence between the two can indicate sarcasm, stress, uncertainty, or suppressed emotion. It is never named, scored, described, or marketed as a lie detector, deception detector, or truthfulness predictor.

## Why this matters enough to be an ADR

Emotion/tone-from-voice signals are easy to over-claim, and "detects when someone is lying" is a common but false and potentially harmful mischaracterization of what acoustic/linguistic mismatch analysis can actually establish. Locking the framing in as an explicit decision - rather than leaving it to whatever language feels natural while implementing Phase 4 under time pressure - prevents that drift.

## Enforcement mechanism

`services/incongruence/interfaces.py`'s `IncongruenceSignal` schema includes a `signal_category: Literal["analytical_not_diagnostic"]` field with no other valid value - a machine-readable guardrail, not just a comment. UI copy (`components/conversation/IntelligencePanel.tsx`) states the non-diagnostic framing explicitly wherever the feature is mentioned, including in its Phase 1 "not yet available" placeholder text, so the correct framing exists from the very first line of copy ever written for this feature, not retrofitted once the real pipeline lands in Phase 4.
