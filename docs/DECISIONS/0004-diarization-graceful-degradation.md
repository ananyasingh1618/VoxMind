# ADR 0004: Diarization failures don't fail the whole speech-processing job

## Decision

`AudioProcessingJob.status` reflects only the *core* pipeline (preprocessing + transcription). Diarization outcome is tracked in a separate field, `diarization_status` (`"completed" | "unavailable" | "failed"`). A missing or invalid diarization configuration - most commonly no `HUGGINGFACE_TOKEN`, since `pyannote/speaker-diarization-3.1` is a gated model - does not fail the job; the transcript still completes normally, with every `speaker_label` left `None`.

## Why

Diarization is a real, common, expected-to-be-absent condition in many deployments (not every environment will have accepted the gated model's terms and configured a token), while transcription is the core value of the feature. Treating a missing optional credential as a hard failure would mean an otherwise perfectly good transcript is thrown away because of an unrelated, unconfigured integration - a worse outcome for the product than shipping a transcript with unattributed speakers.

## Why this couldn't be "just check the exception type from TaskRunner"

The approved (and unmodified) `InProcessTaskRunner.dispatch()` catches any exception a stage raises and collapses it to `JobHandle(status=FAILED, error=str(exc))` - the exception's original *type* does not survive that boundary, only its stringified message. Distinguishing "diarization isn't configured" from "diarization broke" by pattern-matching that string would be fragile and easy to break silently on a future error-message wording change.

## The actual mechanism

`AudioService._run_diarization` checks `settings.HUGGINGFACE_TOKEN` **before** ever dispatching the diarization stage. If it's absent, `diarization_status="unavailable"` is recorded directly, with zero network calls or model-load attempts made - not a caught failure, a decision never to try. If a token *is* configured but the pipeline still fails at runtime (bad token, network issue, etc.), the stage is dispatched for real, and a genuine failure there is recorded as `diarization_status="failed"` with the (best-effort) error message from the collapsed exception.

This lives entirely in the orchestration layer (`voxmind/services/audio_service.py`), not inside `TaskRunner` - `TaskRunner`'s Phase 1 contract was explicitly not to be redesigned for this phase, and didn't need to be.

## Consequence for downstream data

`speaker_segments` is a genuinely empty list (not fabricated placeholder data) when diarization didn't run, and every `aligned_turns.speaker_label` is `None` in that case - the alignment algorithm still runs (it's pure and always succeeds given valid transcript segments), it just has no speaker data to attribute against. See `tests/integration/test_audio_pipeline_real_whisper.py::test_full_pipeline_produces_a_genuine_transcript` for the verified end-to-end behavior.
