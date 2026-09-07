# Speech Pipeline (Phase 2)

Real audio ingestion, preprocessing, speech-to-text, speaker diarization, and transcript/diarization alignment. This document covers what's implemented, what's genuinely been verified in this environment, and exactly what's required to run the parts that couldn't be (diarization's gated model).

## Status summary

| Component | Implementation | Verified in this environment |
|---|---|---|
| Audio ingestion (upload, validation, safe filenames) | Real | Yes - unit + integration tests, real HTTP uploads |
| Audio preprocessing (ffmpeg decode/normalize) | Real | Yes - real ffmpeg, real audio fixtures |
| Speech-to-text (faster-whisper) | Real | **Yes - genuine model download + inference, see below** |
| Speaker diarization (pyannote.audio) | Real | **No - requires a Hugging Face token this environment doesn't have; see "Diarization credentials" below** |
| Transcript/diarization alignment | Real, deterministic | Yes - pure-function unit tests, no ML involved |
| Pipeline orchestration (TaskRunner) | Real | Yes - end-to-end through the actual API |

## Supported audio formats

Ingestion-time validation accepts (by real magic-byte sniffing, never by trusting the client's declared `Content-Type`):

- **wav** - always reliably decodable.
- **mp3** - reliably decodable via ffmpeg.
- **m4a** - reliably decodable via ffmpeg (AAC in an MP4 container).
- **ogg** - reliably decodable via ffmpeg.
- **webm** - accepted at ingestion (browsers commonly record to this via `MediaRecorder`), but decodability depends on which codec is inside the container (Opus is reliable; some less common webm audio codecs may not be). A file that sniffs as webm but fails real decoding is rejected at the *preprocessing* stage with a 422, not silently accepted.

Limits (configurable, see `.env.example`): 25MB max upload size, 10 minutes max duration.

## Preprocessing

Runs the uploaded audio through `ffmpeg` via stdin/stdout pipes (no temp files touch disk) to produce a canonical mono, 16kHz, 16-bit PCM WAV, then independently validates the result with `soundfile`. The original upload is never modified - the canonical WAV is stored as a *separate* `AudioAsset` (`kind="processed"`, linked via `source_asset_id`).

## Speech-to-text: faster-whisper

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2-accelerated Whisper). Model weights come from the **non-gated** `Systran/faster-whisper-<size>` Hugging Face repos - no account or token is required, unlike diarization.

Configuration (`.env.example`):

```
WHISPER_MODEL_SIZE=tiny       # tiny|base|small|medium|large-v3 - larger = more accurate, slower
WHISPER_DEVICE=cpu            # cpu|cuda|auto
WHISPER_COMPUTE_TYPE=int8     # int8 is fastest on CPU; use float16 on a GPU
WHISPER_LANGUAGE=             # empty = auto-detect
WHISPER_BEAM_SIZE=5
```

### Genuine verification performed

This was actually run, not assumed: real speech was synthesized locally with macOS's `say` command ("The quick brown fox jumps over the lazy dog"), converted to a real WAV, and transcribed by a genuinely downloaded and loaded `tiny` model. The model produced: `"the quick brown fox jumps over the lazy dog."` - a correct transcription, with a real per-segment confidence proxy (`exp(avg_logprob)` ≈ 0.68) and real language detection (`en`, probability 0.995). See `tests/integration/test_whisper_real_model.py` and `tests/integration/test_audio_pipeline_real_whisper.py` (marked `@pytest.mark.real_model`), and `tests/fixtures/README.md` for exact fixture generation steps.

Confidence caveat: faster-whisper does not expose a calibrated confidence score. `TranscriptSegment.confidence` is `exp(avg_logprob)`, a proxy that correlates with but is not equal to a true probability - documented as such in the schema, never presented as more precise than it is.

## Speaker diarization: pyannote.audio

Uses [pyannote.audio](https://github.com/pyannote-audio/pyannote-audio)'s `pyannote/speaker-diarization-3.1` pipeline. Unlike Whisper, **this model is gated** on Hugging Face.

### Credentials required to run this for real

1. Create a free Hugging Face account.
2. Accept the user conditions on both model pages (both are required - the pipeline depends on the segmentation model internally):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Create a read-scoped access token: https://huggingface.co/settings/tokens
4. Set `HUGGINGFACE_TOKEN=<your token>` in `.env`.

### What happens without a token (this environment's actual state)

`AudioService` checks `HUGGINGFACE_TOKEN` **before** attempting to dispatch the diarization stage at all - if it's absent, diarization is skipped entirely (no network call, no model load attempt) and the `AudioProcessingJob` records `diarization_status="unavailable"` with a clear reason. Crucially, **this does not fail the whole job** - the transcript still completes normally, with all `speaker_label`s left `None` (never guessed or defaulted to a fake value like `"SPEAKER_00"`).

This was verified for real in this environment: `tests/unit/test_diarization_provider.py` confirms the exact error raised when the token is absent, and `tests/integration/test_audio_pipeline_real_whisper.py::test_full_pipeline_produces_a_genuine_transcript` confirms the whole-pipeline graceful-degradation behavior end-to-end (transcript succeeds, `diarization_status == "unavailable"`).

`tests/integration/test_diarization_real_model.py` contains the real-model diarization test, marked `@pytest.mark.requires_hf_token` and `skipif`-guarded - it is **skipped** in this environment for exactly the reason above, not because it doesn't exist. To actually run it:

```bash
export HUGGINGFACE_TOKEN=hf_...
pytest -m requires_hf_token
```

## Alignment algorithm

Deterministic, pure-function, no ML - see `voxmind/services/speech/alignment.py` for the full algorithm docstring. Summary: each Whisper transcript segment is assigned to whichever diarization speaker segment overlaps it for the longest total duration (ties broken by earliest overlap start, then label); segments with no overlapping speaker are left unattributed (`None`); consecutive same-speaker segments are merged into one turn as long as the gap between them doesn't exceed 2 seconds.

**Known limitation, by design**: when more than one speaker overlaps a single Whisper segment, that segment's text is *not* split between them - it's attributed entirely to whichever speaker has the larger overlap. Whisper's default segments carry no word-level timestamps to split on reliably, and guessing a word boundary would be fabrication. Word-level alignment (splitting mid-segment) is a possible future enhancement, not attempted here.

## Pipeline architecture

Preprocessing → transcription → diarization → alignment, each a `PipelineStage` (`voxmind/services/speech/stages.py`) dispatched through the unmodified, Phase 1-approved `TaskRunner` contract (`dispatch → get_status → get_result`). Under `InProcessTaskRunner` (the only runner through Phase 1-4), this executes synchronously within the `/process` request; the API response shape (`AudioProcessingJobOut` with a `status` field) is deliberately the same shape a truly asynchronous Celery-backed version would return, so nothing above the TaskRunner boundary needs to change when Phase 9 implements `CeleryTaskRunner` for real. See `docs/DECISIONS/0002-task-runner-boundary.md`.

Diarization failing gracefully (rather than failing the whole job) is implemented at the orchestration layer (`AudioService._run_diarization`), not inside `TaskRunner` itself, precisely because `TaskRunner`'s approved contract was not to be modified for this phase.

## API usage

See `docs/api.md` for the full endpoint reference. Quick example:

```bash
# Upload
curl -X POST .../conversations/$CONV_ID/audio \
  -H "Authorization: Bearer $TOKEN" -F "file=@recording.wav"

# Process (runs synchronously under InProcessTaskRunner - may take several
# seconds depending on WHISPER_MODEL_SIZE and audio length)
curl -X POST .../conversations/$CONV_ID/audio/$AUDIO_ID/process \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{}'

# Retrieve the structured transcript once job.message_id is known
curl .../conversations/$CONV_ID/messages/$MESSAGE_ID/transcript \
  -H "Authorization: Bearer $TOKEN"
```

## Testing strategy

- **Unit** (`tests/unit/`): audio validation/sniffing, safe filenames, the alignment algorithm (fully deterministic, no ML), pipeline stage wiring (test-double storage/providers), real ffmpeg-based preprocessing (genuine audio fixtures, no mocking), diarization's configuration guard.
- **Integration** (`tests/integration/`): real HTTP uploads against a real Postgres database, ownership/authorization, DB cascade/constraint behavior for the new tables.
- **Real-model** (`@pytest.mark.real_model`): genuinely downloads and runs faster-whisper - no credentials needed, included by default in this environment's `pytest` runs since network access to Hugging Face was confirmed available.
- **Credential-gated** (`@pytest.mark.requires_hf_token`): genuinely runs pyannote diarization - `skipif`-guarded, skipped without a real token (which is this environment's actual state).

Run everything (mirrors CI): `pytest`. Run only the fast suite: `pytest -m "not real_model and not requires_hf_token"`.

## Security notes specific to audio

- Upload content is classified by real magic-byte sniffing, never by trusting the client's `Content-Type` header.
- Filenames are sanitized (`safe_filename`) before use in any storage key - directory components and path-traversal sequences (`../`) are stripped, never interpolated raw into a storage path.
- Uploads are read in bounded chunks and rejected as soon as they exceed `MAX_AUDIO_UPLOAD_BYTES`, rather than buffering an arbitrarily large payload into memory first.
- No temp files are written to disk anywhere in the pipeline - ffmpeg runs over stdin/stdout pipes, and Whisper/diarization consume in-memory bytes/arrays - so there is nothing to leak or forget to clean up.
- Diarization error messages are deliberately generic (`"...model could not be loaded..."`) and never include the raw `HUGGINGFACE_TOKEN` value or a raw exception `repr()`, even in logs.
- Every audio/processing-job/transcript endpoint re-verifies conversation ownership before returning anything - a 404, not a 403, for both "doesn't exist" and "not yours" (no existence-leak), matching the Phase 1 convention.
