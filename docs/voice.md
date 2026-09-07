# The real-time voice loop (Phase 5)

Real microphone capture → real speech-to-text → real emotion/NLP analysis → the real RAG/LLM pipeline → real text-to-speech → real playback, streamed to the client as genuine progress events - no simulated timing, no fabricated audio, no fake transcript.

## The target loop

```
User speaks
  -> real MediaRecorder capture (browser)
  -> real upload + real Whisper transcription (Phase 2, AudioService)
  -> real emotion/NLP/incongruence analysis (Phase 3/4, best-effort)
  -> real memory + RAG retrieval (Phase 4, RagService.generate_for_message)
  -> real LLM generation, or an honest "unavailable" (Phase 4)
  -> real TTS synthesis (Phase 5)
  -> real audio playback (browser)
```

`VoiceTurnService` (`services/voice_turn_service.py`) is the only place this whole loop is wired together. Every stage it calls is a real, already-implemented service from an earlier phase - nothing is reimplemented, only sequenced and instrumented. In particular, `generate_for_message()` (added to `RagService` in this phase) is the exact same code path `POST /conversations/{id}/ask` uses for a typed question - a spoken question and a typed question produce an answer through identical grounding/citation/persistence logic.

"Guardrails" (mentioned in the originally-requested target loop) do not exist yet - they're Phase 6 per the roadmap - so the loop genuinely skips that step rather than faking a pass-through guardrail check.

## Streaming architecture: Server-Sent Events, not fake progress

`POST /conversations/{id}/voice-turns` accepts a recorded audio blob (multipart) and returns a genuine `text/event-stream` response. Each event is emitted **the moment the real backend stage it describes actually completes** - there is no client-side timer simulating progress. Stage names map directly to UI states:

| SSE `stage` | UI state | What really happened |
|---|---|---|
| `processing` | `processing` | Real upload + real Whisper transcription via `AudioService` |
| `thinking` | `thinking` | Real (best-effort) emotion + NLP + incongruence analysis |
| `retrieving` | `retrieving` | Real hybrid retrieval via `RetrievalService` |
| `generating` | `generating` | Real LLM call (or a genuine "unavailable" report) |
| `speaking` | `speaking` | Real TTS synthesis, then real audio playback client-side |
| `error` | `error` | A genuine stage failure |
| (client-only) | `listening` | Before any audio has been sent - never emitted by the server |
| (client-only) | `interrupted` | The client aborted the connection or stopped playback |

A final `done` event carries the complete result: the transcript, the answer, validated citations, grounding status, an `audio_url` to fetch the synthesized speech from (or `null` if generation was unavailable), and the full `stage_latencies_ms` breakdown.

### Incremental text handling - what's real, what's a deliberate scope decision

The STT transcript is shown to the user as soon as it's ready (in the `processing: completed` event), well before the LLM has even started - genuine incremental feedback, not a placeholder. Token-level LLM streaming (word-by-word generation) is **not** implemented, deliberately: `RagService.generate_for_message()` forces structured tool-call output from the LLM (required so citations can be validated against real retrieved chunks - see docs/rag.md's grounding section), and forced structured/tool-call output does not stream as meaningful partial text the way free-form prose does. Streaming stage-transitions and the real transcript is the incremental behavior this phase implements "where supported"; full token streaming of a tool-forced response is out of scope for a documented, technical reason, not an oversight.

## Real text-to-speech

`services/tts/` mirrors the LLM provider abstraction (`services/llm/`) exactly - a `TextToSpeechProvider` protocol, a factory returning `None` (never a fake provider) when disabled/unconfigured:

- **`local_hf` (the default)** - `facebook/mms-tts-eng`, a VITS model, not gated, freely downloadable. Loaded via plain Hugging Face Transformers (`VitsModel`/`VitsTokenizer`), the same "load the base model directly, no extra package" pattern as every other local model in this project. Output is a raw waveform, WAV-encoded via `soundfile`. Unlike the LLM providers (which default to `local_dev` = off, since no free local LLM was wired in), TTS defaults to a real, always-on local provider specifically so the full voice loop is genuinely demonstrable end-to-end without any credentials.
- **`openai`** - real OpenAI TTS (`audio.speech.create`), credential-gated (`OPENAI_API_KEY`, a developer-console key - see docs/rag.md for the same subscription-vs-API-key distinction). Returns real MP3 bytes.
- **`disabled`** - no TTS at all; the loop still completes, `speaking: unavailable` is reported honestly.

**A real bug was found and fixed while building this provider** - see [DECISIONS/0007](DECISIONS/0007-reranker-float64-precision-bug.md) for the *first* occurrence of this exact class of issue (the retrieval reranker), and this phase's own investigation below for a second, unrelated one.

## Real interruption

A user can stop playback or cancel an in-flight request at any point, and the two are handled differently on purpose:

- **Cancelling an in-flight request** (before the answer is ready): the frontend aborts the `fetch()` call via `AbortController`. This closes the underlying HTTP connection, which Starlette's `StreamingResponse` detects on its own and cancels the server-side task - `VoiceTurnService.run()` catches that cancellation and records `VoiceTurn.status = "interrupted"` before letting the cancellation propagate. See [DECISIONS/0009](DECISIONS/0009-voice-loop-cancellation.md) for the investigation: the first implementation polled `request.is_disconnected()` between stages, and live testing proved that check never actually gets a chance to run - the runtime's own cancellation pre-empts it. The fix is a single `except asyncio.CancelledError` handler with a *shielded* cleanup write (`anyio.CancelScope(shield=True)` - an unshielded `await` inside a cancellation handler would itself be immediately cancelled before reaching the database).
- **Stopping playback** (the answer already arrived and is playing): purely client-side - `useVoiceTurn.interrupt()` pauses the single shared `<audio>` element. Nothing to tell the server; the turn already completed successfully.

**No overlapping speech**: `useVoiceTurn` owns exactly one `Audio` instance. Starting a new turn, or interrupting, always stops whatever is currently playing first (`stopPlayback()`) before anything else happens.

**A known, inherent limitation, observed live and worth stating plainly**: cancelling the coroutine stops the *pipeline* immediately (no further stages run, the row is marked `interrupted` right away), but a CPU-bound model call already dispatched via `asyncio.to_thread()` (e.g. a Whisper transcription already running in a worker thread) keeps executing to completion in the background - Python cannot forcibly kill a running OS thread. Its result is simply discarded; this wastes some CPU time on an interrupted turn but never corrupts state or blocks the interruption from being recorded. A production system wanting to reclaim that CPU time immediately would need killable subprocess workers instead of a thread pool - out of scope here, and mentioned so the behavior isn't mistaken for a bug if it's noticed in logs (a transcription log line can appear briefly *after* the `voice_turn_interrupted` log line for the same request).

## Latency instrumentation

`VoiceTurn.stage_latencies_ms` (JSONB) persists genuinely measured (`time.monotonic()`) durations for exactly the requested buckets - `stt_ms`, `analysis_ms`, `retrieval_ms`, `llm_ms`, `tts_ms`, `total_ms` - never estimated. The same numbers are emitted live in the SSE `done` event and rendered in a "Timing breakdown" panel in the UI (`VoiceSessionControl.tsx`).

## Conversation turn lifecycle

`VoiceTurn.status`: `pending` (created, pipeline running) → one of `completed` (full loop succeeded, including TTS), `partial` (STT and/or LLM succeeded but TTS was unavailable/failed, or the LLM itself was unavailable), `failed` (STT itself failed - there's no transcript to build on), or `interrupted` (the client disconnected before the pipeline finished). A row is never left at `pending` forever in a normal run; if a client disconnects, the cancellation handler always lands it on `interrupted` before the process gives up its work.

## API

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/conversations/{id}/voice-turns` | POST | Bearer token | Multipart `file` upload. Returns a genuine `text/event-stream` of real progress events, ending in `done` or `error`. |
| `/conversations/{id}/voice-turns/{turn_id}` | GET | Bearer token | Fetches a persisted `VoiceTurn`'s final status/latencies. |
| `/conversations/{id}/voice-turns/{turn_id}/audio` | GET | Bearer token | Streams the real synthesized audio bytes (`audio/wav` or `audio/mpeg`, matching whichever provider ran). Requires the same Bearer auth as everything else - the frontend fetches this as a blob (`fetchVoiceTurnAudio()`) rather than using a plain `<audio src>`, since browsers can't attach an `Authorization` header to a bare `<audio>` tag. |

## What has been genuinely verified

- **Live, repeatedly, against a real running server**: real microphone-recorded-style audio (the same `hello_world.wav` fixture Phase 2 uses) → real Whisper transcript → real emotion/NLP analysis → real retrieval → (with the test-only mock LLM enabled, since no real API key exists in this environment) a real assistant answer → real TTS synthesis → a downloaded, valid, non-silent WAV file. With no LLM configured (this environment's real default), the loop honestly reports `generating: unavailable` / `speaking: unavailable` and completes with `status="partial"` - never a fabricated answer or a fake audio clip.
- **Real interruption**, live: killing the client connection during STT and during the analysis stage both produced a clean `status="interrupted"` row and no unhandled server-side exception.
- **Automated tests** (`tests/integration/test_voice_turn_pipeline_real.py`, `tests/integration/test_tts_providers_real.py`, `tests/unit/test_tts_factory.py`, and the frontend's `tests/useVoiceTurn.test.tsx`/`tests/voiceSessionControl.test.tsx`) cover: the full loop with a real transcript and honest unavailable generation, the full loop with real synthesized audio (test-only mock LLM), the TTS provider factory's credential-gating, real local TTS producing non-silent audio, and the cancellation-handling mechanism in isolation (see docs/DECISIONS/0009 for why the full HTTP-level cancellation test is exercised this way rather than through a real aborted HTTP request).
- **Not verified**: a real Anthropic/OpenAI network call, and real OpenAI TTS (both credential-gated tests are implemented and skip cleanly without a key - see docs/rag.md for the same limitation on the text/RAG side). Not verified in an actual browser (no browser tool available in this environment) - verified via component/hook tests, TypeScript compilation, and a production build, per the same honesty standard applied to Phase 4's frontend.

## Environment variables

`TTS_PROVIDER` (`local_hf` default, `openai`, or `disabled`), `TTS_LOCAL_MODEL`, `TTS_DEVICE`, `OPENAI_TTS_MODEL`, `OPENAI_TTS_VOICE`, `VOICE_TURN_MAX_ANSWER_CHARS` (caps how much of a long answer is sent to TTS). `OPENAI_API_KEY` is required only for the `openai` TTS provider - the default `local_hf` provider needs no credentials at all.
