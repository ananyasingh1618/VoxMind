# Test audio fixtures

`hello_world.wav` / `hello_world.mp3` are genuine synthesized speech - not
silence, not noise, not hand-crafted bytes - saying "The quick brown fox
jumps over the lazy dog", generated with:

```bash
say -o hello_world.aiff "The quick brown fox jumps over the lazy dog"
afconvert -f WAVE -d LEI16@16000 -c 1 hello_world.aiff hello_world.wav
ffmpeg -i hello_world.wav -codec:a libmp3lame -qscale:a 4 hello_world.mp3
rm hello_world.aiff
```

These are used by real (non-mocked) preprocessing and Whisper transcription
tests - see `tests/unit/test_preprocessing.py` and
`tests/integration/test_audio_pipeline_real_whisper.py`. The real
`faster-whisper` `tiny` model transcribes `hello_world.wav` as "the quick
brown fox jumps over the lazy dog." (verified manually during Phase 2
development), which is what the real-model test asserts a substring match
against.
