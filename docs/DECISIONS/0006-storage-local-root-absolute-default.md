# ADR 0006: `STORAGE_LOCAL_ROOT` defaults to an absolute path

## The bug

Running the documented training command (`python -m ml.training.train_emotion_model ...` from the repo root) saved a real, correctly-trained checkpoint - but the running API server (documented to run `uvicorn` from `apps/api/`) couldn't find it at inference time, and `/emotion/process` failed with an unhandled `FileNotFoundError`. Both processes used the same relative default, `STORAGE_LOCAL_ROOT="./.data/storage"`, but resolved it against two different working directories, so `LocalFilesystemStorage` silently wrote and read from two different real directories on disk. This was caught during the first real end-to-end inference verification against a genuinely trained model (`docs/emotion.md`), not by any earlier test - every prior test happened to run from a single consistent working directory, so the mismatch never surfaced.

## The fix

`Settings.STORAGE_LOCAL_ROOT`'s default is now computed as an absolute path anchored to the `apps/api` directory itself (`Path(__file__).resolve().parent.parent.parent / ".data" / "storage"`, computed in `core/config.py`), not a string literal resolved against whatever the current working directory happens to be at runtime. The relative override in `.env`/`.env.example` was removed so this safe default actually takes effect; both files now document that any override must also be absolute.

## Why this is the right level of fix

This is a one-line-of-reasoning config bug, not an architecture problem: `StorageBackend`, `LocalFilesystemStorage`, `EmotionCheckpoint`, `ModelVersionRepository`, and `EmotionService` all behaved exactly as designed once they were both pointed at the same real directory - no redesign was needed or made. The single misplaced checkpoint file produced before this fix was moved (not regenerated - it was already a real, validly-trained artifact) to the corrected location.

## Verification

After the fix, a full live end-to-end run (real held-out RAVDESS test sample → upload → Phase 2 transcription → Phase 3 emotion inference) correctly resolved and loaded the real `ravdess-v1` checkpoint and produced a correct prediction (`neutral`, matching the sample's real ground-truth label) - see `docs/emotion.md`'s "Genuine trained model" section for the full result.
