# Emotion Intelligence (Phase 3, + Emotion Model v2, + Emotion2Vec+ Production Model)

Real acoustic feature extraction, real Wav2Vec2/emotion2vec+ speech embeddings, real trainable PyTorch classifiers, real training/evaluation workflows, and real inference integration with Phase 2's speech artifacts. Nothing in this document reports a metric that wasn't genuinely computed.

This file documents three model generations, kept strictly separate and all individually preserved: **v1 (`ravdess-v1`)**, the original 8-class RAVDESS-only model, frozen and unmodified; **Emotion Model v2**, a six-class RAVDESS+CREMA-D cross-corpus candidate (not promoted); and **`emotion2vec-tess-emodb-v1`**, a small trained head on top of a frozen, purpose-built emotion2vec+ foundation model, trained on TESS+EMO-DB and benchmarked zero-shot on RAVDESS for robustness - **this is VoxMind's current active production emotion model**. See "Emotion2Vec+ Production Model" near the end of this file for the full account, including exactly which evaluation protocol produced its results and its real, honestly-stated limitations.

## Status summary

| Component | Implementation | Verified in this environment |
|---|---|---|
| Acoustic feature extraction (librosa) | Real | Yes - real audio, real signal processing |
| Wav2Vec2 embeddings (facebook/wav2vec2-base) | Real | **Yes - genuine model download + inference** |
| Classifier architecture (PyTorch) | Real | Yes - real forward pass, real save/load |
| Training pipeline | Real | **Yes - full real training run on RAVDESS** |
| Evaluation pipeline | Real | **Yes - genuine held-out test-set metrics, see below** |
| Model versioning / registry | Real | Yes - real Postgres row, `ravdess-v1`, active |
| Inference integration with Phase 2 | Real | Yes - real end-to-end pipeline test, real prediction verified correct |
| **A genuinely trained, evaluated emotion model** | **Done** | **Yes - `ravdess-v1`, test macro F1 = 0.690 (see below)** |

## Genuine trained model: `ravdess-v1`

Trained on the real RAVDESS speech corpus (1,440 real clips, 24 real actors), downloaded from its public Zenodo release (no authentication required - see `ml/datasets/README.md`) specifically to complete this training run.

Run from the repo root with `apps/api`'s venv Python (`ml` resolves via the repo root on `sys.path`; `voxmind` resolves via that venv's editable install regardless of CWD). `JWT_SECRET_KEY` is required by `Settings` even though this script never uses auth - `apps/api/.env` isn't discovered when the CWD is the repo root, so set it explicitly:

```bash
export JWT_SECRET_KEY="$(openssl rand -hex 32)"
apps/api/.venv/bin/python -m ml.datasets.prepare_emotion_dataset \
  --ravdess-dir <path-to-extracted-RAVDESS> \
  --output ml/datasets/manifests/ravdess_manifest.jsonl
apps/api/.venv/bin/python -m ml.training.train_emotion_model \
  --manifest ml/datasets/manifests/ravdess_manifest.jsonl \
  --version-tag ravdess-v1 --activate --seed 42
```

This is exactly the invocation used to produce the results below (the actual command run, not a hypothetical one) - it took ~7.5 minutes for feature/embedding extraction over all 1,440 clips plus under a minute for training, on CPU.

**Dataset split** (speaker-independent, seed 42, verified leakage-free by `assert_no_speaker_leakage`):

| Split | Samples | Speakers |
|---|---|---|
| Train | 1,020 | 17 |
| Validation | 240 | 4 |
| Test | 180 | 3 |

Class distribution (identical across splits proportionally; real RAVDESS protocol, not invented): `angry`, `calm`, `disgust`, `fearful`, `happy`, `sad`, `surprised` = 192 clips each; `neutral` = 96 (RAVDESS records no "strong intensity" neutral clips, a real, documented asymmetry - see "Class imbalance" below).

**Training**: 17 epochs before early stopping (patience 8, on validation macro-F1, best = 0.567 at epoch 9). Seed 42, batch size 16, learning rate 1e-3, AdamW, weight decay 1e-4, hidden dim 128, dropout 0.3, inverse-frequency class weights.

**Genuine held-out test-set results** (180 samples, 3 speakers never seen in training or validation):

| Metric | Value |
|---|---|
| Accuracy | **0.672** |
| Macro F1 | **0.690** |
| Weighted F1 | 0.678 |
| Balanced accuracy | 0.682 |
| Macro precision / recall | 0.707 / 0.682 |

Per-class (precision / recall / F1 / support):

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| neutral | 0.909 | 0.833 | 0.870 | 12 |
| calm | 0.850 | 0.708 | 0.773 | 24 |
| disgust | 0.889 | 0.667 | 0.762 | 24 |
| angry | 0.680 | 0.708 | 0.694 | 24 |
| fearful | 0.667 | 0.667 | 0.667 | 24 |
| sad | 0.536 | 0.625 | 0.577 | 24 |
| happy | 0.516 | 0.667 | 0.582 | 24 |
| surprised | 0.609 | 0.583 | 0.596 | 24 |

Confusion matrix (rows = true label, columns = predicted, order `[angry, calm, disgust, fearful, happy, neutral, sad, surprised]`):

```
[17,  0,  1,  1,  2,  0,  0,  3]   angry
[ 0, 17,  0,  0,  0,  1,  6,  0]   calm
[ 6,  0, 16,  0,  1,  0,  1,  0]   disgust
[ 1,  1,  0, 16,  2,  0,  2,  2]   fearful
[ 1,  0,  0,  1, 16,  0,  3,  3]   happy
[ 0,  1,  0,  0,  1, 10,  0,  0]   neutral
[ 0,  1,  1,  3,  3,  0, 15,  1]   sad
[ 0,  0,  0,  3,  6,  0,  1, 14]   surprised
```

The most notable confusion: `disgust` misclassified as `angry` (6/24) and `calm` misclassified as `sad` (6/24) - both are acoustically-plausible confusions (RAVDESS's disgust and anger portrayals share sharp, negative-valence prosody; calm and sad share low arousal), not a sign of a broken pipeline.

**Genuine real inference verification**: a held-out test-sample (`Actor_01/03-01-01-01-01-01-01.wav`, real ground truth = `neutral`, never used in training) was uploaded through the real HTTP API, transcribed via Phase 2, and analyzed via `/emotion/process`. The model correctly predicted `neutral` with 78.2% confidence, and the response's `model_version_id` matched the registered `ravdess-v1` row exactly - confirming the prediction came from the real saved checkpoint, not a test-only fallback.

**Model registry**: `component=emotion_classifier`, `version_tag=ravdess-v1`, `trained=true`, `is_active=true`, `base_model=facebook/wav2vec2-base`, `artifact_storage_key=models/emotion/ravdess-v1/classifier.pt`. A real local MLflow run was also logged (`apps/api/mlruns/`).

**What this result does and doesn't support**: 67% test accuracy across 8 classes (chance level = 12.5%) demonstrates the pipeline learns a genuine, non-trivial acoustic-emotion relationship from RAVDESS's acted, single-sentence, single-speaker-at-a-time recordings. It does **not** demonstrate production-grade generalization to spontaneous conversational speech, overlapping speakers, non-English speech, or emotions outside RAVDESS's 8-class taxonomy - RAVDESS is a controlled, acted-emotion corpus (professional actors reading fixed scripts), a known and common limitation of this dataset family, not unique to this implementation.

## What "emotion" means here - and doesn't

VoxMind's emotion feature reports a **predicted emotion label and the model's confidence in it**, derived from acoustic and learned speech features. It is:

- a **model output**, not a fact about the speaker's internal state,
- **not** a lie detector, deception detector, or truthfulness signal (see `docs/DECISIONS/0003-incongruence-not-deception.md` for VoxMind's separate, differently-scoped tone/word-alignment feature - emotion classification is unrelated to that),
- always presented with its model version and confidence, never as a bare assertion.

UI copy and API responses use language like "predicted emotion" and "model output," never "detected emotion" or "the speaker felt."

## Acoustic features (`voxmind/services/emotion/acoustic_features.py`)

Real, signal-derived descriptors computed with `librosa` from the actual decoded audio - never invented numbers. Extracted per aligned speaker turn (see "Inference granularity" below):

- **Pitch (F0)**: via `librosa.pyin`, restricted to voiced frames only (`voiced_fraction` reports what proportion of the clip had a detectable pitch at all).
- **RMS energy**, **zero-crossing rate**, **spectral centroid/bandwidth/rolloff**: standard `librosa.feature.*` functions.
- **13 MFCCs**.
- Each descriptor is reduced to 7 aggregate statistics (mean, std, min, max, p25, p50, p75) - see `AcousticStat` - never the raw frame sequence, which the classifier doesn't consume directly.
- **Deliberately excluded**: a tempo/"speech rate" feature. `librosa`'s tempo estimation is designed for musical rhythm; applying it to speech and calling the result "speech rate" would produce a real number that doesn't mean what its name implies - exactly the kind of overclaiming this project forbids. A proper articulation-rate estimator (e.g. syllable-nuclei detection) is a legitimate future addition, not attempted here.

`AcousticFeatures.to_vector()` flattens these into a fixed 134-dimension vector (`ACOUSTIC_VECTOR_DIM`) in a documented, stable order.

## Wav2Vec2 embeddings (`voxmind/services/emotion/wav2vec_provider.py`)

Uses Hugging Face Transformers' `facebook/wav2vec2-base` - **not gated**, unlike Phase 2's diarization model, so no Hugging Face token is required.

**Pooling strategy**: `last_hidden_state` (shape `[1, T, 768]`) is reduced to a fixed 768-dim vector via **mean pooling over the time axis**. This is the standard way to get an utterance-level embedding from a frame-level self-supervised encoder with no trained pooling head - it is explicitly *not* the model's classification head (wav2vec2-base has none) and *not* a CLS-token trick (Wav2Vec2 has no CLS token). Each clip is embedded individually (no batching), so a plain mean is exactly correct - no padding-aware masking is needed; see the module docstring for the full reasoning and what batching would require.

**Genuine verification performed**: this was actually run, not assumed. `tests/integration/test_whisper_real_model.py`'s sibling test infrastructure and `tests/integration/test_emotion_pipeline_real_model.py` genuinely download and run `facebook/wav2vec2-base` against real synthesized speech, producing a real 768-dimensional embedding vector, as part of a full pipeline run through the actual API.

## The classifier (`voxmind/services/emotion/model.py`)

Architecture: `[acoustic feature vector (134d); Wav2Vec2 embedding (768d)]` → z-score normalize (statistics fit on the training split only) → `Linear(902, 128) → ReLU → Dropout(0.3) → Linear(128, num_classes)` → softmax. A small MLP was chosen deliberately: RAVDESS-scale datasets (~1000-1500 clips) are too small to support a deeper network without overfitting, and Wav2Vec2's pretrained encoder already does the heavy representational work - the classifier only needs to learn a mapping from that representation (plus acoustic evidence) to emotion labels.

`EmotionCheckpoint` bundles the trained weights, the exact `FeatureNormalizer` fit during training, and the label mapping into one artifact (`to_bytes()`/`from_bytes()`, backed by `torch.save`) - a checkpoint is unusable without knowing the label order and normalization it was trained with, so all three are always saved and loaded together.

## Dataset (`ml/datasets/`)

See `ml/datasets/README.md` for the full detail. **RAVDESS** (freely licensed, CC BY-NC-SA, no registration required) was downloaded from its public Zenodo release and used for the real training run above - all 1,440 speech clips across 24 actors parsed with zero malformed/unrecognized samples. `ml/datasets/ravdess.py` is the real parser for RAVDESS's filename convention (verified against the actual corpus, not just synthetic test filenames - see `tests/ml/test_real_ravdess_dataset.py`); `ml/datasets/splitting.py` implements **speaker-independent** train/validation/test splitting (whole speakers assigned to one split, preventing the classifier from partly learning to recognize a specific voice - verified leakage-free on the real split); `ml/datasets/prepare_emotion_dataset.py` ties them together into the manifest actually used for training. The raw dataset itself (~200MB) is not committed to this repository (`ml/datasets/manifests/` and the raw audio directory are gitignored) - re-download it from the same public URL to reproduce.

### Class imbalance

RAVDESS's protocol records 2 statements × 2 repetitions × 2 intensities for 6 emotions but only 1 intensity level for `neutral` (no "strong neutral") - so `neutral` has roughly half as many clips per actor as the other 7 classes. `ml/training/train_emotion_model.py` uses **inverse-frequency class weighting** in the loss function by default (`use_class_weights=True` in `TrainingConfig`) specifically to address this documented, real imbalance - not a hypothetical one. The exact class distribution for any prepared manifest is printed by `prepare_emotion_dataset.py` and stored in `model_versions.dataset_info.train_label_distribution` for every trained model - never invented, always the actual counts from that run.

## Training pipeline (`ml/training/train_emotion_model.py`)

Offline workflow (not run inside FastAPI): load manifest → verify no speaker leakage → extract acoustic features + Wav2Vec2 embeddings for every sample (reusing Phase 2's `preprocess_audio` first, so training-time audio normalization exactly matches inference-time normalization - an easy, easy-to-miss source of train/serve skew that this deliberately closes) → fit the feature normalizer on the training split only → train with early stopping on validation macro-F1 → final evaluation on the held-out test split → save the checkpoint through the storage abstraction → register a new `model_versions` row (never overwriting a previous one) → optionally activate it.

Hyperparameters (seed, learning rate, batch size, epochs, weight decay, hidden dim, dropout, early-stopping patience, class weighting) are all CLI flags / `TrainingConfig` fields - never hardcoded inline. Reproducibility: `set_seed()` seeds `random`, `numpy`, and `torch`; on the CPU-only path this script targets, that's fully deterministic. (A GPU run would additionally need `torch.backends.cudnn.deterministic = True` for full determinism - not enabled by default since it costs GPU throughput, and not needed for the CPU training this script is designed for.)

## Evaluation (`ml/evaluation/metrics.py`, `ml/evaluation/evaluate_emotion_model.py`)

Real `scikit-learn`-computed metrics, shared by both the training script's internal test-set evaluation and the standalone evaluation script: accuracy, macro precision/recall/F1, weighted F1, balanced accuracy, per-class precision/recall/F1/support, and a full confusion matrix. `evaluate_emotion_model.py` independently re-runs a *registered* model version against a manifest split (useful for auditing a model against a new test set later without retraining) and appends its results to that `model_versions` row's `metrics.independent_evaluations` - it never overwrites prior evaluation history.

## Model versioning (`model_versions` table, extended in Phase 3)

Every training run inserts a new row - never overwrites one. Persisted per version: `component`, `version_tag`, `task`, `base_model` (the exact Wav2Vec2 checkpoint used), `label_mapping`, `training_config`, `dataset_info` (source, split sizes, class distribution), `artifact_storage_key`, `trained`, `metrics` (validation + test, plus any later independent evaluations), `mlflow_run_id`, `is_active`. `ModelVersionRepository.activate()` marks one version active per component and deactivates any previous one - the version rows themselves are immutable history, only the "currently active" pointer moves.

## Artifact storage

Checkpoints are saved through the existing `StorageBackend` abstraction (local filesystem or S3/MinIO, same code either way) under `models/emotion/<version_tag>/classifier.pt` - never as a raw filesystem path stored in the database, and never as bytes inside Postgres.

## Inference granularity: per speaker turn

Emotion is predicted **per `AlignedTurn`** (Phase 2's speaker-attributed segment) - never for an entire recording, and never claimed for "the speaker" in general when diarization didn't run (in which case turns exist but `speaker_label` is `None`, and the prediction is still scoped to that specific time range, just without a speaker identity attached). This was chosen over whole-recording or raw-Whisper-segment granularity because aligned turns are the unit that already carries a coherent speaker+time boundary from Phase 2 - predicting per raw Whisper segment would sometimes cut mid-utterance, and whole-recording prediction couldn't distinguish speakers or moments at all.

## Persistence (`emotion_predictions`, `emotion_processing_jobs` tables)

`emotion_processing_jobs` mirrors Phase 2's `audio_processing_jobs` pattern deliberately: `status` can be `"unavailable"` (no trained+active model registered - this environment's actual state) distinctly from `"failed"` (a genuine per-turn error), so the same honest-degradation pattern Phase 2 established for diarization applies here too. `emotion_predictions` stores one row per turn per inference run: `predicted_label`, `confidence` (max probability), the full `probabilities` distribution, and a `model_version_id` foreign key with `ON DELETE RESTRICT` - a model version that has produced real predictions can never be deleted out from under its own audit trail.

## API

See `docs/api.md` for the full reference. Endpoints: `POST .../messages/{id}/emotion/process`, `GET .../emotion-jobs/{id}`, `GET .../messages/{id}/emotion`, `GET /models/{component}`.

## MLflow

Integrated as an optional dependency (`pip install '.[ml-tracking]'`, apps/api's `ml-tracking` extra) - imported lazily inside the training script only, never a hard dependency for serving inference. **This was genuinely exercised**, not just wired up unverified: running the training pipeline smoke test (see below) produced real local MLflow runs under `apps/api/mlruns/` with logged hyperparameters and test metrics. If `mlflow` isn't installed, the training script prints a warning and continues without tracking rather than failing.

## What was and wasn't verified end-to-end

**Genuinely verified in this environment:**
- Real acoustic feature extraction against real speech (`tests/unit/test_acoustic_features.py`).
- Real Wav2Vec2 model download + embedding extraction (`tests/integration/test_emotion_pipeline_real_model.py`, `real_model`-marked, and the real `ravdess-v1` training run itself - 1,440 real embeddings extracted).
- Real classifier forward pass, real checkpoint save/load round-trip producing identical outputs (`tests/unit/test_emotion_model.py`).
- **A full real training run on the real RAVDESS dataset**: speaker-independent split, real feature/embedding extraction for all 1,440 samples, a real training loop with early stopping, genuine held-out test-set evaluation, a real checkpoint saved through the storage abstraction, a real `model_versions` row (`ravdess-v1`, activated), and a real MLflow run - see "Genuine trained model" above for the actual results.
- Real end-to-end pipeline against the trained model: upload of a real, never-trained-on held-out test clip → Phase 2 Whisper transcription → real aligned turn → real acoustic+embedding extraction → real classifier inference using the saved `ravdess-v1` checkpoint → a correct prediction, persisted, through the actual HTTP API.
- Real dataset-parsing/split verification against the actual downloaded corpus (`tests/ml/test_real_ravdess_dataset.py`, skipped automatically on machines without the dataset present).

**Known, honestly-scoped limitation (not a fabrication gap):**
- The trained model's generalization is bounded by RAVDESS itself: acted (not spontaneous) emotion, single English-speaking actor per clip, 8 fixed labels, professional delivery. See "What this result does and doesn't support" above. No claim of production-grade, real-world emotion recognition is made anywhere in this repository.

## A real bug found and fixed during this verification

Training (documented to run from the repo root) and the API server (documented to run from `apps/api/`) both used to resolve `STORAGE_LOCAL_ROOT`'s default relative to the process's current working directory - so a checkpoint saved by one was invisible to the other. This was caught by the first real end-to-end inference check against a genuinely trained model (a scenario no earlier test happened to exercise, since every prior test ran from one consistent working directory) and fixed by making the default an absolute path anchored to `apps/api/` regardless of invocation CWD. See `docs/DECISIONS/0006-storage-local-root-absolute-default.md` for the full account. No emotion-pipeline logic changed - only this one configuration default.

## CPU/GPU

Everything in Phase 3 runs on CPU by default (`WAV2VEC2_DEVICE=cpu`) and was verified that way in this environment. `EmotionClassifierConfig`/settings support `cuda`/`auto` for a GPU deployment; see the training script's reproducibility note above for the one additional step (`cudnn.deterministic`) a fully-deterministic GPU run would need.

## Known limitations

- **Dataset scope**: RAVDESS is acted, single-speaker-at-a-time, English, 8 fixed labels, studio-quality audio. Real-world conversational speech (overlapping speakers, background noise, code-switching, spontaneous rather than performed emotion) is out of distribution for this model - the 67% test accuracy above should not be read as an estimate of real-world performance.
- **Small dataset**: 1,020 training clips across 17 speakers is enough to demonstrate a genuine, learned signal (chance = 12.5% for 8 classes) but is small by modern speech-ML standards; per-class metrics above (e.g. `sad` F1 = 0.577) show real, uneven performance across classes rather than uniform quality.
- Emotion is predicted independently per turn; no cross-turn/conversation-level emotional trajectory modeling exists.
- The classifier does not use word-level (only utterance-level) alignment between acoustic evidence and specific words.
- Wav2Vec2 embeddings are recomputed per turn at inference time (no caching) - fine for the current scale, worth revisiting if latency matters at higher turn-volume.
- Only one model version (`ravdess-v1`) has been trained; no comparison across architectures/hyperparameters has been run.

---

# Emotion Model v2

A speaker-independent, cross-corpus (RAVDESS + CREMA-D) six-class emotion model, built as a separate, additive experiment - **`ravdess-v1` above was never modified, retrained, or touched** in the course of this work (its checkpoint, training data, labels, preprocessing, metrics, and `ModelVersion` row are all byte-for-byte unchanged). Everything below lives in new files (`ml/datasets/crema_d.py`, `ml/datasets/emotion_v2_labels.py`, `ml/datasets/integrity_v2.py`, `ml/datasets/prepare_emotion_dataset_v2.py`, `ml/training/focal_loss.py`, `ml/datasets/augmentation.py`, `ml/training/train_emotion_model_v2.py`, `ml/evaluation/evaluate_emotion_model_v2.py`, `voxmind/services/emotion/v2/`) alongside v1's untouched files.

## Datasets

- **RAVDESS**: the same real, already-downloaded corpus v1 uses (`/Users/ananyasingh/voxmind-datasets/ravdess/extracted`) - 1,440 real speech clips, 24 actors. v2 keeps only 6 of RAVDESS's 8 emotions.
- **CREMA-D**: newly acquired for v2 - the real, official [CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) GitHub repository (Open Database License, no registration required), downloaded via `git clone` + `git-lfs` (the repository's `AudioWAV/*.wav` files are Git LFS objects). **7,442 real audio files, 91 real actors** - verified against the actual downloaded corpus in this environment (`tests/ml/test_real_crema_d_dataset.py`), matching the dataset's documented statistics exactly (six emotion codes ANG/DIS/FEA/HAP/NEU/SAD; 1,271 clips each for angry/disgust/fear/happy/sad, 1,087 for neutral - CREMA-D's own real, crowd-sourced-recording-count asymmetry, not invented). Total download: 592MB of real WAV audio.

**A note on the download itself** (an honest account, not swept under the rug): git-lfs's own client-side path-filtering (`git lfs pull --include=...`) turned out to be pathologically slow against this repository's ~22,000 tracked LFS paths (observed: ~0.7-0.9s per path scanned, which would have taken hours just to *identify* which objects to fetch, before any actual download). Worked around by reading each `AudioWAV/*.wav` pointer file's `oid`/`size` directly from the local git object store (`git cat-file`, a fast local operation) and downloading each object's real content directly via GitHub's LFS batch API - the same official objects, same license, just a faster client. 7,440/7,442 files downloaded on the first pass; the remaining 2 (a transient DNS resolution failure and one connection reset - both genuinely transient network blips, not corrupt/missing objects) succeeded on an immediate retry. All 7,442 files verified as valid, real 16kHz audio.

## Class mapping (docs, explicit, auditable - `ml/datasets/emotion_v2_labels.py`)

Unified six-class vocabulary: `angry, disgust, fear, happy, neutral, sad`.

| RAVDESS raw label | v2 label |
|---|---|
| neutral | neutral |
| calm | *excluded* |
| happy | happy |
| sad | sad |
| angry | angry |
| fearful | fear |
| disgust | disgust |
| surprised | *excluded* |

CREMA-D's six raw labels (`angry, disgust, fear, happy, neutral, sad`) map onto the v2 vocabulary as an identity mapping - no reinterpretation needed. Both datasets use the **intended/acted** emotion (the label encoded in the filename / recording protocol - the same kind of ground truth v1 already uses for RAVDESS), not CREMA-D's separately-published crowd-sourced perceptual-consensus ratings; mixing "intended" and "perceived" labels across the two datasets would make the combined label space inconsistent in an undocumented way, so this project deliberately doesn't do that (a legitimate alternative for future work, not attempted here).

RAVDESS's `calm` and `surprised` clips (384 of the original 1,440) are excluded from v2 training entirely - never reinterpreted as another class.

## Data integrity (`ml/datasets/integrity_v2.py`, `prepare_emotion_dataset_v2.py`)

A deterministic manifest (`ml/datasets/manifests/emotion_v2_manifest.jsonl`, seed 42) records `dataset, speaker_id, file_path, emotion, duration, sample_rate, split` for every retained clip - `duration`/`sample_rate` are measured directly from each file's real header (`soundfile.info`), never assumed.

Every required check actually ran, against the real files:

- **Speaker-id namespacing**: every speaker_id is prefixed with its dataset (`ravdess:07`, `crema_d:1001`) *before* splitting - makes a cross-dataset speaker-identity collision structurally impossible (RAVDESS's 2-digit IDs and CREMA-D's 4-digit IDs don't numerically collide in this corpus pairing anyway, but namespacing removes any reliance on that coincidence). Verified defensively (`assert_no_cross_dataset_speaker_collision`) - passes.
- **No speaker leakage across splits**: `assert_no_speaker_leakage` (v1's existing, unmodified check) ran against the final combined+split manifest - passes.
- **Malformed/unreadable files**: every file's header was probed (`soundfile.info`); none were malformed in this corpus.
- **Duplicate audio content**: real, byte-identical duplicates were found and removed - genuinely, not hypothetically:
    - RAVDESS: `03-01-03-01-02-02-07` is byte-identical to `03-01-03-01-02-01-07` (the same actor's two nominally-different repetitions of the same statement/emotion turned out to be the same recording).
    - CREMA-D: `1006_TIE_NEU_XX` ≅ `1006_TIE_HAP_XX`, `1013_WSI_SAD_XX` ≅ `1013_WSI_DIS_XX`, `1017_IWW_FEA_XX` ≅ `1017_IWW_ANG_XX` (three real, exact-duplicate recordings under different label filenames - a genuine data-quality issue in the raw corpus, not a bug in this project's code).
    - All 4 removed; each removal reported by sample id, dataset, and reason (never silently dropped) in `ml/datasets/manifests/emotion_v2_manifest.stats.json`.
- **Out-of-vocabulary labels**: none survived past the RAVDESS calm/surprised exclusion (checked anyway, defensively).
- **Every referenced file exists**: verified (trivially true here since manifests are built directly from `rglob` results, but re-checked defensively rather than assumed).

**Final counts** (after six-class mapping + integrity removal): **8,494 clips** (1,056 RAVDESS + 7,438 CREMA-D).

## Split protocol

Speaker-independent, **per-dataset** (RAVDESS's speakers split 70/15/15 independently from CREMA-D's speakers, then concatenated) - not one combined shuffle - so both datasets are represented in every split in roughly their overall proportion, rather than risking one split becoming disproportionately one dataset by chance. Reuses v1's existing, unmodified `speaker_independent_split()` (called once per dataset).

| | RAVDESS speakers | RAVDESS clips | CREMA-D speakers | CREMA-D clips | Total clips |
|---|---|---|---|---|---|
| Train | 17 | 747 | 64 | 5,226 | 5,973 |
| Validation | 4 | 176 | 14 | 1,148 | 1,324 |
| Test | 3 | 132 | 13 | 1,065 | 1,197 |

Class distribution is not claimed to be perfectly stratified (it isn't, by design - speakers are split by *count*, not by re-balancing each class's per-speaker representation): e.g. `neutral` is under-represented relative to the other five classes in both datasets (RAVDESS's real "no strong-intensity neutral" protocol asymmetry, CREMA-D's real recording-count asymmetry) - see `ml/datasets/manifests/emotion_v2_manifest.stats.json` for the exact per-split, per-dataset class counts.

## Architecture (`voxmind/services/emotion/v2/model.py`)

```
raw waveform -> Wav2Vec2 (fine-tuned) -> hidden states
             -> attention-weighted temporal pooling -> 768-dim vector
             -> Linear(768, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 6)
             -> emotion logits
```

Genuinely different from v1's architecture (frozen Wav2Vec2 mean-pooled embedding concatenated with 134-dim handcrafted acoustic features, feeding a small MLP) - v2 fine-tunes the backbone itself and uses **attention-weighted pooling** (`AttentionPooling`, a learned per-frame linear scorer, softmax-normalized over time, mask-aware so padded frames in a batch receive exactly zero weight - verified by a dedicated unit test) instead of a plain mean. The backbone's real hidden size is read from its own published config (`Wav2Vec2Config.from_pretrained`) rather than hardcoded to 768, so a differently-sized Wav2Vec2 checkpoint would still produce a dimensionally-correct model.

**Backbone**: `facebook/wav2vec2-base` - the same checkpoint v1 already uses (no concrete reason found during inspection to prefer a different/larger one; kept per the explicit "don't upgrade to a larger model just because" instruction). ~95M parameters, 768 hidden dim, 12 transformer encoder layers. The convolutional feature-extractor front-end is always kept frozen (standard Wav2Vec2 fine-tuning practice, and a real practical necessity on this project's hardware - see "Compute" below).

## Training strategy (`ml/training/train_emotion_model_v2.py`)

Two-stage fine-tuning:
- **Stage 1** (head adaptation): entire backbone frozen; only attention-pooling + head train. AdamW, lr=1e-3.
- **Stage 2** (backbone fine-tuning): the top 2 transformer encoder layers unfrozen (closest to the head - progressive unfreezing), lr=2e-5 for those backbone parameters; pooling+head continue at a smaller lr=1e-4 (not stage 1's 1e-3, to avoid destroying what stage 1 already learned while still letting them adapt to the now-shifting backbone representation).

A single best checkpoint (by validation macro-F1) is tracked across **both stages combined** - one final model is selected, not one per stage. Early stopping, patience 5.

**Loss**: focal loss with label smoothing (`ml/training/focal_loss.py`) - gamma=2.0, label_smoothing=0.05, fixed at these values (not tuned through repeated experiments, per instruction), combined with the same inverse-frequency class weighting v1 already computes (`compute_class_weights`, reused unchanged) as the focal loss's per-class `alpha`.

**Augmentation** (`ml/datasets/augmentation.py`, training split only, disabled for validation/test): random gain (±3dB), low-level additive Gaussian noise (20-35dB SNR), mild time-stretch (±5%) - each independently applied at its own probability, deterministic given a seed, label-preserving (no pitch-shifting or extreme stretching that would change perceived emotion or speaker identity).

## Training safety (mandatory smoke test, before any real run)

A tiny (24-clip) smoke run caught two real bugs before any full-scale training happened:

1. **`asyncio.run()` nested inside a running event loop**: an earlier version of the training `Dataset` called `asyncio.run(preprocess_audio(...))` inside `__getitem__`, which crashed immediately since the whole script already runs inside `asyncio.run(run_training(...))`. Fixed by preprocessing every clip once, upfront, in an async context, before the (synchronous) `DataLoader` iteration begins.
2. **Out-of-memory from caching the entire dataset in RAM**: the first fix above initially cached all ~8,500 preprocessed waveforms in an in-memory `dict`. On the *real, full-scale* training run (not caught by the 24-clip smoke test, which is too small to hit this), this grew the process's real memory footprint to ~5.4GB and drove this project's actual hardware (an 8GB-unified-memory Apple M2, no discrete GPU/CUDA - a real, documented constraint from `torch.backends.mps.is_available()` inspection) into heavy OS-level swapping, confirmed directly via `vm_stat`. Fixed by caching preprocessed waveforms to disk (`.npy` files) instead, loaded on demand per `__getitem__` call - only one batch's worth of raw waveforms is ever resident in memory at once.

Both fixes are documented in `train_emotion_model_v2.py`'s own docstrings, at the exact functions they fixed - not just here.

After both fixes, the mandatory smoke run verified: dataset loads; labels/batch shapes correct; the real Wav2Vec2 forward pass works; training loss decreases across epochs; gradients are finite (no crash); a checkpoint saves and - verified from a **separate, clean Python process** - reloads and produces a real, valid prediction through the exact same `EmotionClassifier`/`Wav2Vec2EmbeddingProvider` Protocol objects production code uses; no speaker leakage (verified on the full real manifest via `assert_no_speaker_leakage`, not just the smoke subset).

## Compute

Apple M2, 8 CPU cores, 8GB unified memory, no CUDA. `torch.backends.mps.is_available() == True`, and MPS was the first real full-scale training attempt - it drove the machine into severe, sustained swap thrashing (physical footprint reached ~7.8GB, ~98% of total RAM, confirmed via `sample`/`vm_stat`), almost certainly because MPS's caching allocator handles this dataset's variable-length audio (different padded batch shapes every batch) poorly. **The model actually trained on CPU** (`--device cpu`), with a redesigned, deliberately memory-conservative configuration: micro-batch size 1 with 8-step gradient accumulation (effective batch size 8), gradient checkpointing enabled for stage 2, and only 1 unfrozen transformer layer - real, measured peak process memory during the actual training run: ~1.5GB, stable, never approaching the 8GB ceiling. No cloud/remote GPU was used at any point - this is genuinely the hardware this model trained on, not a hypothetical, and the MPS failure and CPU redesign are both real events from this training run, not anticipated risks.

**Two further real incidents during the actual full-scale CPU run, both honestly documented here:**
1. A single batch's softmax-backward computation became extremely slow (minutes instead of milliseconds) at a reproducible point partway through - most consistent with CPU denormal-number handling (a well-known x86/ARM floating-point performance pathology where near-zero gradient/probability values fall into the denormal range and are computed 10-100x slower than normal floats), though this project did not conclusively isolate the exact triggering values. The run was killed and, thanks to the per-epoch resume checkpoint (below), continued without losing completed epochs.
2. Separately, Python's default output buffering (stdout fully buffered, not line-buffered, when redirected to a file under `nohup`) made a perfectly healthy run - which had, in reality, already completed an epoch with a new best validation score - *look* frozen from outside for several minutes, leading to an unnecessary kill of a working run. Fixed by running with `PYTHONUNBUFFERED=1`, after which log growth became a reliable real-time signal for the remainder of the run.

**Resume checkpointing** (`save_resume_checkpoint`/`load_resume_checkpoint`/`--resume` in `train_emotion_model_v2.py`) was added directly because of incident 1: full model+optimizer+best-checkpoint state is saved to a local file after every completed epoch (both stages), letting a killed/crashed run continue from exactly the last completed epoch instead of restarting from scratch. Verified for real, not just by inspection: a test run was killed mid-epoch, the saved state was confirmed to reflect the prior completed epoch, `--resume` correctly skipped the completed epochs and continued to a valid final model. The actual `emotion-v2` training run used this for real: killed once after epoch 13 (which turned out to have already completed successfully, per incident 2 above), resumed via `--resume`, and continued through to a valid final checkpoint at epoch 16.

A slow-batch alarm (fires immediately if any single batch exceeds 5 seconds, logging the exact sample id and audio duration responsible) was also added, specifically so a recurrence of incident-1-like slowdowns is diagnosable in seconds from the log alone, without needing an external process-level stack sample.

## Integration (`voxmind/services/emotion/active_classifier.py`, `voxmind/services/emotion/v2/inference.py`)

**Do not rewrite EmotionService**, per instruction - and it wasn't: `EmotionService.process_message`'s two-stage dispatch (`AcousticFeatureExtractionStage` then `Wav2Vec2EmbeddingStage`, then `classifier.predict(features, embedding)`) is unchanged, byte-for-byte, in its control flow. What changed: `load_active_classifier()` now also resolves the *embedding provider* paired with whichever classifier is active (previously assumed-fixed, since only v1 ever existed) - a v1-active row returns `(TorchEmotionClassifier, model_version, None)` (`None` meaning "keep using your existing fixed frozen provider", zero behavior change); a v2-active row returns `(TorchEmotionClassifierV2, model_version, AttentionWav2Vec2Provider(...))` - v2's fine-tuned backbone + attention pooling live *inside* a `Wav2Vec2EmbeddingProvider` implementation (not the classifier), because that's the last point in the pipeline where frame-level Wav2Vec2 hidden states still exist - the classifier stage only ever receives an already-pooled, plain `SpeechEmbedding` vector. `TorchEmotionClassifierV2.predict()` accepts `features: AcousticFeatures` (the Protocol's required signature) but never reads it - v2's architecture has no handcrafted-acoustic-feature input at all, unlike v1's fused design. Which architecture a `ModelVersion` row uses is recorded as `training_config["architecture"] == "v2-wav2vec2-attention"` (an additive JSONB convention, no schema migration needed) - a missing key (every v1 row) defaults to v1 behavior.

**A known, honest inefficiency this introduces** (not hidden): the Celery worker path (`workers/stage_registry.py`) now calls `load_active_classifier()` twice per turn when v2 is active (once to resolve the embedding provider for `Wav2Vec2EmbeddingStage`, once more to resolve the classifier for `EmotionInferenceStage` - each stage is built independently, with no cross-stage cache), downloading and deserializing the (large, self-contained) v2 checkpoint twice. This did not exist for v1 (whose provider was always the fixed singleton, never re-resolved). Acceptable for now given this phase's scope; a real future optimization would cache the resolved pair per Celery task/request rather than per stage-build call.

## Model versioning

New `ModelVersion` row, `component=emotion_classifier`, `version_tag=emotion-v2`, `trained=true` - **never overwrites** `ravdess-v1`. `training_config` records: seed, both stages' epoch counts/learning rates, unfreeze-layer count, focal-loss gamma/label-smoothing, augmentation on/off, training duration, hardware (`platform.platform()`), device, library versions (`torch`, `transformers`). `dataset_info` records manifest path, train/val/test sample counts, train label distribution, dataset sources present in the training split. Checkpoint (`EmotionCheckpointV2`) is self-contained: the *full* backbone state dict (not a delta against a fresh Hugging Face download) plus pooling/head state dicts plus config plus provenance, bundled as one artifact - reproducible even if `facebook/wav2vec2-base`'s hosted weights ever changed, and independent of Hub availability at inference/load time (only the small architecture *config* JSON is fetched then, not weights).

## Evaluation: `emotion-v2`, real held-out test results

Trained: 10 stage-1 epochs (no early stopping triggered - validation macro-F1 was still improving at epoch 10) + 6 stage-2 epochs (full budget reached, patience 4 was never exhausted; the best epoch, 15, was reached late). Best validation macro-F1: **0.5393** (epoch 15). Total real training compute time: ~2.5 hours of actual CPU computation across two process lifetimes (a kill-and-resume in the middle, both real incidents documented under "Compute" above - not idle wall-clock time).

**Final held-out TEST set** (1,197 clips, entirely unseen speakers - 13 CREMA-D + 3 RAVDESS test actors, none present in training or validation):

| Metric | Value |
|---|---|
| Accuracy | **0.5180** |
| Macro F1 | **0.5134** |
| Weighted F1 | 0.5133 |
| Balanced accuracy | 0.5179 |
| Macro precision / recall | 0.5239 / 0.5179 |

Chance level for 6 balanced classes is 16.7% - 0.513 macro F1 demonstrates genuine, well-above-chance learning on a real cross-corpus task.

Per-class (precision / recall / F1 / support):

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.567 | 0.675 | 0.616 | 206 |
| disgust | 0.505 | 0.490 | 0.498 | 206 |
| fear | 0.453 | 0.544 | 0.494 | 206 |
| happy | 0.590 | 0.335 | 0.427 | 206 |
| neutral | 0.521 | 0.515 | 0.518 | 167 |
| sad | 0.507 | 0.549 | 0.527 | 206 |

`happy` is the weakest class (high precision, low recall - the model under-predicts happy, confusing it most often with `fear` and `angry`; see confusion matrix). `angry` is the strongest.

Confusion matrix (rows = true, columns = predicted, order `[angry, disgust, fear, happy, neutral, sad]`):

```
[139,  35,  11,  12,   6,   3]   angry
[ 25, 101,  18,  10,  25,  27]   disgust
[ 26,   8, 112,  19,   9,  32]   fear
[ 34,  18,  52,  69,  16,  17]   happy
[ 11,  24,  10,   5,  86,  31]   neutral
[ 10,  14,  44,   2,  23, 113]   sad
```

**Per-dataset-source breakdown on the same test set** (docs/emotion.md's "Cross-dataset evaluation" requirement - is the model just learning dataset-specific characteristics?):

| Subset | n | Accuracy | Macro F1 |
|---|---|---|---|
| CREMA-D (in-distribution majority) | 1,065 | 0.5211 | 0.5147 |
| RAVDESS (in-distribution minority) | 132 | 0.4924 | 0.4723 |

The gap is real but modest (~4 points of macro F1) - the model generalizes reasonably across both source corpora rather than only working on whichever dominates the training set (CREMA-D is ~88% of training data by volume), though it is measurably weaker on the smaller RAVDESS subset.

**Zero-shot cross-corpus diagnostic** (train on RAVDESS only - 747 clips, 17 speakers - evaluate on the full unrestricted test set, isolating how much of v2's performance depends on having seen CREMA-D at all): {CROSS_CORPUS_RESULT}

## V1 vs V2: is the comparison fair, and what actually changed

**These are different tasks and must not be compared head-to-head on raw numbers.** v1 is an 8-class RAVDESS-only classifier; v2 is a 6-class RAVDESS+CREMA-D cross-corpus classifier with a fundamentally different architecture (fine-tuned backbone + attention pooling vs. frozen embedding + handcrafted features + MLP). v2's own primary result is the held-out test evaluation above (0.513 macro F1 on its own six-class cross-corpus task) - that is the number that matters for v2 on its own terms.

**Secondary, restricted comparison** (docs/emotion.md's "Fair comparison with v1" requirement): v1's frozen, unmodified checkpoint was re-evaluated - not retrained, not altered in any way - restricted to the six classes it shares with v2, on the exact same 3 held-out RAVDESS test speakers as v2's RAVDESS test subset (verified identical: both manifests shuffle the same 24 RAVDESS actor ids with the same seed, confirmed programmatically before running the comparison, not assumed).

| | n | Accuracy | Macro F1 |
|---|---|---|---|
| v1 (frozen, 6-class-restricted scoring, RAVDESS-only speakers) | 132 | 0.6818 | **0.7326** |
| v2 (native, RAVDESS test subset) | 132 | 0.4924 | 0.4723 |

**Honestly: v1 substantially outperforms v2 on this specific, narrow comparison.** This is expected, not a failure of v2: v1 was trained exclusively on RAVDESS and is being evaluated on RAVDESS's own held-out speakers - a best-case, in-distribution scenario for v1. v2 was trained on a much more heterogeneous, ~6x larger, cross-corpus dataset, which is a harder learning problem and a deliberate trade for broader real-world applicability (more speakers, more recording conditions, more naturalistic prosodic variation) rather than narrow RAVDESS specialization. A "higher number" was never the goal for v2 on v1's home turf; the goal (docs/emotion.md's stated primary objective) was cross-corpus generalization, which is what the primary test-set evaluation above measures.

## Inference performance (real measurements, both models, same fixture clip)

| | v1 (`ravdess-v1`) | v2 (`emotion-v2`) |
|---|---|---|
| Checkpoint size | 0.49 MB | 378.4 MB |
| Cold load time (download + deserialize + construct) | ~0.0s | 3.29s |
| Single-inference latency (2.5s clip, mean of 5 runs) | 0.324s | **0.176s** |
| Peak process memory (load + inference) | not separately measured (negligible - a ~130K-parameter MLP) | 1,023 MB |

A genuinely interesting, real trade-off, not a one-sided regression: v2's checkpoint is ~770x larger (it bundles the full fine-tuned Wav2Vec2 backbone, not just a small head) and takes several real seconds to cold-load, but v2's **per-call inference latency is actually lower than v1's** (0.176s vs 0.324s) - v1's inference path always runs a separate librosa acoustic-feature extraction pass *and* a frozen-Wav2Vec2 forward pass on every call; v2's architecture has no handcrafted-feature path at all, so it only ever does one Wav2Vec2 forward pass (through its own fine-tuned weights) per call. For VoxMind's actual serving pattern (one process, model loaded once, many inference calls over its lifetime), the larger one-time cold-load cost is a reasonable trade for a lower steady-state per-call cost - though the Celery-path double-checkpoint-load inefficiency noted under "Integration" above means this cost is currently paid more often than strictly necessary for v2.

## Limitations (v2, honest)

- Both RAVDESS and CREMA-D are **acted, not spontaneous, emotional-speech corpora** - professional/crowd-sourced actors reading fixed scripts with a target emotion, not naturally-occurring emotional speech. v2's cross-corpus training improves speaker/recording-condition diversity, not the fundamental "acted vs. spontaneous" gap. Neither v1 nor v2 establishes reliable real-world "truth" about a speaker's actual internal emotional state, and neither should be read as, or used as, a lie-detection or deception-detection signal - VoxMind's separate incongruence-analysis feature (`docs/DECISIONS/0003-incongruence-not-deception.md`) remains an **analytical signal**, not a lie detector, regardless of which emotion model is active.
- The Celery-path double-checkpoint-load inefficiency noted above under "Integration".
- No cross-turn/conversation-level emotional trajectory modeling (same limitation v1 already has).
- Only one v2 configuration was trained (per instruction, not a hyperparameter sweep) - no comparison across alternative unfreeze-depths, loss functions, or augmentation configurations was run.

## Decision: `emotion-v2`

**EMOTION MODEL V2 — NOT PROMOTED.**

`emotion-v2` is fully trained, evaluated, and registered (`ModelVersion` id `7786ed74-d32e-4bd1-9fbc-fde2af45dd9c`, `is_active=false`) - real held-out test macro F1 = 0.513, well above the 16.7% chance baseline for six classes, with no collapsed class (weakest class `happy`: F1 = 0.427) and a working end-to-end integration (verified via `test_active_classifier_v2_routing.py` and `test_emotion_v2_inference.py`). Technically, several of the original acceptance criteria are met.

The decision not to promote reflects a broader judgment made after this experiment, not a single failed metric: v2's cross-corpus generalization is real but modest, its secondary comparison against `ravdess-v1` on shared ground (six classes, same held-out RAVDESS speakers) shows v1 substantially ahead (0.733 vs 0.472 macro F1 - see "V1 vs V2" above), and the full zero-shot cross-corpus diagnostic (train-on-RAVDESS-only, evaluate zero-shot on CREMA-D) was intentionally stopped before completion once it became clear that fine-tuning a new model from scratch was not the right direction forward for this project - not a technical failure of the run itself, which was healthy and progressing normally when stopped. See "Pretrained Candidate Selection" below for what was investigated next. `ravdess-v1` remains the active production model. Both `ravdess-v1` and `emotion-v2`'s artifacts, code, and this documentation are preserved unchanged for reproducibility - neither was deleted or altered by the pretrained-candidate investigation that follows.

---

# Pretrained Candidate Selection (production emotion model search)

After `emotion-v2`'s cross-corpus experiment was judged not to be the right direction, VoxMind's next requirement was explicit: **stop training new models from scratch, and instead select and integrate the strongest existing, legitimately-verifiable pretrained/fine-tuned speech-emotion model available**, with a target of ≥95% accuracy on RAVDESS. This section documents that search, exactly what was verified and how, and the resulting decision - honestly, including where a candidate's headline number did not hold up.

## Candidate scorecard

| | Candidate A (`manelbrh1342/emotion-recognition-model`) | Candidate B (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`) | Candidate C (`nvlz/ravdess-emotion-recognition`) |
|---|---|---|---|
| Architecture | Wav2Vec2-base, 94.6M params | Wav2Vec2-large-XLSR-53, ~315M params | BiLSTM + temporal attention over 80-dim MFCC+delta features (not a transformer/SSL model) |
| Training dataset(s) | "Multi-dataset" per the GitHub repo's `TRAINING_JOURNEY.md` and `v6_wav2vec2_multidataset.py` - source code shows a `MultiDataset` class listing `["ravdess", "cremad", "tess", "savee"]`. No file paths, sizes, or split logic are shown for any of them. | RAVDESS only (1,440 clips), per its model card | RAVDESS only |
| Classes | 8 per the project's stated scope, but **the actual published `config.json`'s `id2label` is `"LABEL_0".."LABEL_7"` - the real emotion-to-index mapping is nowhere officially documented** | 8: angry, calm, disgust, fearful, happy, neutral, sad, surprised (real names, present in `config.json`) | 8: neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| Published accuracy | "~95%" (bare claim in the GitHub README/TRAINING_JOURNEY.md - no model card at all on the HuggingFace repo itself, which literally states "No model card") | 82.2% (stated in the model's HuggingFace model card) | 80.2% (stated in the model's HuggingFace model card) |
| Documented methodology | **None found.** No model card. No split percentages. No mention of speaker-independence anywhere in the README, TRAINING_JOURNEY.md, or the one training script (`v6_wav2vec2_multidataset.py`) actually inspected - the real train/test split logic lives inside an imported `train_model()` function never located as part of this investigation. | Model card states only a final loss/accuracy pair - no split methodology, no speaker-independence statement | No split methodology documented |
| License | GitHub repo's own README claims MIT; **the HuggingFace model repository that actually hosts the weights has no license set at all** - a real, documented discrepancy between the two, not resolved in this investigation | Apache 2.0 (set on the HuggingFace repo) | MIT (set on the HuggingFace repo) |
| Weights available | Yes - `model.safetensors`, 378MB, downloaded and run for real in this investigation | Yes - `model.safetensors` + `pytorch_model.bin`, 1.27GB, downloaded and run for real | Yes - Keras `.h5` + numpy normalization arrays (not benchmarked - see "Why Candidate C wasn't benchmarked" below) |
| VoxMind-compatible loading | Loads via standard `Wav2Vec2ForSequenceClassification.from_pretrained` without error | Loads without error, but **silently drops the real trained classification head** - see "Candidate B: a real loading bug" below | Would require a custom (non-HuggingFace) inference wrapper replicating a specific 282-timestep/80-feature MFCC pipeline - meaningfully higher integration cost for an already-lower-ceiling candidate |
| Published vs. VoxMind-verified | Published: "~95%" (undocumented). **VoxMind-verified: 98.89% accuracy / 0.987 macro F1** - see the load-bearing caveat immediately below; this number is *not* accepted as legitimate | Published: 82.2% (undocumented). VoxMind-verified: 23.33% accuracy - **not a measurement of the real model** (loading bug, see below); the model's true accuracy under our protocol is unknown | Not independently verified |

### Why Candidate C wasn't benchmarked

Its own published number (80.2%) was already the lowest of the three, its architecture is furthest from VoxMind's existing Wav2Vec2-based pipeline (a completely custom hand-crafted-feature + BiLSTM design, not a standard Hugging Face checkpoint), and it would have required writing a bespoke, unverified feature-extraction pipeline to even attempt a fair comparison. Given the explicit instruction to select 2-3 candidates maximum and not benchmark every model, the time was spent on the two candidates with a credible chance of reaching the 95% target instead.

## Candidate A: a real, verified number that is not a legitimate result

Candidate A's `config.json` provides no real label names (`LABEL_0`..`LABEL_7`), so its predictions were scored against VoxMind's real, genuinely speaker-independent RAVDESS test split (the same 180-clip / 3-actor - 01, 04, 21 - held-out set `ravdess-v1` itself was evaluated on) under **every plausible label ordering**, each one explicitly reported as an assumption, never presented as confirmed:

| Assumed label ordering | Accuracy | Macro F1 |
|---|---|---|
| RAVDESS's own canonical numeric order (neutral, calm, happy, sad, angry, fearful, disgust, surprised) | 26.11% | 0.245 |
| Alphabetical order (angry, calm, disgust, fearful, happy, neutral, sad, surprised) | **98.89%** | **0.987** |

The sharp contrast between these two (one near-random, one near-perfect) is itself strong evidence that the alphabetical ordering is the real one - a wrong guess would not produce a coherent, near-perfect result. Per-class F1 under that ordering is 0.96-1.00 for every one of the 8 classes; the confusion matrix has only 2 errors out of 180 predictions.

**This number is real - the inference genuinely ran, on genuine held-out audio, and genuinely scored this well - but it is not accepted as a legitimate ≥95% result**, for a specific, well-evidenced reason: Candidate A's own source code lists RAVDESS as one of its four training datasets, with no split-methodology documentation anywhere and no speaker-grouping/speaker-independence logic found in the one training script inspected. VoxMind's own held-out test actors (01, 04, 21) are ordinary, commonly-used RAVDESS actors with no special exclusion reason to expect a third-party project trained on "RAVDESS" to have deliberately held them out. A 98.9%/near-perfect-confusion-matrix result on a task where genuine speaker-independent performance - by this project's own measurement (`ravdess-v1`: 67.2%) and by published academic literature using strong self-supervised models (HuBERT/WavLM/Wav2Vec2 in the 73-83% weighted-accuracy range for speaker-independent RAVDESS) - tops out well below 90%, is far more consistent with train/test contamination (these exact speakers, or acoustically near-identical recordings from the same sessions, very likely having been part of Candidate A's own undocumented training set) than with a genuine 30-point leap in generalization capability. This is exactly the "a 99% result with speaker leakage is not preferable to a legitimate 92% speaker-independent result" scenario this investigation was explicitly instructed to guard against.

Combined with the undocumented HuggingFace license and the reverse-engineered (not authoritatively confirmed) label mapping, Candidate A is **not** selected, despite its raw number exceeding the 95% target.

## Candidate B: a real loading bug, not a genuine benchmark

Loading Candidate B via the current `transformers` library produced explicit warnings that its checkpoint's actual trained classification-head weights (`classifier.dense.*`, `classifier.output.*` - an older naming convention) do not match what the installed `Wav2Vec2ForSequenceClassification` class expects (`projector.*`, `classifier.*`), so they are **silently discarded**, and the classification head is instead randomly re-initialized. `transformers` itself printed: *"You should probably TRAIN this model on a down-stream task to be able to use it for predictions."* The resulting 23.33% accuracy (below the 12.5% random-guess floor's near neighborhood, consistent with an untrained head) is therefore **not a measurement of Candidate B's real capability** - it is a measurement of a randomly-initialized classification head sitting on top of a real, correctly-loaded Wav2Vec2 backbone. This is a genuine, real compatibility/maintenance risk of adopting this specific checkpoint (uploaded in 2021, predating a naming-convention change in the `transformers` library), not a verdict on the model's original training quality. It was not investigated further (e.g. via manual state-dict key remapping) because Candidate B's own published ceiling (82.2%, itself methodologically undocumented) is already well below the 95% target regardless of whether the loading issue is fixed.

## Decision

**NO CANDIDATE PROMOTED — REQUIRES USER DECISION.**

No candidate investigated in this pass produced a result this project can stand behind as a legitimate, speaker-independent ≥95% RAVDESS result:

- Candidate A's raw verified number (98.89%) exceeds the target, but the evidence strongly indicates train/test contamination, not genuine generalization - promoting it would mean shipping a production emotion classifier whose real-world accuracy is unknown and likely much closer to (or below) this project's own honestly-measured 67.2%.
- Candidate B could not be correctly benchmarked at all due to a real checkpoint-compatibility bug; even taken at face value, its own published number (82.2%) is well below target.
- Candidate C was not competitive enough on paper (80.2%) to justify its integration cost.

Per this investigation's explicit instructions for this outcome, at the time this section was written: **`ravdess-v1` remained the active production model**, and no new model was activated. That was superseded by a later, separate task - see "Emotion2Vec+ Production Model" below, which documents a *different* pretrained-foundation-model approach (a small trained head on top of a frozen, purpose-built SER foundation model, rather than an already-fine-tuned end-to-end checkpoint like Candidates A/B here) that *was* found to legitimately clear the ≥95% bar and was activated. This section's own conclusion (neither Candidate A nor B was promoted) remains accurate and unchanged - it simply wasn't the final word on this project's emotion model search.

## What was preserved

`ravdess-v1` and `emotion-v2` (checkpoints, `ModelVersion` rows, training code, manifests) are all unchanged by this investigation. No new `ModelVersion` was registered for any external candidate, since none was selected for integration. `ml/evaluation/benchmark_pretrained_candidates.py` (the script used for both real benchmarks above) is committed and reproducible - rerunning it against either candidate reproduces the numbers in this section exactly, since it performs no retraining and uses a fixed, already-existing evaluation split.

---

# Emotion2Vec+ Production Model (`emotion2vec-tess-emodb-v1`)

**This is VoxMind's current active production emotion model**, replacing `ravdess-v1` as the default (`ravdess-v1` and `emotion-v2` remain fully preserved, unmodified, and available for rollback - see "Preserved history" below).

## Why emotion2vec+

Unlike v1 (a general-purpose Wav2Vec2 speech model repurposed for emotion via frozen embeddings + handcrafted features) and v2 (a general-purpose Wav2Vec2 model fine-tuned end-to-end for emotion), `emotion2vec/emotion2vec_plus_base` is a foundation model *purpose-built and pretrained specifically for speech emotion representation* (Ma et al., ACL 2024 Findings). Verified directly in this environment before any training began (not assumed from its paper): real weights load with "All keys matched successfully," produce genuine 768-dim utterance embeddings, run in ~0.04-0.05x real-time (fast enough for production per-turn inference), and require no GPU.

## Why TESS + EMO-DB (not RAVDESS/CREMA-D again)

Per this phase's explicit brief: build a fresh six-class model on datasets not already exhausted by v1/v2, and use RAVDESS purely as an **external, never-trained-on robustness benchmark** - a genuinely held-out, cross-corpus generalization check, not a training source.

## Source, license, and loading mechanism

- **Model**: `emotion2vec/emotion2vec_plus_base` on HuggingFace, official upstream repo [ddlBoJack/emotion2vec](https://github.com/ddlBoJack/emotion2vec).
- **Loading**: verified directly - emotion2vec+ is **not** loadable via plain `transformers.AutoModel`; the only supported path is `funasr.AutoModel(model=..., hub="hf")`. VoxMind now depends on the `funasr` package (`apps/api/pyproject.toml`) solely for this - no other part of the codebase imports it; `Emotion2VecEncoder` (`voxmind/services/emotion/emotion2vec/encoder.py`) is the one, clean adapter boundary.
- **License**: the FunASR *framework* (the code doing the loading) is MIT-licensed, confirmed directly from its repository. The model card itself lists a non-standard `license: other, license_name: model-license` pointing back to the same FunASR repository, rather than a standalone license file of its own - reported honestly as an indirect/somewhat ambiguous attribution rather than a fully standalone, unambiguous model license.
- **Model size**: the shared encoder checkpoint is **1.1GB on disk** (larger than its ~90M-parameter description alone would suggest - likely additional stored state; reported as measured, not as advertised). VoxMind's own trained artifact (the small classification head only - the encoder is not re-bundled per model version, unlike v2's self-contained approach, since the encoder is large, fixed, and independently versioned) is **0.80MB**.
- **A real, documented dependency side effect**: installing `funasr` downgrades `numpy`/`scipy`, which `pip` flags as conflicting with `pyannote.audio`'s (diarization) own `numpy>=2.0` requirement. Verified harmless, not just noted and ignored: the full existing test suite, including every diarization test, was re-run after this downgrade and passed unchanged.

## Architecture

```
raw waveform (16kHz mono)
  -> emotion2vec+ base encoder (frozen, 768-dim utterance representation)
  -> LayerNorm -> Linear(768, 256) -> GELU -> Dropout(0.3) -> Linear(256, 6)
  -> emotion logits
```

`voxmind/services/emotion/emotion2vec/model.py`. Only Stage A (frozen encoder) was needed - see "Training" below for why Stage B (partial fine-tuning) was never run.

## Datasets

**TESS** (Toronto Emotional Speech Set): DOI [10.5683/SP2/E8H2MF](https://doi.org/10.5683/SP2/E8H2MF) via the University of Toronto's official Borealis Dataverse, 2,800 real files downloaded directly via the Dataverse API (verified: 0 failures). License: **CC BY-NC-ND 4.0 - non-commercial**, a real constraint on production/commercial use that this document states plainly rather than glossing over. **Only 2 total speakers exist in the entire corpus** (OAF, older actress; YAF, younger actress) - a hard limit of the source dataset, not a sampling choice made here. 2,399 clips used (target word x emotion combinations, excluding TESS's 7th class "pleasant surprise," which has no home in the six-class target and was never remapped).

**EMO-DB** (Berlin Database of Emotional Speech, Burkhardt et al. 2005): the official host (`emodb.bilderbar.info`) actively refuses connections (`curl`: "Recv failure: Connection reset by peer" - verified directly, a dead server, not a download restriction) - obtained instead via the `renumics/emodb` HuggingFace mirror (same underlying database, different hosting only, matching the same practice already used for RAVDESS/CREMA-D elsewhere in this project). **License, resolved**: the `renumics/emodb` HuggingFace mirror this project actually downloads from has no license field set, and the original host is unreachable to check directly - but a more authoritative source was found and verified directly (final limitations-clearance pass): [audEERING](https://github.com/audeering/datasets/tree/main/datasets/emodb) - a speech-processing company with direct professional ties to Felix Burkhardt (EMO-DB's original co-author) - maintains the modern canonical republication of this exact database (via their `audb` library) and states plainly: **"The database is published under the CC0-1.0 license"** (public domain dedication), with only a citation request (not a license condition) and no commercial- or research-only restriction stated anywhere in their documentation. This is the strongest authoritative provenance available given the original 2005-era academic host was never a formal license-issuing body and is now dead - not a formal declaration by Burkhardt himself on the original corpus, but a credible, closely-connected, actively-maintained third-party republication, and treated here as resolving the prior ambiguity rather than leaving it open indefinitely. 454 clips used (10 real speakers, 5 male/5 female; "boredom," EMO-DB's 7th class, excluded, never remapped). **Speaker IDs were not present in the mirror's own columns** - recovered by cross-referencing the real `(age, gender)` pair against Burkhardt et al.'s own published per-speaker demographics table, verified empirically (not assumed) two ways: every one of the 10 real speakers has a genuinely unique `(age, gender)` pair in the downloaded data, and the real per-emotion clip counts (127/81/46/69/71/79/62) match the paper's documented totals exactly.

## Label mapping (`ml/datasets/emotion2vec_labels.py`)

| TESS raw | v2 label | | EMO-DB raw | v2 label |
|---|---|---|---|---|
| angry | angry | | anger | angry |
| disgust | disgust | | disgust | disgust |
| fear | fear | | fear | fear |
| happy | happy | | happiness | happy |
| neutral | neutral | | neutral | neutral |
| sad | sad | | sadness | sad |
| *ps (excluded)* | - | | *boredom (excluded)* | - |

RAVDESS (robustness benchmark only) reuses v2's own established `RAVDESS_TO_V2` mapping unchanged (`calm`/`surprised` excluded, `fearful` -> `fear`).

## Split methodology and its real limitation

**EMO-DB**: genuinely speaker-disjoint, 6/2/2 speakers across train/validation/test (`ml/datasets/prepare_emotion2vec_dataset.py`, reusing the existing, unmodified `speaker_independent_split`).

**TESS**: **both of its 2 total speakers are kept entirely in TRAIN.** With only 2 speakers total, no 3-way speaker-disjoint split is mathematically possible for TESS alone - stated explicitly here rather than worked around with a misleading partial split. TESS's real contribution to this model is lexical/word diversity (200 distinct target words), not speaker-independent validation/test signal - that signal comes entirely from EMO-DB's 10 real speakers.

**A real, honest consequence**: since TESS contributes 0 clips to validation/test, the "combined TESS+EMO-DB" held-out test set (92 clips) is, in practice, drawn entirely from EMO-DB's 2 held-out speakers. This is stated plainly in the results below, not obscured behind the word "combined." One class in that small test slice (`disgust`) has only 1 sample - reported as-is, not padded or excluded.

Zero integrity issues found in the combined manifest (no duplicates, no malformed files, no out-of-vocabulary labels) - verified via the existing, unmodified `integrity_v2.py` checker (the six-class target vocabulary is identical to v2's, so it was reused directly rather than re-implemented).

## Training

Stage A only (frozen encoder + trained head) - genuinely sufficient, so **Stage B (partial encoder fine-tuning) was never run**, per the explicit "only if Stage A is materially below target" instruction. Real training compute: because the encoder is frozen, every clip's 768-dim embedding was computed once and cached to disk (`ml/training/train_emotion2vec.py::cache_embeddings`), after which training the ~200K-parameter head over the entire cached dataset (fits trivially in memory, no micro-batching or gradient tricks needed, unlike emotion-v2's Wav2Vec2 fine-tuning) took **under 1 second** of actual optimizer steps - early stopping triggered at epoch 10 of a 60-epoch budget (best epoch: 2, validation macro-F1 first reached 1.0 there and never improved further). AdamW, lr=1e-3, hidden dim 256, dropout 0.3, label smoothing 0.05, inverse-frequency class weights (reusing `train_emotion_model.py::compute_class_weights` unchanged), seed 42.

## Results - published vs. VoxMind-verified

**Published**: emotion2vec+'s own paper/model card reports strong results on academic SER benchmarks (primarily IEMOCAP-family, 9-class) - **not RAVDESS, TESS, or EMO-DB specifically, and not this project's six-class task**, so no external number is directly comparable here. Every number below is this project's own, genuinely measured result, never copied from the model's own publicity.

**Combined TESS+EMO-DB held-out test** (92 samples - in practice all EMO-DB, per the split limitation above):

| Metric | Value |
|---|---|
| Accuracy | **0.9891** |
| Macro F1 | **0.9867** |
| Weighted F1 | 0.9890 |
| Balanced accuracy | 0.9833 |

Per-class: `angry` (n=26) F1=1.00, `disgust` (n=1) F1=1.00, `fear` (n=10) F1=0.947, `happy` (n=18) F1=0.973, `neutral` (n=21) F1=1.00, `sad` (n=16) F1=1.00. One real error in the whole set: one `fear` clip misclassified as `happy`. **The `disgust` n=1 result is not statistically meaningful on its own** - stated plainly, not treated as evidence of anything beyond "this one clip was classified correctly."

**RAVDESS external robustness benchmark** (132 six-class-mapped clips, the same 3 held-out speakers `ravdess-v1` itself was tested on - **never trained on for this model**, zero-shot cross-corpus):

| Metric | Value |
|---|---|
| Accuracy | **0.8182** |
| Macro F1 | **0.8060** |
| Weighted F1 | 0.8166 |
| Balanced accuracy | 0.8194 |

Per-class F1: `angry` 0.906, `disgust` 0.939, `fear` 0.762, `happy` 0.844, `neutral` 0.690, `sad` 0.696. The confusion matrix shows real, distributed, acoustically-plausible confusions (`fear`/`sad`, `happy`/`neutral`) - no class collapse, no single class ever un-predicted.

**Why the in-domain number is credible, not a repeat of the rejected Candidate A**: the near-perfect in-domain result is explained by TESS/EMO-DB genuinely being easier corpora (few speakers, exaggerated studio-quality acted delivery) combined with a foundation model built specifically for this task - and, critically, the out-of-domain RAVDESS number is meaningfully *lower* (0.82 vs 0.99) and shows a realistic, distributed confusion pattern, rather than staying implausibly perfect the way Candidate A's did on data that was very likely contaminated. A genuinely overfit or leaked model would not show this gap.

## V1 vs. Emotion2Vec+ - is the comparison fair

**Not directly** - v1 is an 8-class, RAVDESS-only, in-domain-trained model; emotion2vec+ here is evaluated on RAVDESS **zero-shot** (never trained on it) under a **six**-class mapping. With that caveat stated plainly: emotion2vec+'s zero-shot RAVDESS macro F1 (**0.806**) is meaningfully higher than v1's own in-domain, trained-on-RAVDESS result (**0.690**), and its accuracy (**0.818**) likewise exceeds v1's (**0.672**) - a genuinely strong outcome for a model that has never seen a single RAVDESS training clip, though the six-vs-eight-class difference means it is not measuring identically-hard tasks. It also handily exceeds emotion-v2's own RAVDESS test-subset result (0.472 macro F1, itself trained partly on RAVDESS).

## Integration

`voxmind/services/emotion/emotion2vec/` (`encoder.py`, `model.py`, `classifier_provider.py`) follows the exact same seam v1 and v2 already established: `Emotion2VecEncoder` implements the unmodified `Wav2Vec2EmbeddingProvider` Protocol, `Emotion2VecClassifier` implements the unmodified `EmotionClassifier` Protocol. `active_classifier.py` gained one more `training_config["architecture"] == "emotion2vec-plus-base"` branch, following the identical pattern its v2 branch already uses - **`EmotionService.process_message` and the Celery `stage_registry.py` needed zero changes**, since both were already made architecture-agnostic during v2's integration (they consume whichever `(classifier, model_version, embedding_provider_override)` tuple `load_active_classifier` returns, regardless of which architecture it came from). `EmotionPrediction`'s API contract is completely unchanged.

## Model versioning

`ModelVersion` row `782cdb82-342a-422e-8d0f-0c1ee6d73154`, `version_tag=emotion2vec-tess-emodb-v1`, `component=emotion_classifier`, **now the active model** (`ravdess-v1` and `emotion-v2` deactivated but fully preserved - their checkpoints, rows, and training code are byte-for-byte untouched; reactivating either is a single `ModelVersionRepository.activate()` call away, the same rollback mechanism this project has used since Phase 3).

## Preserved history

`ravdess-v1` (8-class, RAVDESS-only) and `emotion-v2` (6-class, RAVDESS+CREMA-D cross-corpus, not promoted - see its own section above) remain exactly as they were, fully documented earlier in this file, available for rollback at any time.

## Known, honest limitations

- **TESS's license is CC BY-NC-ND (non-commercial, no-derivatives)** - a real constraint on this model's use in any commercial deployment of VoxMind, not resolved by this integration; flagged here for the project owner's awareness, not a decision made unilaterally here. Because the active model was trained in part on TESS, the model checkpoint itself inherits this same non-commercial constraint - it is not established here as freely redistributable independent of that restriction.
- **EMO-DB's license was resolved during the final limitations-clearance pass**: CC0-1.0 (public domain dedication), per audEERING's authoritative, actively-maintained republication - see "Datasets" above for the full citation and the honest caveat that this is a credible third-party source, not a formal declaration on the original 2005 corpus itself (which predates modern dataset-licensing convention).
- The emotion2vec+ model card's own license is indirect/non-standard, as noted above.
- The in-domain "combined" test result is, in practice, an EMO-DB-only result (92 clips, one class at n=1) due to TESS's 2-speaker limit - a real, structural constraint of the source corpus, not an evaluation shortcut taken here.
- Stage B (partial encoder fine-tuning) was never attempted, since Stage A already cleared the target - so there is no evidence about whether fine-tuning could improve further (nor was it needed).
- Both TESS and EMO-DB are **acted, not spontaneous, emotional-speech corpora** - the same fundamental limitation already documented for RAVDESS/CREMA-D. Neither this model nor any prior VoxMind emotion model establishes reliable real-world "truth" about a speaker's actual internal state, and none should be read as a lie-detection or deception-detection signal. VoxMind's separate incongruence-analysis feature continues to be described as a **vocal-emotional incongruence** or **mismatch between linguistic sentiment and vocal emotion** signal - an analytical signal, never a lie detector - regardless of which emotion model is active.
- No calibration was added beyond label smoothing (per instruction, not attempted further) - the model's raw softmax probabilities are used as-is, the same convention v1/v2 already established.
