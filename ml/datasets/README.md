# Emotion Dataset

VoxMind's emotion classifier is designed to train on **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) - chosen because it's freely licensed (CC BY-NC-SA 4.0, no registration/application required), speaker-labeled (24 professional actors, exactly what speaker-independent splitting needs), and uses a clean, parseable filename convention with no separate label file to keep in sync.

**This dataset has been downloaded and trained on in this environment** (the public Zenodo release, no authentication required) - see `docs/emotion.md`'s "Genuine trained model: ravdess-v1" section for the actual, measured results (test macro F1 = 0.690). The raw corpus (~200MB) is not committed to the repository - re-download it from the URL below to reproduce.

## Getting RAVDESS

1. Download the "Audio-only (16bit, 48kHz) speech" subset from Zenodo: https://zenodo.org/records/1188976 (file `Audio_Speech_Actors_01-24.zip`, ~200MB). No login required.
2. Extract it - you'll get 24 `Actor_NN/` folders, each containing ~60 `.wav` files named like `03-01-06-01-02-01-12.wav`.

## Filename convention (parsed by `ravdess.py`)

`{modality}-{vocal_channel}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav`

| Position | Field | Values |
|---|---|---|
| 1 | Modality | 01=full-AV, 02=video-only, 03=audio-only |
| 2 | Vocal channel | 01=speech, 02=song (only speech is used) |
| 3 | Emotion | 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised |
| 4 | Intensity | 01=normal, 02=strong |
| 5 | Statement | 01, 02 (two fixed sentences) |
| 6 | Repetition | 01, 02 |
| 7 | Actor | 01-24 (odd=male, even=female) - used as `speaker_id` |

## Preparing the manifest

Run from the repo root, using apps/api's venv Python directly (so `ml` resolves via the repo root on `sys.path` and `voxmind` resolves via the editable install in that venv):

```bash
apps/api/.venv/bin/python -m ml.datasets.prepare_emotion_dataset \
  --ravdess-dir /path/to/Audio_Speech_Actors_01-24 \
  --output ml/datasets/manifests/ravdess_manifest.jsonl
```

This scans for RAVDESS-named `.wav` files, keeps only the speech (not song) subset, applies a **speaker-independent** train/validation/test split (default 70/15/15 by actor count, seeded, see `splitting.py`), and writes a JSONL manifest. The script prints the resulting label distribution per split - inspect it before training; RAVDESS has a real, expected imbalance (see `docs/emotion.md`'s "Class imbalance" section).

## Manifest format

One JSON object per line:

```json
{"sample_id": "03-01-06-01-02-01-12", "audio_path": "/abs/path/.../03-01-06-01-02-01-12.wav", "label": "fearful", "dataset_source": "ravdess", "speaker_id": "12", "split": "train"}
```

## Speaker-independent splitting

Splitting is done by **actor** (speaker), not by individual clip - every clip from a given actor goes entirely into one split. This prevents the classifier from partially learning to recognize a specific person's voice (which would inflate test-set accuracy in a way that wouldn't generalize to a new speaker VoxMind has never heard). See `splitting.py`'s docstring and `assert_no_speaker_leakage`, which the training script runs as a real check before training starts, not just a design intention.

## Using a different dataset

`ml/datasets/schema.py`'s `EmotionSample`/manifest format is dataset-agnostic. To use CREMA-D, IEMOCAP, or another corpus, write a parser analogous to `ravdess.py` (real filename/metadata parsing for that dataset, no invented labels) producing the same `EmotionSample` list, then reuse `splitting.py` and the training script unchanged.

## CREMA-D, TESS, EMO-DB (Emotion Model v2 / Emotion2Vec+ production model)

See `docs/emotion.md`'s "Emotion Model v2" and "Emotion2Vec+ Production Model" sections for the full account (why each dataset was chosen, license terms, and exact reproduction commands). Briefly:

- **CREMA-D**: official [CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) GitHub repo (Git LFS - `git lfs pull`), ODbL license. Parsed by `crema_d.py`.
- **TESS**: official University of Toronto Borealis Dataverse, DOI [10.5683/SP2/E8H2MF](https://doi.org/10.5683/SP2/E8H2MF), CC BY-NC-ND 4.0 (**non-commercial**) - only 2 total speakers in the whole corpus, a real limitation, not a sampling choice. Parsed by `tess.py`.
- **EMO-DB**: the original host (`emodb.bilderbar.info`) is unreachable (verified - the server actively resets connections); obtained via the `renumics/emodb` HuggingFace mirror instead. **License: CC0-1.0** (public domain dedication), per [audEERING](https://github.com/audeering/datasets/tree/main/datasets/emodb)'s authoritative, actively-maintained republication (audEERING has direct professional ties to Felix Burkhardt, EMO-DB's original co-author) - resolved during the final limitations-clearance pass; see `docs/emotion.md`'s "Datasets" section for the full citation and honest caveats. Parsed by `emodb.py`, which also recovers real speaker ids from the mirror's `(age, gender)` columns against the dataset's own published speaker demographics table (verified empirically against the real downloaded data - see the module's docstring).
