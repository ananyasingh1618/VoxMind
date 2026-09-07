# ml/

Scaffolding for the emotion classifier's training/evaluation lifecycle (Phase 3) and later RAG evaluation (Phase 4/7). Empty by design until those phases land - see the root [README](../README.md) and [docs/architecture.md](../docs/architecture.md) for the roadmap. Nothing here is implemented yet; no model has been trained, and no metrics exist to report.

- `datasets/` - dataset download/prep scripts, data card
- `preprocessing/` - audio feature extraction pipelines shared with `apps/api/voxmind/services/emotion`
- `training/` - the emotion classifier training script
- `evaluation/` - STT/emotion/retrieval/grounding evaluation scripts (Phase 7)
- `models/` - versioned checkpoints (git-ignored; tracked via MLflow, not committed to the repo)
- `notebooks/` - exploratory analysis
