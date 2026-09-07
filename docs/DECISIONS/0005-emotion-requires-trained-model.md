# ADR 0005: Emotion inference requires a trained, active model - no fallback

## Decision

`EmotionService` (the only code path reachable from `api/v1/endpoints/emotion.py`) never produces an `EmotionPrediction` unless a `ModelVersion` row exists with `component="emotion_classifier"`, `trained=True`, and `is_active=True`. If none exists, `/emotion/process` reports `EmotionProcessingJob.status="unavailable"` with a clear, actionable reason - it never falls back to running an untrained or randomly-initialized network and presenting its output as a real prediction.

## Why

An untrained (or randomly-initialized) classifier's softmax output is a real tensor computation, but it carries no genuine signal about emotion - it's noise shaped like a probability distribution. Serving it through the same API surface a real trained model would use, even with a `trained: false` flag buried in the response, risks that flag being ignored by a client (or a future engineer extending the UI) and the number being presented to an end user as if it meant something. The only safe way to guarantee that never happens is to make the untrained path structurally unreachable from the product-facing service, not merely labeled.

## What this environment's actual state is

**Update**: a real labeled dataset (RAVDESS) has since been supplied and trained on - see `docs/emotion.md`'s "Genuine trained model: ravdess-v1" section for the real, measured results (test macro F1 = 0.690). A `model_versions` row now exists with `component="emotion_classifier"`, `trained=True`, `is_active=True`, so `/emotion/process` now genuinely serves real predictions rather than reporting `"unavailable"`.

This section is left here, unedited below, as the accurate historical record of the state this ADR was written against - the *decision* (no untrained fallback, ever) did not change, and was exactly what made it possible to add a real trained model later without touching `EmotionService`'s gating logic at all.

> No labeled emotion dataset has been supplied to this environment (see `docs/emotion.md`, `ml/datasets/README.md`). Consequently, no row in `model_versions` for `component="emotion_classifier"` has `trained=True` in this deployment, and every real request to `/emotion/process` made against this codebase as shipped will report `"unavailable"`. This is the honest, correct behavior, not a bug to be worked around.

## Where the untrained path *is* legitimately used

`services/emotion/classifier_provider.build_untrained_baseline()` constructs a real (randomly-initialized) classifier for exactly one purpose: proving the acoustic-feature-extraction → Wav2Vec2-embedding → classifier-forward-pass → persistence pipeline is wired correctly, in tests (`tests/unit/test_classifier_provider.py`, `tests/unit/test_emotion_stages.py`, `tests/integration/test_emotion_pipeline_real_model.py`). It is never imported by `services/emotion_service.py` or any API endpoint. `tests/integration/test_training_pipeline_smoke.py` similarly registers a real-but-untrained model version purely to verify the training script's mechanics - it is never activated, so `EmotionService`'s gate never serves it.

## Consequence for the roadmap

Phase 3 ships the complete real pipeline (feature extraction, embeddings, classifier architecture, training script, evaluation script, model registry, inference stages, API, UI) in a state that is honestly "not yet producing real predictions in this deployment" until someone runs `ml/training/train_emotion_model.py` against a real labeled dataset and activates the resulting model version. This is the correct state to ship in, per the project's explicit rule against claiming training occurred when it did not.
