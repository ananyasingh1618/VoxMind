from __future__ import annotations

import pytest

from ml.evaluation.metrics import compute_classification_metrics


def test_perfect_predictions_yield_perfect_metrics():
    y_true = ["happy", "sad", "angry", "happy", "sad"]
    y_pred = ["happy", "sad", "angry", "happy", "sad"]

    metrics = compute_classification_metrics(y_true, y_pred, ["happy", "sad", "angry"])

    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.weighted_f1 == 1.0
    assert metrics.balanced_accuracy == 1.0


def test_known_confusion_matrix_shape_and_values():
    y_true = ["happy", "happy", "sad", "sad"]
    y_pred = ["happy", "sad", "sad", "sad"]

    metrics = compute_classification_metrics(y_true, y_pred, ["happy", "sad"])

    # rows=true, cols=pred, order=["happy", "sad"]
    assert metrics.confusion_matrix == [[1, 1], [0, 2]]
    assert metrics.n_samples == 4


def test_per_class_metrics_have_correct_support():
    y_true = ["happy", "happy", "happy", "sad"]
    y_pred = ["happy", "happy", "sad", "sad"]

    metrics = compute_classification_metrics(y_true, y_pred, ["happy", "sad"])

    assert metrics.per_class["happy"].support == 3
    assert metrics.per_class["sad"].support == 1
    assert metrics.per_class["happy"].recall == pytest.approx(2 / 3)


def test_completely_wrong_predictions_yield_zero_accuracy():
    y_true = ["happy", "happy"]
    y_pred = ["sad", "sad"]

    metrics = compute_classification_metrics(y_true, y_pred, ["happy", "sad"])

    assert metrics.accuracy == 0.0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compute_classification_metrics(["happy"], ["happy", "sad"], ["happy", "sad"])


def test_empty_evaluation_set_raises_rather_than_reporting_fake_metrics():
    with pytest.raises(ValueError):
        compute_classification_metrics([], [], ["happy", "sad"])


def test_to_dict_is_json_serializable():
    import json

    metrics = compute_classification_metrics(["happy", "sad"], ["happy", "sad"], ["happy", "sad"])
    json.dumps(metrics.to_dict())  # must not raise
