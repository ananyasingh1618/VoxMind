"""Real classification metrics via scikit-learn - shared by the training
script's internal test-set evaluation and the standalone
evaluate_emotion_model.py, so there is exactly one metric computation to
trust. Nothing here is invented; every number is a function of the
`y_true`/`y_pred` arrays actually passed in.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class PerClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    per_class: dict[str, PerClassMetrics]
    confusion_matrix: list[list[int]]
    label_order: list[str]
    n_samples: int

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


def compute_classification_metrics(
    y_true: list[str], y_pred: list[str], label_names: list[str]
) -> ClassificationMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length.")
    if not y_true:
        raise ValueError("Cannot compute metrics over an empty evaluation set.")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_names, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_names, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(y_true, y_pred, labels=label_names, average="weighted", zero_division=0)

    per_class = {
        label: PerClassMetrics(
            precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]), support=int(support[i])
        )
        for i, label in enumerate(label_names)
    }

    cm = confusion_matrix(y_true, y_pred, labels=label_names)

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        label_order=list(label_names),
        n_samples=len(y_true),
    )
