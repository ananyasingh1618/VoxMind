from __future__ import annotations

import torch

from ml.training.focal_loss import FocalLossWithLabelSmoothing


def test_loss_is_finite_and_positive_for_random_logits():
    criterion = FocalLossWithLabelSmoothing(num_classes=4)
    logits = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))

    loss = criterion(logits, targets)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_confident_correct_predictions_yield_lower_loss_than_confident_wrong_ones():
    criterion = FocalLossWithLabelSmoothing(num_classes=3, label_smoothing=0.0)
    targets = torch.tensor([0, 1])

    correct_logits = torch.tensor([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
    wrong_logits = torch.tensor([[-10.0, 10.0, -10.0], [10.0, -10.0, -10.0]])

    assert criterion(correct_logits, targets).item() < criterion(wrong_logits, targets).item()


def test_higher_gamma_downweights_easy_examples_more(monkeypatch=None):
    targets = torch.tensor([0])
    # A confidently-correct ("easy") example.
    easy_logits = torch.tensor([[8.0, -8.0]])

    loss_gamma0 = FocalLossWithLabelSmoothing(num_classes=2, gamma=0.0, label_smoothing=0.0)(easy_logits, targets)
    loss_gamma2 = FocalLossWithLabelSmoothing(num_classes=2, gamma=2.0, label_smoothing=0.0)(easy_logits, targets)

    # Higher gamma down-weights an already-easy example's loss further.
    assert loss_gamma2.item() < loss_gamma0.item()


def test_label_smoothing_prevents_zero_loss_on_a_perfect_prediction():
    targets = torch.tensor([0])
    perfect_logits = torch.tensor([[100.0, -100.0]])

    loss_no_smoothing = FocalLossWithLabelSmoothing(num_classes=2, gamma=0.0, label_smoothing=0.0)(perfect_logits, targets)
    loss_with_smoothing = FocalLossWithLabelSmoothing(num_classes=2, gamma=0.0, label_smoothing=0.1)(perfect_logits, targets)

    assert loss_with_smoothing.item() > loss_no_smoothing.item()


def test_alpha_upweights_the_specified_class():
    targets = torch.tensor([1])
    logits = torch.tensor([[-2.0, 2.0]])  # confidently correct for class 1

    default_loss = FocalLossWithLabelSmoothing(num_classes=2, label_smoothing=0.0)(logits, targets)
    weighted_loss = FocalLossWithLabelSmoothing(
        num_classes=2, label_smoothing=0.0, alpha=torch.tensor([1.0, 5.0])
    )(logits, targets)

    assert weighted_loss.item() > default_loss.item()


def test_rejects_invalid_label_smoothing():
    import pytest

    with pytest.raises(ValueError):
        FocalLossWithLabelSmoothing(num_classes=3, label_smoothing=1.0)


def test_rejects_negative_gamma():
    import pytest

    with pytest.raises(ValueError):
        FocalLossWithLabelSmoothing(num_classes=3, gamma=-1.0)
