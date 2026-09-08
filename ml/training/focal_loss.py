"""Focal loss with label smoothing (docs/emotion.md, "Loss" section) - used
by Emotion Model v2's training script only; v1's training script
(train_emotion_model.py) is untouched and keeps using plain
`CrossEntropyLoss` with inverse-frequency class weights.

Configuration is fixed at the values specified for this project
(gamma=2.0, label_smoothing=0.05) rather than tuned through repeated
experiments, per docs/emotion.md's explicit instruction not to sweep
gamma across "dozens of experiments." `alpha` (optional per-class weights)
is supported and, when supplied, reuses the exact same inverse-frequency
weighting v1 already computes (`train_emotion_model.py::compute_class_weights`)
- not a second, different weighting scheme.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLossWithLabelSmoothing(nn.Module):
    """Standard multi-class focal loss (Lin et al., 2017) composed with
    label smoothing, computed directly on log-probabilities so both
    effects combine in one numerically-stable expression rather than one
    being applied as a post-hoc correction to the other.

    gamma: down-weights the loss contribution of already-easy (high
        predicted-probability-of-true-class) examples, so hard/minority-
        class examples dominate the gradient more than they would under
        plain cross-entropy - the documented purpose (docs/emotion.md).
    label_smoothing: replaces the one-hot target with
        `(1 - smoothing)` on the true class and
        `smoothing / (num_classes - 1)` spread over the rest, reducing
        the model's incentive to produce extreme, overconfident logits.
    alpha: optional per-class weight tensor (e.g. inverse-frequency
        weights), applied multiplicatively alongside the focal term - not
        a substitute for it.
    """

    # A real, PyTorch-recommended fix, not a suppression: `register_buffer`
    # sets `self.alpha` dynamically via `nn.Module.__setattr__`, which
    # mypy's torch stubs can't reliably infer through - without this
    # explicit class-level annotation, mypy instead resolves `self.alpha`
    # via `nn.Module.__getattr__`'s generic fallback and (a known stub
    # limitation for exactly this pattern) infers it as the `Tensor` type
    # object itself rather than an instance, making `self.alpha.to(...)`
    # look like an attempt to call the class. Declaring the real runtime
    # type here is the officially documented way to type a PyTorch buffer.
    alpha: torch.Tensor

    def __init__(self, *, num_classes: int, gamma: float = 2.0, label_smoothing: float = 0.05, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if gamma < 0.0:
            raise ValueError("gamma must be >= 0.")
        self.num_classes = num_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("alpha", alpha if alpha is not None else torch.ones(num_classes), persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (batch, num_classes) raw logits. targets: (batch,) int64 class indices."""
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        with torch.no_grad():
            smoothed_targets = torch.full_like(log_probs, self.label_smoothing / (self.num_classes - 1))
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Per-class focal weight (1 - p_t)^gamma, broadcast against every
        # class's smoothed target mass (not just the argmax class) so label
        # smoothing's "soft" targets are still down-weighted consistently
        # with the focal term's intent.
        focal_weight = (1.0 - probs).pow(self.gamma)

        alpha = self.alpha.to(logits.device).unsqueeze(0)  # (1, num_classes)
        per_class_loss = -alpha * focal_weight * smoothed_targets * log_probs
        return per_class_loss.sum(dim=-1).mean()
