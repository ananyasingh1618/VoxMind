"""Externalized training hyperparameters - never scattered through the
training script itself. Every field here is a genuine knob the training
script reads; nothing is hardcoded elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    seed: int = 42
    learning_rate: float = 1e-3
    batch_size: int = 16
    epochs: int = 50
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    dropout: float = 0.3
    optimizer: str = "adamw"
    early_stopping_patience: int = 8
    use_class_weights: bool = True
    train_fraction: float = 0.7
    validation_fraction: float = 0.15

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "optimizer": self.optimizer,
            "early_stopping_patience": self.early_stopping_patience,
            "use_class_weights": self.use_class_weights,
        }
