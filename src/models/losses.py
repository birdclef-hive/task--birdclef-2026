"""Loss functions for BirdCLEF 2026 multilabel classification.

Includes FocalLoss, combined BCE+Focal (2nd place), and differentiable
SoftAUC loss (3rd place) for direct metric optimization.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in multilabel classification.

    Down-weights well-classified examples, focusing training on hard negatives.
    Essential for 650+ species with extreme long-tail distribution.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        return (self.alpha * focal_weight * bce).mean()


class FocalBCELoss(nn.Module):
    """Linear combination of BCE and Focal Loss with label smoothing.

    2nd place BirdCLEF 2025 approach: balanced blend prevents Focal Loss
    from over-suppressing easy examples while still handling imbalance.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        focal_weight: float = 0.5,
        label_smoothing: float = 0.05,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.focal_weight = focal_weight
        self.label_smoothing = label_smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            num_classes = target.shape[-1]
            smooth_pos = 1.0 - self.label_smoothing
            smooth_neg = self.label_smoothing / num_classes
            target = target * smooth_pos + (1 - target) * smooth_neg

        bce = F.binary_cross_entropy(pred, target)
        focal = self.focal(pred, target)
        return self.bce_weight * bce + self.focal_weight * focal


class SoftAUCLoss(nn.Module):
    """Differentiable AUC approximation for direct metric optimization.

    3rd place BirdCLEF 2025 innovation. Approximates pairwise AUC by
    comparing all positive-negative prediction pairs per class.
    Supports soft labels from pseudo-labeling.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for c in range(pred.shape[1]):
            pos_mask = target[:, c] > 0.5
            neg_mask = target[:, c] <= 0.5
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            pos_preds = pred[:, c][pos_mask]
            neg_preds = pred[:, c][neg_mask]
            # Pairwise differences: each positive should score higher than each negative
            diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)
            loss = F.binary_cross_entropy_with_logits(
                diff * self.margin, torch.ones_like(diff)
            )
            losses.append(loss)
        if losses:
            return torch.stack(losses).mean()
        # No valid classes in batch — return zero gradient
        return pred.sum() * 0


def build_loss(config: dict[str, Any]) -> nn.Module:
    """Factory function to build loss from config."""
    loss_cfg = config.get("loss", config)
    loss_type = loss_cfg.get("type", "focal_bce")

    if loss_type == "focal_bce":
        return FocalBCELoss(
            label_smoothing=loss_cfg.get("label_smoothing", 0.05),
            focal_alpha=loss_cfg.get("focal_alpha", 0.25),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        )
    elif loss_type == "focal":
        return FocalLoss(
            alpha=loss_cfg.get("focal_alpha", 0.25),
            gamma=loss_cfg.get("focal_gamma", 2.0),
        )
    elif loss_type == "soft_auc":
        return SoftAUCLoss(margin=loss_cfg.get("margin", 1.0))
    elif loss_type == "bce":
        return nn.BCELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
