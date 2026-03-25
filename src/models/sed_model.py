"""Sound Event Detection model with timm backbones for BirdCLEF 2026.

SED architecture with attention-based pooling for frame-level predictions,
supporting diverse backbone families via timm: EfficientNet, NFNet, RegNet.
"""

from __future__ import annotations

import logging
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AttentionHead(nn.Module):
    """Attention-based pooling head for Sound Event Detection.

    Produces both clip-level and frame-level predictions by learning
    attention weights over time frames, enabling the model to focus
    on frames with vocalizations.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.attention = nn.Linear(hidden_dim, num_classes)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, time_frames, features)

        Returns:
            clip_pred: (batch, num_classes) — clip-level predictions
            frame_preds: (batch, time_frames, num_classes) — frame-level
        """
        x = self.dropout(F.relu(self.fc1(x)))
        attention_weights = torch.softmax(self.attention(x), dim=1)
        frame_preds = torch.sigmoid(self.classifier(x))
        clip_pred = (attention_weights * frame_preds).sum(dim=1)
        return clip_pred, frame_preds


class SEDModel(nn.Module):
    """Sound Event Detection model with timm backbone.

    Supports any timm model as backbone. Tested with:
    tf_efficientnet_b0_ns, tf_efficientnet_b3_ns, tf_efficientnetv2_s.in21k,
    tf_efficientnetv2_b3, eca_nfnet_l0, regnety_008, regnety_016
    """

    def __init__(
        self,
        backbone_name: str = "tf_efficientnet_b0_ns",
        num_classes: int = 650,
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout_backbone: float = 0.25,
        dropout_head: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="",
        )

        # Infer feature dimension from a dummy forward pass
        with torch.no_grad():
            dummy = torch.randn(1, 1, 128, 312)  # ~5s at hop=512
            features = self.backbone(dummy)
            feat_dim = features.shape[1]

        logger.info(
            "Backbone %s: feature_dim=%d, params=%.1fM",
            backbone_name,
            feat_dim,
            sum(p.numel() for p in self.backbone.parameters()) / 1e6,
        )

        # Adaptive pool: collapse frequency dim, keep time dim
        self.freq_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.dropout_backbone = nn.Dropout(dropout_backbone)
        self.head = AttentionHead(feat_dim, num_classes, hidden_dim, dropout_head)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, 1, n_mels, time_frames) mel spectrogram

        Returns:
            clip_pred: (batch, num_classes)
            frame_preds: (batch, time_frames, num_classes)
        """
        features = self.backbone(x)  # (B, C, H, W)
        features = self.freq_pool(features).squeeze(-1)  # (B, C, T)
        features = features.permute(0, 2, 1)  # (B, T, C)
        features = self.dropout_backbone(features)
        clip_pred, frame_preds = self.head(features)
        return clip_pred, frame_preds

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SEDModel":
        """Create model from a config dict (model section of base.yaml)."""
        model_cfg = config.get("model", config)
        return cls(
            backbone_name=model_cfg.get("backbone", "tf_efficientnet_b0_ns"),
            num_classes=model_cfg.get("num_classes", 650),
            pretrained=model_cfg.get("pretrained", True),
            hidden_dim=model_cfg.get("hidden_dim", 512),
            dropout_backbone=model_cfg.get("dropout_backbone", 0.25),
            dropout_head=model_cfg.get("dropout_head", 0.5),
        )
