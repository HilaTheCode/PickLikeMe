"""Preference model: swappable backbone + a small linear preference head.

V3 (docs/roadmap.md) replaces the V1/V2 custom CNN backbone with a pretrained
vision backbone, preferred order DINOv3 > SigLIP2 > ConvNeXt, while keeping
PreferenceHead's interface (construction, forward, checkpoint keys) stable so
training/eval/ranking code and saved V1/V2 checkpoints are unaffected. The
original CNN stays available via backbone="cnn" so V2-vs-V3 comparison runs
stay possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

try:
    import timm  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    timm = None

DINOV3_BACKBONE = "vit_small_patch16_dinov3"


@dataclass
class ModelConfig:
    backbone: str = DINOV3_BACKBONE
    pretrained: bool = True
    freeze_backbone: bool = True
    input_channels: int = 3
    cnn_embedding_dim: int = 128


def _build_cnn_backbone(config: ModelConfig) -> tuple[nn.Module, int]:
    backbone = nn.Sequential(
        nn.Conv2d(config.input_channels, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    return backbone, config.cnn_embedding_dim


def _build_pretrained_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if timm is None:
        raise RuntimeError(
            "timm is required for pretrained backbones (pip install -U timm). "
            "docs/roadmap.md V3 preferred order is DINOv3 > SigLIP2 > ConvNeXt; "
            "e.g. ModelConfig(backbone='convnext_tiny') as a fallback if DINOv3 "
            "weights cannot be downloaded."
        )
    try:
        backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
    except TypeError:
        # Some architectures (e.g. ConvNeXt) are resolution-agnostic and don't
        # accept dynamic_img_size at all.
        backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
    return backbone, backbone.num_features


class PreferenceHead(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.freeze_backbone = self.config.freeze_backbone and self.config.backbone != "cnn"

        if self.config.backbone == "cnn":
            self.backbone, embedding_dim = _build_cnn_backbone(self.config)
        else:
            self.backbone, embedding_dim = _build_pretrained_backbone(self.config.backbone, self.config.pretrained)
            if self.freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.backbone.eval()

        self.classifier = nn.Linear(embedding_dim, 1)

    def train(self, mode: bool = True) -> "PreferenceHead":
        super().train(mode)
        if self.freeze_backbone:
            # Keep a frozen backbone in eval mode (no dropout / BN update)
            # even though the outer model is put in train() during training.
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        features = torch.flatten(features, start_dim=1)
        logits = self.classifier(features)
        return logits
