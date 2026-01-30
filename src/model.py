"""Model architecture for wound classification."""

import torch
import torch.nn as nn

# Try timm first, fallback to torchvision
try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights


class WoundClassifier(nn.Module):
    """Pretrained CNN for 6-class wound classification.

    Supports MobileNetV3-Small (CPU-optimized) and ResNet18 backbones.
    """

    BACKBONES = {"mobilenet_v3_small", "resnet18"}

    def __init__(
        self,
        num_classes: int = 6,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.2,
    ):
        """Initialize the classifier.

        Args:
            num_classes: Number of output classes.
            backbone: Backbone architecture ("mobilenet_v3_small" or "resnet18").
            pretrained: Use ImageNet pretrained weights.
            freeze_backbone: Freeze backbone weights (train only classifier).
            dropout: Dropout rate in classifier head.
        """
        super().__init__()

        if backbone not in self.BACKBONES:
            raise ValueError(f"backbone must be one of {self.BACKBONES}, got '{backbone}'")

        self.backbone_name = backbone
        self.num_classes = num_classes

        # Build backbone and classifier
        if backbone == "mobilenet_v3_small":
            self._build_mobilenet(pretrained, dropout, num_classes)
        else:
            self._build_resnet(pretrained, dropout, num_classes)

        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()

    def _build_mobilenet(self, pretrained: bool, dropout: float, num_classes: int) -> None:
        """Build MobileNetV3-Small backbone with custom classifier."""
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)

        # Extract backbone (features)
        self.backbone = base.features

        # Pooling layer
        self.pool = base.avgpool

        # Get classifier input features
        in_features = base.classifier[0].in_features

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),  # MobileNetV3 uses Hardswish
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def _build_resnet(self, pretrained: bool, dropout: float, num_classes: int) -> None:
        """Build ResNet18 backbone with custom classifier."""
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)

        # Extract backbone (everything except fc)
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )

        # Pooling layer
        self.pool = base.avgpool

        # Get classifier input features
        in_features = base.fc.in_features

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (for fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        # Feature extraction
        x = self.backbone(x)

        # Global pooling
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Classification
        x = self.classifier(x)

        return x


# =============================================================================
# timm EfficientNet Wrapper (for 2-phase fine-tuning)
# =============================================================================


class EfficientNetClassifier(nn.Module):
    """EfficientNet classifier using timm library.

    Designed for 2-phase fine-tuning:
    - Phase A: Freeze backbone, train head only
    - Phase B: Unfreeze last N blocks, train with differential LR

    EfficientNet structure (timm):
    - model.conv_stem, model.bn1: stem layers
    - model.blocks: list of 7 stages (blocks[0] to blocks[6])
    - model.conv_head, model.bn2: head conv layers
    - model.classifier: final FC layer

    For partial unfreezing, we unfreeze blocks[-2:] (last 2 stages).
    """

    SUPPORTED_MODELS = {"tf_efficientnetv2_s", "efficientnet_b0"}

    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_s",
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """Initialize EfficientNet classifier.

        Args:
            model_name: timm model name.
            num_classes: Number of output classes.
            pretrained: Use ImageNet pretrained weights.
            dropout: Dropout rate before classifier.
        """
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for EfficientNet. Install with: pip install timm")

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_name must be one of {self.SUPPORTED_MODELS}")

        self.model_name = model_name
        self.num_classes = num_classes

        # Create timm model with custom head
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier, we'll add our own
            drop_rate=dropout,
        )

        # Get feature dimension
        in_features = self.model.num_features

        # Custom classifier head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        # Track frozen state
        self._backbone_frozen = False

    def freeze_backbone(self) -> None:
        """Freeze entire backbone (Phase A: head-only training)."""
        for param in self.model.parameters():
            param.requires_grad = False
        self._backbone_frozen = True

    def unfreeze_backbone_partial(self, num_blocks: int = 2) -> None:
        """Unfreeze last N blocks for fine-tuning (Phase B).

        Args:
            num_blocks: Number of final blocks to unfreeze (default 2).
        """
        # First, ensure everything is frozen
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last N blocks
        # EfficientNet has model.blocks as a Sequential of stages
        total_blocks = len(self.model.blocks)
        unfreeze_from = max(0, total_blocks - num_blocks)

        for i in range(unfreeze_from, total_blocks):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = True

        # Also unfreeze conv_head and bn2 (between blocks and classifier)
        if hasattr(self.model, "conv_head"):
            for param in self.model.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.model, "bn2"):
            for param in self.model.bn2.parameters():
                param.requires_grad = True

        self._backbone_frozen = False
        print(f"Unfroze blocks[{unfreeze_from}:{total_blocks}] + conv_head/bn2")

    def unfreeze_backbone_full(self) -> None:
        """Unfreeze entire backbone."""
        for param in self.model.parameters():
            param.requires_grad = True
        self._backbone_frozen = False

    def get_param_groups(self, backbone_lr: float, head_lr: float) -> list:
        """Get parameter groups with differential learning rates.

        Args:
            backbone_lr: LR for backbone (unfrozen parts).
            head_lr: LR for classifier head.

        Returns:
            List of param group dicts for optimizer.
        """
        # Backbone params (only trainable ones)
        backbone_params = [p for p in self.model.parameters() if p.requires_grad]

        # Head params
        head_params = list(self.classifier.parameters())

        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # timm model with num_classes=0 returns features after global pool
        features = self.model(x)
        logits = self.classifier(features)
        return logits


# =============================================================================
# Factory Functions
# =============================================================================


def build_model(
    num_classes: int = 6,
    backbone: str = "mobilenet_v3_small",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.2,
) -> WoundClassifier:
    """Factory function to build a WoundClassifier (torchvision).

    Args:
        num_classes: Number of output classes.
        backbone: Backbone architecture ("mobilenet_v3_small" or "resnet18").
        pretrained: Use ImageNet pretrained weights.
        freeze_backbone: Freeze backbone weights.
        dropout: Dropout rate in classifier head.

    Returns:
        Configured WoundClassifier model.
    """
    return WoundClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )


def build_efficientnet(
    model_name: str = "tf_efficientnetv2_s",
    num_classes: int = 6,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = True,
) -> EfficientNetClassifier:
    """Factory function to build an EfficientNet classifier (timm).

    Args:
        model_name: "tf_efficientnetv2_s" (default) or "efficientnet_b0" (lighter).
        num_classes: Number of output classes.
        pretrained: Use ImageNet pretrained weights.
        dropout: Dropout rate in classifier head.
        freeze_backbone: If True, freeze backbone for Phase A training.

    Returns:
        Configured EfficientNetClassifier model.
    """
    model = EfficientNetClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )

    if freeze_backbone:
        model.freeze_backbone()

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Testing WoundClassifier (torchvision)")
    print("=" * 60)
    model = build_model(backbone="mobilenet_v3_small", pretrained=True)
    print(f"Model: {model.backbone_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")

    if TIMM_AVAILABLE:
        print("\n" + "=" * 60)
        print("Testing EfficientNetClassifier (timm)")
        print("=" * 60)

        # Test with backbone frozen (Phase A)
        model = build_efficientnet(
            model_name="efficientnet_b0",  # Lighter for testing
            pretrained=True,
            freeze_backbone=True,
        )
        print(f"Model: {model.model_name}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(
            f"Trainable (Phase A): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        # Test partial unfreeze (Phase B)
        model.unfreeze_backbone_partial(num_blocks=2)
        print(
            f"Trainable (Phase B): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        print(f"Input: {x.shape} -> Output: {out.shape}")
    else:
        print("\ntimm not installed. Skipping EfficientNet test.")
