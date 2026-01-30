"""Image transforms for wound classification."""

from torchvision import transforms

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Build transforms for training images.

    Uses conservative augmentations suitable for medical wound images.
    Aggressive augmentations (e.g., vertical flip, large rotations) are avoided
    as they may create unrealistic wound presentations.

    Args:
        image_size: Target size for images (will be cropped to image_size x image_size).

    Returns:
        Composed transforms for training.
    """
    return transforms.Compose(
        [
            # RandomResizedCrop: scale 0.8-1.0 for slight zoom variation
            # Preserves wound context while adding augmentation
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            # Horizontal flip is safe: wounds can appear on either side of body
            transforms.RandomHorizontalFlip(p=0.5),
            # Small rotation: ±10 degrees for realistic capture angle variation
            # Larger rotations avoided to preserve wound orientation/drainage direction
            transforms.RandomRotation(degrees=10),
            # Mild color jitter: accounts for lighting variation in clinical settings
            # Conservative values preserve diagnostic color information
            transforms.ColorJitter(
                brightness=0.1,  # Mild: clinical lighting varies
                contrast=0.1,  # Mild: camera/screen differences
                saturation=0.05,  # Very low: tissue color is diagnostically important
                hue=0.0,  # ZERO: hue changes could mask infection signs (redness)
            ),
            # Optional grayscale: helps model focus on texture/shape features
            # Low probability to maintain color as primary signal
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_test_transforms(image_size: int = 224) -> transforms.Compose:
    """Build transforms for test/validation images.

    Deterministic transforms only for reproducible evaluation.

    Args:
        image_size: Target size for images (will be cropped to image_size x image_size).

    Returns:
        Composed transforms for testing/validation.
    """
    return transforms.Compose(
        [
            # Resize to 256, then center crop to target size
            # Matches training preprocessing while being deterministic
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_train_transforms_light(image_size: int = 224) -> transforms.Compose:
    """Build lighter augmentations for Phase A (head-only training).

    Minimal augmentation to establish baseline before fine-tuning.

    Args:
        image_size: Target size for images.

    Returns:
        Composed transforms for training.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
