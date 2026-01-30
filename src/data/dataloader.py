"""DataLoader utilities for wound classification."""

from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def build_train_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    generator=None,
) -> DataLoader:
    """Build DataLoader for training.

    Args:
        dataset: Training dataset.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        generator: Optional torch.Generator for reproducible shuffling.

    Returns:
        DataLoader for training.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )


def build_test_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    generator=None,
) -> DataLoader:
    """Build DataLoader for testing/validation.

    Args:
        dataset: Test/validation dataset.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        DataLoader for testing/validation.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )


def build_weighted_train_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    generator=None,
) -> DataLoader:
    """Build DataLoader for training with class imbalance handling.

    Uses WeightedRandomSampler with inverse-frequency weights to handle
    class imbalance by oversampling minority classes.

    Args:
        dataset: Training dataset with a `samples` attribute containing
            (path, label) tuples.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        DataLoader for training with weighted sampling.
    """
    # Extract labels from dataset.samples
    labels = [label for _, label in dataset.samples]

    # Compute class counts
    class_counts = Counter(labels)

    # Compute inverse-frequency weights for each class
    num_samples = len(labels)
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}

    # Assign weight to each sample based on its class
    sample_weights = torch.tensor([class_weights[label] for label in labels])

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Must be False when using sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
