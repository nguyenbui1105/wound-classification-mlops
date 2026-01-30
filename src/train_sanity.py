"""Sanity training run to validate the pipeline."""

import sys
from pathlib import Path

# Add project root to sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
from collections import Counter

import torch
import torch.nn as nn
from torchvision import models

from src.data.dataloader import build_test_dataloader, build_weighted_train_dataloader
from src.data.dataset import WoundDataset
from src.data.transforms import build_test_transforms, build_train_transforms

# Default config
BATCH_SIZE = 32
IMAGE_SIZE = 224
LEARNING_RATE = 1e-3
MAX_STEPS = 30
MAX_EVAL_STEPS = 10
SEED = 42
NUM_CLASSES = 6


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sanity training run")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-eval-steps", type=int, default=MAX_EVAL_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def build_model(num_classes: int) -> nn.Module:
    """Build ResNet18 model with custom classifier."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int,
    log_interval: int = 10,
) -> float:
    """Train for one epoch (or max_steps)."""
    model.train()
    total_loss = 0.0
    num_steps = 0

    for step, (images, labels) in enumerate(loader):
        if step >= max_steps:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_steps += 1

        if (step + 1) % log_interval == 0:
            print(f"  Step {step + 1}/{max_steps} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    return avg_loss


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    max_steps: int,
) -> float:
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            if step >= max_steps:
                break

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def main():
    args = parse_args()

    # Set seed for determinism
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Transforms
    train_transform = build_train_transforms(args.image_size)
    test_transform = build_test_transforms(args.image_size)

    # Datasets
    train_dataset = WoundDataset(
        root_dir="data/processed",
        split="train",
        transform=train_transform,
    )
    test_dataset = WoundDataset(
        root_dir="data/processed",
        split="test",
        transform=test_transform,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # DataLoaders
    train_loader = build_weighted_train_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,  # Use 0 for sanity check to avoid multiprocessing issues
    )
    test_loader = build_test_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # Log first batch info
    first_images, first_labels = next(iter(train_loader))
    print(f"Batch shape: {first_images.shape}")
    print(f"First batch label distribution: {Counter(first_labels.tolist())}")

    # Model
    model = build_model(NUM_CLASSES)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    print(f"\nTraining for {args.max_steps} steps...")
    avg_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_steps=args.max_steps,
    )
    print(f"Epoch average loss: {avg_loss:.4f}")

    # Evaluate
    print(f"\nEvaluating on {args.max_eval_steps} test batches...")
    accuracy = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        max_steps=args.max_eval_steps,
    )
    print(f"Test accuracy: {accuracy:.4f}")

    print("\nSanity check complete!")


if __name__ == "__main__":
    main()
