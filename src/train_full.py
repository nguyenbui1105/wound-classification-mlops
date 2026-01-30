"""Full training script for wound classification."""

import sys
from pathlib import Path

# Add project root to sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
from collections import Counter

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.data.dataloader import build_test_dataloader, build_weighted_train_dataloader
from src.data.dataset import WoundDataset
from src.data.transforms import build_test_transforms, build_train_transforms

# Constants
NUM_CLASSES = 6
CLASS_NAMES = ["BG", "D", "N", "P", "S", "V"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Full training for wound classification")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4, help="Classifier LR")
    parser.add_argument(
        "--backbone-lr", type=float, default=1e-5, help="Backbone LR (for fine-tuning)"
    )
    parser.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=0,
        help="Epoch to unfreeze backbone (0=start unfrozen)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=-1, help="-1 means no limit")
    parser.add_argument("--max-eval-steps", type=int, default=-1, help="-1 means no limit")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """Build ResNet18 model with pretrained weights and custom classifier.

    Uses ImageNet pretrained weights for transfer learning, which provides
    a strong initialization for medical image classification.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def get_param_groups(model: nn.Module, backbone_lr: float, classifier_lr: float) -> list:
    """Create parameter groups with differential learning rates.

    Args:
        model: ResNet18 model with .fc as classifier.
        backbone_lr: Learning rate for backbone (feature extractor).
        classifier_lr: Learning rate for classifier head.

    Returns:
        List of param group dicts for optimizer.
    """
    # Backbone: all params except fc
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    # Classifier: only fc params
    classifier_params = model.fc.parameters()

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": classifier_params, "lr": classifier_lr},
    ]


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone parameters (train only classifier)."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all backbone parameters for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    max_steps: int = -1,
    log_interval: int = 50,
) -> float:
    """Train for one epoch with mixed precision.

    Args:
        model: The neural network model.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: GradScaler for AMP.
        device: Device to run on.
        epoch: Current epoch number (for logging).
        max_steps: Maximum steps per epoch (-1 for no limit).
        log_interval: Print loss every N steps.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_steps = 0

    for step, (images, labels) in enumerate(loader):
        if max_steps > 0 and step >= max_steps:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_steps += 1

        if (step + 1) % log_interval == 0:
            print(f"  Epoch {epoch} | Step {step + 1} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    return avg_loss


def compute_metrics_from_confusion_matrix(
    conf_matrix: torch.Tensor,
    eps: float = 1e-8,
) -> dict:
    """Compute metrics from confusion matrix (pure PyTorch, no sklearn).

    Args:
        conf_matrix: NxN confusion matrix where conf_matrix[i,j] = count of
                     samples with true label i predicted as j.
        eps: Small value to avoid division by zero.

    Returns:
        Dict with per-class and aggregate metrics.
    """
    num_classes = conf_matrix.size(0)

    # Per-class metrics
    tp = conf_matrix.diag()  # True positives: diagonal
    fp = conf_matrix.sum(dim=0) - tp  # False positives: column sum - diagonal
    fn = conf_matrix.sum(dim=1) - tp  # False negatives: row sum - diagonal

    # Recall = TP / (TP + FN) = TP / row_sum
    recall = tp / (tp + fn + eps)

    # Precision = TP / (TP + FP) = TP / col_sum
    precision = tp / (tp + fp + eps)

    # F1 = 2 * P * R / (P + R)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # Macro averages (unweighted mean across classes)
    macro_f1 = f1.mean().item()
    balanced_accuracy = recall.mean().item()  # = macro recall

    return {
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "f1": f1.tolist(),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
    }


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    max_steps: int = -1,
) -> dict:
    """Evaluate model on test set with comprehensive metrics.

    Args:
        model: The neural network model.
        loader: Test data loader.
        device: Device to run on.
        max_steps: Maximum steps for evaluation (-1 for no limit).

    Returns:
        Dict with accuracy, macro_f1, balanced_accuracy, per-class recall/precision/f1.
    """
    model.eval()

    # Accumulate all predictions and targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            if max_steps > 0 and step >= max_steps:
                break

            images = images.to(device)
            labels = labels.to(device)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_preds.append(predicted.cpu())
            all_targets.append(labels.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Overall accuracy
    total = all_targets.size(0)
    correct = (all_preds == all_targets).sum().item()
    accuracy = correct / total if total > 0 else 0.0

    # Build confusion matrix (pure PyTorch)
    conf_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for t, p in zip(all_targets, all_preds):
        conf_matrix[t.item(), p.item()] += 1

    # Compute metrics from confusion matrix
    metrics = compute_metrics_from_confusion_matrix(conf_matrix.float())

    return {
        "accuracy": accuracy,
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "per_class_recall": metrics["recall"],
        "per_class_precision": metrics["precision"],
        "per_class_f1": metrics["f1"],
        "confusion_matrix": conf_matrix,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    accuracy: float,
    macro_f1: float,
    path: Path,
) -> None:
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        },
        path,
    )


def main():
    args = parse_args()

    # Reproducibility
    set_seed(args.seed)

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
        num_workers=args.num_workers,
    )
    test_loader = build_test_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Log first batch info
    first_images, first_labels = next(iter(train_loader))
    print(f"Batch shape: {first_images.shape}")
    print(f"First batch label distribution: {Counter(first_labels.tolist())}")

    # Model
    model = build_model(NUM_CLASSES, device)
    print(f"Model: ResNet18 (pretrained) with {NUM_CLASSES} output classes")

    # Optionally freeze backbone for initial epochs
    if args.unfreeze_epoch > 0:
        freeze_backbone(model)
        print(f"Backbone frozen until epoch {args.unfreeze_epoch}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()

    # AdamW with differential learning rates:
    # - Backbone: very low LR to preserve pretrained features
    # - Classifier: higher LR for faster adaptation
    param_groups = get_param_groups(model, args.backbone_lr, args.lr)
    optimizer = torch.optim.AdamW(param_groups)

    # Verify param groups are set correctly
    print(f"Optimizer: AdamW with {len(optimizer.param_groups)} param groups")
    print(
        f"  Group 0 (backbone): lr={optimizer.param_groups[0]['lr']:.2e}, params={len(optimizer.param_groups[0]['params'])}"
    )
    print(
        f"  Group 1 (classifier): lr={optimizer.param_groups[1]['lr']:.2e}, params={len(list(optimizer.param_groups[1]['params']))}"
    )

    # CosineAnnealingLR: smooth decay to near-zero LR, works well for fine-tuning
    # Chosen over OneCycleLR because:
    # 1. Simpler to configure (no warmup/max_lr tuning needed)
    # 2. More stable for medical imaging where we want conservative updates
    # 3. Works reliably with AdamW for transfer learning scenarios
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mixed precision scaler
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Checkpoint paths
    checkpoint_dir = ROOT / "artifacts" / "checkpoints"
    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"

    best_macro_f1 = 0.0
    best_accuracy = 0.0

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone at specified epoch
        if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            unfreeze_backbone(model)
            print(f">>> Backbone unfrozen at epoch {epoch} for fine-tuning")

        # Train
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            max_steps=args.max_train_steps,
        )

        # Step scheduler after each epoch
        scheduler.step()

        # Evaluate (returns dict with comprehensive metrics)
        metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            max_steps=args.max_eval_steps,
        )

        accuracy = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]
        balanced_acc = metrics["balanced_accuracy"]
        per_class_recall = metrics["per_class_recall"]
        per_class_f1 = metrics["per_class_f1"]

        # Log results
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Macro-F1: {macro_f1:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print("  Per-class Recall:")
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            print(
                f"    {cls_name}: {per_class_recall[cls_id]:.4f} (F1: {per_class_f1[cls_id]:.4f})"
            )
        # Log actual optimizer LRs (scheduler updates these in-place)
        backbone_lr = optimizer.param_groups[0]["lr"]
        classifier_lr = optimizer.param_groups[1]["lr"]
        print(f"  LR: backbone={backbone_lr:.2e}, classifier={classifier_lr:.2e}")
        print("-" * 60)

        # Save last checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, accuracy, macro_f1, last_checkpoint_path
        )

        # Save best checkpoint (primary: macro-F1, tie-breaker: accuracy)
        is_best = (macro_f1 > best_macro_f1) or (
            macro_f1 == best_macro_f1 and accuracy > best_accuracy
        )
        if is_best:
            best_macro_f1 = macro_f1
            best_accuracy = accuracy
            save_checkpoint(
                model, optimizer, scheduler, epoch, accuracy, macro_f1, best_checkpoint_path
            )
            print(f"  New best model! Macro-F1: {best_macro_f1:.4f}, Accuracy: {best_accuracy:.4f}")

    print("\nTraining complete!")
    print(f"Best Macro-F1: {best_macro_f1:.4f}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
