"""2-Phase EfficientNet training for wound classification.

Phase A: Freeze backbone, train head only (3-5 epochs)
Phase B: Unfreeze last 2 blocks, fine-tune with differential LR (10-20 epochs)
"""

import sys
from pathlib import Path

# Add project root to sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.dataloader import build_test_dataloader, build_weighted_train_dataloader
from src.data.dataset import WoundDataset
from src.data.transforms import (
    build_test_transforms,
    build_train_transforms,
    build_train_transforms_light,
)
from src.model import build_efficientnet

# Constants
NUM_CLASSES = 6
CLASS_NAMES = ["BG", "D", "N", "P", "S", "V"]


def compute_class_weights(dataset, num_classes: int) -> torch.Tensor:
    """Compute inverse frequency class weights from dataset labels.

    Args:
        dataset: Dataset object (may have .labels, .targets, .samples, or none).
        num_classes: Number of classes in the dataset.

    Returns:
        Tensor of shape [num_classes] with normalized inverse frequency weights.
    """

    # Extract labels using priority: labels > targets > samples > iterate
    if hasattr(dataset, "labels"):
        labels = dataset.labels
    elif hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "samples"):
        labels = [y for (_, y) in dataset.samples]
    else:
        # Fallback: iterate through dataset
        labels = []
        for i in range(len(dataset)):
            _, y = dataset[i]
            # Handle tensor or int
            if isinstance(y, torch.Tensor):
                labels.append(y.item())
            else:
                labels.append(int(y))

    label_counts = Counter(labels)

    # Inverse frequency: weight = total / (num_classes * count)
    total = len(dataset)
    weights = []
    for cls in range(num_classes):
        count = label_counts.get(cls, 1)  # Avoid division by zero
        weights.append(total / (num_classes * count))

    # Normalize so weights sum to num_classes (keeps loss scale similar)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    weights_tensor = weights_tensor / weights_tensor.sum() * num_classes

    return weights_tensor


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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="2-Phase EfficientNet training for wound classification"
    )
    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b0",
        choices=["tf_efficientnetv2_s", "efficientnet_b0"],
        help="EfficientNet variant (b0 is lighter/faster for CPU)",
    )
    parser.add_argument("--dropout", type=float, default=0.3)

    # Phase A: Head-only training
    parser.add_argument("--phase-a-epochs", type=int, default=5)
    parser.add_argument("--phase-a-lr", type=float, default=3e-3, help="Head LR for Phase A")

    # Phase B: Fine-tuning
    parser.add_argument("--phase-b-epochs", type=int, default=15)
    parser.add_argument("--phase-b-head-lr", type=float, default=1e-3, help="Head LR for Phase B")
    parser.add_argument(
        "--phase-b-backbone-lr", type=float, default=3e-4, help="Backbone LR for Phase B"
    )
    parser.add_argument(
        "--unfreeze-blocks", type=int, default=2, help="Number of blocks to unfreeze"
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=16, help="Smaller for CPU memory")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0, help="0 for Windows CPU")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)

    # Class imbalance strategy (choose one, not both)
    parser.add_argument(
        "--use-sampler", type=bool, default=True, help="Use WeightedRandomSampler for class balance"
    )
    parser.add_argument(
        "--use-class-weights",
        type=bool,
        default=False,
        help="Use class-weighted CE loss (disables sampler)",
    )

    # Early stopping
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience (epochs)")

    # Class P boost (index 3)
    parser.add_argument(
        "--class-p-boost", type=float, default=1.5, help="Loss weight multiplier for class P"
    )

    # Fast dev run
    parser.add_argument(
        "--fast-dev-run", action="store_true", help="Quick test: 1 epoch, 10 batches"
    )

    # Debug/limits
    parser.add_argument("--max-train-steps", type=int, default=-1, help="-1 means no limit")
    parser.add_argument("--max-eval-steps", type=int, default=-1, help="-1 means no limit")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep CPU-friendly: don't force full determinism (slower)
    torch.use_deterministic_algorithms(False)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    phase: str,
    max_steps: int = -1,
    log_interval: int = 20,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_steps = 0

    for step, (images, labels) in enumerate(loader):
        if max_steps > 0 and step >= max_steps:
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
            print(f"  [{phase}] Epoch {epoch} | Step {step + 1} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    return avg_loss


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    max_steps: int = -1,
) -> dict:
    """Evaluate model on test set with comprehensive metrics.

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
    phase: str,
    accuracy: float,
    macro_f1: float,
    path: Path,
) -> None:
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "phase": phase,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        },
        path,
    )


def save_metrics_artifacts(
    metrics: dict,
    args: argparse.Namespace,
    ckpt_path: Path,
    phase: str,
    epoch: int,
    run_id: str,
    class_names: list,
) -> None:
    """Save evaluation metrics artifacts for auditability (healthcare ML best practice).

    Args:
        metrics: Dict from evaluate() containing accuracy, macro_f1, confusion_matrix, etc.
        args: Parsed command line arguments with all hyperparameters.
        ckpt_path: Path to the checkpoint file that was just saved.
        phase: Training phase ("A" or "B").
        epoch: Current epoch number.
        run_id: Unique run identifier (timestamp YYYYMMDD_HHMMSS).
        class_names: List of class names for confusion matrix CSV.
    """
    # Create output directory for this run's metrics
    out_dir = ROOT / "artifacts" / "metrics" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save metrics.json with all relevant metadata
    metrics_dict = {
        # Training state
        "phase": phase,
        "epoch": epoch,
        # Performance metrics
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "per_class_recall": metrics["per_class_recall"],
        "per_class_precision": metrics["per_class_precision"],
        "per_class_f1": metrics["per_class_f1"],
        # Model config
        "model": args.model,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        # Training strategy
        "use_class_weights": args.use_class_weights,
        "use_sampler": args.use_sampler,
        "class_p_boost": args.class_p_boost,
        "label_smoothing": args.label_smoothing,
        "patience": args.patience,
        # Phase config
        "unfreeze_blocks": args.unfreeze_blocks,
        "phase_a_epochs": args.phase_a_epochs,
        "phase_b_epochs": args.phase_b_epochs,
        # Optimizer config
        "phase_a_lr": args.phase_a_lr,
        "phase_b_head_lr": args.phase_b_head_lr,
        "phase_b_backbone_lr": args.phase_b_backbone_lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        # Checkpoint reference
        "checkpoint_path": str(ckpt_path),
    }

    metrics_json_path = out_dir / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # 2. Save confusion_matrix.csv
    conf_matrix = metrics["confusion_matrix"]  # torch.Tensor of shape [num_classes, num_classes]
    conf_csv_path = out_dir / "confusion_matrix.csv"

    with open(conf_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header row: empty cell + class names
        writer.writerow([""] + class_names)
        # Data rows: class name + values
        for i, row_name in enumerate(class_names):
            row_values = conf_matrix[i].tolist()
            writer.writerow([row_name] + row_values)

    print(f"  Metrics artifacts saved to: {out_dir}")


def print_epoch_stats(
    epoch: int,
    total_epochs: int,
    phase: str,
    train_loss: float,
    metrics: dict,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Print epoch statistics with full metrics."""
    accuracy = metrics["accuracy"]
    macro_f1 = metrics["macro_f1"]
    balanced_acc = metrics["balanced_accuracy"]
    per_class_recall = metrics["per_class_recall"]
    per_class_f1 = metrics["per_class_f1"]

    print(f"\n[{phase}] Epoch {epoch}/{total_epochs}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print("  Per-class Recall (F1):")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        print(f"    {cls_name}: {per_class_recall[cls_id]:.4f} (F1: {per_class_f1[cls_id]:.4f})")

    # Print LRs from optimizer param groups
    if len(optimizer.param_groups) == 1:
        # Phase A: single param group (head only)
        print(f"  LR (head): {optimizer.param_groups[0]['lr']:.2e}")
    else:
        # Phase B: two param groups (backbone, head)
        print(f"  LR (backbone): {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  LR (head): {optimizer.param_groups[1]['lr']:.2e}")
    print("-" * 60)


def main():
    args = parse_args()

    # Create unique run ID for metrics artifacts
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Fast dev run overrides
    if args.fast_dev_run:
        args.phase_a_epochs = 1
        args.phase_b_epochs = 0  # Skip Phase B
        args.max_train_steps = 10
        args.max_eval_steps = 10
        print("*** FAST DEV RUN: 1 epoch, 10 batches ***")

    # Reproducibility
    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    # Create generator for DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Device (CPU for this setup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Transforms
    train_transform_light = build_train_transforms_light(args.image_size)
    train_transform_full = build_train_transforms(args.image_size)
    test_transform = build_test_transforms(args.image_size)

    # Datasets
    train_dataset_light = WoundDataset(
        root_dir="data/processed",
        split="train",
        transform=train_transform_light,
    )
    train_dataset_full = WoundDataset(
        root_dir="data/processed",
        split="train",
        transform=train_transform_full,
    )
    test_dataset = WoundDataset(
        root_dir="data/processed",
        split="test",
        transform=test_transform,
    )

    print(f"Train dataset size: {len(train_dataset_light)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # =========================================================================
    # Class imbalance strategy: sampler vs class-weighted loss
    # =========================================================================
    use_class_weights = args.use_class_weights

    # CPU-friendly settings
    pin_memory = device.type == "cuda"  # False for CPU to avoid warnings

    if use_class_weights:
        # Compute class weights from training data (inverse frequency)
        class_weights = compute_class_weights(train_dataset_light, NUM_CLASSES)
        print("Using class-weighted loss (sampler disabled)")
        print(f"  Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

        # DataLoaders WITHOUT sampler (shuffle=True)
        train_loader_light = DataLoader(
            train_dataset_light,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            generator=g,
        )
        train_loader_full = DataLoader(
            train_dataset_full,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            generator=g,
        )
    else:
        # Use WeightedRandomSampler (default)
        print("Using WeightedRandomSampler for class balance")
        class_weights = None
        train_loader_light = build_weighted_train_dataloader(
            train_dataset_light,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            generator=g,
        )
        train_loader_full = build_weighted_train_dataloader(
            train_dataset_full,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            generator=g,
        )

    test_loader = build_test_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        generator=g,
    )

    # Model: EfficientNet with frozen backbone
    print(f"\nBuilding {args.model} with frozen backbone...")
    model = build_efficientnet(
        model_name=args.model,
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout=args.dropout,
        freeze_backbone=True,
    )
    model = model.to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable (Phase A): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss: weighted or unweighted based on strategy
    # Always apply class P boost (index 3) to focus learning on underperforming class
    loss_weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
    loss_weights[3] = args.class_p_boost  # Class P boost
    if use_class_weights:
        loss_weights = loss_weights * class_weights
    print(
        f"Loss weights (with P boost={args.class_p_boost}): {[f'{w:.3f}' for w in loss_weights.tolist()]}"
    )
    print(f"Label smoothing: {args.label_smoothing}")

    criterion = nn.CrossEntropyLoss(
        weight=loss_weights,
        label_smoothing=args.label_smoothing,
    )

    # Checkpoint paths
    checkpoint_dir = ROOT / "artifacts" / "checkpoints"
    best_checkpoint_path = checkpoint_dir / "best_efficientnet.pt"
    last_checkpoint_path = checkpoint_dir / "last_efficientnet.pt"

    best_macro_f1 = 0.0
    best_accuracy = 0.0
    epochs_without_improvement = 0

    # =========================================================================
    # PHASE A: Head-only training
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE A: Head-only training (backbone frozen)")
    print("=" * 60)

    # Optimizer for Phase A: only head params
    optimizer_a = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.phase_a_lr,
        weight_decay=args.weight_decay,
    )
    # CosineAnnealingLR decays from initial LR to eta_min over T_max epochs
    # eta_min=1e-4 ensures LR doesn't drop too low for short Phase A
    scheduler_a = CosineAnnealingLR(optimizer_a, T_max=args.phase_a_epochs, eta_min=1e-4)

    print(f"Phase A optimizer: lr={optimizer_a.param_groups[0]['lr']:.2e}")

    for epoch in range(1, args.phase_a_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader_light,
            criterion=criterion,
            optimizer=optimizer_a,
            device=device,
            epoch=epoch,
            phase="A",
            max_steps=args.max_train_steps,
        )

        # Evaluate with full metrics
        metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            max_steps=args.max_eval_steps,
        )

        accuracy = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]

        # Print stats with full metrics
        print_epoch_stats(epoch, args.phase_a_epochs, "A", train_loss, metrics, optimizer_a)

        # Step scheduler AFTER logging (prepares LR for next epoch)
        scheduler_a.step()

        # Determine if this is the best model (primary: macro-F1, tie-breaker: accuracy)
        is_best = (macro_f1 > best_macro_f1) or (
            macro_f1 == best_macro_f1 and accuracy > best_accuracy
        )

        # Update best metrics tracking (always, even in fast-dev-run)
        if is_best:
            best_macro_f1 = macro_f1
            best_accuracy = accuracy
            epochs_without_improvement = 0
            print(f"  New best model! Macro-F1: {best_macro_f1:.4f}, Accuracy: {best_accuracy:.4f}")
        else:
            epochs_without_improvement += 1

        # Skip checkpoint/artifact saving in fast dev run
        if args.fast_dev_run:
            if epochs_without_improvement >= args.patience:
                print(f"  Early stopping: no improvement for {args.patience} epochs")
                break
            continue

        # Save last checkpoint
        save_checkpoint(
            model, optimizer_a, scheduler_a, epoch, "A", accuracy, macro_f1, last_checkpoint_path
        )

        # Save best checkpoint and metrics artifacts
        if is_best:
            save_checkpoint(
                model,
                optimizer_a,
                scheduler_a,
                epoch,
                "A",
                accuracy,
                macro_f1,
                best_checkpoint_path,
            )
            save_metrics_artifacts(
                metrics, args, best_checkpoint_path, "A", epoch, run_id, CLASS_NAMES
            )
        else:
            if epochs_without_improvement >= args.patience:
                print(f"  Early stopping: no improvement for {args.patience} epochs")
                break

    # =========================================================================
    # PHASE B: Fine-tuning with partial unfreeze
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"PHASE B: Fine-tuning (unfreezing last {args.unfreeze_blocks} blocks)")
    print("=" * 60)

    # Unfreeze last N blocks
    model.unfreeze_backbone_partial(num_blocks=args.unfreeze_blocks)
    print(f"Trainable (Phase B): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer for Phase B: differential LR
    param_groups = model.get_param_groups(
        backbone_lr=args.phase_b_backbone_lr,
        head_lr=args.phase_b_head_lr,
    )
    optimizer_b = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Verify param groups
    print(f"Optimizer: AdamW with {len(optimizer_b.param_groups)} param groups")
    for i, pg in enumerate(optimizer_b.param_groups):
        group_name = "backbone" if i == 0 else "head"
        print(f"  Group {i} ({group_name}): lr={pg['lr']:.2e}, params={len(pg['params'])}")

    # CosineAnnealingLR with eta_min that preserves meaningful learning
    # eta_min=5e-6 for backbone, but scheduler applies same ratio to all groups
    scheduler_b = CosineAnnealingLR(optimizer_b, T_max=args.phase_b_epochs, eta_min=1e-5)

    # Reset early stopping counter for Phase B
    epochs_without_improvement = 0

    for epoch in range(1, args.phase_b_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader_full,
            criterion=criterion,
            optimizer=optimizer_b,
            device=device,
            epoch=epoch,
            phase="B",
            max_steps=args.max_train_steps,
        )

        # Evaluate with full metrics
        metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            max_steps=args.max_eval_steps,
        )

        accuracy = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]

        # Print stats with full metrics
        print_epoch_stats(epoch, args.phase_b_epochs, "B", train_loss, metrics, optimizer_b)

        # Step scheduler AFTER logging (prepares LR for next epoch)
        scheduler_b.step()

        # Save last checkpoint
        save_checkpoint(
            model, optimizer_b, scheduler_b, epoch, "B", accuracy, macro_f1, last_checkpoint_path
        )

        # Save best checkpoint (primary: macro-F1, tie-breaker: accuracy)
        is_best = (macro_f1 > best_macro_f1) or (
            macro_f1 == best_macro_f1 and accuracy > best_accuracy
        )
        if is_best:
            best_macro_f1 = macro_f1
            best_accuracy = accuracy
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                optimizer_b,
                scheduler_b,
                epoch,
                "B",
                accuracy,
                macro_f1,
                best_checkpoint_path,
            )
            save_metrics_artifacts(
                metrics, args, best_checkpoint_path, "B", epoch, run_id, CLASS_NAMES
            )
            print(f"  New best model! Macro-F1: {best_macro_f1:.4f}, Accuracy: {best_accuracy:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  Early stopping: no improvement for {args.patience} epochs")
                break

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best Macro-F1: {best_macro_f1:.4f}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
