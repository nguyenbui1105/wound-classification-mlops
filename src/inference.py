"""Inference pipeline for wound classification."""

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add project root to sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model import build_efficientnet

# Constants
NUM_CLASSES = 6
CLASS_NAMES = ["BG", "D", "N", "P", "S", "V"]
IMAGE_SIZE = 224


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        device: Device to load model on

    Returns:
        Loaded model in eval mode

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint is invalid
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Build model architecture (must match training)
    model = build_efficientnet(
        model_name="efficientnet_b0",
        num_classes=NUM_CLASSES,
        pretrained=False,  # We're loading weights from checkpoint
        dropout=0.3,
        freeze_backbone=False,  # Inference mode, no freezing needed
    )

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"  Phase: {checkpoint.get('phase', 'unknown')}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Accuracy: {checkpoint.get('accuracy', 0.0):.4f}")
        print(f"  Macro-F1: {checkpoint.get('macro_f1', 0.0):.4f}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    model = model.to(device)

    # CPU optimization: convert to channels_last memory format
    if device.type == "cpu":
        model = model.to(memory_format=torch.channels_last)

    model.eval()
    return model


def preprocess_image(image_path: Path) -> torch.Tensor:
    """Preprocess image for inference.

    Args:
        image_path: Path to input image

    Returns:
        Preprocessed image tensor of shape [1, 3, 224, 224]

    Raises:
        FileNotFoundError: If image doesn't exist
        RuntimeError: If image can't be loaded
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")

    # Preprocessing: resize, center crop, normalize (same as test transforms)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_tensor = transform(image)
    # Add batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def predict(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> dict:
    """Run inference and return predictions.

    Args:
        model: Loaded model in eval mode
        image_tensor: Preprocessed image tensor [1, 3, 224, 224]
        device: Device to run inference on

    Returns:
        Dict with top-1 class, probability, and all class probabilities
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)

    # Get top-1 prediction
    top1_prob, top1_idx = torch.max(probs, dim=1)
    top1_class = CLASS_NAMES[top1_idx.item()]
    top1_prob_value = top1_prob.item()

    # Get all class probabilities
    all_probs = probs[0].cpu().tolist()
    all_probs_dict = {class_name: prob for class_name, prob in zip(CLASS_NAMES, all_probs)}

    return {
        "top1_class": top1_class,
        "top1_prob": top1_prob_value,
        "all_probs": all_probs_dict,
    }


def save_prediction(
    prediction: dict,
    image_path: Path,
    output_dir: Path,
) -> Path:
    """Save prediction to JSON file.

    Args:
        prediction: Prediction dict from predict()
        image_path: Original image path
        output_dir: Directory to save prediction.json

    Returns:
        Path to saved prediction.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output dict
    output = {
        "image": str(image_path),
        "top1_class": prediction["top1_class"],
        "top1_prob": round(prediction["top1_prob"], 4),
        "all_probs": {k: round(v, 4) for k, v in prediction["all_probs"].items()},
    }

    # Save to JSON
    output_path = output_dir / "prediction.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path


def print_prediction(prediction: dict):
    """Print human-readable prediction to console.

    Args:
        prediction: Prediction dict from predict()
    """
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Top-1 Class: {prediction['top1_class']}")
    print(f"Top-1 Probability: {prediction['top1_prob']:.4f}")
    print("\nAll Class Probabilities:")

    # Sort by probability (descending)
    sorted_probs = sorted(prediction["all_probs"].items(), key=lambda x: x[1], reverse=True)

    for class_name, prob in sorted_probs:
        bar_length = int(prob * 50)  # Scale to 50 chars max
        bar = "█" * bar_length
        print(f"  {class_name}: {prob:.4f} {bar}")

    print("=" * 60)


def scan_image_directory(image_dir: Path, max_images: int = None) -> list[Path]:
    """Scan directory for image files.

    Args:
        image_dir: Directory to scan
        max_images: Optional limit on number of images to return

    Returns:
        List of image file paths

    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no images found
    """
    if not image_dir.exists():
        error_msg = "Error: The image directory does not exist.\n"
        error_msg += f"  Attempted path: {image_dir.absolute()}\n"
        error_msg += f"  Current working directory: {Path.cwd()}\n\n"
        error_msg += "Tip: Create the directory or use --init-sample-dir:\n"
        error_msg += f"  mkdir -p {image_dir}\n"
        error_msg += "  python src/inference.py --init-sample-dir"
        raise FileNotFoundError(error_msg)
    if not image_dir.is_dir():
        raise ValueError(f"Not a directory: {image_dir}")

    # Scan for common image extensions
    extensions = [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"]
    image_files = []

    for ext in extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))

    # Sort for consistent ordering
    image_files = sorted(set(image_files))

    if len(image_files) == 0:
        error_msg = f"No images found in {image_dir}.\n"
        error_msg += "Supported formats: .jpg .jpeg .png .webp"
        raise ValueError(error_msg)

    # Apply max_images limit if specified
    if max_images is not None and max_images > 0:
        image_files = image_files[:max_images]
        print(f"Limited to first {len(image_files)} images (--max-images={max_images})")

    return image_files


def preprocess_image_safe(image_path: Path) -> tuple[Path, torch.Tensor, Exception]:
    """Safely preprocess an image, returning tensor or exception.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (image_path, tensor_or_None, exception_or_None)
    """
    try:
        tensor = preprocess_image(image_path)
        return (image_path, tensor, None)
    except Exception as e:
        return (image_path, None, e)


def run_batch_inference(
    model: torch.nn.Module,
    image_files: list[Path],
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
) -> tuple[list[dict], list[tuple[Path, Exception]]]:
    """Run optimized batched inference on images with parallel preprocessing.

    Args:
        model: Loaded model in eval mode
        image_files: List of image file paths
        device: Device to run inference on
        batch_size: Batch size for model forward pass
        num_workers: Number of thread workers for parallel preprocessing

    Returns:
        Tuple of (successful_predictions, failed_images)
        - successful_predictions: List of dicts with {image_path, top1_class, top1_prob, all_probs}
        - failed_images: List of (image_path, exception) tuples
    """
    predictions = []
    failures = []

    print(
        f"\nProcessing {len(image_files)} images with batch_size={batch_size}, workers={num_workers}..."
    )

    # Parallel preprocessing with ThreadPoolExecutor
    preprocessed_items = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(preprocess_image_safe, img_path): img_path for img_path in image_files
        }

        completed = 0
        for future in as_completed(futures):
            img_path, tensor, error = future.result()
            if error is None:
                preprocessed_items.append((img_path, tensor))
            else:
                failures.append((img_path, error))
                print(f"  Failed (preprocess): {img_path.name} - {error}")

            completed += 1
            if completed % 50 == 0 or completed == len(image_files):
                print(f"  Preprocessed {completed}/{len(image_files)} images")

    # Sort by original order (image_files order)
    path_to_idx = {path: idx for idx, path in enumerate(image_files)}
    preprocessed_items.sort(key=lambda x: path_to_idx[x[0]])

    if len(preprocessed_items) == 0:
        print("No images successfully preprocessed.")
        return predictions, failures

    # Batched inference
    print("Running batched inference...")
    num_batches = (len(preprocessed_items) + batch_size - 1) // batch_size

    with torch.inference_mode():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(preprocessed_items))
            batch_items = preprocessed_items[start_idx:end_idx]

            # Stack tensors: List of [1, C, H, W] -> [B, C, H, W]
            batch_paths = [item[0] for item in batch_items]
            batch_tensors = [item[1] for item in batch_items]
            batch_tensor = torch.cat(batch_tensors, dim=0).to(device)

            # Convert to channels_last for CPU optimization
            if device.type == "cpu":
                batch_tensor = batch_tensor.to(memory_format=torch.channels_last)

            # Forward pass
            logits = model(batch_tensor)
            probs = F.softmax(logits, dim=1)

            # Extract predictions for each image in batch
            for i, img_path in enumerate(batch_paths):
                img_probs = probs[i].cpu()
                top1_prob, top1_idx = torch.max(img_probs, dim=0)
                top1_class = CLASS_NAMES[top1_idx.item()]
                top1_prob_value = top1_prob.item()

                all_probs_dict = {
                    class_name: img_probs[j].item() for j, class_name in enumerate(CLASS_NAMES)
                }

                result = {
                    "image": str(img_path),
                    "top1_class": top1_class,
                    "top1_prob": top1_prob_value,
                    "all_probs": all_probs_dict,
                }
                predictions.append(result)

            # Progress indicator
            processed = end_idx
            if (batch_idx + 1) % 10 == 0 or processed == len(preprocessed_items):
                print(f"  Processed {processed}/{len(preprocessed_items)} images")

    return predictions, failures


def save_batch_report(
    predictions: list[dict],
    report_dir: Path,
) -> dict[str, Path]:
    """Save batch inference report files.

    Args:
        predictions: List of prediction dicts
        report_dir: Directory to save report files

    Returns:
        Dict with paths to created files: {jsonl, csv, per_image_dir}
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save predictions.jsonl (one JSON per line)
    jsonl_path = report_dir / "predictions.jsonl"
    with open(jsonl_path, "w") as f:
        for pred in predictions:
            # Round probabilities for cleaner output
            pred_rounded = {
                "image": pred["image"],
                "top1_class": pred["top1_class"],
                "top1_prob": round(pred["top1_prob"], 4),
                "all_probs": {k: round(v, 4) for k, v in pred["all_probs"].items()},
            }
            f.write(json.dumps(pred_rounded) + "\n")

    # 2. Save summary.csv
    csv_path = report_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        # CSV columns: image, top1_class, top1_prob, prob_BG, prob_D, prob_N, prob_P, prob_S, prob_V
        fieldnames = ["image", "top1_class", "top1_prob"] + [f"prob_{c}" for c in CLASS_NAMES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for pred in predictions:
            row = {
                "image": pred["image"],
                "top1_class": pred["top1_class"],
                "top1_prob": round(pred["top1_prob"], 4),
            }
            # Add per-class probabilities
            for class_name in CLASS_NAMES:
                row[f"prob_{class_name}"] = round(pred["all_probs"][class_name], 4)

            writer.writerow(row)

    # 3. Save per-image predictions
    per_image_dir = report_dir / "per_image"
    per_image_dir.mkdir(parents=True, exist_ok=True)

    for pred in predictions:
        image_path = Path(pred["image"])
        output_file = per_image_dir / f"{image_path.stem}.json"

        pred_rounded = {
            "image": pred["image"],
            "top1_class": pred["top1_class"],
            "top1_prob": round(pred["top1_prob"], 4),
            "all_probs": {k: round(v, 4) for k, v in pred["all_probs"].items()},
        }

        with open(output_file, "w") as f:
            json.dump(pred_rounded, f, indent=2)

    return {
        "jsonl": jsonl_path,
        "csv": csv_path,
        "per_image_dir": per_image_dir,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Wound classification inference pipeline")
    parser.add_argument(
        "--image", type=str, default=None, help="Path to input image (for single-image inference)"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Path to directory containing images (for batch inference)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/checkpoints/best_efficientnet.pt",
        help="Path to model checkpoint (default: artifacts/checkpoints/best_efficientnet.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for prediction.json (single-image mode, default: artifacts/inference_outputs/<timestamp>)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="artifacts/inference_reports",
        help="Output directory for batch inference reports (default: artifacts/inference_reports)",
    )
    parser.add_argument(
        "--report-id",
        type=str,
        default=None,
        help="Report ID for batch inference (default: timestamp YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--create-sample-dir",
        action="store_true",
        help="Create the image directory if it doesn't exist (with README.txt)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for batch inference (default: 16)"
    )
    parser.add_argument(
        "--num-preprocess-workers",
        type=int,
        default=4,
        help="Number of thread workers for parallel image preprocessing (default: 4)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (for quick sanity runs)",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Set torch CPU thread count (default: None, uses PyTorch defaults)",
    )
    parser.add_argument(
        "--init-sample-dir",
        action="store_true",
        help="Initialize assets/sample_images directory with instructions and exit",
    )

    args = parser.parse_args()

    # Handle --init-sample-dir early exit
    if args.init_sample_dir:
        sample_dir = Path("assets/sample_images")
        sample_dir.mkdir(parents=True, exist_ok=True)
        readme_path = sample_dir / "README.txt"
        with open(readme_path, "w") as f:
            f.write("Sample Images Directory\n")
            f.write("=" * 40 + "\n\n")
            f.write("Put your wound classification test images here.\n")
            f.write("Supported formats: .jpg, .jpeg, .png, .webp\n\n")
            f.write("Run batch inference:\n")
            f.write("  python src/inference.py --image-dir assets/sample_images\n")
        print(f"Initialized directory: {sample_dir.absolute()}")
        print("Added README.txt with instructions")
        sys.exit(0)

    # Validation: either --image or --image-dir must be provided
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir must be provided")
    if args.image and args.image_dir:
        parser.error("Cannot use both --image and --image-dir simultaneously")

    return args


def main():
    """Main inference pipeline."""
    args = parse_args()

    # Set CPU thread count if specified
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
        print(f"Set torch CPU threads: {args.cpu_threads}")

    # Setup paths
    checkpoint_path = Path(args.checkpoint)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        # Load model
        print("\nLoading model...")
        model = load_model(checkpoint_path, device)

        # Single-image mode
        if args.image:
            image_path = Path(args.image)

            # Output directory with timestamp
            if args.output_dir:
                output_dir = Path(args.output_dir)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = ROOT / "artifacts" / "inference_outputs" / timestamp

            # Preprocess image
            print(f"\nPreprocessing image: {image_path}")
            image_tensor = preprocess_image(image_path)

            # Run inference
            print("Running inference...")
            prediction = predict(model, image_tensor, device)

            # Save prediction
            output_path = save_prediction(prediction, image_path, output_dir)
            print(f"\nPrediction saved to: {output_path}")

            # Print results
            print_prediction(prediction)

        # Batch mode
        elif args.image_dir:
            image_dir = Path(args.image_dir)

            # Create sample directory if flag is set and directory doesn't exist
            if args.create_sample_dir and not image_dir.exists():
                image_dir.mkdir(parents=True, exist_ok=True)
                readme_path = image_dir / "README.txt"
                with open(readme_path, "w") as f:
                    f.write("Put your test images in this folder.\n")
                print(f"Created directory: {image_dir}")
                print("Added README.txt with instructions")

            # Setup report directory
            if args.report_id:
                report_id = args.report_id
            else:
                report_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            report_dir = Path(args.report_dir) / report_id

            # Scan for images
            print(f"\nScanning directory: {image_dir}")
            image_files = scan_image_directory(image_dir, max_images=args.max_images)
            print(f"Found {len(image_files)} images")

            # Run optimized batch inference
            predictions, failures = run_batch_inference(
                model,
                image_files,
                device,
                batch_size=args.batch_size,
                num_workers=args.num_preprocess_workers,
            )

            # Save report
            print(f"\nSaving report to: {report_dir}")
            report_files = save_batch_report(predictions, report_dir)

            # Print summary
            print("\n" + "=" * 60)
            print("BATCH INFERENCE SUMMARY")
            print("=" * 60)
            print(f"Total images: {len(image_files)}")
            print(f"Successful: {len(predictions)}")
            print(f"Failed: {len(failures)}")
            print("\nReport files:")
            print(f"  - JSONL: {report_files['jsonl']}")
            print(f"  - CSV: {report_files['csv']}")
            print(f"  - Per-image JSONs: {report_files['per_image_dir']}")

            if failures:
                print("\nFailed images:")
                for img_path, error in failures:
                    print(f"  - {img_path.name}: {error}")

            print("=" * 60)

    except FileNotFoundError as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
