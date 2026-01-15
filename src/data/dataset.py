# src/data/dataset.py
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class WoundDataset(Dataset):
    # Frozen label encoding (must match documentation)
    CLASS_TO_ID = {
        "BG": 0,
        "D": 1,
        "N": 2,
        "P": 3,
        "S": 4,
        "V": 5,
    }

    def __init__(self, root_dir: str, split: str, transform=None):
        """
        Args:
            root_dir (str): Path to processed data directory (e.g., "data/processed")
            split (str): "train" or "test"
            transform: Optional transform applied to each image
        """
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Build index: list of (image_path, label_id)
        self.samples: list[tuple[Path, int]] = []
        for class_name, label_id in self.CLASS_TO_ID.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, label_id))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    ds = WoundDataset(root_dir="data/processed", split="train")
    print("Number of training samples:", len(ds))
    print("First 5 samples:")
    for i in range(5):
        print(ds.samples[i])
