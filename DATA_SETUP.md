# 📊 Data Setup Guide

This guide explains how to organize your wound classification dataset for training and inference.

---

## 📁 Required Directory Structure

The training scripts expect data organized in the following structure:

```
data/
└── processed/
    ├── train/
    │   ├── BG/          # Background images
    │   │   ├── image_001.jpg
    │   │   ├── image_002.jpg
    │   │   └── ...
    │   ├── D/           # D-type wound images
    │   │   ├── image_001.jpg
    │   │   └── ...
    │   ├── N/           # N-type wound images
    │   ├── P/           # P-type wound images
    │   ├── S/           # S-type wound images
    │   └── V/           # V-type wound images
    └── test/
        ├── BG/
        ├── D/
        ├── N/
        ├── P/
        ├── S/
        └── V/
```

---

## 🏷️ Class Labels

| Class | Description | Example Count |
|-------|-------------|---------------|
| **BG** | Background / No wound | ~500 images |
| **D** | D-type wound | ~400 images |
| **N** | N-type wound | ~450 images |
| **P** | P-type wound | ~350 images |
| **S** | S-type wound | ~420 images |
| **V** | V-type wound | ~380 images |

**Total recommended:** ~2500 images (train + test)

---

## 🚀 Quick Setup

### Method 1: Manual Organization

```bash
# Create directory structure
mkdir -p data/processed/{train,test}/{BG,D,N,P,S,V}

# Move your images into appropriate folders
# Example:
cp /path/to/background_images/* data/processed/train/BG/
cp /path/to/d_type_wounds/* data/processed/train/D/
# ... repeat for all classes
```

### Method 2: Automated Script (Python)

Create a script `organize_data.py`:

```python
#!/usr/bin/env python3
"""Organize wound images into train/test splits."""

import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
SOURCE_DIR = Path("data/raw")  # Your source images
OUTPUT_DIR = Path("data/processed")
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Class mapping (filename prefix -> class label)
CLASS_MAPPING = {
    "bg_": "BG",
    "d_": "D",
    "n_": "N",
    "p_": "P",
    "s_": "S",
    "v_": "V",
}

def organize_images():
    """Organize images into train/test splits."""
    
    # Create output directories
    for split in ["train", "test"]:
        for cls in CLASS_MAPPING.values():
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for prefix, class_name in CLASS_MAPPING.items():
        # Find all images with this prefix
        images = list(SOURCE_DIR.glob(f"{prefix}*.jpg"))
        
        if not images:
            print(f"Warning: No images found for class {class_name}")
            continue
        
        # Split into train/test
        train_imgs, test_imgs = train_test_split(
            images, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE
        )
        
        # Copy to train directory
        for img in train_imgs:
            dst = OUTPUT_DIR / "train" / class_name / img.name
            shutil.copy(img, dst)
        
        # Copy to test directory
        for img in test_imgs:
            dst = OUTPUT_DIR / "test" / class_name / img.name
            shutil.copy(img, dst)
        
        print(f"Class {class_name}: {len(train_imgs)} train, {len(test_imgs)} test")
    
    print(f"\n✅ Dataset organized successfully!")
    print(f"Train directory: {OUTPUT_DIR / 'train'}")
    print(f"Test directory: {OUTPUT_DIR / 'test'}")

if __name__ == "__main__":
    organize_images()
```

Run the script:

```bash
python organize_data.py
```

---

## 📏 Dataset Requirements

### Image Format

- **Supported formats:** `.jpg`, `.jpeg`, `.png`, `.webp`
- **Recommended:** JPEG for smaller file size
- **Resolution:** Minimum 224x224 pixels (will be resized automatically)

### Image Quality

- ✅ Clear, well-lit images
- ✅ Wound area clearly visible
- ✅ Minimal motion blur
- ❌ Avoid heavily compressed images
- ❌ Avoid extreme angles or occlusion

### Dataset Balance

Aim for roughly balanced class distribution:

```python
# Recommended minimum per class
BG: 300+ images
D:  250+ images
N:  250+ images
P:  250+ images
S:  250+ images
V:  250+ images
```

**Note:** The training script uses `WeightedRandomSampler` to handle class imbalance automatically, but balanced datasets train faster and generalize better.

---

## ✅ Verify Dataset

After organizing your data, verify the structure:

```bash
# Check directory structure
tree -L 3 data/processed

# Count images per class
find data/processed/train -name "*.jpg" | wc -l
find data/processed/test -name "*.jpg" | wc -l

# Per-class counts
for cls in BG D N P S V; do
    echo "$cls: $(find data/processed/train/$cls -name "*.jpg" | wc -l) train, $(find data/processed/test/$cls -name "*.jpg" | wc -l) test"
done
```

Expected output:

```
BG: 400 train, 100 test
D: 320 train, 80 test
N: 360 train, 90 test
P: 280 train, 70 test
S: 336 train, 84 test
V: 304 train, 76 test
```

---

## 🧪 Sanity Check

Test that the dataset loads correctly:

```python
from src.data.dataset import WoundDataset

# Load train dataset
train_ds = WoundDataset(
    root_dir="data/processed",
    split="train",
    transform=None
)

print(f"Train dataset size: {len(train_ds)}")
print(f"First sample: {train_ds[0]}")

# Load test dataset
test_ds = WoundDataset(
    root_dir="data/processed",
    split="test",
    transform=None
)

print(f"Test dataset size: {len(test_ds)}")
```

---

## 🔄 Data Augmentation

During training, images are automatically augmented with:

- Random resized crop (scale 0.8-1.0)
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation)
- Random grayscale (5% chance)

See `src/data/transforms.py` for full augmentation pipeline.

---

## 📊 Train/Test Split

**Recommended split:** 80% train, 20% test

```python
# Example split calculation
Total images: 2500
Train: 2000 (80%)
Test: 500 (20%)
```

### Split Strategies

1. **Random Split** (simplest):
   ```python
   from sklearn.model_selection import train_test_split
   train, test = train_test_split(images, test_size=0.2, random_state=42)
   ```

2. **Stratified Split** (maintains class balance):
   ```python
   train, test = train_test_split(
       images, labels, 
       test_size=0.2, 
       stratify=labels,
       random_state=42
   )
   ```

3. **Patient-Based Split** (prevents data leakage):
   ```python
   # Split by patient ID, not by image
   unique_patients = df['patient_id'].unique()
   train_patients, test_patients = train_test_split(
       unique_patients, test_size=0.2
   )
   ```

---

## 🎯 Sample Images

For inference testing without training, use sample images:

```bash
# Create sample directory
mkdir -p assets/sample_images

# Add some test images (download or copy)
cp data/processed/test/P/*.jpg assets/sample_images/
```

Then test inference:

```bash
python src/inference.py --image-dir assets/sample_images
```

---

## 🔐 Data Privacy

**Important for medical data:**

- ✅ Remove all patient identifiers (HIPAA compliance)
- ✅ Anonymize metadata
- ✅ Use secure storage
- ✅ Implement access controls
- ❌ Never commit raw data to git
- ❌ Never share patient data publicly

Add to `.gitignore`:

```gitignore
# Data directories (never commit patient data)
data/raw/
data/processed/
*.jpg
*.png
*.jpeg
```

---

## 📦 Data Versioning (Optional)

Use DVC (Data Version Control) for large datasets:

```bash
# Initialize DVC
dvc init

# Track data directory
dvc add data/processed

# Commit DVC files (not the actual data)
git add data/processed.dvc .gitignore
git commit -m "Add dataset v1.0"

# Configure remote storage (S3, GCS, etc.)
dvc remote add -d storage s3://mybucket/wound-data
dvc push
```

---

## ❓ Troubleshooting

### Issue: "FileNotFoundError: Missing class directory"

**Solution:** Ensure all 6 class folders exist in both train and test:

```bash
for split in train test; do
    for cls in BG D N P S V; do
        mkdir -p data/processed/$split/$cls
    done
done
```

### Issue: "No images found"

**Solution:** Check file extensions and naming:

```bash
# List all files in class directory
ls -la data/processed/train/BG/

# Check for hidden characters or wrong extensions
file data/processed/train/BG/*
```

### Issue: "Class imbalance warning"

**Solution:** Use `WeightedRandomSampler` (enabled by default):

```python
# In train_efficientnet.py
python src/train_efficientnet.py --use-sampler
```

---

## 🔗 Next Steps

After setting up your data:

1. ✅ Verify dataset structure
2. ✅ Run sanity training: `python src/train_sanity.py`
3. ✅ Start full training: `python src/train_efficientnet.py`
4. ✅ Monitor training metrics in `artifacts/metrics/`
5. ✅ Test inference on validation images

---

## 📞 Need Help?

- Check [TESTING.md](TESTING.md) for troubleshooting
- Open an issue: [GitHub Issues](https://github.com/yourusername/wound-classification-mlops/issues)
- Email: your.email@example.com

---

**Good luck with your training! 🚀**
