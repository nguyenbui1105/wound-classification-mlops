# 🩹 Wound Classification MLOps

[![Tests](https://github.com/nguyenbui1105/wound-classification-mlops/actions/workflows/tests.yml/badge.svg)](https://github.com/nguyenbui1105/wound-classification-mlops/actions/workflows/tests.yml)
[![Docker Build](https://github.com/nguyenbui1105/wound-classification-mlops/actions/workflows/docker-build.yml/badge.svg)](https://github.com/nguyenbui1105/wound-classification-mlops/actions/workflows/docker-build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace Demo](https://img.shields.io/badge/🤗-Demo-yellow.svg)](https://huggingface.co/spaces/nguyenbui1105/wound-classification-demo)

End-to-end wound image classification system with production-ready MLOps pipeline. Built with FastAPI, Streamlit UI, Docker deployment, comprehensive testing, and CI/CD automation.

---

## ✨ Features

- 🔬 **EfficientNet-B0** model for 6-class wound classification (BG, D, N, P, S, V)
- 🚀 **FastAPI** inference server with single & batch prediction
- 🎨 **Modern Streamlit UI** with glassmorphic design
- 🐳 **Docker-ready** with CPU-optimized deployment
- ✅ **Comprehensive tests** (30 tests with pytest)
- 🔄 **CI/CD pipeline** via GitHub Actions
- 📊 **Production monitoring** with structured logging
- 🔒 **Security-first** with non-root Docker containers

## 🎮 Live Demo

Try the model instantly without installation:

**[🤗 Interactive Demo on HuggingFace Spaces](https://huggingface.co/spaces/nguyenbui1105/wound-classification-demo)**

- Upload wound images and get instant predictions
- Trained EfficientNet-B0 model (40MB)
- Probability scores for all 6 wound types
- No setup required - runs in your browser

---

## 📋 Table of Contents

- [Live Demo](#live-demo)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Training](#training)
- [Testing](#testing)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- Model checkpoint file (download or train your own)

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/wound-classification-mlops.git
cd wound-classification-mlops
```

### 2. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 3. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements_docker.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Start API Server

```bash
uvicorn src.api:app --reload
```

API will be available at: http://localhost:8000

### 5. Start UI (Optional)

In a new terminal:

```bash
streamlit run src/ui_streamlit.py
```

UI will be available at: http://localhost:8501

---

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and start services
docker compose up --build

# Or run in background
docker compose up -d
```

### Manual Docker Build

```bash
# Build image
docker build -t wound-ai-api:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  -e CHECKPOINT_PATH=/app/artifacts/checkpoints/best_efficientnet.pt \
  wound-ai-api:latest
```

---

## 📖 Usage

### Single Image Prediction

#### Via API

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/wound_image.jpg"
```

#### Via Python

```python
import requests

with open("wound_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

#### Via UI

1. Open http://localhost:8501
2. Navigate to "Single Prediction"
3. Upload image
4. Click "Classify Image"

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_dir": "assets/sample_images",
    "report_id": "my_experiment",
    "batch_size": 16
  }'
```

### Command Line Inference

```bash
# Single image
python src/inference.py --image path/to/image.jpg

# Batch processing
python src/inference.py --image-dir assets/sample_images
```

See [API_USAGE.md](API_USAGE.md) for complete API documentation.

---

## 🎓 Training

### 2-Phase EfficientNet Training

The model uses a 2-phase fine-tuning strategy:
- **Phase A**: Freeze backbone, train head only (5 epochs)
- **Phase B**: Unfreeze last 2 blocks, fine-tune (15 epochs)

```bash
# Start training
python src/train_efficientnet.py \
  --phase-a-epochs 5 \
  --phase-b-epochs 15 \
  --batch-size 16 \
  --use-sampler

# Fast development run (sanity check)
python src/train_efficientnet.py --fast-dev-run
```

### Training Options

```bash
# Custom hyperparameters
python src/train_efficientnet.py \
  --model efficientnet_b0 \
  --dropout 0.3 \
  --phase-a-lr 3e-3 \
  --phase-b-head-lr 1e-3 \
  --phase-b-backbone-lr 3e-4 \
  --class-p-boost 1.5 \
  --patience 2
```

### Data Setup

See [DATA_SETUP.md](DATA_SETUP.md) for instructions on organizing your dataset.

---

## 🧪 Testing

### Run All Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# With coverage
pytest --cov=src --cov-report=term-missing
```

### Test Categories

```bash
# Unit tests only
pytest -m "not slow"

# Integration tests (requires model checkpoint)
pytest -m integration

# Specific test file
pytest tests/test_health.py -v
```

See [TESTING.md](TESTING.md) for detailed testing guide.

---

## 📦 Deployment

### Production Checklist

- [ ] Set strong `SECRET_KEY` in environment
- [ ] Use HTTPS/TLS certificates
- [ ] Enable rate limiting
- [ ] Configure logging level
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure backups for model checkpoints
- [ ] Review security settings in Dockerfile

### CI/CD Setup

See [.github/DOCKER_CI_SETUP.md](.github/DOCKER_CI_SETUP.md) for GitHub Actions setup.

### Environment Variables

```bash
# Required
CHECKPOINT_PATH=artifacts/checkpoints/best_efficientnet.pt

# Optional
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
WOUND_API_TESTING=0
MAX_UPLOAD_SIZE=10485760  # 10MB
```

---

## 📂 Project Structure

```
wound-classification-mlops/
├── .github/
│   └── workflows/
│       ├── tests.yml           # CI tests
│       └── docker-build.yml    # Docker build & push
├── src/
│   ├── api.py                  # FastAPI server
│   ├── inference.py            # Batch inference
│   ├── model.py                # Model architectures
│   ├── train_efficientnet.py  # 2-phase training
│   ├── ui_streamlit.py         # Streamlit UI
│   └── data/                   # Data loading
│       ├── dataset.py
│       ├── transforms.py
│       └── dataloader.py
├── tests/
│   ├── conftest.py             # Pytest config
│   ├── test_health.py          # Health endpoint tests
│   ├── test_predict.py         # Prediction tests
│   └── test_batch_predict.py  # Batch prediction tests
├── artifacts/
│   ├── checkpoints/            # Model weights
│   ├── metrics/                # Training metrics
│   └── inference_reports/      # Batch predictions
├── data/
│   └── processed/              # Training data
│       ├── train/
│       └── test/
├── docker-compose.yml          # Docker Compose config
├── Dockerfile                  # Container definition
├── requirements_docker.txt     # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── .env.example                # Environment template
├── API_USAGE.md                # API documentation
├── TESTING.md                  # Testing guide
├── DATA_SETUP.md               # Data setup guide
└── README.md                   # This file
```

---

## 🔧 Configuration

### Model Configuration

Edit `src/model.py` to change model architecture:

```python
# Available models
- efficientnet_b0 (default, balanced)
- tf_efficientnetv2_s (larger, more accurate)
- mobilenet_v3_small (fastest, CPU-optimized)
- resnet18 (baseline)
```

### API Configuration

Edit `src/api.py` or set environment variables:

```python
CHECKPOINT_PATH = "artifacts/checkpoints/best_efficientnet.pt"
IMAGE_SIZE = 224
BATCH_SIZE = 16
MAX_WORKERS = 4
```

---

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 87.34% |
| Macro F1-Score | 0.8512 |
| Balanced Accuracy | 0.8489 |
| Inference Time (CPU) | ~50ms/image |
| Model Size | ~17MB |

### Per-Class Performance

| Class | Recall | Precision | F1-Score |
|-------|--------|-----------|----------|
| BG | 0.92 | 0.89 | 0.90 |
| D | 0.85 | 0.87 | 0.86 |
| N | 0.88 | 0.85 | 0.86 |
| P | 0.81 | 0.84 | 0.82 |
| S | 0.86 | 0.88 | 0.87 |
| V | 0.89 | 0.87 | 0.88 |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Write tests for new features
- Update documentation
- Run linters before committing:

```bash
# Format code
black .
isort .

# Check code quality
ruff check .
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- FastAPI framework by Sebastián Ramírez
- Streamlit for the amazing UI framework
- PyTorch team for the deep learning framework

---

## 📞 Support

- 📧 Email: nguyenbd1105@gmail.com

## 🔗 Links

- [🤗 Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/nguyenbui1105/wound-classification-demo)
- [API Documentation](http://localhost:8000/docs) (when server is running)
- [Streamlit UI](http://localhost:8501) (when UI is running)
- [Docker Hub](https://hub.docker.com/r/yourusername/wound-ai-api)

---

**Made with ❤️ by Nguyen Bui**
