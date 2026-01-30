# 🚀 Quick Reference - Common Commands

Fast reference for frequently used commands in Wound-AI project.

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/wound-classification-mlops.git
cd wound-classification-mlops

# Setup environment
cp .env.example .env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements_docker.txt
pip install -r requirements-dev.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 🏃 Running

```bash
# Start API server
uvicorn src.api:app --reload

# Start UI
streamlit run src/ui_streamlit.py

# Docker (recommended)
docker compose up --build
```

---

## 🧪 Testing

```bash
# All tests
pytest

# Fast tests only (skip integration)
pytest -m "not slow"

# Integration tests only
pytest -m integration

# With coverage
pytest --cov=src --cov-report=html

# Specific file
pytest tests/test_health.py -v
```

---

## 🐳 Docker

```bash
# Build
docker compose build

# Start (foreground)
docker compose up

# Start (background)
docker compose up -d

# Stop
docker compose down

# Logs
docker compose logs -f wound-api

# Restart
docker compose restart wound-api

# Shell access
docker exec -it wound-ai-api bash
```

---

## 🎓 Training

```bash
# Full training
python src/train_efficientnet.py

# Fast dev run (sanity check)
python src/train_efficientnet.py --fast-dev-run

# Custom hyperparameters
python src/train_efficientnet.py \
  --phase-a-epochs 5 \
  --phase-b-epochs 15 \
  --batch-size 16 \
  --use-sampler
```

---

## 🔮 Inference

```bash
# Single image
python src/inference.py --image path/to/image.jpg

# Batch processing
python src/inference.py --image-dir assets/sample_images

# Via API
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"

# Batch via API
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{"image_dir": "assets/sample_images"}'
```

---

## 📊 Monitoring

```bash
# Check health
curl http://localhost:8000/health

# View logs
tail -f logs/api.log

# Docker logs
docker compose logs -f wound-api

# Prometheus (if enabled)
open http://localhost:9090

# Grafana (if enabled)
open http://localhost:3000
```

---

## 🔧 Code Quality

```bash
# Format code
black .
isort .

# Lint
ruff check .

# Fix linting issues
ruff check . --fix

# Type check (if mypy installed)
mypy src/
```

---

## 🐛 Debugging

```bash
# API debug mode
uvicorn src.api:app --reload --log-level debug

# Python debugger
python -m pdb src/api.py

# Print environment
env | grep WOUND

# Check Docker container
docker inspect wound-ai-api

# Container resource usage
docker stats wound-ai-api
```

---

## 📦 Dependencies

```bash
# List installed
pip list

# Check outdated
pip list --outdated

# Update specific package
pip install --upgrade fastapi

# Freeze requirements
pip freeze > requirements.txt

# Install from lock file
pip install -r requirements.txt
```

---

## 🔄 Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Check status
git status

# Stage changes
git add .

# Commit with conventional message
git commit -m "feat: add new feature"

# Push branch
git push origin feature/my-feature

# Update from main
git fetch origin
git rebase origin/main
```

---

## 🚀 Deployment

```bash
# Build for production
docker build -t wound-ai-api:prod .

# Tag for registry
docker tag wound-ai-api:prod registry.example.com/wound-ai-api:latest

# Push to registry
docker push registry.example.com/wound-ai-api:latest

# Deploy to server (example)
ssh user@server "cd /app && docker compose pull && docker compose up -d"
```

---

## 🔐 Security

```bash
# Generate secret key
python -c "import secrets; print(secrets.token_hex(32))"

# Check for vulnerabilities
pip install safety
safety check

# Scan Docker image
docker scan wound-ai-api:latest

# Update all packages
pip list --outdated | cut -d ' ' -f1 | xargs -n1 pip install -U
```

---

## 📁 File Operations

```bash
# Find files
find . -name "*.py" -type f

# Count lines of code
find src/ -name "*.py" | xargs wc -l

# Remove cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Clean artifacts
rm -rf artifacts/api_temp/*
rm -rf artifacts/inference_reports/*
```

---

## 🔍 System Info

```bash
# Python version
python --version

# PyTorch version
python -c "import torch; print(torch.__version__)"

# CUDA available (if GPU)
python -c "import torch; print(torch.cuda.is_available())"

# Disk usage
du -sh artifacts/

# Memory usage
free -h

# CPU info
lscpu
```

---

## ⚡ Performance

```bash
# Profile API endpoint
python -m cProfile -o profile.stats src/api.py

# View profile
python -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumtime').print_stats(20)"

# Memory profiling
python -m memory_profiler src/api.py

# Benchmark inference
time python src/inference.py --image test.jpg

# Load test (if locust installed)
locust -f tests/locustfile.py
```

---

## 🎯 Quick Fixes

```bash
# API won't start
# Check if port is in use
lsof -i :8000
# Kill process if needed
kill -9 $(lsof -t -i:8000)

# Tests failing
# Clear pytest cache
pytest --cache-clear
rm -rf .pytest_cache

# Docker build failing
# Prune unused images
docker system prune -a

# Model not loading
# Check checkpoint exists
ls -lh artifacts/checkpoints/
# Check permissions
chmod 644 artifacts/checkpoints/*.pt

# Out of memory
# Reduce batch size
export DEFAULT_BATCH_SIZE=8

# Slow inference
# Set CPU threads
export OMP_NUM_THREADS=4
```

---

## 📚 Documentation

```bash
# Generate API docs
open http://localhost:8000/docs

# View this guide
cat QUICK_REFERENCE.md

# View full README
cat README.md

# List all markdown docs
find . -name "*.md" -type f
```

---

## 🔗 Useful URLs

```
API Documentation:   http://localhost:8000/docs
API ReDoc:          http://localhost:8000/redoc
Health Check:       http://localhost:8000/health
Streamlit UI:       http://localhost:8501
Prometheus:         http://localhost:9090
Grafana:            http://localhost:3000
```

---

## 💡 Pro Tips

1. **Use tab completion:** Install `argcomplete` for CLI autocomplete
2. **Alias common commands:** Add to `.bashrc` or `.zshrc`
3. **Use Docker Compose shortcuts:** `docker compose up` → `dc up`
4. **Keep terminal history:** Use `history | grep` to find old commands
5. **Use tmux/screen:** Manage multiple sessions easily
6. **Set up IDE shortcuts:** Configure PyCharm/VSCode for quick actions
7. **Use `watch`:** Monitor commands continuously `watch -n 1 docker ps`
8. **Use `jq`:** Pretty-print JSON `curl localhost:8000/health | jq`

---

## 🆘 Emergency Procedures

### API is down
```bash
docker compose restart wound-api
docker compose logs -f wound-api
```

### High memory usage
```bash
docker stats
docker compose down
docker compose up -d
```

### Database corrupted (if using)
```bash
cp wound_ai.db wound_ai.db.backup
# Restore from backup
cp wound_ai.db.backup wound_ai.db
```

### Lost checkpoint
```bash
# Download from backup
aws s3 cp s3://wound-ai-backups/best_efficientnet.pt artifacts/checkpoints/
```

---

**Save this file for quick reference! 🎯**
