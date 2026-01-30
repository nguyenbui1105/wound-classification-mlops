# 🤝 Contributing to Wound-AI Classification

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

---

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for everyone, regardless of:
- Experience level
- Gender identity and expression
- Sexual orientation
- Disability
- Personal appearance
- Race or ethnicity
- Age
- Religion
- Nationality

### Expected Behavior

- Be respectful and considerate
- Use welcoming and inclusive language
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment, trolling, or discriminatory comments
- Personal or political attacks
- Publishing others' private information
- Spam or off-topic comments
- Any other conduct inappropriate in a professional setting

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Git
- Docker (optional, for containerized development)
- Basic understanding of:
  - FastAPI
  - PyTorch
  - Machine Learning concepts

### Quick Setup

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/wound-classification-mlops.git
cd wound-classification-mlops

# 3. Add upstream remote
git remote add upstream https://github.com/ORIGINAL-OWNER/wound-classification-mlops.git

# 4. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements_docker.txt
pip install -r requirements-dev.txt

# 6. Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

---

## 💻 Development Setup

### Running in Development Mode

```bash
# Start API server with auto-reload
uvicorn src.api:app --reload

# Start Streamlit UI
streamlit run src/ui_streamlit.py

# Run tests
pytest
```

### Using Docker for Development

```bash
# Build development image
docker compose -f docker-compose.dev.yml build

# Start services
docker compose -f docker-compose.dev.yml up

# Attach to container for debugging
docker exec -it wound-ai-api bash
```

---

## 🎯 How to Contribute

### Types of Contributions

We welcome:

1. **Bug Fixes**
   - Fix broken functionality
   - Improve error handling
   - Resolve edge cases

2. **New Features**
   - Add new API endpoints
   - Implement new model architectures
   - Enhance UI/UX

3. **Documentation**
   - Improve README
   - Add code comments
   - Write tutorials

4. **Tests**
   - Add unit tests
   - Add integration tests
   - Improve test coverage

5. **Performance Improvements**
   - Optimize inference speed
   - Reduce memory usage
   - Improve batch processing

### Workflow

1. **Check existing issues/PRs**
   - Avoid duplicate work
   - Join ongoing discussions

2. **Create/claim an issue**
   - Describe what you want to do
   - Wait for approval before starting

3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

4. **Make changes**
   - Write code
   - Add tests
   - Update documentation

5. **Test locally**
   ```bash
   # Run tests
   pytest
   
   # Check code quality
   black .
   isort .
   ruff check .
   ```

6. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add feature X"
   ```

7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create Pull Request**
   - Go to GitHub
   - Click "New Pull Request"
   - Fill in the template

---

## 📝 Coding Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these specifics:

```python
# Line length: 100 characters (configured in pyproject.toml)
MAX_LINE_LENGTH = 100

# Use black for formatting
black .

# Use isort for import sorting
isort .

# Use ruff for linting
ruff check .
```

### Code Organization

```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import torch
import numpy as np
from fastapi import FastAPI

# Local imports
from src.model import build_model
from src.data import WoundDataset
```

### Naming Conventions

```python
# Variables and functions: snake_case
def process_image(image_path: Path) -> torch.Tensor:
    processed_data = preprocess(image_path)
    return processed_data

# Classes: PascalCase
class WoundClassifier:
    def __init__(self):
        pass

# Constants: UPPER_SNAKE_CASE
MAX_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = 224

# Private methods: _leading_underscore
def _internal_helper():
    pass
```

### Type Hints

Always use type hints for function signatures:

```python
from typing import List, Dict, Optional

def predict_batch(
    images: List[Path],
    batch_size: int = 16,
    device: Optional[str] = None
) -> Dict[str, float]:
    """Predict on batch of images.
    
    Args:
        images: List of image paths
        batch_size: Batch size for inference
        device: Device to run on (cpu/cuda)
    
    Returns:
        Dictionary with predictions
    """
    pass
```

### Documentation

Use Google-style docstrings:

```python
def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint.
    
    This function loads a trained model from a checkpoint file and
    prepares it for inference.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint is invalid
    
    Example:
        >>> device = torch.device("cpu")
        >>> model = load_model(Path("model.pt"), device)
        >>> print(model)
    """
    pass
```

---

## 🧪 Testing Guidelines

### Writing Tests

1. **Follow existing structure:**
   ```
   tests/
   ├── conftest.py          # Shared fixtures
   ├── test_health.py       # Health endpoint tests
   ├── test_predict.py      # Prediction tests
   └── test_integration.py  # Integration tests
   ```

2. **Test naming:**
   ```python
   def test_health_endpoint_returns_200():
       """Test that health endpoint returns 200 status."""
       pass
   
   def test_predict_rejects_invalid_file_type():
       """Test that predict endpoint rejects non-image files."""
       pass
   ```

3. **Use fixtures:**
   ```python
   @pytest.fixture
   def sample_image():
       """Create a sample test image."""
       img = Image.new("RGB", (224, 224))
       return img
   
   def test_preprocessing(sample_image):
       """Test image preprocessing."""
       tensor = preprocess_image(sample_image)
       assert tensor.shape == (1, 3, 224, 224)
   ```

4. **Mark slow tests:**
   ```python
   @pytest.mark.slow
   def test_full_training_loop():
       """Test complete training pipeline."""
       pass
   ```

### Running Tests

```bash
# All tests
pytest

# Fast tests only
pytest -m "not slow"

# With coverage
pytest --cov=src --cov-report=html

# Specific file
pytest tests/test_health.py -v

# Specific test
pytest tests/test_health.py::test_health_endpoint_returns_200
```

### Test Coverage

Aim for **>80% code coverage**:

```bash
# Generate coverage report
pytest --cov=src --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```

---

## 🔄 Pull Request Process

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## How Has This Been Tested?
Describe testing done

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings
```

### Review Process

1. **Automated checks must pass:**
   - All tests pass
   - Code style checks pass
   - No merge conflicts

2. **Code review:**
   - At least one approval required
   - Address all review comments
   - Keep discussion professional

3. **Merging:**
   - Squash and merge for clean history
   - Delete branch after merge

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**

```bash
# Feature
git commit -m "feat(api): add batch prediction endpoint"

# Bug fix
git commit -m "fix(model): resolve memory leak in inference"

# Documentation
git commit -m "docs: update deployment guide"

# Breaking change
git commit -m "feat(api)!: change response format

BREAKING CHANGE: API response format changed from {...} to {...}"
```

---

## 🐛 Issue Reporting

### Bug Reports

Use this template:

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. See error

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- Docker version (if applicable): [e.g., 24.0.5]

## Additional Context
Screenshots, logs, etc.
```

### Feature Requests

Use this template:

```markdown
## Feature Description
Clear description of the feature

## Motivation
Why is this feature needed?

## Proposed Solution
How should it work?

## Alternatives Considered
Other approaches considered

## Additional Context
Mockups, examples, etc.
```

---

## 🏆 Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Thanked in documentation

---

## 📞 Getting Help

- **Questions:** Open a [Discussion](https://github.com/yourusername/wound-classification-mlops/discussions)
- **Bugs:** Open an [Issue](https://github.com/yourusername/wound-classification-mlops/issues)
- **Email:** your.email@example.com

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Git Flow Guide](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
- [How to Write Good Commit Messages](https://chris.beams.io/posts/git-commit/)

---

**Thank you for contributing to Wound-AI! 🎉**
