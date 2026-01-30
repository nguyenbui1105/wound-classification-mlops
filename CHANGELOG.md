# Changelog

All notable changes to the Wound-AI Classification project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Structured logging with request ID tracking
- Environment-based configuration
- Prometheus monitoring support
- Docker multi-stage build for smaller images
- Non-root user in Docker container for security
- Integration tests for real model validation
- Comprehensive deployment guide
- Contributing guidelines

### Changed
- Improved Dockerfile with security best practices
- Enhanced docker-compose.yml with resource limits
- Updated pytest configuration with new markers

### Fixed
- Memory optimization for CPU inference
- Healthcheck reliability in Docker

---

## [1.0.0] - 2026-01-29

### Added
- Initial release
- FastAPI server with single and batch prediction
- EfficientNet-B0 model with 6-class classification
- 2-phase fine-tuning training script
- Streamlit UI with glassmorphic design
- Docker deployment support
- Comprehensive test suite (30 tests)
- GitHub Actions CI/CD workflows
- API documentation
- Testing guide

### Model Performance
- Test Accuracy: 87.34%
- Macro F1-Score: 0.8512
- Balanced Accuracy: 0.8489
- Inference Time: ~50ms/image (CPU)
- Model Size: ~17MB

### Infrastructure
- Python 3.11
- FastAPI 0.109.0
- PyTorch (CPU-optimized)
- Docker & Docker Compose

---

## Release Notes

### Version 1.0.0 (2026-01-29)

**Highlights:**
- Production-ready wound classification API
- Modern Streamlit UI for easy interaction
- Comprehensive testing and CI/CD
- Docker deployment with health checks

**Breaking Changes:**
- None (initial release)

**Known Issues:**
- Model checkpoint not included (must be trained or downloaded separately)
- Documentation assumes basic ML/Docker knowledge

**Migration Guide:**
- N/A (initial release)

---

## Upgrade Instructions

### From 0.x to 1.0.0
N/A - Initial release

---

## Future Roadmap

### Version 1.1.0 (Planned)
- [ ] Add authentication support (JWT)
- [ ] Implement rate limiting
- [ ] Add caching layer (Redis)
- [ ] Support GPU inference
- [ ] Add more model architectures

### Version 1.2.0 (Planned)
- [ ] Add database integration for prediction history
- [ ] Implement user management
- [ ] Add API key management
- [ ] Support model versioning
- [ ] Add A/B testing framework

### Version 2.0.0 (Future)
- [ ] Multi-model ensemble support
- [ ] Real-time inference via WebSocket
- [ ] Mobile app support
- [ ] Advanced analytics dashboard
- [ ] DICOM image support

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Suggesting features
- Submitting pull requests

---

## Support

For questions or issues:
- GitHub Issues: [Report a bug](https://github.com/yourusername/wound-classification-mlops/issues)
- Discussions: [Ask a question](https://github.com/yourusername/wound-classification-mlops/discussions)
- Email: your.email@example.com

---

[Unreleased]: https://github.com/yourusername/wound-classification-mlops/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/wound-classification-mlops/releases/tag/v1.0.0
