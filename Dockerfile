# =============================================================================
# Dockerfile for Wound-AI FastAPI Server (Production-Ready)
# =============================================================================
# Build: docker compose build
# Run: docker compose up
#
# Features:
# - CPU-optimized PyTorch
# - Non-root user for security
# - Multi-stage build for smaller image
# - Health check with curl
# - Proper logging configuration
# =============================================================================

FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements_docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_docker.txt

# Install PyTorch CPU-only (smaller image size)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# =============================================================================
# Final stage - smaller image
# =============================================================================

FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code
COPY --chown=appuser:appuser src/ ./src/

# Create necessary directories with proper permissions
RUN mkdir -p /app/artifacts /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check using curl (more reliable than python -c)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO

# Run FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
