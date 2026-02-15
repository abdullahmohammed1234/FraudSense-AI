# =============================================================================
# FraudSense AI - Production Dockerfile
# =============================================================================
# Multi-stage build for optimized production image

# -----------------------------------------------------------------------------
# Base stage - Python runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Dependencies stage - Install Python dependencies
# -----------------------------------------------------------------------------
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Application stage - Final production image
# -----------------------------------------------------------------------------
FROM base

# Create non-root user for security
RUN groupadd -r fraudsense && \
    useradd -r -g fraudsense fraudsense

# Copy installed dependencies from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY creditcard.csv ./creditcard.csv

# Create necessary directories
RUN mkdir -p /app/logs /app/backend/logs && \
    chown -R fraudsense:fraudsense /app

# Switch to non-root user
USER fraudsense

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables (configurable at runtime)
ENV PYTHONPATH=/app \
    LOG_LEVEL=INFO \
    API_KEYS=dev-key-001:Admin:1000,analyst-key-001:Analyst:500,auditor-key-001:Auditor:200

# Run the application
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
