# =============================================================
# SoundShield-AI Docker Image
# Multi-stage build for optimized production deployment
# =============================================================

# --- Stage 1: Builder ---
FROM python:3.10-slim AS builder

WORKDIR /build

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.10-slim AS runtime

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY *.py ./
COPY inappropriate_words.json ./
COPY templates/ templates/
COPY static/ static/

# Create required directories
RUN mkdir -p uploads reports models

# Environment defaults (override via docker-compose or -e flags)
ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=5000
ENV WEB_DEBUG=false
ENV UPLOAD_FOLDER=uploads
ENV REPORTS_FOLDER=reports
ENV DATABASE_PATH=/app/data/soundshield.db
ENV PYTHONUNBUFFERED=1

# Expose web port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Run the web application
CMD ["python", "web_app.py"]
