# ─────────────────────────────────────────────────────────────────────────────
# Zoho Face Recognition Module — Dockerfile
# Uses slim Python image with dlib compiled from source.
# This image is used by Render for deployment.
# Build time: ~10–15 minutes on first deploy (dlib compilation).
# Subsequent deploys use Docker layer cache and are much faster.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bullseye

# ── System build dependencies for dlib ────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libatlas-base-dev \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install heavy packages first (cached unless requirements change) ───────────
# dlib must be installed before face_recognition
RUN pip install --no-cache-dir cmake==3.27.9
RUN pip install --no-cache-dir dlib==19.24.4
RUN pip install --no-cache-dir face-recognition==1.3.0

# ── Copy requirements and install remaining packages ──────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Create non-root user for security ─────────────────────────────────────────
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# ── Expose port (Render sets $PORT; gunicorn binds to it) ─────────────────────
EXPOSE 5000

# ── Startup command ───────────────────────────────────────────────────────────
# 2 workers: enough for a free-tier instance, won't OOM on Render 512MB RAM
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 --log-level info"]
