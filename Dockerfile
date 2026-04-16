# ─────────────────────────────────────────────────────────────────────────────
# Zoho Face Recognition Module — Dockerfile
# Uses slim Python image with dlib compiled from source.
# Build time: ~10–15 minutes on first deploy (dlib compilation).
# Subsequent deploys use Docker layer cache and are much faster.
#
# FIX: cmake installed via apt-get (system binary), NOT pip.
# pip's cmake wrapper breaks dlib's build subprocess with:
#   ModuleNotFoundError: No module named 'cmake'
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bullseye

# ── System dependencies — cmake installed here via apt (real binary) ──────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libatlas-base-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Verify cmake is available as a real binary ────────────────────────────────
RUN cmake --version

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install dlib and face-recognition (heavy, cached layer) ──────────────────
# No pip cmake needed — apt cmake above is on PATH and works correctly
RUN pip install --no-cache-dir dlib==19.24.2
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
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 --log-level info"]
