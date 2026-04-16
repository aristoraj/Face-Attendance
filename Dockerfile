# ─────────────────────────────────────────────────────────────────────────────
# Zoho Face Recognition Module — Dockerfile
#
# ROOT CAUSE FIX HISTORY:
#   Attempt 1: pip install cmake → cmake module not found in dlib subprocess
#   Attempt 2: apt cmake + pip dlib → pip creates isolated build env, pulls in
#              cmake 3.27+ from PyPI, which breaks dlib's bundled old pybind11:
#              "Compatibility with CMake < 3.5 has been removed from CMake"
#
# FINAL FIX:
#   --no-build-isolation on dlib install forces pip to use the SYSTEM cmake
#   (3.18.4 from apt) instead of creating an isolated env that downloads 3.27+
#   Also pin setuptools < 72 to avoid "test command deprecated" abort
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim-bullseye

# ── System packages ───────────────────────────────────────────────────────────
# cmake here is the REAL system binary (3.18.4) — not the pip wrapper
# python3-dev needed so dlib can find Python.h headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libatlas-base-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Confirm system cmake is available ────────────────────────────────────────
RUN cmake --version

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Step 1: Pin setuptools to avoid "test command disabled" abort in dlib ─────
# setuptools >= 72 raises an ERROR for packages that use tests_require
# dlib 19.24.x uses tests_require → must pin below 72
RUN pip install --no-cache-dir "setuptools==68.2.0" wheel

# ── Step 2: Install dlib WITHOUT build isolation ──────────────────────────────
# --no-build-isolation tells pip: do NOT create a fresh venv for the build.
# This prevents pip from downloading cmake 3.27+ into an isolated build env.
# Instead dlib's cmake invocation uses /usr/bin/cmake (3.18.4 from apt above)
# which is fully compatible with dlib's bundled old pybind11.
RUN pip install --no-cache-dir --no-build-isolation dlib==19.24.2

# ── Step 3: Install face-recognition (depends on dlib) ───────────────────────
RUN pip install --no-cache-dir face-recognition==1.3.0

# ── Step 4: Install remaining Python dependencies ────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY . .

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# Render sets $PORT automatically; fall back to 5000 locally
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 --log-level info"]
