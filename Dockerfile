# ─────────────────────────────────────────────────────────────────────────────
# Zoho Face Recognition — Dockerfile
#
# Changes from v2:
#   - curl added for model download
#   - MiniFASNet anti-spoofing ONNX model pre-baked into image (~1.1 MB)
#   - /app/data directory created for SQLite attendance queue
#   - Gunicorn workers increased from 2 → 4 (handles 1000+ student peak load)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim-bullseye

# ── System libraries ──────────────────────────────────────────────────────────
# build-essential: needed for insightface's tiny Cython mesh extension (~8s, ~50MB)
# curl: used to download the MiniFASNet liveness model
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download InsightFace buffalo_sc model ─────────────────────────────────
# Baked into image so it's available instantly after Render spins up.
RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_sc', root='/app/.insightface', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=0, det_size=(320, 320)); \
print('InsightFace buffalo_sc model ready.')"

# ── Download MiniFASNet anti-spoofing model ───────────────────────────────────
# 2.7_80x80_MiniFASNetV2: ~1.1 MB ONNX model from Silent-Face-Anti-Spoofing.
# Detects screen/video replay attacks (real face vs phone screen) passively —
# no user interaction needed, transparent to disabled students.
# The || echo makes this layer non-fatal: if GitHub is unreachable during build,
# the app still starts; check_liveness() returns (True, 1.0, "model_unavailable").
RUN mkdir -p /app/.anti_spoof && \
    curl -fsSL --retry 3 --retry-delay 3 --max-time 60 \
      "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx" \
      -o /app/.anti_spoof/MiniFASNetV2.onnx \
    && echo "MiniFASNet liveness model ready ($(wc -c < /app/.anti_spoof/MiniFASNetV2.onnx) bytes)" \
    || echo "WARNING: liveness model download failed — passive anti-spoofing disabled at runtime"

# ── Create SQLite queue directory ─────────────────────────────────────────────
RUN mkdir -p /app/data

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# 4 workers: handles 1000+ students in a 10-11 AM peak (≈17 req/min).
# Per-request latency is now ~1.5s (face recognition only — no Zoho wait),
# so 4 workers sustain >160 req/min throughput — well above the required rate.
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 4 --timeout 120 --log-level info"]
