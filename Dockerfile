# ─────────────────────────────────────────────────────────────────────────────
# Zoho Face Recognition — Dockerfile (insightface edition)
#
# WHY WE DROPPED dlib/face_recognition:
#   dlib compiles from C++ source — on Render's build servers it spawned
#   parallel g++ processes consuming 8GB+ RAM, crashing the build every time.
#
# WHY insightface:
#   - Pre-built wheel — zero compilation, installs in seconds
#   - ArcFace model — more accurate than dlib's HOG-based approach
#   - ONNX runtime — lightweight inference, no TensorFlow/PyTorch needed
#   - buffalo_sc model is ~74MB, baked into the image during build
#
# Build time: ~3–5 minutes (no compilation)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim-bullseye

# ── System libraries ─────────────────────────────────────────────────────────
# build-essential (g++) needed for insightface's tiny Cython mesh extension.
# Unlike dlib (thousands of C++ files, 8GB OOM), this is ONE small file
# that compiles in ~8 seconds using ~50MB RAM — totally safe.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install all Python dependencies ──────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download insightface model into the image ─────────────────────────────
# buffalo_sc = small ArcFace model (~74MB: detector + recogniser)
# Baking it in here means zero download time at runtime / after Render sleeps
RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_sc', root='/app/.insightface', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=0, det_size=(320, 320)); \
print('InsightFace buffalo_sc model ready.')"

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 --log-level info"]
