"""
Face recognition utilities using InsightFace (ArcFace / ONNX).

Replaces the previous dlib/face_recognition implementation.

Why insightface:
  - Pre-built wheels — no C++ compilation, deploys in seconds
  - buffalo_sc model: ArcFace recognition + RetinaFace detection
  - Cosine similarity on 512-d normed embeddings
  - More accurate than HOG-based dlib on varied lighting / angles

Similarity thresholds for buffalo_sc:
  > 0.35  likely same person
  > 0.50  confident match
  > 0.65  very high confidence
"""

import io
import json
import time
import base64
import logging
import threading

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── InsightFace model singleton (loaded once per process) ──────────────────────
_face_app = None
_face_app_lock = threading.Lock()


def _get_face_app():
    global _face_app
    if _face_app is None:
        with _face_app_lock:
            if _face_app is None:
                from insightface.app import FaceAnalysis
                logger.info("Loading InsightFace buffalo_sc model...")
                app = FaceAnalysis(
                    name="buffalo_sc",
                    root="/app/.insightface",
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=(320, 320))
                _face_app = app
                logger.info("InsightFace model loaded successfully.")
    return _face_app


# ── Student face encoding cache ───────────────────────────────────────────────

class FaceCache:
    """Thread-safe in-memory cache for student face encodings."""

    def __init__(self, ttl: int = 3600):
        self._data = None
        self._timestamp = 0.0
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self._data is not None and (time.time() - self._timestamp) < self._ttl:
                return self._data
            return None

    def set(self, data):
        with self._lock:
            self._data = data
            self._timestamp = time.time()
            logger.info(f"Face cache updated: {len(data)} student encodings.")

    def invalidate(self):
        with self._lock:
            self._data = None

    @property
    def age_seconds(self):
        return time.time() - self._timestamp if self._data else None

    @property
    def size(self):
        return len(self._data) if self._data else 0


# ── Image helpers ─────────────────────────────────────────────────────────────

def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode a base64 image (with or without data URI prefix) to RGB numpy array."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


# ── Face encoding ─────────────────────────────────────────────────────────────

def encode_face_from_array(image_array: np.ndarray):
    """
    Detect and encode the primary face in an RGB numpy array.
    Returns (512-d normed embedding, None) or (None, error_message).
    """
    app = _get_face_app()
    faces = app.get(image_array)

    if not faces:
        return None, "No face detected in the image."

    largest = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    embedding = largest.normed_embedding
    if embedding is None:
        return None, "Face detected but could not generate embedding (try facing the camera directly)."
    return embedding, None


def encode_face_from_bytes(image_bytes: bytes):
    """Encode a face from raw image bytes (e.g. downloaded from Zoho Creator URL)."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return encode_face_from_array(np.array(image))
    except Exception as e:
        return None, str(e)


# ── Face matching ─────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.25  # buffalo_sc: lowered from 0.35 — real-world lighting/angle variance


def find_best_match(submitted_embedding: np.ndarray, students: list, tolerance: float = 0.55):
    """
    Find the best-matching student using cosine similarity.
    insightface normed embeddings: dot product = cosine similarity.
    """
    if not students:
        return None, 0.0

    known_embeddings = np.array([s["encoding"] for s in students])
    similarities = np.dot(known_embeddings, submitted_embedding)

    best_idx = int(np.argmax(similarities))
    best_sim = float(similarities[best_idx])

    logger.info(f"Best similarity: {best_sim:.3f} (threshold: {SIMILARITY_THRESHOLD}) — student: {students[best_idx]['name']}")

    if best_sim >= SIMILARITY_THRESHOLD:
        confidence = round(
            min((best_sim - SIMILARITY_THRESHOLD) / (0.65 - SIMILARITY_THRESHOLD) * 100, 99.9), 1
        )
        return students[best_idx], confidence

    return None, 0.0


# ── Embedding serialisation ───────────────────────────────────────────────────

def embedding_to_json(embedding: np.ndarray) -> str:
    """Serialise a 512-d numpy embedding to a compact JSON string for storage."""
    return json.dumps([round(float(v), 6) for v in embedding])


def json_to_embedding(json_str: str) -> np.ndarray:
    """Deserialise a JSON string back to a normalised numpy embedding."""
    arr = np.array(json.loads(json_str), dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm   # re-normalise in case of float rounding
    return arr
