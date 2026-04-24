"""
Face recognition utilities using InsightFace (ArcFace / ONNX).

Model: buffalo_l (ResNet100 ArcFace + RetinaFace detector)
  - 512-d normed embeddings, cosine similarity matching
  - Handles similar Indian faces, angled photos, low-quality enrollment images
  - det_size=640: larger detection grid for distant / tilted faces

Similarity thresholds for buffalo_l normed embeddings:
  > 0.30  likely same person (low quality / extreme angle)
  > 0.40  confident match  ← default FACE_MATCH_TOLERANCE
  > 0.55  high confidence
  > 0.70  very high confidence
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
                logger.info("Loading InsightFace buffalo_l model...")
                app = FaceAnalysis(
                    name="buffalo_l",
                    root="/app/.insightface",
                    providers=["CPUExecutionProvider"],
                )
                # 640×640 detection grid: catches faces at distance and odd angles
                app.prepare(ctx_id=0, det_size=(640, 640))
                _face_app = app
                logger.info("InsightFace buffalo_l model loaded successfully.")
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


def encode_face_with_bbox(image_array: np.ndarray):
    """
    Detect and encode the primary face, also returning its bounding box.
    Returns (embedding, bbox, None) or (None, None, error_message).
    The bbox is needed by liveness_utils.check_liveness().
    """
    app = _get_face_app()
    faces = app.get(image_array)

    if not faces:
        return None, None, "No face detected in the image."

    largest = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    embedding = largest.normed_embedding
    if embedding is None:
        return None, None, "Face detected but could not generate embedding (try facing the camera directly)."
    return embedding, largest.bbox, None


def encode_face_from_bytes(image_bytes: bytes):
    """Encode a face from raw image bytes (e.g. downloaded from Zoho Creator URL)."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return encode_face_from_array(np.array(image))
    except Exception as e:
        return None, str(e)


# ── Face matching ─────────────────────────────────────────────────────────────

def find_best_match(submitted_embedding: np.ndarray, students: list, tolerance: float = 0.40):
    """
    Find the best-matching student using cosine similarity.
    InsightFace normed embeddings: dot product = cosine similarity in [-1, 1].

    tolerance: minimum cosine similarity to accept as a match.
      buffalo_l recommended range: 0.35 (permissive) – 0.45 (strict).
      Default 0.40 balances accuracy vs false-reject rate for 1200 Indian students.
    """
    if not students:
        return None, 0.0

    known_embeddings = np.array([s["encoding"] for s in students])
    similarities = np.dot(known_embeddings, submitted_embedding)

    best_idx = int(np.argmax(similarities))
    best_sim = float(similarities[best_idx])

    logger.info(
        f"Best similarity: {best_sim:.3f} (tolerance: {tolerance}) "
        f"— student: {students[best_idx]['name']}"
    )

    if best_sim >= tolerance:
        # Scale to 0–100% between tolerance and 0.75 (practical upper bound)
        confidence = round(
            min((best_sim - tolerance) / (0.75 - tolerance) * 100, 99.9), 1
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
