"""
Face recognition utilities with in-memory caching.
Handles face encoding, comparison, and student cache management.
"""

import time
import logging
import threading
import numpy as np
import face_recognition
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)


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
            logger.info(f"Face cache updated with {len(data)} student encodings.")

    def invalidate(self):
        with self._lock:
            self._data = None
            logger.info("Face cache invalidated.")

    @property
    def age_seconds(self):
        return time.time() - self._timestamp if self._data else None

    @property
    def size(self):
        return len(self._data) if self._data else 0


def decode_base64_image(b64_string: str) -> np.ndarray:
    """
    Decode a base64 image string (with or without data URI prefix)
    into a NumPy RGB array suitable for face_recognition.
    """
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    image_bytes = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def encode_face_from_array(image_array: np.ndarray):
    """
    Detect and encode the first face found in an image array.
    Returns the encoding (128-d vector) or None if no face found.
    """
    locations = face_recognition.face_locations(image_array, model="hog")
    if not locations:
        return None, "No face detected in the image."

    encodings = face_recognition.face_encodings(image_array, locations)
    if not encodings:
        return None, "Could not encode detected face."

    return encodings[0], None


def encode_face_from_bytes(image_bytes: bytes):
    """
    Encode a face from raw image bytes (e.g., downloaded from a URL).
    Returns (encoding, error_message).
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)
        return encode_face_from_array(image_array)
    except Exception as e:
        return None, str(e)


def find_best_match(submitted_encoding: np.ndarray, students: list, tolerance: float = 0.55):
    """
    Compare a submitted face encoding against a list of student encodings.

    Args:
        submitted_encoding: 128-d face encoding of the captured photo
        students: list of dicts with keys: id, name, roll_number, class, encoding
        tolerance: max face distance to count as a match (lower = stricter)

    Returns:
        (best_match_dict, confidence_percent) or (None, 0)
    """
    if not students:
        return None, 0.0

    known_encodings = [s["encoding"] for s in students]
    distances = face_recognition.face_distance(known_encodings, submitted_encoding)

    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])

    if best_distance <= tolerance:
        confidence = round((1.0 - best_distance) * 100, 1)
        return students[best_idx], confidence

    return None, 0.0
