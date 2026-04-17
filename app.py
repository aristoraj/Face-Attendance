"""
Zoho Creator Face Recognition Attendance Module
Flask backend — serves the webcam UI and handles face verification.

Endpoints:
  GET  /                    → Serve the webcam frontend
  GET  /api/health          → Health check (also used by keepalive ping)
  GET  /api/cache/status    → Cache status info
  POST /api/cache/refresh   → Force refresh student face cache
  POST /api/verify          → Verify face + post attendance
  GET  /api/debug/students  → Debug raw Zoho records
"""

import logging
import os
import threading
import time

import requests as req
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from config import (
    PORT, DEBUG, SECRET_KEY, FACE_MATCH_TOLERANCE,
    CACHE_TTL_SECONDS, SELF_URL, ZOHO_STUDENT_REPORT, ZOHO_ATTENDANCE_FORM,
)
from face_utils import FaceCache, decode_base64_image, encode_face_from_array, find_best_match
from zoho_api import ZohoCreatorAPI

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
app.secret_key = SECRET_KEY
CORS(app, resources={r"/api/*": {"origins": "*"}})

zoho = ZohoCreatorAPI()

# ─── Per-batch face cache ──────────────────────────────────────────────────────
# Key: batch_id string (or "ALL" for no batch filter).
# Each batch gets its own FaceCache so refreshes are scoped.
_batch_caches: dict[str, FaceCache] = {}
_batch_caches_lock = threading.Lock()


def _get_cache(batch_id: str = None) -> FaceCache:
    key = batch_id or "ALL"
    with _batch_caches_lock:
        if key not in _batch_caches:
            _batch_caches[key] = FaceCache(ttl=CACHE_TTL_SECONDS)
        return _batch_caches[key]


def get_students_cached(batch_id: str = None) -> list:
    cache = _get_cache(batch_id)
    students = cache.get()
    if students is None:
        scope = f"batch {batch_id}" if batch_id else "all students"
        logger.info(f"Cache miss — loading {scope} from Zoho Creator...")
        students = zoho.get_students(batch_id=batch_id)
        cache.set(students)
    else:
        logger.info(f"Cache hit — {cache.size} students (age: {cache.age_seconds:.0f}s)")
    return students


# ─── Always-on keepalive (Render free tier) ───────────────────────────────────
def _keepalive_worker():
    """
    Ping our own /api/health every 14 minutes so Render's free tier doesn't
    spin down the instance. Set SELF_URL env var to activate.
    """
    if not SELF_URL:
        logger.info("SELF_URL not set — keepalive disabled (set it to your Render URL)")
        return

    ping_url = SELF_URL.rstrip("/") + "/api/health"
    logger.info(f"Keepalive started — pinging {ping_url} every 14 min")

    while True:
        time.sleep(14 * 60)   # 14 minutes
        try:
            r = req.get(ping_url, timeout=10)
            logger.info(f"Keepalive ping → HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"Keepalive ping failed: {e}")


_keepalive_thread = threading.Thread(target=_keepalive_worker, daemon=True)
_keepalive_thread.start()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health")
def health():
    total_cached = sum(c.size for c in _batch_caches.values())
    return jsonify({
        "status":           "ok",
        "version":          "2.0.0",
        "total_cached":     total_cached,
        "batch_scopes":     list(_batch_caches.keys()),
        "keepalive_active": bool(SELF_URL),
    })


@app.route("/api/cache/status")
def cache_status():
    status = {}
    for key, cache in _batch_caches.items():
        status[key] = {
            "students_cached": cache.size,
            "age_seconds":     cache.age_seconds,
            "ttl_seconds":     CACHE_TTL_SECONDS,
        }
    return jsonify(status if status else {"ALL": {"students_cached": 0}})


@app.route("/api/cache/refresh", methods=["POST"])
def cache_refresh():
    batch_id = request.args.get("batch_id") or (request.get_json(silent=True) or {}).get("batch_id")
    try:
        cache = _get_cache(batch_id)
        cache.invalidate()
        students = get_students_cached(batch_id=batch_id)
        return jsonify({
            "success":         True,
            "students_loaded": len(students),
            "batch_id":        batch_id or "ALL",
            "message":         f"Cache refreshed. {len(students)} student encodings loaded.",
        })
    except Exception as e:
        logger.exception("Cache refresh failed")
        msg = str(e)
        if "400" in msg and "oauth" in msg.lower():
            hint = "Zoho OAuth token is invalid or expired — regenerate ZOHO_REFRESH_TOKEN in Render."
        elif "401" in msg:
            hint = "Zoho authentication failed — check your OAuth credentials in Render."
        else:
            hint = msg
        return jsonify({"success": False, "error": hint}), 500



@app.route("/api/verify", methods=["POST"])
def verify():
    """
    Verify a captured photo against the student database.

    Expected JSON body:
    {
        "image":          "<base64 JPEG, with or without data URI prefix>",
        "blink_verified": true,
        "batch_id":       "4445260000003610007",  ← optional, scopes recognition
        "session_id":     "4445260000003999001"   ← optional, links attendance to session
    }
    """
    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({"success": False, "error": "Empty request body."}), 400
        if "image" not in data:
            return jsonify({"success": False, "error": "Missing 'image' field."}), 400
        if not data.get("blink_verified", False):
            return jsonify({
                "success": False,
                "error": "Liveness check failed. Please blink naturally in front of the camera.",
            }), 400

        batch_id   = data.get("batch_id")   or None
        session_id = data.get("session_id") or None

        # ── Decode and encode submitted face ──────────────────────────────────
        try:
            image_array = decode_base64_image(data["image"])
        except Exception as e:
            return jsonify({"success": False, "error": f"Image decode failed: {e}"}), 400

        submitted_encoding, err = encode_face_from_array(image_array)
        if err:
            return jsonify({"success": False, "error": err}), 422
        if submitted_encoding is None:
            return jsonify({
                "success": False,
                "error": "Could not generate face embedding. Please try again.",
            }), 422

        # ── Load student encodings (batch-scoped cache) ───────────────────────
        students = get_students_cached(batch_id=batch_id)
        if not students:
            return jsonify({
                "success": False,
                "error":   "No students with face photos found in this batch.",
            }), 404

        # ── Match ─────────────────────────────────────────────────────────────
        best_match, confidence = find_best_match(
            submitted_encoding, students, tolerance=FACE_MATCH_TOLERANCE
        )

        if not best_match:
            logger.info("No face match found.")
            return jsonify({
                "success": True,
                "matched": False,
                "message": "Face not recognised. Please try again or contact admin.",
            })

        logger.info(f"Match: {best_match['name']} ({confidence}% confidence)")

        # ── Duplicate attendance guard ─────────────────────────────────────────
        from datetime import datetime
        today_str = datetime.now().strftime("%d-%b-%Y")
        if zoho.check_duplicate_attendance(best_match["id"], today_str, session_id=session_id):
            logger.info(f"Duplicate attendance blocked for {best_match['name']}")
            return jsonify({
                "success":    True,
                "matched":    True,
                "duplicate":  True,
                "student": {
                    "id":   best_match["id"],
                    "name": best_match["name"],
                },
                "confidence":         confidence,
                "attendance_posted":  False,
                "message": f"{best_match['name']} is already marked present today.",
            })

        # ── Post attendance ───────────────────────────────────────────────────
        att_result = zoho.post_attendance(
            student_id=best_match["id"],
            student_name=best_match["name"],
            verification_type="face_blink_verified",
            session_id=session_id,
        )

        return jsonify({
            "success":    True,
            "matched":    True,
            "duplicate":  False,
            "student": {
                "id":           best_match["id"],
                "name":         best_match["name"],
                "roll_number":  best_match.get("student_number", ""),
            },
            "confidence":        confidence,
            "attendance_posted": att_result.get("success", False),
            "message": (
                f"Welcome, {best_match['name']}! Attendance marked successfully."
                if att_result.get("success")
                else f"Matched as {best_match['name']} but attendance posting failed. Please contact admin."
            ),
        })

    except Exception as e:
        logger.exception("Unexpected error in /api/verify")
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500



@app.route("/api/debug/students")
def debug_students():
    """Debug — raw student records to verify field names."""
    try:
        token = zoho._refresh_token()
        headers = {"Authorization": f"Zoho-oauthtoken {token}"}

        s_url = f"{zoho._base_url}/report/{ZOHO_STUDENT_REPORT}"
        s_resp = req.get(s_url, headers=headers, params={"from": 1, "limit": 3}, timeout=20)
        s_resp.raise_for_status()
        s_records = s_resp.json().get("data", [])

        a_url = f"{zoho._base_url}/report/All_Attendances"
        a_resp = req.get(a_url, headers=headers, params={"from": 1, "limit": 3}, timeout=20)
        a_records = a_resp.json().get("data", []) if a_resp.status_code == 200 else []

        return jsonify({
            "student_field_keys":    list(s_records[0].keys()) if s_records else [],
            "student_sample":        [{k: str(v)[:100] for k, v in r.items()} for r in s_records[:2]],
            "attendance_field_keys": list(a_records[0].keys()) if a_records else [],
            "attendance_sample":     [{k: str(v)[:100] for k, v in r.items()} for r in a_records[:2]],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
