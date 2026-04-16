"""
Zoho Creator Face Recognition Attendance Module
Flask backend — serves the webcam UI and handles face verification.

Endpoints:
  GET  /                  → Serve the webcam frontend
  GET  /api/health        → Health check
  GET  /api/cache/status  → Cache status info
  POST /api/cache/refresh → Force refresh student face cache
  POST /api/verify        → Verify face + post attendance
"""

import logging
import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from config import PORT, DEBUG, SECRET_KEY, FACE_MATCH_TOLERANCE, CACHE_TTL_SECONDS
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
face_cache = FaceCache(ttl=CACHE_TTL_SECONDS)


# ─── Helper: load students (with cache) ───────────────────────────────────────
def get_students_cached() -> list:
    students = face_cache.get()
    if students is None:
        logger.info("Cache miss — fetching students from Zoho Creator...")
        students = zoho.get_students()
        face_cache.set(students)
    else:
        logger.info(f"Cache hit — {face_cache.size} students (age: {face_cache.age_seconds:.0f}s)")
    return students


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "version": "1.0.0",
        "cache_size": face_cache.size,
        "cache_age_seconds": face_cache.age_seconds,
    })


@app.route("/api/cache/status")
def cache_status():
    return jsonify({
        "students_cached": face_cache.size,
        "age_seconds": face_cache.age_seconds,
        "ttl_seconds": CACHE_TTL_SECONDS,
    })


@app.route("/api/cache/refresh", methods=["POST"])
def cache_refresh():
    face_cache.invalidate()
    students = get_students_cached()
    return jsonify({
        "success": True,
        "students_loaded": len(students),
        "message": f"Cache refreshed. {len(students)} student encodings loaded.",
    })


@app.route("/api/debug/students")
def debug_students():
    """Debug — raw student records to verify field names."""
    try:
        import requests as req
        from config import ZOHO_STUDENT_REPORT, ZOHO_ATTENDANCE_FORM
        token = zoho._refresh_token()
        headers = {"Authorization": f"Zoho-oauthtoken {token}"}

        # Fetch student records
        s_url = f"{zoho._base_url}/report/{ZOHO_STUDENT_REPORT}"
        s_resp = req.get(s_url, headers=headers, params={"from": 1, "limit": 3}, timeout=20)
        s_resp.raise_for_status()
        s_records = s_resp.json().get("data", [])

        # Fetch attendance records to see real field names
        a_url = f"{zoho._base_url}/report/All_Attendances"
        a_resp = req.get(a_url, headers=headers, params={"from": 1, "limit": 3}, timeout=20)
        a_records = a_resp.json().get("data", []) if a_resp.status_code == 200 else []

        return jsonify({
            "student_field_keys": list(s_records[0].keys()) if s_records else [],
            "student_sample": [{k: str(v)[:100] for k, v in r.items()} for r in s_records[:2]],
            "attendance_field_keys": list(a_records[0].keys()) if a_records else [],
            "attendance_sample": [{k: str(v)[:100] for k, v in r.items()} for r in a_records[:2]],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/verify", methods=["POST"])
def verify():
    """
    Verify a captured photo against the student database.

    Expected JSON body:
    {
        "image": "<base64 image string, with or without data URI prefix>",
        "blink_verified": true   ← must be true (set by frontend after blink detected)
    }

    Returns:
    {
        "success": bool,
        "matched": bool,
        "student": { "id", "name", "roll_number", "class" },   ← on match
        "confidence": float,                                     ← on match
        "attendance_posted": bool,                               ← on match
        "message": str
    }
    """
    try:
        data = request.get_json(force=True)

        # ── Validate input ────────────────────────────────────────────────────
        if not data:
            return jsonify({"success": False, "error": "Empty request body."}), 400

        if "image" not in data:
            return jsonify({"success": False, "error": "Missing 'image' field."}), 400

        if not data.get("blink_verified", False):
            return jsonify({
                "success": False,
                "error": "Liveness check failed. Please blink naturally in front of the camera.",
            }), 400

        # ── Decode and encode submitted face ──────────────────────────────────
        try:
            image_array = decode_base64_image(data["image"])
        except Exception as e:
            return jsonify({"success": False, "error": f"Image decode failed: {e}"}), 400

        submitted_encoding, err = encode_face_from_array(image_array)
        if err:
            return jsonify({"success": False, "error": err}), 422
        if submitted_encoding is None:
            return jsonify({"success": False, "error": "Could not generate face embedding. Please try again."}), 422

        # ── Load student encodings (cached) ───────────────────────────────────
        students = get_students_cached()
        if not students:
            return jsonify({
                "success": False,
                "error": "No students with face photos found in the database.",
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

        logger.info(f"Match found: {best_match['name']} ({confidence}% confidence)")

        # ── Post attendance ───────────────────────────────────────────────────
        att_result = zoho.post_attendance(
            student_id=best_match["id"],
            student_name=best_match["name"],
            verification_type="face_blink_verified",
        )

        return jsonify({
            "success": True,
            "matched": True,
            "student": {
                "id": best_match["id"],
                "name": best_match["name"],
                "roll_number": best_match.get("roll_number", ""),
                "class": best_match.get("class", ""),
            },
            "confidence": confidence,
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


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
