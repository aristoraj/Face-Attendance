"""
Zoho Creator API Client.
Handles OAuth token refresh, fetching student records with photos,
and posting attendance records.

Changes:
  - Batch-scoped student loading (batch_id filter)
  - Pre-computed embedding: reads Face_Embedding field before downloading photo
  - save_embedding(): writes 512-d vector back to the student record
  - post_attendance(): optional session_id context
  - check_duplicate_attendance(): blocks same student being marked twice on same date
"""

import logging
import os
import time
import requests
from datetime import datetime

from config import (
    ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN,
    ZOHO_ACCOUNT_OWNER, ZOHO_APP_NAME, ZOHO_DATA_CENTER,
    ZOHO_STUDENT_REPORT, ZOHO_ATTENDANCE_FORM, ZOHO_ATTENDANCE_REPORT,
    FIELD_STUDENT_ID, FIELD_STUDENT_NUMBER, FIELD_STUDENT_NAME,
    FIELD_STUDENT_PHOTO, FIELD_STUDENT_BATCH, FIELD_STUDENT_EMBEDDING,
    FIELD_ATT_STUDENT, FIELD_ATT_DATE, FIELD_ATT_STATUS, FIELD_ATT_SESSION,
)
from face_utils import encode_face_from_bytes, embedding_to_json, json_to_embedding

logger = logging.getLogger(__name__)


class ZohoCreatorAPI:
    """Client for Zoho Creator REST API v2."""

    BASE_URL_TEMPLATE  = "https://creator.zoho.{dc}/api/v2/{owner}/{app}"
    TOKEN_URL_TEMPLATE = "https://accounts.zoho.{dc}/oauth/v2/token"

    def __init__(self):
        self._access_token = None
        self._token_expiry = 0.0   # Unix timestamp; 0 means "not yet fetched"
        self._base_url = self.BASE_URL_TEMPLATE.format(
            dc=ZOHO_DATA_CENTER,
            owner=ZOHO_ACCOUNT_OWNER,
            app=ZOHO_APP_NAME,
        )
        self._token_url = self.TOKEN_URL_TEMPLATE.format(dc=ZOHO_DATA_CENTER)

    # ─── Auth ──────────────────────────────────────────────────────────────────

    def _refresh_token(self) -> str:
        """Exchange the refresh token for a new access token. Stores expiry."""
        resp = requests.post(
            self._token_url,
            params={
                # Read from os.environ every time so hot-reload via /admin/reauth works instantly
                "refresh_token": os.environ.get("ZOHO_REFRESH_TOKEN", ZOHO_REFRESH_TOKEN),
                "client_id":     ZOHO_CLIENT_ID,
                "client_secret": ZOHO_CLIENT_SECRET,
                "grant_type":    "refresh_token",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if "access_token" not in data:
            raise RuntimeError(f"Token refresh failed: {data}")
        self._access_token = data["access_token"]
        # Refresh 90 s before actual expiry to cover clock skew and request latency
        self._token_expiry = time.time() + data.get("expires_in", 3600) - 90
        logger.info("Zoho access token refreshed (valid for ~%.0f min).",
                    (self._token_expiry - time.time()) / 60)
        return self._access_token

    def _get_token(self) -> str:
        """
        Return a valid access token, calling Zoho's OAuth endpoint only when
        the cached token is missing or within 90 s of expiry.

        This is the critical fix for the rate-limit error:
          "You have made too many requests continuously."
        Previously _headers() called _refresh_token() on every single API
        request, which fires dozens of token refreshes while loading the student
        cache (one per photo download). Zoho allows ~10 token requests per
        minute per client — loading 18 students exceeded that instantly.
        """
        if self._access_token and time.time() < self._token_expiry:
            return self._access_token
        return self._refresh_token()

    def _headers(self) -> dict:
        token = self._get_token()   # cached — only calls Zoho when token expires
        return {
            "Authorization": f"Zoho-oauthtoken {token}",
            "Content-Type":  "application/json",
        }

    # ─── Students ──────────────────────────────────────────────────────────────

    def get_students(self, batch_id: str = None) -> list[dict]:
        """
        Fetch student records from Zoho Creator, encode face embeddings, and
        return a list of student dicts.

        Args:
            batch_id: Zoho system record ID of the Batch to scope to (optional).
                      If None, loads ALL students.
        """
        url = f"{self._base_url}/report/{ZOHO_STUDENT_REPORT}"
        students = []
        page_start = 1
        page_size = 200

        scope_label = f"batch {batch_id}" if batch_id else "all batches"
        logger.info(f"Fetching students from Zoho Creator ({scope_label})...")

        while True:
            params = {"from": page_start, "limit": page_size}
            resp = requests.get(url, headers=self._headers(), params=params, timeout=30)
            resp.raise_for_status()
            records = resp.json().get("data", [])

            if not records:
                break

            for record in records:
                # ── Batch filter (Python-side — avoids Zoho criteria quirks) ──
                if batch_id:
                    batch_field = record.get(FIELD_STUDENT_BATCH)
                    record_batch_id = None
                    if isinstance(batch_field, dict):
                        record_batch_id = (
                            batch_field.get("ID")
                            or batch_field.get("id")
                            or batch_field.get("display_value")
                        )
                    elif isinstance(batch_field, str):
                        record_batch_id = batch_field
                    if record_batch_id != batch_id:
                        continue

                student = self._process_record(record)
                if student:
                    students.append(student)

            logger.info(
                f"Page {page_start}: {len(records)} records fetched, "
                f"{len(students)} valid face encodings so far."
            )

            if len(records) < page_size:
                break
            page_start += page_size

        logger.info(f"Total students loaded ({scope_label}): {len(students)}")
        return students

    def get_students_list(self, batch_id: str = None) -> list[dict]:
        """
        Lightweight fetch of student names + IDs only (no photo download / encoding).
        Used for the manual attendance dropdown.
        """
        url = f"{self._base_url}/report/{ZOHO_STUDENT_REPORT}"
        students = []
        page_start = 1
        page_size = 200

        while True:
            params = {"from": page_start, "limit": page_size}
            resp = requests.get(url, headers=self._headers(), params=params, timeout=30)
            resp.raise_for_status()
            records = resp.json().get("data", [])

            if not records:
                break

            for record in records:
                if batch_id:
                    batch_field = record.get(FIELD_STUDENT_BATCH)
                    record_batch_id = None
                    if isinstance(batch_field, dict):
                        record_batch_id = (
                            batch_field.get("ID")
                            or batch_field.get("id")
                            or batch_field.get("display_value")
                        )
                    elif isinstance(batch_field, str):
                        record_batch_id = batch_field
                    if record_batch_id != batch_id:
                        continue

                student_id = record.get("ID") or record.get("id")
                name_raw = record.get(FIELD_STUDENT_NAME)
                if isinstance(name_raw, dict):
                    name = (
                        name_raw.get("display_value")
                        or f"{name_raw.get('first_name', '')} {name_raw.get('last_name', '')}".strip()
                        or "Unknown"
                    )
                else:
                    name = str(name_raw).strip() if name_raw else "Unknown"

                students.append({
                    "id":             student_id,
                    "name":           name,
                    "student_number": str(record.get(FIELD_STUDENT_NUMBER, "")),
                })

            if len(records) < page_size:
                break
            page_start += page_size

        return students

    def _process_record(self, record: dict) -> dict | None:
        """Parse a raw Zoho Creator record into a student dict with face encoding."""
        student_id = record.get("ID") or record.get("id")

        # Name component field returns a dict
        name_raw = record.get(FIELD_STUDENT_NAME)
        if isinstance(name_raw, dict):
            name = (
                name_raw.get("display_value")
                or f"{name_raw.get('first_name', '')} {name_raw.get('last_name', '')}".strip()
                or "Unknown"
            )
        else:
            name = str(name_raw).strip() if name_raw else "Unknown"

        student_number = str(record.get(FIELD_STUDENT_NUMBER, "")).strip()

        # ── 1. Try pre-computed embedding first (skip photo download) ──────────
        embedding_raw = record.get(FIELD_STUDENT_EMBEDDING, "")
        if embedding_raw and isinstance(embedding_raw, str) and embedding_raw.strip().startswith("["):
            try:
                embedding = json_to_embedding(embedding_raw.strip())
                logger.info(f"Pre-computed embedding loaded for '{name}' ({student_number})")
                return {
                    "id":             student_id,
                    "student_number": student_number,
                    "name":           name,
                    "encoding":       embedding,
                }
            except Exception as e:
                logger.warning(f"Bad stored embedding for '{name}': {e} — falling back to photo")

        # ── 2. Fallback: download photo and encode ─────────────────────────────
        photo = record.get(FIELD_STUDENT_PHOTO)
        if not photo:
            logger.warning(f"Skipping '{name}' — no photo and no stored embedding.")
            return None

        if isinstance(photo, dict):
            photo_url = photo.get("url") or photo.get("value") or photo.get("download_url")
        else:
            photo_url = str(photo)

        if not photo_url:
            logger.warning(f"Skipping '{name}' — could not extract photo URL: {repr(photo)[:100]}")
            return None

        if photo_url.startswith("/"):
            photo_url = f"https://creator.zoho.{ZOHO_DATA_CENTER}{photo_url}"

        try:
            encoding = self._download_and_encode(photo_url)
        except Exception as e:
            logger.warning(f"Skipping '{name}': {e}")
            return None

        if encoding is None:
            logger.warning(f"No face in photo for '{name}'.")
            return None

        logger.info(f"Encoded face from photo for '{name}' ({student_number})")

        # ── 3. Save embedding back to Zoho Creator for next cache load ─────────
        try:
            self.save_embedding(student_id, encoding)
        except Exception as e:
            logger.warning(f"Could not save embedding for '{name}': {e} (non-fatal)")

        return {
            "id":             student_id,
            "student_number": student_number,
            "name":           name,
            "encoding":       encoding,
        }

    def _download_and_encode(self, url: str):
        resp = requests.get(url, headers=self._headers(), timeout=20)
        resp.raise_for_status()
        encoding, err = encode_face_from_bytes(resp.content)
        if err:
            raise ValueError(err)
        return encoding

    def save_embedding(self, student_system_id: str, embedding) -> None:
        """
        Write the 512-d embedding to the Face_Embedding field on the Student record.
        Future cache loads will read from here — no photo download needed.
        NOTE: Requires a Multi Line field named 'Face_Embedding' on Student Database form.
        """
        url = f"{self._base_url}/report/{ZOHO_STUDENT_REPORT}/{student_system_id}"
        payload = {"data": {FIELD_STUDENT_EMBEDDING: embedding_to_json(embedding)}}
        resp = requests.patch(url, headers=self._headers(), json=payload, timeout=15)
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"PATCH embedding failed HTTP {resp.status_code}: {resp.text[:200]}"
            )
        logger.info(f"Saved embedding for student {student_system_id}")

    # ─── Duplicate Attendance Guard ────────────────────────────────────────────

    def check_duplicate_attendance(
        self,
        student_id: str,
        date_str:   str,
        session_id: str = None,
    ) -> bool:
        """
        Returns True if attendance already exists for this student on date_str.
        If session_id is provided, only blocks duplicates within the same session.
        """
        try:
            url = f"{self._base_url}/report/{ZOHO_ATTENDANCE_REPORT}"
            criteria = f'({FIELD_ATT_DATE}=="{date_str}")'
            resp = requests.get(
                url,
                headers=self._headers(),
                params={"criteria": criteria, "limit": 200},
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning(f"Duplicate check query HTTP {resp.status_code} — allowing")
                return False

            records = resp.json().get("data", [])
            for rec in records:
                rec_student = rec.get(FIELD_ATT_STUDENT)
                if isinstance(rec_student, dict):
                    rec_sid = (
                        rec_student.get("ID")
                        or rec_student.get("display_value", "")
                    )
                else:
                    rec_sid = str(rec_student or "")

                if rec_sid == student_id:
                    if session_id:
                        # Scoped check — only duplicate if same session
                        rec_session = rec.get(FIELD_ATT_SESSION)
                        if isinstance(rec_session, dict):
                            rec_session_id = (
                                rec_session.get("ID")
                                or rec_session.get("display_value", "")
                            )
                        else:
                            rec_session_id = str(rec_session or "")
                        if rec_session_id == session_id:
                            return True
                    else:
                        return True

            return False

        except Exception as e:
            logger.warning(f"Duplicate check error: {e} — allowing attendance")
            return False

    # ─── Attendance ────────────────────────────────────────────────────────────

    def post_attendance(
        self,
        student_id:        str,
        student_name:      str,
        verification_type: str = "face_blink_verified",
        session_id:        str = None,
    ) -> dict:
        """
        Post a new attendance record to Zoho Creator.
        Optionally links to a Session record via session_id (system record ID).
        """
        url = f"{self._base_url}/form/{ZOHO_ATTENDANCE_FORM}"
        now = datetime.now()

        data_payload = {
            FIELD_ATT_STUDENT: student_id,
            FIELD_ATT_DATE:    now.strftime("%d-%b-%Y"),
            FIELD_ATT_STATUS:  "Present",
        }

        if session_id:
            data_payload[FIELD_ATT_SESSION] = session_id

        payload = {"data": data_payload}
        logger.info(f"Posting attendance — {student_name} | payload: {payload}")

        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=15)
            logger.info(f"Zoho response HTTP {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()

            result = resp.json()
            zoho_code = result.get("code")
            if zoho_code is not None and zoho_code != 3000:
                logger.error(f"Zoho error code={zoho_code}: {result.get('message', '')}")
                return {"success": False, "error": f"Zoho error {zoho_code}: {result.get('message', '')}"}

            logger.info(f"Attendance posted for {student_name} (ID: {student_id})")
            return {"success": True, "data": result}

        except requests.HTTPError as e:
            logger.error(f"HTTP error: {e} — {e.response.text}")
            return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"success": False, "error": str(e)}

    # ─── Utility ───────────────────────────────────────────────────────────────

    def test_connection(self) -> dict:
        try:
            url = f"https://creator.zoho.{ZOHO_DATA_CENTER}/api/v2/meta/app/{ZOHO_APP_NAME}"
            resp = requests.get(url, headers=self._headers(), timeout=10)
            return {"connected": resp.status_code == 200, "status_code": resp.status_code}
        except Exception as e:
            return {"connected": False, "error": str(e)}
