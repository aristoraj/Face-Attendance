"""
Zoho Creator API Client.
Handles OAuth token refresh, fetching student records with photos,
and posting attendance records.
"""

import logging
import requests
from datetime import datetime
from config import (
    ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN,
    ZOHO_ACCOUNT_OWNER, ZOHO_APP_NAME, ZOHO_STUDENT_REPORT,
    ZOHO_ATTENDANCE_FORM, ZOHO_DATA_CENTER,
    FIELD_STUDENT_ID, FIELD_STUDENT_NUMBER, FIELD_STUDENT_NAME, FIELD_STUDENT_PHOTO,
    FIELD_ATT_STUDENT, FIELD_ATT_DATE, FIELD_ATT_STATUS,
)
from face_utils import encode_face_from_bytes

logger = logging.getLogger(__name__)


class ZohoCreatorAPI:
    """Client for Zoho Creator REST API v2."""

    BASE_URL_TEMPLATE = "https://creator.zoho.{dc}/api/v2/{owner}/{app}"
    TOKEN_URL_TEMPLATE = "https://accounts.zoho.{dc}/oauth/v2/token"

    def __init__(self):
        self._access_token: str | None = None
        self._base_url = self.BASE_URL_TEMPLATE.format(
            dc=ZOHO_DATA_CENTER,
            owner=ZOHO_ACCOUNT_OWNER,
            app=ZOHO_APP_NAME,
        )
        self._token_url = self.TOKEN_URL_TEMPLATE.format(dc=ZOHO_DATA_CENTER)

    # ─── Auth ──────────────────────────────────────────────────────────────────

    def _refresh_token(self) -> str:
        """Exchange refresh token for a fresh access token."""
        resp = requests.post(
            self._token_url,
            params={
                "refresh_token": ZOHO_REFRESH_TOKEN,
                "client_id": ZOHO_CLIENT_ID,
                "client_secret": ZOHO_CLIENT_SECRET,
                "grant_type": "refresh_token",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if "access_token" not in data:
            raise RuntimeError(f"Token refresh failed: {data}")
        self._access_token = data["access_token"]
        logger.debug("Zoho access token refreshed.")
        return self._access_token

    def _headers(self) -> dict:
        token = self._refresh_token()
        return {
            "Authorization": f"Zoho-oauthtoken {token}",
            "Content-Type": "application/json",
        }

    # ─── Students ──────────────────────────────────────────────────────────────

    def get_students(self) -> list[dict]:
        """
        Fetch all student records from Zoho Creator, encode their photos,
        and return a list of student dicts with face encodings.
        """
        url = f"{self._base_url}/report/{ZOHO_STUDENT_REPORT}"
        students = []
        page_start = 1
        page_size = 200

        logger.info("Fetching students from Zoho Creator...")

        while True:
            resp = requests.get(
                url,
                headers=self._headers(),
                params={"from": page_start, "limit": page_size},
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
            records = payload.get("data", [])

            if not records:
                break

            for record in records:
                student = self._process_record(record)
                if student:
                    students.append(student)

            logger.info(f"Fetched {len(records)} records (page start {page_start}); encoded {len(students)} faces so far.")

            if len(records) < page_size:
                break  # Last page
            page_start += page_size

        logger.info(f"Total students with valid face encodings: {len(students)}")
        return students

    def _process_record(self, record: dict) -> dict | None:
        """Parse a raw Zoho Creator record into a student dict with face encoding."""
        # System record ID — always present in every Zoho Creator record
        student_id = record.get("ID") or record.get("id")

        # Zoho Creator Name component fields return a dict — extract display_value
        name_raw = record.get(FIELD_STUDENT_NAME)
        if isinstance(name_raw, dict):
            name = (
                name_raw.get("display_value")
                or f"{name_raw.get('first_name', '')} {name_raw.get('last_name', '')}".strip()
                or "Unknown"
            )
        else:
            name = str(name_raw).strip() if name_raw else "Unknown"

        logger.info(f"Processing student '{name}' (ID: {student_id}), raw record keys: {list(record.keys())}")

        # Photo field — Zoho Creator returns image fields as a dict with a 'url' key
        photo = record.get(FIELD_STUDENT_PHOTO)
        if not photo:
            logger.warning(f"Skipping student '{name}' — photo field '{FIELD_STUDENT_PHOTO}' is missing or empty. Record keys: {list(record.keys())}")
            return None

        logger.info(f"Student '{name}' photo field value: {repr(photo)[:200]}")

        if isinstance(photo, dict):
            photo_url = photo.get("url") or photo.get("value") or photo.get("download_url")
        else:
            photo_url = str(photo)

        if not photo_url:
            logger.warning(f"Skipping student '{name}' — could not extract URL from photo field: {repr(photo)[:200]}")
            return None

        # Zoho Creator sometimes returns a relative URL — make it absolute
        if photo_url.startswith("/"):
            photo_url = f"https://creator.zoho.{ZOHO_DATA_CENTER}{photo_url}"
            logger.info(f"Converted relative photo URL to absolute: {photo_url}")

        logger.info(f"Downloading photo for '{name}' from: {photo_url[:100]}")

        try:
            encoding = self._download_and_encode(photo_url)
        except Exception as e:
            logger.warning(f"Skipping student '{name}': {e}")
            return None

        if encoding is None:
            logger.warning(f"No face embedding generated for '{name}' — photo may lack a clear frontal face.")
            return None

        # Student ID field (e.g. 1001) — used as the lookup display value in Attendance form
        student_number = str(record.get(FIELD_STUDENT_NUMBER, "")).strip()
        logger.info(f"Successfully encoded face for student '{name}' (Student_ID: {student_number}).")
        return {
            "id": student_id,            # Zoho system record ID
            "student_number": student_number,  # e.g. "1001" — used for lookup posting
            "name": name,
            "encoding": encoding,
        }

    def _download_and_encode(self, url: str):
        """Download an image from a Zoho Creator URL and return its face encoding."""
        resp = requests.get(url, headers=self._headers(), timeout=20)
        resp.raise_for_status()

        encoding, err = encode_face_from_bytes(resp.content)
        if err:
            raise ValueError(err)
        return encoding

    # ─── Attendance ────────────────────────────────────────────────────────────

    def post_attendance(
        self,
        student_id: str,
        student_name: str,
        student_number: str = "",
        verification_type: str = "face_blink_verified",
    ) -> dict:
        """
        Post a new attendance record to Zoho Creator.

        Student_ID is a LOOKUP field — Zoho Creator v2 expects the display value
        of the linked record (e.g. "1001"), not a nested {"ID": "..."} object.

        Returns dict with 'success' bool and optional 'data'/'error' keys.
        """
        url = f"{self._base_url}/form/{ZOHO_ATTENDANCE_FORM}"
        now = datetime.now()

        # Use student_number (e.g. "1001") as the lookup display value
        lookup_value = student_number if student_number else student_name
        payload = {
            "data": {
                FIELD_ATT_STUDENT: lookup_value,
                FIELD_ATT_DATE: now.strftime("%d-%b-%Y"),
                FIELD_ATT_STATUS: "Present",   # value must match your Attendance dropdown option exactly
            }
        }

        logger.info(f"Posting attendance — URL: {url}")
        logger.info(f"Payload: {payload}")

        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=15)
            logger.info(f"Zoho attendance response HTTP {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()

            result = resp.json()
            # Zoho Creator returns code 3000 on success; other codes indicate errors
            zoho_code = result.get("code")
            if zoho_code is not None and zoho_code != 3000:
                logger.error(f"Zoho rejected attendance record: code={zoho_code} message={result.get('message', '')} details={result}")
                return {"success": False, "error": f"Zoho error code {zoho_code}: {result.get('message', '')}"}

            logger.info(f"Attendance posted successfully for {student_name} (ID: {student_id}) — Zoho response: {result}")
            return {"success": True, "data": result}

        except requests.HTTPError as e:
            logger.error(f"HTTP error posting attendance: {e} — {e.response.text}")
            return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
        except Exception as e:
            logger.error(f"Unexpected error posting attendance: {e}")
            return {"success": False, "error": str(e)}

    # ─── Utility ───────────────────────────────────────────────────────────────

    def test_connection(self) -> dict:
        """Quick connectivity test — returns org info."""
        try:
            url = f"https://creator.zoho.{ZOHO_DATA_CENTER}/api/v2/meta/app/{ZOHO_APP_NAME}"
            resp = requests.get(url, headers=self._headers(), timeout=10)
            return {"connected": resp.status_code == 200, "status_code": resp.status_code}
        except Exception as e:
            return {"connected": False, "error": str(e)}
