"""
Configuration for Zoho Face Recognition Module.
All values are loaded from environment variables.
Update your Render environment variables to match your Zoho Creator setup.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Zoho OAuth Credentials ───────────────────────────────────────────────────
ZOHO_CLIENT_ID = os.environ.get("ZOHO_CLIENT_ID", "")
ZOHO_CLIENT_SECRET = os.environ.get("ZOHO_CLIENT_SECRET", "")
ZOHO_REFRESH_TOKEN = os.environ.get("ZOHO_REFRESH_TOKEN", "")

# ─── Zoho Creator App Config ──────────────────────────────────────────────────
# The Zoho account username/email owner of the Creator app
ZOHO_ACCOUNT_OWNER = os.environ.get("ZOHO_ACCOUNT_OWNER", "")

# The link name of your Zoho Creator application (from the app URL)
ZOHO_APP_NAME = os.environ.get("ZOHO_APP_NAME", "student_management")

# Link name of the report that lists all students (with photo field)
ZOHO_STUDENT_REPORT = os.environ.get("ZOHO_STUDENT_REPORT", "All_Student_Databases")

# Link name of the attendance form to post records into
ZOHO_ATTENDANCE_FORM = os.environ.get("ZOHO_ATTENDANCE_FORM", "Attendance")

# Zoho data center: com | eu | in | com.au | jp
ZOHO_DATA_CENTER = os.environ.get("ZOHO_DATA_CENTER", "com")

# ─── Zoho Creator Field Mappings ──────────────────────────────────────────────
# Student Database field names (link names, not display names)
# "ID" is the Zoho Creator system record ID — always present, no custom field needed
FIELD_STUDENT_ID   = "ID"        # System record ID (used for lookup posting)
FIELD_STUDENT_NAME = os.environ.get("FIELD_STUDENT_NAME", "Name")
FIELD_STUDENT_PHOTO = os.environ.get("FIELD_STUDENT_PHOTO", "Photo")

# Attendance form field names
# Student is a LOOKUP field — posted as {"Student": {"ID": "<record_id>"}}
FIELD_ATT_STUDENT  = os.environ.get("FIELD_ATT_STUDENT", "Student")   # lookup field
FIELD_ATT_DATE     = os.environ.get("FIELD_ATT_DATE", "Date")
FIELD_ATT_STATUS   = os.environ.get("FIELD_ATT_STATUS", "Attendance") # your dropdown field

# ─── Face Recognition Settings ────────────────────────────────────────────────
# Lower = stricter matching (0.4–0.6 recommended)
FACE_MATCH_TOLERANCE = float(os.environ.get("FACE_MATCH_TOLERANCE", "0.55"))

# Student face cache TTL in seconds (3600 = 1 hour)
# Avoids re-fetching all student photos on every request
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))

# ─── App Settings ─────────────────────────────────────────────────────────────
PORT = int(os.environ.get("PORT", 5000))
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-key-in-production")
