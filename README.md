# Zoho Creator — Face Recognition Attendance Module

A real-time face verification system that integrates with **Zoho Creator**. It pulls student photos from your Creator database, performs live face matching with **liveness detection** (eye blink verification), and automatically posts attendance records.

---

## Architecture

```
Browser (Webcam)
   │  face-api.js / MediaPipe Face Mesh
   │  → Eye Aspect Ratio (EAR) blink detection
   │  → Capture frame on confirmed blink
   │
   ▼ POST /api/verify (base64 image)
Flask Backend (Render)
   │  face_recognition + dlib
   │  → Fetch student photos from Zoho Creator API
   │  → Encode + compare faces
   │  → If match: POST attendance to Zoho Creator
   │
   ▼
Zoho Creator
   → Student_Database report (photos)
   → Attendance form (new record)
```

---

## Features

- **Real-time face detection** using MediaPipe Face Mesh (468 landmarks, runs in browser)
- **Liveness detection** via Eye Aspect Ratio (EAR) — requires the user to blink twice
- **Face recognition** using `face_recognition` (dlib) — 128-d face encodings
- **Smart caching** — student photos fetched and encoded once per hour (configurable)
- **Zoho Creator API** integration for both reading students and writing attendance
- **Embeddable** — works as an iframe inside a Zoho Creator page/URL widget

---

## Project Structure

```
zoho-face-recognition/
├── app.py              # Flask app + API routes
├── zoho_api.py         # Zoho Creator API client (OAuth, students, attendance)
├── face_utils.py       # Face encoding, comparison, caching
├── config.py           # All config loaded from environment variables
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker build (dlib compiled from source)
├── render.yaml         # Render deployment config (Infrastructure as Code)
├── .env.example        # Environment variable template (copy → .env locally)
├── .gitignore
└── static/
    └── index.html      # Webcam UI with MediaPipe blink detection
```

---

## Step 1 — Zoho Creator Setup

### 1a. Your Student Database form must have:
| Field Display Name | Field Link Name      | Type              |
|--------------------|----------------------|-------------------|
| Student ID         | `Student_ID`         | Text / Auto-Number|
| Name               | `Name`               | Text              |
| Photo              | `Photo`              | Image / File      |
| Roll Number        | `Roll_Number`        | Text (optional)   |
| Class              | `Class`              | Text (optional)   |

Create a **Report** that includes all students. Note its **link name** (e.g., `All_Students`).

### 1b. Your Attendance form must have:
| Field Display Name | Field Link Name      | Type     |
|--------------------|----------------------|----------|
| Student ID         | `Student_ID`         | Text     |
| Student Name       | `Student_Name`       | Text     |
| Date               | `Date`               | Date     |
| Time               | `Time`               | Time     |
| Status             | `Status`             | Text     |
| Verification Type  | `Verification_Type`  | Text     |

> **Field link names** can be viewed in Zoho Creator → Form → Settings → field link name column.

---

## Step 2 — Zoho OAuth Credentials

1. Go to **https://api-console.zoho.com**
2. Create a **Self Client** application
3. Generate a token with these scopes:
   ```
   ZohoCreator.report.READ,ZohoCreator.form.CREATE
   ```
4. Use the generated code to get a **refresh token** via Postman or curl:

```bash
curl -X POST "https://accounts.zoho.com/oauth/v2/token" \
  -d "code=YOUR_CODE" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "redirect_uri=https://www.zoho.com" \
  -d "grant_type=authorization_code"
```

Save the `refresh_token` from the response.

---

## Step 3 — GitHub Setup

```bash
# Create a new repo on github.com, then:
git init
git remote add origin https://github.com/YOUR_USERNAME/zoho-face-recognition.git
git add .
git commit -m "Initial commit: Zoho face recognition attendance module"
git push -u origin main
```

---

## Step 4 — Deploy to Render

1. Go to **https://render.com** → **New → Web Service**
2. Connect your GitHub repository
3. Render will auto-detect `render.yaml` — click **Apply**
4. In the Render dashboard, go to **Environment** and set these variables:

| Key                   | Value                        |
|-----------------------|------------------------------|
| `ZOHO_CLIENT_ID`      | From api-console.zoho.com    |
| `ZOHO_CLIENT_SECRET`  | From api-console.zoho.com    |
| `ZOHO_REFRESH_TOKEN`  | From Step 2                  |
| `ZOHO_ACCOUNT_OWNER`  | Your Zoho username           |
| `SECRET_KEY`          | Any random 32-char string    |

> All other env vars have defaults in `render.yaml` — update if your field names differ.

5. Click **Deploy** — first build takes ~15 min (dlib compiles from source).
6. Your service will be live at: `https://zoho-face-recognition.onrender.com`

---

## Step 5 — Embed in Zoho Creator

1. In Zoho Creator, open your app → **Pages** → Add a new page
2. Add a **URL widget** or **HTML widget**
3. Set the URL to your Render service:
   ```
   https://zoho-face-recognition.onrender.com
   ```
4. Or embed via HTML:
   ```html
   <iframe
     src="https://zoho-face-recognition.onrender.com"
     width="100%"
     height="700"
     frameborder="0"
     allow="camera; microphone"
   ></iframe>
   ```

> **Important:** Make sure the `allow="camera"` attribute is present on the iframe, otherwise the browser will block webcam access.

---

## Local Development

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/zoho-face-recognition.git
cd zoho-face-recognition

# Install dlib (Mac/Linux)
pip install cmake dlib face-recognition

# Install other dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your Zoho credentials

# Run
python app.py
# → http://localhost:5000
```

---

## API Reference

### `GET /api/health`
Returns service health and cache status.
```json
{ "status": "ok", "version": "1.0.0", "cache_size": 42, "cache_age_seconds": 300 }
```

### `POST /api/verify`
Verify a captured photo against the student database.

**Request:**
```json
{
  "image": "data:image/jpeg;base64,...",
  "blink_verified": true
}
```

**Response (match):**
```json
{
  "success": true,
  "matched": true,
  "student": { "id": "S001", "name": "Jane Doe", "roll_number": "R21", "class": "10A" },
  "confidence": 87.3,
  "attendance_posted": true,
  "message": "Welcome, Jane Doe! Attendance marked successfully."
}
```

**Response (no match):**
```json
{
  "success": true,
  "matched": false,
  "message": "Face not recognised. Please try again or contact admin."
}
```

### `POST /api/cache/refresh`
Force-refresh the student face encoding cache.

### `GET /api/cache/status`
Returns cache age and number of students loaded.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Render build fails | Check Docker build logs — usually a dlib compilation issue on older Python versions |
| "No students found" | Verify `ZOHO_STUDENT_REPORT` matches the report link name exactly |
| "Token refresh failed" | Zoho refresh tokens expire if unused for 60 days — re-generate |
| Low confidence scores | Lower `FACE_MATCH_TOLERANCE` to 0.5 or ensure student photos are clear, front-facing |
| Blink not detected | Adjust `EAR_CLOSE_THRESH` in `static/index.html` (default: 0.18) |
| Camera blocked in iframe | Add `allow="camera"` attribute to the `<iframe>` tag in Zoho Creator |
| Slow first request | Students are fetched + encoded on first request — subsequent requests use cache |

---

## Security Notes

- Never commit `.env` — it contains OAuth secrets
- Set a strong, random `SECRET_KEY` in Render environment variables
- The Render free tier sleeps after 15 min inactivity — upgrade to Starter for always-on
- Zoho refresh tokens should be rotated periodically

---

## License

MIT — built for integration with Zoho Creator.
#   N e w F a c e A t t e n d a n c e  
 