#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# push_to_github.sh
# Run this script ONCE from inside the zoho-face-recognition folder.
# It creates the GitHub repo and pushes all code automatically.
#
# Usage:
#   cd path/to/zoho-face-recognition
#   chmod +x push_to_github.sh
#   ./push_to_github.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # Exit on any error

# ── Config (pre-filled) ───────────────────────────────────────────────────────
GITHUB_USERNAME="aristoraj"
REPO_NAME="zoho-face-recognition"
REPO_DESC="Real-time face recognition attendance module for Zoho Creator with eye blink liveness detection"
VISIBILITY="public"   # Change to "private" if you want

# ── Ask for PAT securely (not echoed to terminal) ─────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Zoho Face Recognition — GitHub Push Script            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "→ GitHub username : $GITHUB_USERNAME"
echo "→ Repository      : $REPO_NAME ($VISIBILITY)"
echo ""
read -s -p "Paste your GitHub Personal Access Token (input hidden): " GITHUB_PAT
echo ""

if [ -z "$GITHUB_PAT" ]; then
  echo "❌ No token provided. Exiting."
  exit 1
fi

# ── Step 1: Create the repo on GitHub ────────────────────────────────────────
echo ""
echo "▶ Step 1/4 — Creating GitHub repository..."

HTTP_STATUS=$(curl -s -o /tmp/gh_create_resp.json -w "%{http_code}" \
  -X POST \
  -H "Authorization: token $GITHUB_PAT" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d "{
    \"name\": \"$REPO_NAME\",
    \"description\": \"$REPO_DESC\",
    \"private\": $([ \"$VISIBILITY\" = \"private\" ] && echo true || echo false),
    \"auto_init\": false
  }")

if [ "$HTTP_STATUS" = "201" ]; then
  echo "   ✓ Repository created: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
elif [ "$HTTP_STATUS" = "422" ]; then
  echo "   ℹ  Repository already exists — continuing with push..."
else
  echo "   ❌ Failed to create repo (HTTP $HTTP_STATUS)"
  cat /tmp/gh_create_resp.json
  exit 1
fi

# ── Step 2: Init git ──────────────────────────────────────────────────────────
echo ""
echo "▶ Step 2/4 — Initialising git..."

git init -q
git config user.name  "$GITHUB_USERNAME"
git config user.email "$GITHUB_USERNAME@users.noreply.github.com"

# ── Step 3: Commit all files ──────────────────────────────────────────────────
echo ""
echo "▶ Step 3/4 — Staging and committing files..."

git add .
git commit -q -m "feat: initial commit — Zoho Creator face recognition attendance module

- Flask backend with face_recognition (dlib) for face matching
- Zoho Creator API integration: fetch student photos, post attendance
- MediaPipe Face Mesh frontend with Eye Aspect Ratio blink detection
- Docker + Render deployment config (render.yaml)
- In-memory student encoding cache (1hr TTL)
- Full environment variable driven configuration"

echo "   ✓ Committed $(git diff --cached --name-only HEAD~1 2>/dev/null | wc -l || echo 'all') files"

# ── Step 4: Push to GitHub ────────────────────────────────────────────────────
echo ""
echo "▶ Step 4/4 — Pushing to GitHub..."

REMOTE_URL="https://$GITHUB_PAT@github.com/$GITHUB_USERNAME/$REPO_NAME.git"

git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"
git push -u origin main -q

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   ✓  All done! Your repository is live.                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  GitHub repo : https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo ""
echo "  Next steps:"
echo "  1. Go to https://render.com → New → Web Service"
echo "  2. Connect this GitHub repo"
echo "  3. Render auto-detects render.yaml"
echo "  4. Set your 4 secret env vars in Render dashboard:"
echo "     → ZOHO_CLIENT_ID"
echo "     → ZOHO_CLIENT_SECRET"
echo "     → ZOHO_REFRESH_TOKEN"
echo "     → ZOHO_ACCOUNT_OWNER  (your Zoho username)"
echo "     → SECRET_KEY          (any random 32-char string)"
echo "  5. Click Deploy — first build ~15 min (dlib compiles)"
echo "  6. Embed in Zoho Creator with:"
echo "     <iframe src=\"https://YOUR-APP.onrender.com\" allow=\"camera\" />"
echo ""

# Clean up PAT from git remote (replace with clean URL)
git remote set-url origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo "  (PAT removed from git remote config ✓)"
echo ""
