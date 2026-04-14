#!/usr/bin/env bash
# ============================================================================
# notarize.sh — Submit Outlier.app to Apple for notarization
# ============================================================================
# Usage:
#   ./desktop_app/build/notarize.sh
#
# Required environment variables (or pass as arguments):
#   APPLE_ID        — your Apple ID email
#   APP_PASSWORD    — app-specific password (from appleid.apple.com)
#   TEAM_ID         — Apple Developer Team ID
#
# Or pass inline:
#   APPLE_ID=you@example.com APP_PASSWORD=xxxx-xxxx-xxxx-xxxx TEAM_ID=XXXXXXXXXX \
#       ./desktop_app/build/notarize.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DESKTOP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_PATH="$DESKTOP_DIR/dist/Outlier.app"
ZIP_PATH="$DESKTOP_DIR/dist/Outlier.zip"

echo "============================================"
echo "  Outlier Notarization"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Preflight checks
# ---------------------------------------------------------------------------
if [ ! -d "$APP_PATH" ]; then
    echo "[ERROR] Outlier.app not found at: $APP_PATH"
    echo "        Run build_app.sh and sign.sh first."
    exit 1
fi

# Check the app is actually signed (not ad-hoc)
if ! codesign --verify --deep --strict "$APP_PATH" 2>/dev/null; then
    echo "[ERROR] Outlier.app is not properly signed."
    echo "        Run sign.sh with a valid Developer ID certificate first."
    exit 1
fi

# Check credentials
if [ -z "${APPLE_ID:-}" ]; then
    echo "[ERROR] APPLE_ID not set."
    echo "        export APPLE_ID=your@apple-id-email.com"
    exit 1
fi

if [ -z "${APP_PASSWORD:-}" ]; then
    echo "[ERROR] APP_PASSWORD not set."
    echo "        Generate an app-specific password at https://appleid.apple.com"
    echo "        export APP_PASSWORD=xxxx-xxxx-xxxx-xxxx"
    exit 1
fi

if [ -z "${TEAM_ID:-}" ]; then
    echo "[ERROR] TEAM_ID not set."
    echo "        Find your Team ID at https://developer.apple.com/account"
    echo "        export TEAM_ID=XXXXXXXXXX"
    exit 1
fi

echo "[1/4] Preflight checks passed."

# ---------------------------------------------------------------------------
# 2. Create zip for submission
# ---------------------------------------------------------------------------
echo "[2/4] Creating zip archive for submission..."
rm -f "$ZIP_PATH"

# ditto preserves resource forks and extended attributes
ditto -c -k --keepParent "$APP_PATH" "$ZIP_PATH"

echo "       Created: $ZIP_PATH ($(du -h "$ZIP_PATH" | cut -f1))"

# ---------------------------------------------------------------------------
# 3. Submit to Apple notary service
# ---------------------------------------------------------------------------
echo "[3/4] Submitting to Apple notary service..."
echo "       This may take several minutes..."

xcrun notarytool submit "$ZIP_PATH" \
    --apple-id "$APPLE_ID" \
    --password "$APP_PASSWORD" \
    --team-id "$TEAM_ID" \
    --wait \
    --timeout 30m

SUBMIT_EXIT=$?

if [ $SUBMIT_EXIT -ne 0 ]; then
    echo ""
    echo "[ERROR] Notarization submission failed (exit code $SUBMIT_EXIT)."
    echo ""
    echo "  Common causes:"
    echo "    - Invalid Apple ID / app-specific password"
    echo "    - The app was signed with an ad-hoc or self-signed certificate"
    echo "    - Hardened runtime not enabled (sign.sh uses --options runtime)"
    echo ""
    echo "  To check the log of the last submission:"
    echo "    xcrun notarytool log <submission-id> --apple-id \$APPLE_ID --password \$APP_PASSWORD --team-id \$TEAM_ID"
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Staple the notarization ticket to the app
# ---------------------------------------------------------------------------
echo "[4/4] Stapling notarization ticket..."

xcrun stapler staple "$APP_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "  Notarization SUCCEEDED."
    echo "  The app is signed, notarized, and stapled."
    echo ""
    echo "  Next: ./desktop_app/build/create_dmg.sh"
else
    echo ""
    echo "  [ERROR] Stapling failed. The notarization may still be processing."
    echo "          Wait a few minutes and retry:"
    echo "          xcrun stapler staple $APP_PATH"
    exit 1
fi

# Clean up
rm -f "$ZIP_PATH"
echo "  Cleaned up $ZIP_PATH"
