#!/usr/bin/env bash
# ============================================================================
# sign.sh — Code-sign the Outlier.app with a Developer ID certificate
# ============================================================================
# Usage:
#   ./desktop_app/build/sign.sh                          # uses env var or prompts
#   ./desktop_app/build/sign.sh "Developer ID Application: Your Name (TEAMID)"
#   CODESIGN_IDENTITY="Developer ID Application: ..." ./desktop_app/build/sign.sh
#
# For ad-hoc (local testing only):
#   ./desktop_app/build/sign.sh --adhoc
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DESKTOP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_PATH="$DESKTOP_DIR/dist/Outlier.app"
ENTITLEMENTS="$SCRIPT_DIR/entitlements.plist"

echo "============================================"
echo "  Outlier Code Signing"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Verify .app exists
# ---------------------------------------------------------------------------
if [ ! -d "$APP_PATH" ]; then
    echo "[ERROR] Outlier.app not found at: $APP_PATH"
    echo "        Run build_app.sh first."
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Determine signing identity
# ---------------------------------------------------------------------------
ADHOC=false

if [ "${1:-}" = "--adhoc" ]; then
    ADHOC=true
    IDENTITY="-"
    echo "[INFO] Using ad-hoc signature (local testing only — will NOT pass Gatekeeper)."
elif [ -n "${1:-}" ]; then
    IDENTITY="$1"
elif [ -n "${CODESIGN_IDENTITY:-}" ]; then
    IDENTITY="$CODESIGN_IDENTITY"
else
    echo "[ERROR] No signing identity provided."
    echo ""
    echo "  Provide an identity via one of:"
    echo "    1. Command-line argument:"
    echo "         ./sign.sh \"Developer ID Application: Your Name (TEAMID)\""
    echo "    2. Environment variable:"
    echo "         export CODESIGN_IDENTITY=\"Developer ID Application: Your Name (TEAMID)\""
    echo "    3. Ad-hoc (testing only):"
    echo "         ./sign.sh --adhoc"
    echo ""
    echo "  To list available identities:"
    echo "         security find-identity -v -p codesigning"
    exit 1
fi

echo "[1/3] Signing identity: $IDENTITY"

# ---------------------------------------------------------------------------
# 3. Create entitlements if missing
# ---------------------------------------------------------------------------
if [ ! -f "$ENTITLEMENTS" ]; then
    echo "[INFO] Creating default entitlements.plist..."
    cat > "$ENTITLEMENTS" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    <key>com.apple.security.cs.allow-dyld-environment-variables</key>
    <true/>
    <key>com.apple.security.network.client</key>
    <true/>
    <key>com.apple.security.network.server</key>
    <true/>
</dict>
</plist>
PLIST
fi

# ---------------------------------------------------------------------------
# 4. Sign the app (deep — all embedded frameworks & binaries)
# ---------------------------------------------------------------------------
echo "[2/3] Signing Outlier.app..."

ENTITLEMENTS_FLAG=""
if [ "$ADHOC" = false ] && [ -f "$ENTITLEMENTS" ]; then
    ENTITLEMENTS_FLAG="--entitlements $ENTITLEMENTS"
fi

codesign \
    --deep \
    --force \
    --verify \
    --verbose \
    --timestamp \
    --options runtime \
    --sign "$IDENTITY" \
    $ENTITLEMENTS_FLAG \
    "$APP_PATH"

# ---------------------------------------------------------------------------
# 5. Verify the signature
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Verifying signature..."

codesign --verify --deep --strict --verbose=2 "$APP_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "  Code signing SUCCEEDED."
    if [ "$ADHOC" = true ]; then
        echo "  WARNING: Ad-hoc signature — Gatekeeper will reject this."
        echo "           Use a Developer ID certificate for distribution."
    else
        echo "  The app is signed and ready for notarization."
        echo "  Next: ./desktop_app/build/notarize.sh"
    fi
else
    echo ""
    echo "  [ERROR] Signature verification FAILED."
    exit 1
fi
