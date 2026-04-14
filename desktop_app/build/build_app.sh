#!/usr/bin/env bash
# ============================================================================
# build_app.sh — Bundle Outlier desktop app into a macOS .app via PyInstaller
# ============================================================================
# Usage: ./desktop_app/build/build_app.sh
# Output: desktop_app/dist/Outlier.app
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DESKTOP_DIR="$PROJECT_ROOT/desktop_app"
DIST_DIR="$DESKTOP_DIR/dist"
BUILD_DIR="$DESKTOP_DIR/pyinstaller_build"
ENTRY_POINT="$DESKTOP_DIR/backend/server.py"
FRONTEND_DIR="$DESKTOP_DIR/frontend"
OUTLIER_PKG="$PROJECT_ROOT/outlier"
ICON_PATH="$SCRIPT_DIR/icon.icns"

APP_NAME="Outlier"

echo "============================================"
echo "  Outlier macOS App Builder"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Preflight checks
# ---------------------------------------------------------------------------
if ! command -v pyinstaller &>/dev/null; then
    echo "[ERROR] PyInstaller not found. Install it:"
    echo "        pip install pyinstaller"
    exit 1
fi

if [ ! -f "$ENTRY_POINT" ]; then
    echo "[ERROR] Entry point not found: $ENTRY_POINT"
    exit 1
fi

if [ ! -f "$FRONTEND_DIR/index.html" ]; then
    echo "[ERROR] Frontend not found: $FRONTEND_DIR/index.html"
    exit 1
fi

echo "[1/5] Preflight checks passed."

# ---------------------------------------------------------------------------
# 2. Clean previous build artifacts
# ---------------------------------------------------------------------------
echo "[2/5] Cleaning previous build artifacts..."
rm -rf "$DIST_DIR/${APP_NAME}.app" "$BUILD_DIR"

# ---------------------------------------------------------------------------
# 3. Determine icon flag
# ---------------------------------------------------------------------------
ICON_FLAG=""
if [ -f "$ICON_PATH" ]; then
    ICON_FLAG="--icon=$ICON_PATH"
    echo "[3/5] Using icon: $ICON_PATH"
else
    echo "[3/5] No .icns icon found — using default macOS icon."
    echo "       (Place icon.icns in desktop_app/build/ to customize.)"
fi

# ---------------------------------------------------------------------------
# 4. Run PyInstaller
# ---------------------------------------------------------------------------
echo "[4/5] Running PyInstaller..."

# Build a one-folder .app bundle for macOS
pyinstaller \
    --name "$APP_NAME" \
    --windowed \
    --noconfirm \
    --distpath "$DIST_DIR" \
    --workpath "$BUILD_DIR" \
    --specpath "$BUILD_DIR" \
    --add-data "$FRONTEND_DIR:frontend" \
    --add-data "$OUTLIER_PKG:outlier" \
    --hidden-import uvicorn \
    --hidden-import uvicorn.logging \
    --hidden-import uvicorn.loops \
    --hidden-import uvicorn.loops.auto \
    --hidden-import uvicorn.protocols \
    --hidden-import uvicorn.protocols.http \
    --hidden-import uvicorn.protocols.http.auto \
    --hidden-import uvicorn.protocols.websockets \
    --hidden-import uvicorn.protocols.websockets.auto \
    --hidden-import uvicorn.lifespan \
    --hidden-import uvicorn.lifespan.on \
    --hidden-import fastapi \
    --hidden-import starlette \
    $ICON_FLAG \
    "$ENTRY_POINT"

# ---------------------------------------------------------------------------
# 5. Verify output
# ---------------------------------------------------------------------------
if [ -d "$DIST_DIR/${APP_NAME}.app" ]; then
    echo ""
    echo "[5/5] Build succeeded!"
    echo "       Output: $DIST_DIR/${APP_NAME}.app"
    echo ""
    echo "  Next steps:"
    echo "    1. ./desktop_app/build/sign.sh       # Code-sign the .app"
    echo "    2. ./desktop_app/build/notarize.sh    # Notarize with Apple"
    echo "    3. ./desktop_app/build/create_dmg.sh  # Package as .dmg"
else
    echo ""
    echo "[ERROR] Build failed — $DIST_DIR/${APP_NAME}.app not found."
    exit 1
fi
