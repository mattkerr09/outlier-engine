#!/usr/bin/env bash
# ============================================================================
# create_dmg.sh — Package Outlier.app into a distributable .dmg
# ============================================================================
# Usage: ./desktop_app/build/create_dmg.sh
# Output: desktop_app/dist/Outlier-Installer.dmg
#
# Uses `create-dmg` if installed (brew install create-dmg) for a polished
# drag-to-install DMG with background and icon layout. Falls back to
# hdiutil for a basic but functional DMG.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DESKTOP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_PATH="$DESKTOP_DIR/dist/Outlier.app"
DMG_PATH="$DESKTOP_DIR/dist/Outlier-Installer.dmg"
VOLUME_NAME="Outlier"
STAGING_DIR="$DESKTOP_DIR/dist/.dmg_staging"

echo "============================================"
echo "  Outlier DMG Creator"
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
# 2. Clean previous DMG
# ---------------------------------------------------------------------------
rm -f "$DMG_PATH"

# ---------------------------------------------------------------------------
# 3. Choose method and create DMG
# ---------------------------------------------------------------------------
if command -v create-dmg &>/dev/null; then
    echo "[INFO] Using create-dmg for polished installer..."
    echo ""

    create-dmg \
        --volname "$VOLUME_NAME" \
        --volicon "$APP_PATH/Contents/Resources/Outlier.icns" 2>/dev/null \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "Outlier.app" 150 200 \
        --app-drop-link 450 200 \
        --hide-extension "Outlier.app" \
        --no-internet-enable \
        "$DMG_PATH" \
        "$APP_PATH" \
    || {
        # create-dmg returns non-zero if icon isn't found — retry without volicon
        echo "[WARN] Retrying without volume icon..."
        rm -f "$DMG_PATH"
        create-dmg \
            --volname "$VOLUME_NAME" \
            --window-pos 200 120 \
            --window-size 600 400 \
            --icon-size 100 \
            --icon "Outlier.app" 150 200 \
            --app-drop-link 450 200 \
            --hide-extension "Outlier.app" \
            --no-internet-enable \
            "$DMG_PATH" \
            "$APP_PATH"
    }
else
    echo "[INFO] create-dmg not found — using hdiutil (basic DMG)."
    echo "       For a polished installer: brew install create-dmg"
    echo ""

    # Create a staging directory with the app and Applications symlink
    rm -rf "$STAGING_DIR"
    mkdir -p "$STAGING_DIR"
    cp -R "$APP_PATH" "$STAGING_DIR/"
    ln -s /Applications "$STAGING_DIR/Applications"

    # Create the DMG
    hdiutil create \
        -volname "$VOLUME_NAME" \
        -srcfolder "$STAGING_DIR" \
        -ov \
        -format UDZO \
        -imagekey zlib-level=9 \
        "$DMG_PATH"

    # Clean up staging
    rm -rf "$STAGING_DIR"
fi

# ---------------------------------------------------------------------------
# 4. Verify output
# ---------------------------------------------------------------------------
if [ -f "$DMG_PATH" ]; then
    DMG_SIZE=$(du -h "$DMG_PATH" | cut -f1)
    echo ""
    echo "  DMG created successfully!"
    echo "  Output: $DMG_PATH ($DMG_SIZE)"
    echo ""
    echo "  Distribution checklist:"
    echo "    [x] .app built with PyInstaller"
    echo "    [ ] Signed with Developer ID (sign.sh)"
    echo "    [ ] Notarized with Apple (notarize.sh)"
    echo "    [x] Packaged as .dmg"
    echo ""
    echo "  To verify the DMG mounts correctly:"
    echo "    open $DMG_PATH"
else
    echo "[ERROR] DMG creation failed."
    exit 1
fi
