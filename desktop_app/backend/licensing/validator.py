"""
LicenseValidator — offline-first license validation for the desktop app.

Check order:
1. Local license file (~/.outlier/license.key)
2. If local file valid & not expired  -> ALLOW
3. If local file missing or expired   -> attempt online validation (placeholder)
4. Grace period: 30 days offline after last successful online check
5. On revoke: switch to read-only mode
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Union

import jwt

from .keys import (
    GRACE_PERIOD_DAYS,
    load_public_key,
    validate_license_key,
)


class LicenseStatus:
    """Immutable result of a license check."""

    __slots__ = ("valid", "tier", "mode", "message", "payload", "days_remaining")

    def __init__(
        self,
        valid: bool,
        tier: str = "free",
        mode: str = "full",
        message: str = "",
        payload: Optional[dict] = None,
        days_remaining: Optional[int] = None,
    ):
        self.valid = valid
        self.tier = tier
        self.mode = mode  # "full" | "read_only" | "expired"
        self.message = message
        self.payload = payload or {}
        self.days_remaining = days_remaining

    def __repr__(self) -> str:
        return (
            f"LicenseStatus(valid={self.valid}, tier={self.tier!r}, "
            f"mode={self.mode!r}, days_remaining={self.days_remaining})"
        )


# Default paths
DEFAULT_LICENSE_PATH = Path.home() / ".outlier" / "license.key"
DEFAULT_META_PATH = Path.home() / ".outlier" / "license_meta.json"

# Placeholder online validation URL (replace with real endpoint)
ONLINE_VALIDATION_URL = "https://api.outlier-ai.com/v1/license/validate"


class LicenseValidator:
    """Offline-first license validator.

    Parameters
    ----------
    license_path : path to the local license.key file
    meta_path    : path to license_meta.json (stores last-validated timestamp)
    public_key_path : path to the Ed25519 public key PEM
    """

    def __init__(
        self,
        license_path: Optional[Union[str, Path]] = None,
        meta_path: Optional[Union[str, Path]] = None,
        public_key_path: Optional[Union[str, Path]] = None,
    ):
        self.license_path = Path(license_path) if license_path else DEFAULT_LICENSE_PATH
        self.meta_path = Path(meta_path) if meta_path else DEFAULT_META_PATH
        self.public_key_path = public_key_path  # None → use test key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> LicenseStatus:
        """Run the full validation pipeline and return a LicenseStatus."""
        # Step 1: try local file
        token = self._read_local_token()
        if token is None:
            # No local license — try online
            return self._attempt_online_validation()

        # Step 2: cryptographic verification
        try:
            payload = validate_license_key(
                token, public_key_path=self.public_key_path
            )
        except jwt.ExpiredSignatureError:
            return self._handle_expired()
        except jwt.InvalidTokenError as exc:
            return LicenseStatus(
                valid=False,
                mode="expired",
                message=f"Invalid license: {exc}",
            )

        # Step 3: check revocation flag in meta
        if self._is_revoked():
            return LicenseStatus(
                valid=True,
                tier=payload.get("tier", "free"),
                mode="read_only",
                message="License revoked — read-only mode.",
                payload=payload,
            )

        # Step 4: record successful local validation time
        self._save_meta({"last_validated": int(time.time())})

        days_remaining = max(
            0, (payload["exp"] - int(time.time())) // 86400
        )

        return LicenseStatus(
            valid=True,
            tier=payload.get("tier", "free"),
            mode="full",
            message="License valid.",
            payload=payload,
            days_remaining=days_remaining,
        )

    def revoke(self) -> None:
        """Mark the local license as revoked (read-only mode)."""
        meta = self._load_meta()
        meta["revoked"] = True
        self._save_meta(meta)

    def activate(self, token: str) -> LicenseStatus:
        """Write a new license token to the local file and validate it."""
        self.license_path.parent.mkdir(parents=True, exist_ok=True)
        self.license_path.write_text(token, encoding="utf-8")
        # Clear revocation
        meta = self._load_meta()
        meta.pop("revoked", None)
        meta["last_validated"] = int(time.time())
        self._save_meta(meta)
        return self.check()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_local_token(self) -> Optional[str]:
        if self.license_path.exists():
            return self.license_path.read_text(encoding="utf-8").strip()
        return None

    def _load_meta(self) -> dict:
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_meta(self, meta: dict) -> None:
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def _is_revoked(self) -> bool:
        return self._load_meta().get("revoked", False)

    def _handle_expired(self) -> LicenseStatus:
        """Check grace period after token expiry."""
        meta = self._load_meta()
        last_validated = meta.get("last_validated", 0)
        grace_end = last_validated + GRACE_PERIOD_DAYS * 86400
        now = int(time.time())

        if now < grace_end:
            days_left = max(0, (grace_end - now) // 86400)
            return LicenseStatus(
                valid=True,
                tier="free",  # degrade tier during grace
                mode="full",
                message=f"License expired — grace period active ({days_left} days left).",
                days_remaining=days_left,
            )

        return LicenseStatus(
            valid=False,
            mode="expired",
            message="License expired and grace period elapsed.",
        )

    def _attempt_online_validation(self) -> LicenseStatus:
        """Placeholder for online license validation.

        In production this would POST to ONLINE_VALIDATION_URL with device
        fingerprint and receive a fresh token.  For now it returns a
        'no license' status.
        """
        # TODO: implement actual HTTP call when backend is deployed
        # import httpx
        # resp = httpx.post(ONLINE_VALIDATION_URL, json={...})
        return LicenseStatus(
            valid=False,
            mode="expired",
            message=(
                "No local license found. "
                "Purchase at https://outlier-ai.com/pricing or enter a key."
            ),
        )
