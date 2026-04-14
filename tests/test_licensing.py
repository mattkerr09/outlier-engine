"""
Tests for the Outlier licensing system.

Covers:
  - Key generation and validation (Ed25519 + JWT)
  - Expired key rejection
  - Grace period logic
  - Webhook signature verification
  - Full webhook flow (checkout -> license, subscription deleted -> revoke)
  - LicenseValidator offline-first logic
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import jwt
import pytest

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from desktop_app.backend.licensing.keys import (
    GRACE_PERIOD_DAYS,
    VALID_TIERS,
    generate_ed25519_keypair,
    generate_license_key,
    load_private_key,
    load_public_key,
    save_test_keypair,
    validate_license_key,
)
from desktop_app.backend.licensing.validator import LicenseStatus, LicenseValidator
from desktop_app.backend.licensing.webhook import (
    clear_license_store,
    get_license_store,
    verify_stripe_signature,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_license_store():
    """Reset the in-memory webhook license store between tests."""
    clear_license_store()
    yield
    clear_license_store()


@pytest.fixture
def keypair_dir(tmp_path):
    """Generate a fresh keypair in a temp directory."""
    priv_path, pub_path = save_test_keypair(str(tmp_path))
    return priv_path, pub_path


@pytest.fixture
def license_dir(tmp_path):
    """Return paths for license file and meta within a temp directory."""
    lic = tmp_path / "license.key"
    meta = tmp_path / "license_meta.json"
    return lic, meta


# ====================================================================
# 1. Key generation & validation
# ====================================================================

class TestKeyGeneration:
    def test_generate_keypair_returns_pem(self):
        priv_pem, pub_pem = generate_ed25519_keypair()
        assert b"BEGIN PRIVATE KEY" in priv_pem
        assert b"BEGIN PUBLIC KEY" in pub_pem

    def test_save_and_load_keypair(self, keypair_dir):
        priv_path, pub_path = keypair_dir
        priv = load_private_key(priv_path)
        pub = load_public_key(pub_path)
        assert priv is not None
        assert pub is not None

    def test_generate_and_validate_roundtrip(self, keypair_dir):
        priv_path, pub_path = keypair_dir
        token = generate_license_key(
            user_id="matt@outlier.ai",
            tier="pro",
            duration_days=365,
            private_key_path=priv_path,
        )
        payload = validate_license_key(token, public_key_path=pub_path)
        assert payload["user_id"] == "matt@outlier.ai"
        assert payload["tier"] == "pro"
        assert payload["grace_period_days"] == GRACE_PERIOD_DAYS
        assert payload["expires_at"] > payload["issued_at"]

    def test_all_valid_tiers_accepted(self, keypair_dir):
        priv_path, pub_path = keypair_dir
        for tier in VALID_TIERS:
            token = generate_license_key("u1", tier, private_key_path=priv_path)
            p = validate_license_key(token, public_key_path=pub_path)
            assert p["tier"] == tier

    def test_invalid_tier_rejected(self, keypair_dir):
        priv_path, _ = keypair_dir
        with pytest.raises(ValueError, match="Invalid tier"):
            generate_license_key("u1", "platinum", private_key_path=priv_path)

    def test_different_keypair_rejects_token(self, keypair_dir, tmp_path):
        priv_path, _ = keypair_dir
        token = generate_license_key("u1", "pro", private_key_path=priv_path)

        # Generate a SECOND keypair
        _, other_pub = save_test_keypair(str(tmp_path / "other"))
        with pytest.raises(jwt.InvalidTokenError):
            validate_license_key(token, public_key_path=other_pub)


# ====================================================================
# 2. Expired key rejection
# ====================================================================

class TestExpiredKey:
    def test_expired_token_rejected(self, keypair_dir):
        priv_path, pub_path = keypair_dir
        # Issue a token that already expired (duration_days=0 still gives
        # a non-expired token at "now", so we patch time)
        priv_key = load_private_key(priv_path)
        past = int(time.time()) - 86400 * 2  # 2 days ago
        payload = {
            "sub": "u1",
            "tier": "pro",
            "iat": past,
            "exp": past + 86400,  # expired 1 day ago
            "grace_period_days": 30,
            "jti": "test-jti",
        }
        token = jwt.encode(payload, priv_key, algorithm="EdDSA")

        with pytest.raises(jwt.ExpiredSignatureError):
            validate_license_key(token, public_key_path=pub_path)


# ====================================================================
# 3. Grace period logic
# ====================================================================

class TestGracePeriod:
    def test_grace_period_active(self, keypair_dir, license_dir):
        """Expired token + recent last_validated -> grace period active."""
        priv_path, pub_path = keypair_dir
        lic_path, meta_path = license_dir

        # Create an expired token
        priv_key = load_private_key(priv_path)
        past = int(time.time()) - 86400 * 2
        payload = {
            "sub": "u1",
            "tier": "pro",
            "iat": past,
            "exp": past + 86400,
            "grace_period_days": 30,
            "jti": "gp-test",
        }
        token = jwt.encode(payload, priv_key, algorithm="EdDSA")
        lic_path.write_text(token)

        # Write meta with recent last_validated
        meta = {"last_validated": int(time.time()) - 86400}  # 1 day ago
        meta_path.write_text(json.dumps(meta))

        validator = LicenseValidator(
            license_path=lic_path,
            meta_path=meta_path,
            public_key_path=pub_path,
        )
        status = validator.check()
        assert status.valid is True
        assert status.mode == "full"
        assert "grace period" in status.message.lower()

    def test_grace_period_elapsed(self, keypair_dir, license_dir):
        """Expired token + old last_validated -> fully expired."""
        priv_path, pub_path = keypair_dir
        lic_path, meta_path = license_dir

        priv_key = load_private_key(priv_path)
        past = int(time.time()) - 86400 * 60
        payload = {
            "sub": "u1",
            "tier": "pro",
            "iat": past,
            "exp": past + 86400,
            "grace_period_days": 30,
            "jti": "gp-elapsed",
        }
        token = jwt.encode(payload, priv_key, algorithm="EdDSA")
        lic_path.write_text(token)

        # last_validated > 30 days ago
        meta = {"last_validated": int(time.time()) - 86400 * 35}
        meta_path.write_text(json.dumps(meta))

        validator = LicenseValidator(
            license_path=lic_path,
            meta_path=meta_path,
            public_key_path=pub_path,
        )
        status = validator.check()
        assert status.valid is False
        assert status.mode == "expired"


# ====================================================================
# 4. Webhook signature verification
# ====================================================================

class TestWebhookSignature:
    def test_valid_signature(self):
        secret = "whsec_test123"
        payload = b'{"type": "checkout.session.completed"}'
        ts = str(int(time.time()))
        signed = f"{ts}.".encode() + payload
        sig = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        header = f"t={ts},v1={sig}"

        assert verify_stripe_signature(payload, header, secret) is True

    def test_invalid_signature(self):
        secret = "whsec_test123"
        payload = b'{"type": "checkout.session.completed"}'
        header = "t=1234567890,v1=badsignature"

        assert verify_stripe_signature(payload, header, secret) is False

    def test_expired_timestamp(self):
        secret = "whsec_test123"
        payload = b'{"type": "test"}'
        old_ts = str(int(time.time()) - 600)  # 10 min ago, tolerance is 5 min
        signed = f"{old_ts}.".encode() + payload
        sig = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        header = f"t={old_ts},v1={sig}"

        assert verify_stripe_signature(payload, header, secret, tolerance=300) is False

    def test_malformed_header(self):
        assert verify_stripe_signature(b"x", "garbage", "s") is False
        assert verify_stripe_signature(b"x", "", "s") is False


# ====================================================================
# 5. Full webhook flow (requires FastAPI TestClient)
# ====================================================================

class TestWebhookEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from desktop_app.backend.licensing.webhook import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_checkout_completed_creates_license(self, client):
        # The webhook uses the default test keypair from the repo
        default_pub_path = (
            Path(__file__).parent.parent
            / "desktop_app" / "backend" / "licensing" / "test_keys" / "test_public.pem"
        )
        event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "customer_email": "buyer@example.com",
                    "metadata": {"tier": "pro"},
                }
            },
        }
        resp = client.post("/webhook/stripe", json=event)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "license_created"
        assert body["email"] == "buyer@example.com"
        assert body["tier"] == "pro"

        # The returned license key should be a valid JWT
        key = body["license_key"]
        payload = validate_license_key(key, public_key_path=default_pub_path)
        assert payload["user_id"] == "buyer@example.com"

        # Verify it's in the store
        store = get_license_store()
        assert store["buyer@example.com"]["active"] is True

    def test_subscription_deleted_revokes(self, client):
        # First create a license
        event_create = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "customer_email": "cancel@example.com",
                    "metadata": {"tier": "pro"},
                }
            },
        }
        client.post("/webhook/stripe", json=event_create)

        # Then delete subscription
        event_delete = {
            "type": "customer.subscription.deleted",
            "data": {
                "object": {
                    "metadata": {"email": "cancel@example.com"},
                }
            },
        }
        resp = client.post("/webhook/stripe", json=event_delete)
        assert resp.status_code == 200
        assert resp.json()["status"] == "license_revoked"

        store = get_license_store()
        assert store["cancel@example.com"]["active"] is False

    def test_unknown_event_ignored(self, client):
        event = {"type": "invoice.paid", "data": {"object": {}}}
        resp = client.post("/webhook/stripe", json=event)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"


# ====================================================================
# 6. LicenseValidator end-to-end
# ====================================================================

class TestLicenseValidator:
    def test_no_license_file_returns_expired(self, license_dir, keypair_dir):
        lic_path, meta_path = license_dir
        _, pub_path = keypair_dir
        validator = LicenseValidator(
            license_path=lic_path,
            meta_path=meta_path,
            public_key_path=pub_path,
        )
        status = validator.check()
        assert status.valid is False
        assert status.mode == "expired"

    def test_valid_license_returns_full(self, keypair_dir, license_dir):
        priv_path, pub_path = keypair_dir
        lic_path, meta_path = license_dir

        token = generate_license_key("u1", "pro", private_key_path=priv_path)
        lic_path.write_text(token)

        validator = LicenseValidator(
            license_path=lic_path,
            meta_path=meta_path,
            public_key_path=pub_path,
        )
        status = validator.check()
        assert status.valid is True
        assert status.tier == "pro"
        assert status.mode == "full"
        assert status.days_remaining > 300

    def test_revoked_license_read_only(self, keypair_dir, license_dir):
        priv_path, pub_path = keypair_dir
        lic_path, meta_path = license_dir

        token = generate_license_key("u1", "pro", private_key_path=priv_path)
        lic_path.write_text(token)

        validator = LicenseValidator(
            license_path=lic_path,
            meta_path=meta_path,
            public_key_path=pub_path,
        )
        validator.revoke()
        status = validator.check()
        assert status.valid is True
        assert status.mode == "read_only"

    def test_activate_writes_and_validates(self, keypair_dir, license_dir):
        priv_path, pub_path = keypair_dir
        lic_path, meta_path = license_dir

        token = generate_license_key("u2", "team", private_key_path=priv_path)

        validator = LicenseValidator(
            license_path=lic_path,
            meta_path=meta_path,
            public_key_path=pub_path,
        )
        status = validator.activate(token)
        assert status.valid is True
        assert status.tier == "team"
        assert lic_path.exists()
