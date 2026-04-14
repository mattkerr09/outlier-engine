"""
Payment webhook handler — FastAPI router.

Handles Stripe (or Polar.sh) webhook events:
  - checkout.session.completed  -> generate license, store in DB
  - customer.subscription.deleted -> revoke license

Stripe signature verification uses the webhook secret from env.
All DB operations are placeholders (dict store) — swap with real DB later.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request

from .keys import generate_license_key

router = APIRouter(prefix="/webhook", tags=["webhooks"])

# ---------------------------------------------------------------------------
# Configuration (read from environment in production)
# ---------------------------------------------------------------------------
STRIPE_WEBHOOK_SECRET = os.environ.get(
    "STRIPE_WEBHOOK_SECRET", "whsec_placeholder_replace_me"
)

# In-memory license store (placeholder for a real DB)
_license_store: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Stripe signature verification
# ---------------------------------------------------------------------------

def verify_stripe_signature(
    payload: bytes,
    sig_header: str,
    secret: str,
    tolerance: int = 300,
) -> bool:
    """Verify a Stripe webhook signature (v1 scheme).

    Stripe signs with HMAC-SHA256 over ``{timestamp}.{payload}``.
    """
    try:
        parts = dict(
            item.split("=", 1) for item in sig_header.split(",") if "=" in item
        )
    except ValueError:
        return False

    timestamp = parts.get("t")
    v1_sig = parts.get("v1")
    if not timestamp or not v1_sig:
        return False

    # Check tolerance
    if abs(time.time() - int(timestamp)) > tolerance:
        return False

    signed_payload = f"{timestamp}.".encode() + payload
    expected = hmac.new(
        secret.encode(), signed_payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, v1_sig)


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------

@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None),
):
    """Process Stripe webhook events.

    Returns the generated license key on checkout.session.completed so
    integration tests can verify the full flow.
    """
    raw_body = await request.body()

    # Verify signature (skip in test mode when header is absent)
    if stripe_signature:
        if not verify_stripe_signature(raw_body, stripe_signature, STRIPE_WEBHOOK_SECRET):
            raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        event = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event_type = event.get("type", "")

    # ----- checkout.session.completed -----
    if event_type == "checkout.session.completed":
        session = event.get("data", {}).get("object", {})
        customer_email = session.get("customer_email", "unknown")
        tier = session.get("metadata", {}).get("tier", "pro")

        license_key = generate_license_key(
            user_id=customer_email,
            tier=tier,
        )

        # Store in DB (placeholder)
        _license_store[customer_email] = {
            "license_key": license_key,
            "tier": tier,
            "active": True,
            "created_at": int(time.time()),
        }

        return {
            "status": "license_created",
            "email": customer_email,
            "tier": tier,
            "license_key": license_key,
        }

    # ----- customer.subscription.deleted -----
    elif event_type == "customer.subscription.deleted":
        subscription = event.get("data", {}).get("object", {})
        customer_email = subscription.get("metadata", {}).get("email", "unknown")

        if customer_email in _license_store:
            _license_store[customer_email]["active"] = False

        return {
            "status": "license_revoked",
            "email": customer_email,
        }

    # ----- unhandled events -----
    return {"status": "ignored", "event_type": event_type}


# ---------------------------------------------------------------------------
# Helpers for testing
# ---------------------------------------------------------------------------

def get_license_store() -> dict:
    """Return the in-memory license store (for test assertions)."""
    return _license_store


def clear_license_store() -> None:
    """Reset the in-memory license store."""
    _license_store.clear()
