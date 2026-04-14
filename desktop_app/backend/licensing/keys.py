"""
License key generation and validation using JWT + Ed25519.

Keys are JWTs signed with an Ed25519 private key. The desktop app ships with
the public key and can verify offline. The server holds the private key and
issues new licenses on successful checkout.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional, Tuple, Union

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
    load_pem_private_key,
    load_pem_public_key,
)

# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------
VALID_TIERS = {"free", "pro", "team", "enterprise"}

GRACE_PERIOD_DAYS = 30

# ---------------------------------------------------------------------------
# Key generation helpers
# ---------------------------------------------------------------------------

def generate_ed25519_keypair() -> Tuple[bytes, bytes]:
    """Generate a new Ed25519 keypair. Returns (private_pem, public_pem)."""
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
    )
    public_pem = private_key.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )
    return private_pem, public_pem


def save_test_keypair(directory: Optional[str] = None) -> Tuple[Path, Path]:
    """Generate and persist a test keypair to *directory*.

    Returns (private_key_path, public_key_path).
    """
    if directory is None:
        directory = str(Path(__file__).parent / "test_keys")
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)

    priv_pem, pub_pem = generate_ed25519_keypair()

    priv_path = dest / "test_private.pem"
    pub_path = dest / "test_public.pem"

    priv_path.write_bytes(priv_pem)
    pub_path.write_bytes(pub_pem)

    return priv_path, pub_path


def load_private_key(path: Union[str, Path]) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from a PEM file."""
    raw = Path(path).read_bytes()
    key = load_pem_private_key(raw, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("Expected Ed25519 private key")
    return key


def load_public_key(path: Union[str, Path]) -> Ed25519PublicKey:
    """Load an Ed25519 public key from a PEM file."""
    raw = Path(path).read_bytes()
    key = load_pem_public_key(raw)
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("Expected Ed25519 public key")
    return key


# ---------------------------------------------------------------------------
# License key creation / validation
# ---------------------------------------------------------------------------

def generate_license_key(
    user_id: str,
    tier: str,
    duration_days: int = 365,
    private_key: Optional[Ed25519PrivateKey] = None,
    private_key_path: Optional[Union[str, Path]] = None,
) -> str:
    """Create a signed JWT license key.

    Supply either *private_key* (an Ed25519PrivateKey object) or
    *private_key_path* (path to a PEM file). If neither is provided the test
    key is used.

    Returns a compact JWT string.
    """
    if tier not in VALID_TIERS:
        raise ValueError(f"Invalid tier {tier!r}; must be one of {VALID_TIERS}")

    if private_key is None:
        if private_key_path is None:
            private_key_path = Path(__file__).parent / "test_keys" / "test_private.pem"
        private_key = load_private_key(private_key_path)

    now = int(time.time())
    payload = {
        "sub": user_id,
        "tier": tier,
        "iat": now,
        "exp": now + duration_days * 86400,
        "grace_period_days": GRACE_PERIOD_DAYS,
        "jti": str(uuid.uuid4()),
    }
    token: str = jwt.encode(payload, private_key, algorithm="EdDSA")
    return token


def validate_license_key(
    token: str,
    public_key: Optional[Ed25519PublicKey] = None,
    public_key_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Verify a JWT license key and return its payload.

    Raises jwt.ExpiredSignatureError if the token has expired and
    jwt.InvalidTokenError for any other verification failure.
    """
    if public_key is None:
        if public_key_path is None:
            public_key_path = Path(__file__).parent / "test_keys" / "test_public.pem"
        public_key = load_public_key(public_key_path)

    payload: dict = jwt.decode(
        token,
        public_key,
        algorithms=["EdDSA"],
        options={"require": ["sub", "tier", "iat", "exp"]},
    )

    # Normalise field names for downstream consumers
    payload.setdefault("user_id", payload.get("sub"))
    payload.setdefault("issued_at", payload.get("iat"))
    payload.setdefault("expires_at", payload.get("exp"))

    return payload
