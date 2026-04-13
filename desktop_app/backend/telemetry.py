"""
Outlier Desktop — Telemetry (opt-in only)

Integrates Sentry (crash reporting) and PostHog (product analytics).
All telemetry is opt-in: user must explicitly consent on first launch.
User is identified by a hashed machine ID, never by email or name.

Usage:
    from telemetry import init_telemetry, track_event, capture_error

    init_telemetry(consent=True)  # Call once on startup after user opts in
    track_event("first_chat_sent", {"profile": "medical"})
    capture_error(exception)
"""
import hashlib
import os
import platform
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# Lazy imports — don't fail if SDKs not installed
_sentry_sdk = None
_posthog = None
_initialized = False
_consent = False
_machine_id: Optional[str] = None

SENTRY_DSN = os.environ.get("OUTLIER_SENTRY_DSN", "")
POSTHOG_API_KEY = os.environ.get("OUTLIER_POSTHOG_KEY", "")
POSTHOG_HOST = os.environ.get("OUTLIER_POSTHOG_HOST", "https://us.i.posthog.com")

# Consent file — persisted between launches
CONSENT_FILE = Path.home() / ".outlier" / "telemetry_consent.txt"


def _get_machine_id() -> str:
    """Generate a stable, non-reversible machine identifier."""
    global _machine_id
    if _machine_id is not None:
        return _machine_id
    # Use platform node (MAC address hash) + hostname for stability
    raw = f"{platform.node()}-{uuid.getnode()}"
    _machine_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return _machine_id


def has_consent() -> bool:
    """Check if user has previously consented to telemetry."""
    return CONSENT_FILE.exists() and CONSENT_FILE.read_text().strip() == "yes"


def set_consent(consent: bool) -> None:
    """Persist user's telemetry consent choice."""
    CONSENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONSENT_FILE.write_text("yes" if consent else "no")


def init_telemetry(consent: Optional[bool] = None) -> bool:
    """Initialize telemetry SDKs if user has consented.

    Args:
        consent: If True/False, sets consent and initializes accordingly.
                 If None, checks persisted consent file.

    Returns:
        True if telemetry is active, False otherwise.
    """
    global _sentry_sdk, _posthog, _initialized, _consent

    if consent is not None:
        set_consent(consent)
        _consent = consent
    else:
        _consent = has_consent()

    if not _consent:
        return False

    # Initialize Sentry
    if SENTRY_DSN:
        try:
            import sentry_sdk
            sentry_sdk.init(
                dsn=SENTRY_DSN,
                traces_sample_rate=0.1,
                profiles_sample_rate=0.1,
                release=f"outlier-desktop@{_get_app_version()}",
                environment="production",
                before_send=_scrub_pii,
            )
            sentry_sdk.set_user({"id": _get_machine_id()})
            _sentry_sdk = sentry_sdk
        except ImportError:
            pass  # Sentry SDK not installed — telemetry degrades gracefully

    # Initialize PostHog
    if POSTHOG_API_KEY:
        try:
            import posthog
            posthog.api_key = POSTHOG_API_KEY
            posthog.host = POSTHOG_HOST
            posthog.disabled = False
            _posthog = posthog
        except ImportError:
            pass  # PostHog not installed

    _initialized = True
    return True


def track_event(event: str, properties: Optional[Dict[str, Any]] = None) -> None:
    """Track a product analytics event (PostHog)."""
    if not _consent or _posthog is None:
        return
    props = {
        "platform": platform.system(),
        "arch": platform.machine(),
        **(properties or {}),
    }
    try:
        _posthog.capture(_get_machine_id(), event, props)
    except Exception:
        pass  # Never crash on telemetry failure


def capture_error(exception: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Report an error to Sentry."""
    if not _consent or _sentry_sdk is None:
        return
    try:
        with _sentry_sdk.push_scope() as scope:
            if context:
                for key, value in context.items():
                    scope.set_extra(key, value)
            _sentry_sdk.capture_exception(exception)
    except Exception:
        pass


def _scrub_pii(event, hint):
    """Remove any PII before sending to Sentry."""
    # Remove user IP
    if "user" in event and "ip_address" in event["user"]:
        del event["user"]["ip_address"]
    # Remove file paths that contain username
    if "exception" in event:
        for exc in event["exception"].get("values", []):
            for frame in exc.get("stacktrace", {}).get("frames", []):
                if "filename" in frame:
                    frame["filename"] = _anonymize_path(frame["filename"])
    return event


def _anonymize_path(path: str) -> str:
    """Replace home directory with ~ in file paths."""
    home = str(Path.home())
    return path.replace(home, "~") if home in path else path


def _get_app_version() -> str:
    """Get the current app version."""
    try:
        from desktop_app.backend.server import APP_VERSION
        return APP_VERSION
    except Exception:
        return "unknown"


# Predefined event names for consistency
EVENTS = {
    "app_installed": "app_installed",
    "app_first_run": "app_first_run",
    "first_chat_sent": "first_chat_sent",
    "first_response_received": "first_response_received",
    "profile_created": "profile_created",
    "profile_switched": "profile_switched",
    "model_loaded": "model_loaded",
    "warmup_completed": "warmup_completed",
    "crash": "crash",
    "error": "error",
}
