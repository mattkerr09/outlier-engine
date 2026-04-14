"""
Outlier Desktop App — Licensing Module

Handles license key generation, validation, and payment webhook processing.
Designed as a preparation track: swap placeholder secrets with real ones to go live.
"""

from .keys import generate_license_key, validate_license_key
from .validator import LicenseValidator

__all__ = [
    "generate_license_key",
    "validate_license_key",
    "LicenseValidator",
]
