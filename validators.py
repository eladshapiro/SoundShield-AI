"""
Input Validation & SSRF Protection for SoundShield-AI

Validates all user inputs at API boundaries:
- Webhook URLs (SSRF prevention)
- Language parameters
- Threshold values
- Audio file content verification
"""

import ipaddress
import logging
import os
import struct
from urllib.parse import urlparse
from typing import Optional, Tuple

from config import config

logger = logging.getLogger(__name__)

# Private/reserved IP ranges that should not be used for webhooks
_PRIVATE_NETWORKS = [
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('127.0.0.0/8'),
    ipaddress.ip_network('169.254.0.0/16'),
    ipaddress.ip_network('::1/128'),
    ipaddress.ip_network('fc00::/7'),
    ipaddress.ip_network('fe80::/10'),
]

# Audio file magic bytes (header signatures)
_AUDIO_SIGNATURES = {
    'wav': [(0, b'RIFF'), (8, b'WAVE')],
    'mp3': [(0, b'\xff\xfb'), (0, b'\xff\xf3'), (0, b'\xff\xf2'), (0, b'ID3')],
    'flac': [(0, b'fLaC')],
    'ogg': [(0, b'OggS')],
    'm4a': [(4, b'ftyp')],
    'aac': [(0, b'\xff\xf1'), (0, b'\xff\xf9')],
}

# Threshold validation bounds
_THRESHOLD_BOUNDS = {
    'float': (0.0, 10.0),       # Most thresholds are 0-1 but allow up to 10
    'int': (0, 100000),         # Integer thresholds
    'frequency': (20, 20000),   # Hz range
    'duration': (0.0, 7200.0),  # Seconds, up to 2 hours
}


def validate_webhook_url(url: str) -> Tuple[bool, str]:
    """Validate webhook URL for SSRF protection.

    Rules:
    - Must be https:// (or http:// in debug mode)
    - Must have a valid hostname
    - Must not resolve to private/reserved IP ranges
    - Must not use file://, ftp://, or other dangerous schemes

    Returns: (is_valid, error_message)
    """
    if not url or not isinstance(url, str):
        return False, "URL is required"

    url = url.strip()

    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL format"

    # Scheme check
    allowed_schemes = ['https']
    if os.environ.get('FLASK_DEBUG') == '1' or os.environ.get('WEB_DEBUG', '').lower() in ('1', 'true'):
        allowed_schemes.append('http')

    if parsed.scheme not in allowed_schemes:
        return False, f"URL scheme must be one of: {allowed_schemes}. Got: '{parsed.scheme}'"

    # Host check
    hostname = parsed.hostname
    if not hostname:
        return False, "URL must have a valid hostname"

    # Block localhost variants
    localhost_names = ['localhost', '0.0.0.0', '127.0.0.1', '::1', '[::1]']
    if hostname.lower() in localhost_names:
        return False, "Webhook URL cannot point to localhost"

    # Try to resolve and check for private IPs
    try:
        ip = ipaddress.ip_address(hostname)
        for network in _PRIVATE_NETWORKS:
            if ip in network:
                return False, f"Webhook URL cannot point to private/reserved IP range: {network}"
    except ValueError:
        # hostname is a domain name, not an IP — that's fine
        # We could do DNS resolution here, but that's a network call
        # For now, just block obvious private patterns
        if hostname.endswith('.local') or hostname.endswith('.internal'):
            return False, "Webhook URL cannot point to local/internal domains"

    # Port check — block common internal service ports
    if parsed.port and parsed.port in [6379, 11211, 5432, 3306, 27017, 9200, 2379]:
        return False, f"Webhook URL uses a blocked port: {parsed.port}"

    return True, ""


def validate_language(language: str) -> Tuple[bool, str]:
    """Validate language parameter against supported languages."""
    if not language or not isinstance(language, str):
        return False, "Language parameter is required"

    supported = config.pipeline.supported_languages
    if language.strip().lower() not in [l.lower() for l in supported]:
        return False, f"Unsupported language: '{language}'. Supported: {list(supported)}"

    return True, ""


def validate_threshold_value(detector: str, field: str, value) -> Tuple[bool, str]:
    """Validate a threshold update value is within sane bounds."""
    if value is None:
        return False, "Value is required"

    # Must be numeric
    try:
        numeric_val = float(value)
    except (TypeError, ValueError):
        return False, f"Value must be numeric, got: {type(value).__name__}"

    # Check for special float values
    if numeric_val != numeric_val:  # NaN check
        return False, "Value cannot be NaN"
    if numeric_val == float('inf') or numeric_val == float('-inf'):
        return False, "Value cannot be infinity"

    # Frequency fields
    if 'freq' in field.lower() or 'frequency' in field.lower():
        lo, hi = _THRESHOLD_BOUNDS['frequency']
        if not (lo <= numeric_val <= hi):
            return False, f"Frequency value must be between {lo} and {hi} Hz"
    # Duration fields
    elif 'duration' in field.lower() or 'seconds' in field.lower() or 'window' in field.lower():
        lo, hi = _THRESHOLD_BOUNDS['duration']
        if not (lo <= numeric_val <= hi):
            return False, f"Duration value must be between {lo} and {hi} seconds"
    # Integer fields
    elif isinstance(value, int) or (isinstance(value, float) and value == int(value) and 'threshold' not in field.lower()):
        lo, hi = _THRESHOLD_BOUNDS['int']
        if not (lo <= numeric_val <= hi):
            return False, f"Integer value must be between {lo} and {hi}"
    # General float thresholds
    else:
        lo, hi = _THRESHOLD_BOUNDS['float']
        if not (lo <= numeric_val <= hi):
            return False, f"Threshold value must be between {lo} and {hi}"

    return True, ""


def validate_audio_file(filepath: str, claimed_extension: str = None) -> Tuple[bool, str]:
    """Validate audio file by checking magic bytes match the extension.

    Returns: (is_valid, error_message)
    """
    if not os.path.exists(filepath):
        return False, "File does not exist"

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return False, "File is empty"

    if file_size > config.web.max_content_length:
        return False, f"File exceeds maximum size of {config.web.max_content_length // (1024*1024)}MB"

    # Determine extension
    if claimed_extension:
        ext = claimed_extension.lower().lstrip('.')
    else:
        ext = os.path.splitext(filepath)[1].lower().lstrip('.')

    if ext not in _AUDIO_SIGNATURES:
        # Extension not in our signature database — allow it through
        # (some formats like AAC may have variants)
        return True, ""

    # Read first 12 bytes for magic byte check
    try:
        with open(filepath, 'rb') as f:
            header = f.read(12)
    except IOError as e:
        return False, f"Cannot read file: {e}"

    if len(header) < 4:
        return False, "File too small to be a valid audio file"

    # Check magic bytes
    signatures = _AUDIO_SIGNATURES[ext]
    for offset, magic in signatures:
        if offset + len(magic) <= len(header) and header[offset:offset + len(magic)] == magic:
            return True, ""

    return False, f"File content does not match claimed format '{ext}'. Possible file type mismatch."


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal."""
    # Remove path separators and null bytes
    filename = filename.replace('/', '').replace('\\', '').replace('\x00', '')
    # Remove leading dots (hidden files)
    filename = filename.lstrip('.')
    # Fallback
    if not filename:
        filename = 'unnamed_upload'
    return filename
