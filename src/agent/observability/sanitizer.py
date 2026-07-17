"""Recursive structured-log sanitization."""

from __future__ import annotations

from typing import Any
import logging
import re


REDACTED_FIELDS = {"api_key", "authorization", "password", "secret", "token"}
SOURCE_FIELDS = {"source_code", "prompt", "file_content", "unified_diff", "tool_output"}


_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|password|secret)\s*[:=]\s*[^\s,;]+"
    ),
)


def sanitize(value: Any, key: str | None = None) -> Any:
    normalized_key = (key or "").lower()
    if normalized_key in REDACTED_FIELDS:
        return "***REDACTED***"
    if normalized_key in SOURCE_FIELDS:
        return f"[{len(str(value))} chars]"
    if isinstance(value, dict):
        return {item_key: sanitize(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


class SanitizingProcessor:
    def __call__(self, logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        return sanitize(event_dict)


def sanitize_text(message: str) -> str:
    sanitized = message
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub("***REDACTED***", sanitized)
    return sanitized


class SanitizingFilter(logging.Filter):
    """Apply conservative secret redaction to standard-library logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = sanitize_text(record.getMessage())
        record.args = ()
        return True
