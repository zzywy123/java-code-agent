"""Structured logging configuration."""

from __future__ import annotations

import logging

import structlog

from agent.observability.sanitizer import SanitizingFilter, SanitizingProcessor


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(message)s")
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if not any(isinstance(item, SanitizingFilter) for item in handler.filters):
            handler.addFilter(SanitizingFilter())
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            SanitizingProcessor(),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
