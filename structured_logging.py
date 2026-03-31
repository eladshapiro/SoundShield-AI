"""
Structured Logging for SoundShield-AI

Provides JSON-formatted logging with correlation IDs for tracing
analysis requests through the 7-step detection pipeline.
"""

import logging
import uuid
import time
import os
from contextvars import ContextVar
from typing import Optional

from config import config

# Context variable for correlation ID (thread-safe)
_correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')

# Context variable for current user
_current_user: ContextVar[str] = ContextVar('current_user', default='anonymous')


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return _correlation_id.get()


def set_correlation_id(cid: str = None) -> str:
    """Set correlation ID. Generates a new UUID if none provided."""
    cid = cid or uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


def set_current_user(username: str):
    """Set the current user for log context."""
    _current_user.set(username)


def get_current_user() -> str:
    """Get the current user."""
    return _current_user.get()


class CorrelationFilter(logging.Filter):
    """Adds correlation_id and user to every log record."""

    def filter(self, record):
        record.correlation_id = _correlation_id.get() or '-'
        record.user = _current_user.get() or 'anonymous'
        return True


class StepTimer:
    """Context manager for timing pipeline steps with structured logging."""

    def __init__(self, step_name: str, step_number: int = 0, total_steps: int = 7, logger_instance: logging.Logger = None):
        self.step_name = step_name
        self.step_number = step_number
        self.total_steps = total_steps
        self.logger = logger_instance or logging.getLogger('soundshield.pipeline')
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(
            f"Step {self.step_number}/{self.total_steps}: {self.step_name} started",
            extra={
                'step_name': self.step_name,
                'step_number': self.step_number,
                'total_steps': self.total_steps,
                'event': 'step_start',
            }
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        if exc_type:
            self.logger.error(
                f"Step {self.step_number}/{self.total_steps}: {self.step_name} failed ({duration_ms:.0f}ms)",
                extra={
                    'step_name': self.step_name,
                    'step_number': self.step_number,
                    'duration_ms': round(duration_ms),
                    'event': 'step_error',
                    'error': str(exc_val),
                }
            )
        else:
            self.logger.info(
                f"Step {self.step_number}/{self.total_steps}: {self.step_name} completed ({duration_ms:.0f}ms)",
                extra={
                    'step_name': self.step_name,
                    'step_number': self.step_number,
                    'duration_ms': round(duration_ms),
                    'event': 'step_complete',
                }
            )
        return False  # Don't suppress exceptions


def setup_logging():
    """Configure structured logging for the application."""
    log_cfg = config.logging_config
    root_logger = logging.getLogger()

    # Clear existing handlers to avoid duplication
    root_logger.handlers.clear()

    level = getattr(logging, log_cfg.log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    # Add correlation filter
    correlation_filter = CorrelationFilter()

    if log_cfg.log_format == 'json':
        try:
            from pythonjsonlogger import jsonlogger

            formatter = jsonlogger.JsonFormatter(
                fmt='%(asctime)s %(name)s %(levelname)s %(message)s %(correlation_id)s %(user)s',
                rename_fields={
                    'asctime': 'timestamp',
                    'levelname': 'level',
                    'name': 'logger',
                },
                datefmt='%Y-%m-%dT%H:%M:%S'
            )
        except ImportError:
            # Fallback to text if python-json-logger not installed
            formatter = logging.Formatter(
                '%(asctime)s [%(correlation_id)s] %(name)s %(levelname)s: %(message)s'
            )
    else:
        formatter = logging.Formatter(
            '%(asctime)s [%(correlation_id)s] %(name)s %(levelname)s: %(message)s'
        )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(correlation_filter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_cfg.log_file:
        file_handler = logging.FileHandler(log_cfg.log_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(correlation_filter)
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger = logging.getLogger('soundshield')
    logger.info("Structured logging initialized", extra={'event': 'logging_init', 'format': log_cfg.log_format})

    return logger
