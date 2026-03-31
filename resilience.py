"""
Resilience Patterns for SoundShield-AI

Provides retry logic with exponential backoff and circuit breaker
pattern for graceful degradation of ML model inference.
"""

import time
import logging
import threading
import functools
from enum import Enum
from typing import Tuple, Type

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = 'closed'       # Normal operation
    OPEN = 'open'           # Failing, reject calls
    HALF_OPEN = 'half_open' # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern for protecting against cascading failures.

    States:
    - CLOSED: Normal operation, counting failures
    - OPEN: Service failing, reject immediately (fallback)
    - HALF_OPEN: Allow one test request to check recovery
    """

    def __init__(self, name: str, failure_threshold: int = 3,
                 reset_timeout: float = 60.0, half_open_max: int = 1):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max = half_open_max

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.reset_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
            return self._state

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' recovered -> CLOSED")
            else:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self):
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' OPEN after {self._failure_count} failures")

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.half_open_max:
                    self._half_open_calls += 1
                    return True
            return False
        return False  # OPEN

    def get_status(self) -> dict:
        """Get circuit breaker status."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self._failure_count,
            'failure_threshold': self.failure_threshold,
            'reset_timeout': self.reset_timeout,
        }


def retry(max_attempts: int = 3, backoff_base: float = 1.0,
          retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """Decorator for retry with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        backoff_base: Base delay in seconds (doubles each retry)
        retryable_exceptions: Tuple of exception types to retry on
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        delay = backoff_base * (2 ** (attempt - 1))
                        logger.warning(
                            f"Retry {attempt}/{max_attempts} for {func.__name__}: {e}. "
                            f"Waiting {delay:.1f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator


class MemoryGuard:
    """Checks process memory usage before resource-intensive operations."""

    def __init__(self, limit_mb: int = 4096):
        self.limit_mb = limit_mb
        self._psutil_available = False
        try:
            import psutil
            self._psutil_available = True
        except ImportError:
            logger.warning("psutil not available — memory guard disabled")

    def check(self) -> Tuple[bool, float]:
        """Check if memory usage is within limits.

        Returns: (is_ok, current_usage_mb)
        """
        if not self._psutil_available:
            return True, 0.0

        import psutil
        process = psutil.Process()
        rss_mb = process.memory_info().rss / (1024 * 1024)
        return rss_mb < self.limit_mb, rss_mb

    def get_usage(self) -> dict:
        """Get current memory usage info."""
        if not self._psutil_available:
            return {'available': False, 'message': 'psutil not installed'}

        import psutil
        process = psutil.Process()
        mem = process.memory_info()
        return {
            'available': True,
            'rss_mb': round(mem.rss / (1024 * 1024), 1),
            'limit_mb': self.limit_mb,
            'within_limit': mem.rss / (1024 * 1024) < self.limit_mb,
        }


# Global instances
_circuit_breakers = {}

def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get or create a named circuit breaker."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, **kwargs)
    return _circuit_breakers[name]

def get_all_circuit_breakers() -> dict:
    """Get status of all circuit breakers."""
    return {name: cb.get_status() for name, cb in _circuit_breakers.items()}
