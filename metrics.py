"""
Metrics Collection for SoundShield-AI

Provides application metrics for monitoring: analysis counts,
pipeline step timing, error rates, and incident detection stats.
Uses an in-memory approach (no external dependency required).
"""

import time
import threading
import logging
from collections import defaultdict, deque
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Thread-safe metrics collection for monitoring and observability."""

    def __init__(self):
        self._lock = threading.Lock()
        # Counters
        self._counters = defaultdict(int)
        # Histograms (store last 1000 observations)
        self._histograms = defaultdict(lambda: deque(maxlen=1000))
        # Error tracking with timestamps
        self._errors = deque(maxlen=5000)
        # Step timings
        self._step_timings = defaultdict(lambda: deque(maxlen=500))

    def increment(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Increment a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

    def record_step_timing(self, step_name: str, duration_ms: float):
        """Record pipeline step timing."""
        with self._lock:
            self._step_timings[step_name].append({
                'duration_ms': duration_ms,
                'timestamp': time.time()
            })

    def record_error(self, error_type: str, endpoint: str = ''):
        """Record an error occurrence."""
        with self._lock:
            self._errors.append({
                'error_type': error_type,
                'endpoint': endpoint,
                'timestamp': time.time()
            })
            self._counters[f'errors_total:{error_type}'] += 1

    def get_error_rate(self, window_minutes: int = 5) -> float:
        """Get error rate in the sliding window."""
        cutoff = time.time() - (window_minutes * 60)
        with self._lock:
            recent = sum(1 for e in self._errors if e['timestamp'] > cutoff)
            total_requests = self._counters.get('requests_total', 1)
        return recent / max(total_requests, 1)

    def get_error_summary(self, window_minutes: int = 5) -> Dict:
        """Get error summary for the sliding window."""
        cutoff = time.time() - (window_minutes * 60)
        with self._lock:
            recent_errors = [e for e in self._errors if e['timestamp'] > cutoff]

        by_type = defaultdict(int)
        by_endpoint = defaultdict(int)
        for e in recent_errors:
            by_type[e['error_type']] += 1
            if e['endpoint']:
                by_endpoint[e['endpoint']] += 1

        return {
            'total_errors': len(recent_errors),
            'window_minutes': window_minutes,
            'by_type': dict(by_type),
            'by_endpoint': dict(by_endpoint)
        }

    def get_step_timings(self) -> Dict:
        """Get pipeline step timing percentiles."""
        import statistics
        result = {}
        with self._lock:
            for step, timings in self._step_timings.items():
                values = [t['duration_ms'] for t in timings]
                if not values:
                    continue
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                result[step] = {
                    'count': n,
                    'p50': sorted_vals[n // 2] if n > 0 else 0,
                    'p95': sorted_vals[int(n * 0.95)] if n > 1 else sorted_vals[-1],
                    'p99': sorted_vals[int(n * 0.99)] if n > 1 else sorted_vals[-1],
                    'mean': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                }
        return result

    def get_counters(self) -> Dict:
        """Get all counter values."""
        with self._lock:
            return dict(self._counters)

    def get_all_metrics(self) -> Dict:
        """Get comprehensive metrics snapshot."""
        return {
            'counters': self.get_counters(),
            'step_timings': self.get_step_timings(),
            'error_summary': self.get_error_summary(),
        }

    def generate_prometheus_text(self) -> str:
        """Generate Prometheus text exposition format."""
        lines = []
        with self._lock:
            for key, value in self._counters.items():
                safe_key = key.replace(':', '_').replace('-', '_').replace('.', '_')
                lines.append(f'soundshield_{safe_key} {value}')
        return '\n'.join(lines) + '\n'

    @staticmethod
    def _make_key(name: str, labels: Dict[str, str] = None) -> str:
        if not labels:
            return name
        label_str = ','.join(f'{k}={v}' for k, v in sorted(labels.items()))
        return f'{name}:{label_str}'


# Global metrics instance
metrics = MetricsCollector()
