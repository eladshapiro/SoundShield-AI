"""Tests for resilience patterns: retry, circuit breaker, memory guard."""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resilience import retry, CircuitBreaker, CircuitState, MemoryGuard


class TestRetry(unittest.TestCase):

    def test_retry_success_first_attempt(self):
        call_count = [0]
        @retry(max_attempts=3, backoff_base=0.01)
        def succeed():
            call_count[0] += 1
            return 'ok'
        result = succeed()
        self.assertEqual(result, 'ok')
        self.assertEqual(call_count[0], 1)

    def test_retry_success_after_failures(self):
        call_count = [0]
        @retry(max_attempts=3, backoff_base=0.01)
        def fail_then_succeed():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("not yet")
            return 'ok'
        result = fail_then_succeed()
        self.assertEqual(result, 'ok')
        self.assertEqual(call_count[0], 3)

    def test_retry_exhaustion(self):
        @retry(max_attempts=2, backoff_base=0.01)
        def always_fail():
            raise ValueError("always fails")
        with self.assertRaises(ValueError):
            always_fail()

    def test_retry_specific_exceptions(self):
        @retry(max_attempts=3, backoff_base=0.01, retryable_exceptions=(IOError,))
        def raise_value_error():
            raise ValueError("not retryable")
        with self.assertRaises(ValueError):
            raise_value_error()


class TestCircuitBreaker(unittest.TestCase):

    def test_initial_state_closed(self):
        cb = CircuitBreaker('test', failure_threshold=3)
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_opens_after_threshold(self):
        cb = CircuitBreaker('test', failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)

    def test_rejects_when_open(self):
        cb = CircuitBreaker('test', failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        self.assertFalse(cb.allow_request())

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker('test', failure_threshold=2, reset_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        time.sleep(0.15)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)

    def test_recovery_to_closed(self):
        cb = CircuitBreaker('test', failure_threshold=2, reset_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        self.assertTrue(cb.allow_request())
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_get_status(self):
        cb = CircuitBreaker('test_status', failure_threshold=5)
        status = cb.get_status()
        self.assertEqual(status['name'], 'test_status')
        self.assertEqual(status['state'], 'closed')


class TestMemoryGuard(unittest.TestCase):

    def test_memory_check(self):
        mg = MemoryGuard(limit_mb=99999)
        ok, usage = mg.check()
        # Should be within a very high limit
        self.assertTrue(ok)

    def test_memory_usage_info(self):
        mg = MemoryGuard()
        info = mg.get_usage()
        # Either psutil works or it reports unavailable
        self.assertIn('available', info)

    def test_memory_over_limit(self):
        mg = MemoryGuard(limit_mb=1)  # 1 MB — any process exceeds this
        ok, usage = mg.check()
        if mg._psutil_available:
            self.assertFalse(ok)
            self.assertGreater(usage, 1)


class TestResilienceEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ['AUTH_ENABLED'] = 'False'
        from web_app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_circuit_breakers_endpoint(self):
        response = self.client.get('/api/v1/metrics/circuit-breakers')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('circuit_breakers', data)

    def test_memory_endpoint(self):
        response = self.client.get('/api/v1/metrics/memory')
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
