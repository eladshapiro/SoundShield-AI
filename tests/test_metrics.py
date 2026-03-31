"""Tests for the metrics collection system."""
import os
import sys
import time
import unittest
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import MetricsCollector, metrics


class TestMetricsCollector(unittest.TestCase):

    def setUp(self):
        self.mc = MetricsCollector()

    def test_counter_increment(self):
        self.mc.increment('test_counter')
        self.mc.increment('test_counter', 5)
        counters = self.mc.get_counters()
        self.assertEqual(counters['test_counter'], 6)

    def test_counter_with_labels(self):
        self.mc.increment('requests', labels={'method': 'GET'})
        self.mc.increment('requests', labels={'method': 'POST'})
        counters = self.mc.get_counters()
        self.assertEqual(counters['requests:method=GET'], 1)
        self.assertEqual(counters['requests:method=POST'], 1)

    def test_histogram_observation(self):
        for v in [10, 20, 30, 40, 50]:
            self.mc.observe('latency', v)
        # Just verify no crash
        self.assertEqual(len(self.mc._histograms['latency']), 5)

    def test_step_timing_recording(self):
        self.mc.record_step_timing('audio_analysis', 150.0)
        self.mc.record_step_timing('audio_analysis', 200.0)
        self.mc.record_step_timing('emotion_detection', 50.0)
        timings = self.mc.get_step_timings()
        self.assertIn('audio_analysis', timings)
        self.assertEqual(timings['audio_analysis']['count'], 2)
        self.assertIn('emotion_detection', timings)

    def test_error_recording(self):
        self.mc.record_error('ValueError', '/upload')
        self.mc.record_error('IOError', '/upload')
        summary = self.mc.get_error_summary(window_minutes=1)
        self.assertEqual(summary['total_errors'], 2)
        self.assertIn('ValueError', summary['by_type'])

    def test_error_rate_calculation(self):
        self.mc.increment('requests_total', 100)
        for _ in range(5):
            self.mc.record_error('Error')
        rate = self.mc.get_error_rate(window_minutes=1)
        self.assertAlmostEqual(rate, 0.05, places=2)

    def test_prometheus_text_format(self):
        self.mc.increment('analyses_total', 42)
        text = self.mc.generate_prometheus_text()
        self.assertIn('soundshield_analyses_total 42', text)

    def test_thread_safety(self):
        def increment_many():
            for _ in range(100):
                self.mc.increment('concurrent_test')
        threads = [threading.Thread(target=increment_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(self.mc.get_counters()['concurrent_test'], 1000)

    def test_get_all_metrics(self):
        self.mc.increment('test')
        result = self.mc.get_all_metrics()
        self.assertIn('counters', result)
        self.assertIn('step_timings', result)
        self.assertIn('error_summary', result)

    def test_empty_step_timings(self):
        timings = self.mc.get_step_timings()
        self.assertEqual(timings, {})


class TestMetricsEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ['AUTH_ENABLED'] = 'False'
        from web_app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_prometheus_endpoint(self):
        response = self.client.get('/metrics')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/plain', response.content_type)

    def test_metrics_summary(self):
        response = self.client.get('/api/v1/metrics/summary')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('counters', data)

    def test_error_rates_endpoint(self):
        response = self.client.get('/api/v1/metrics/error-rates?window=5')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('error_rate', data)

    def test_pipeline_timing_endpoint(self):
        response = self.client.get('/api/v1/metrics/pipeline-timing')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('steps', data)


if __name__ == '__main__':
    unittest.main()
