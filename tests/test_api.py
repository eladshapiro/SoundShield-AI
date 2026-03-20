"""
API Smoke Tests for SoundShield-AI

Tests Flask endpoints return correct status codes and response formats.
Uses Flask test client (no server required).
"""

import unittest
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAPIEndpoints(unittest.TestCase):
    """Smoke tests for API v1 endpoints."""

    @classmethod
    def setUpClass(cls):
        from web_app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_health_endpoint(self):
        resp = self.client.get('/health')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('version', data)
        self.assertIn('models', data)

    def test_api_v1_health(self):
        resp = self.client.get('/api/v1/health')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('version', data)

    def test_api_analyses_list(self):
        resp = self.client.get('/api/v1/analyses')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('data', data)

    def test_api_analyses_not_found(self):
        resp = self.client.get('/api/v1/analyses/99999')
        self.assertEqual(resp.status_code, 404)

    def test_api_stats(self):
        resp = self.client.get('/api/v1/stats')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('data', data)
        self.assertIn('total_analyses', data['data'])

    def test_api_audit_log(self):
        resp = self.client.get('/api/v1/audit-log')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('data', data)

    def test_api_notifications(self):
        resp = self.client.get('/api/v1/notifications')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('data', data)
        self.assertIn('unread_count', data)

    def test_api_notifications_mark_all_read(self):
        resp = self.client.post('/api/v1/notifications/read-all')
        self.assertEqual(resp.status_code, 200)

    def test_api_webhooks_list(self):
        resp = self.client.get('/api/v1/webhooks')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('data', data)

    def test_api_config_thresholds_get(self):
        resp = self.client.get('/api/v1/config/thresholds')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('data', data)
        self.assertIn('cry', data['data'])
        self.assertIn('violence', data['data'])

    def test_api_config_thresholds_put_invalid(self):
        resp = self.client.put('/api/v1/config/thresholds',
                               data=json.dumps({'detector': 'invalid', 'key': 'x', 'value': 1}),
                               content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    def test_api_system_info(self):
        resp = self.client.get('/api/v1/system/info')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('data', data)
        self.assertIn('python_version', data['data'])

    def test_admin_page_loads(self):
        resp = self.client.get('/admin')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'SoundShield', resp.data)

    def test_index_page_loads(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)

    def test_404_returns_json(self):
        resp = self.client.get('/api/v1/nonexistent')
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertIn('error', data)


if __name__ == '__main__':
    unittest.main()
