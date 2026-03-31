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

    def test_export_pdf_not_found(self):
        """Test PDF export for nonexistent analysis returns 404."""
        response = self.client.get('/api/v1/analyses/99999/export?format=pdf')
        self.assertIn(response.status_code, [404, 503])

    def test_export_unsupported_format(self):
        """Test export with unsupported format returns 400."""
        response = self.client.get('/api/v1/analyses/1/export?format=xlsx')
        self.assertEqual(response.status_code, 400)

    def test_404_returns_json(self):
        resp = self.client.get('/api/v1/nonexistent')
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertIn('error', data)

    def test_security_headers_present(self):
        """Test that security headers are set on responses."""
        response = self.client.get('/api/v1/health')
        self.assertEqual(response.headers.get('X-Content-Type-Options'), 'nosniff')
        self.assertEqual(response.headers.get('X-Frame-Options'), 'DENY')
        self.assertEqual(response.headers.get('X-XSS-Protection'), '1; mode=block')
        self.assertIn('Content-Security-Policy', response.headers)

    def test_secret_key_configured(self):
        """Test that Flask app has a secret key configured."""
        from web_app import app
        self.assertIsNotNone(app.config.get('SECRET_KEY'))
        self.assertNotEqual(app.config.get('SECRET_KEY'), '')

    def test_batch_upload_no_files(self):
        """Test batch upload with no files returns error."""
        response = self.client.post('/api/v1/batch-upload')
        self.assertEqual(response.status_code, 400)

    def test_batch_upload_success(self):
        """Test batch upload with valid files returns 202."""
        import io
        import numpy as np
        import soundfile as sf

        # Create two small test WAV files
        files = []
        for name in ['test1.wav', 'test2.wav']:
            buf = io.BytesIO()
            audio = np.random.randn(16000).astype(np.float32) * 0.3
            sf.write(buf, audio, 16000, format='WAV')
            buf.seek(0)
            files.append((buf, name))

        data = {}
        file_tuples = []
        for buf, name in files:
            file_tuples.append((io.BytesIO(buf.read()), name))
            buf.seek(0)

        response = self.client.post('/api/v1/batch-upload',
            data={
                'files': [(buf, name) for buf, name in files],
                'language': 'en'
            },
            content_type='multipart/form-data'
        )
        self.assertIn(response.status_code, [200, 202])
        data = response.get_json()
        self.assertIn('batch_id', data)

    def test_batch_status_not_found(self):
        """Test batch status for nonexistent job returns 404."""
        response = self.client.get('/api/v1/batch/nonexistent/status')
        self.assertEqual(response.status_code, 404)

    def test_auth_login_success(self):
        """Test login with valid credentials returns token."""
        response = self.client.post('/api/v1/auth/login',
            json={'username': 'admin', 'password': 'admin'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('token', data)
        self.assertEqual(data['user']['role'], 'admin')

    def test_auth_login_invalid(self):
        """Test login with wrong password returns 401."""
        response = self.client.post('/api/v1/auth/login',
            json={'username': 'admin', 'password': 'wrong'})
        self.assertEqual(response.status_code, 401)

    def test_auth_login_missing_fields(self):
        """Test login with missing fields returns 400."""
        response = self.client.post('/api/v1/auth/login', json={'username': 'admin'})
        self.assertEqual(response.status_code, 400)

    def test_auth_register_creates_user(self):
        """Test user registration (auth disabled, so should work)."""
        response = self.client.post('/api/v1/auth/register',
            json={'username': 'testuser', 'password': 'testpass123', 'role': 'viewer'})
        self.assertIn(response.status_code, [201, 400])  # 400 if user already exists from prev run

    def test_auth_list_users(self):
        """Test listing users returns data."""
        response = self.client.get('/api/v1/auth/users')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)

    # --- Pagination Tests ---

    def test_audit_log_pagination(self):
        """Test audit log supports pagination."""
        response = self.client.get('/api/v1/audit-log?page=1&per_page=5')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)
        self.assertIn('meta', data)
        self.assertEqual(data['meta']['page'], 1)
        self.assertEqual(data['meta']['per_page'], 5)
        self.assertIn('total', data['meta'])

    def test_audit_log_pagination_defaults(self):
        """Test audit log works without explicit pagination params (backward compat)."""
        response = self.client.get('/api/v1/audit-log')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)
        self.assertIn('meta', data)
        self.assertEqual(data['meta']['page'], 1)
        self.assertEqual(data['meta']['per_page'], 50)

    def test_audit_log_per_page_cap(self):
        """Test audit log caps per_page at 100."""
        response = self.client.get('/api/v1/audit-log?per_page=500')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['meta']['per_page'], 100)

    def test_notifications_pagination(self):
        """Test notifications supports pagination."""
        response = self.client.get('/api/v1/notifications?page=1&per_page=5')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)
        self.assertIn('meta', data)
        self.assertEqual(data['meta']['page'], 1)
        self.assertEqual(data['meta']['per_page'], 5)
        self.assertIn('total', data['meta'])
        self.assertIn('unread_count', data)

    def test_notifications_pagination_defaults(self):
        """Test notifications works without explicit pagination params (backward compat)."""
        response = self.client.get('/api/v1/notifications')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)
        self.assertIn('meta', data)
        self.assertEqual(data['meta']['page'], 1)
        self.assertEqual(data['meta']['per_page'], 50)

    def test_notifications_per_page_cap(self):
        """Test notifications caps per_page at 100."""
        response = self.client.get('/api/v1/notifications?per_page=999')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['meta']['per_page'], 100)

    # --- Input Validation & SSRF Protection Tests ---

    def test_webhook_ssrf_blocked_localhost(self):
        """Test that localhost webhook URLs are rejected."""
        response = self.client.post('/api/v1/webhooks',
            json={'url': 'http://127.0.0.1:6379/evil'})
        self.assertEqual(response.status_code, 400)

    def test_webhook_ssrf_blocked_file_scheme(self):
        """Test that file:// webhook URLs are rejected."""
        response = self.client.post('/api/v1/webhooks',
            json={'url': 'file:///etc/passwd'})
        self.assertEqual(response.status_code, 400)

    def test_webhook_ssrf_blocked_private_ip(self):
        """Test that private IP webhook URLs are rejected."""
        response = self.client.post('/api/v1/webhooks',
            json={'url': 'http://192.168.1.1/hook'})
        self.assertEqual(response.status_code, 400)

    def test_invalid_language_rejected(self):
        """Test that unsupported language parameter is rejected."""
        import io, numpy as np, soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format='WAV')
        buf.seek(0)
        response = self.client.post('/upload',
            data={'file': (buf, 'test.wav'), 'language': 'xx'},
            content_type='multipart/form-data')
        self.assertIn(response.status_code, [400, 302])

    def test_validator_threshold_rejects_nan(self):
        """Test that NaN threshold values are rejected."""
        from validators import validate_threshold_value
        is_valid, _ = validate_threshold_value('violence', 'energy', float('nan'))
        self.assertFalse(is_valid)

    def test_validator_threshold_rejects_infinity(self):
        """Test that infinity threshold values are rejected."""
        from validators import validate_threshold_value
        is_valid, _ = validate_threshold_value('violence', 'energy', float('inf'))
        self.assertFalse(is_valid)

    def test_validator_audio_file_empty(self):
        """Test that empty files are rejected."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'')
            tmppath = f.name
        from validators import validate_audio_file
        is_valid, _ = validate_audio_file(tmppath, 'wav')
        self.assertFalse(is_valid)
        os.remove(tmppath)

    # --- Structured Logging Tests ---

    def test_logs_endpoint_returns_data(self):
        """Test /api/v1/logs returns log entries."""
        response = self.client.get('/api/v1/logs?limit=10')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)

    def test_logs_endpoint_with_filters(self):
        """Test /api/v1/logs accepts filter parameters."""
        response = self.client.get('/api/v1/logs?level=ERROR&limit=5')
        self.assertEqual(response.status_code, 200)

    def test_auth_me_returns_user(self):
        """Test /auth/me returns current user info."""
        response = self.client.get('/api/v1/auth/me')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('user', data)
        self.assertIn('username', data['user'])

    def test_auth_refresh_returns_new_token(self):
        """Test /auth/refresh returns a new token."""
        # First login to get a token
        login_resp = self.client.post('/api/v1/auth/login',
            json={'username': 'admin', 'password': 'admin'})
        token = login_resp.get_json()['token']

        # Use token to refresh
        response = self.client.post('/api/v1/auth/refresh',
            headers={'Authorization': f'Bearer {token}'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('token', data)

    def test_auth_me_with_token(self):
        """Test /auth/me with valid JWT token."""
        login_resp = self.client.post('/api/v1/auth/login',
            json={'username': 'admin', 'password': 'admin'})
        token = login_resp.get_json()['token']

        response = self.client.get('/api/v1/auth/me',
            headers={'Authorization': f'Bearer {token}'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['user']['username'], 'admin')
        self.assertEqual(data['user']['role'], 'admin')

    def test_user_role_update(self):
        """Test updating a user's role."""
        # First create a user
        self.client.post('/api/v1/auth/register',
            json={'username': 'roletest', 'password': 'testpass', 'role': 'viewer'})
        # Get user list to find the ID
        resp = self.client.get('/api/v1/auth/users')
        users = resp.get_json()['data']
        user = next((u for u in users if u['username'] == 'roletest'), None)
        if user:
            # Update role
            response = self.client.put(f'/api/v1/users/{user["id"]}/role',
                json={'role': 'analyst'})
            self.assertEqual(response.status_code, 200)

    def test_user_deactivate(self):
        """Test deactivating a user."""
        # Create a user to deactivate
        self.client.post('/api/v1/auth/register',
            json={'username': 'deltest', 'password': 'testpass', 'role': 'viewer'})
        resp = self.client.get('/api/v1/auth/users')
        users = resp.get_json()['data']
        user = next((u for u in users if u['username'] == 'deltest'), None)
        if user:
            response = self.client.delete(f'/api/v1/users/{user["id"]}')
            self.assertEqual(response.status_code, 200)

    def test_request_id_in_response_headers(self):
        """Test that X-Request-ID is present in response headers."""
        response = self.client.get('/api/v1/health')
        self.assertIn('X-Request-ID', response.headers)
        self.assertTrue(len(response.headers['X-Request-ID']) > 0)

    def test_request_id_propagated(self):
        """Test that incoming X-Request-ID is propagated."""
        response = self.client.get('/api/v1/health',
            headers={'X-Request-ID': 'test-req-123'})
        self.assertEqual(response.headers.get('X-Request-ID'), 'test-req-123')


if __name__ == '__main__':
    unittest.main()
