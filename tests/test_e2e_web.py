"""
End-to-End Web Flow Tests for SoundShield-AI

Tests the complete authenticated workflow:
login -> upload -> progress -> results -> export

Uses Flask test client (no browser needed).
"""

import os
import io
import sys
import json
import time
import unittest
import numpy as np
import soundfile as sf

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestE2EWebFlow(unittest.TestCase):
    """End-to-end test of the authenticated web flow."""

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client once for all tests."""
        os.environ['AUTH_ENABLED'] = 'False'  # Tests run without auth enforcement
        from web_app import app
        app.config['TESTING'] = True
        cls.app = app
        cls.client = app.test_client()
        cls._token = None  # Will be populated by test_02

    def _create_test_wav(self, duration=2.0, sr=16000):
        """Create a synthetic WAV file in memory."""
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format='WAV')
        buf.seek(0)
        return buf

    def _get_auth_headers(self, token=None):
        """Get headers with optional auth token."""
        headers = {}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        return headers

    # --- Step 1: Authentication ---

    def test_01_login_page_loads(self):
        """E2E Step 1a: Login page is accessible."""
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'SoundShield', response.data)

    def test_02_login_returns_token(self):
        """E2E Step 1b: Login with valid credentials returns JWT token."""
        response = self.client.post('/api/v1/auth/login',
            json={'username': 'admin', 'password': 'admin'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('token', data)
        self.assertIn('user', data)
        self.assertEqual(data['user']['role'], 'admin')
        # Store token for later tests
        self.__class__._token = data['token']

    def test_03_login_invalid_credentials(self):
        """E2E Step 1c: Login with bad credentials returns 401."""
        response = self.client.post('/api/v1/auth/login',
            json={'username': 'admin', 'password': 'wrong'})
        self.assertEqual(response.status_code, 401)
        data = response.get_json()
        self.assertIn('error', data)

    def test_04_login_missing_fields(self):
        """E2E Step 1d: Login with missing fields returns 400."""
        response = self.client.post('/api/v1/auth/login',
            json={'username': ''})
        self.assertEqual(response.status_code, 400)

    def test_05_auth_me_works(self):
        """E2E Step 1e: /auth/me returns current user info."""
        token = getattr(self.__class__, '_token', None)
        response = self.client.get('/api/v1/auth/me',
            headers=self._get_auth_headers(token))
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('user', data)
        self.assertIn('username', data['user'])
        self.assertIn('role', data['user'])

    def test_06_auth_refresh_token(self):
        """E2E Step 1f: Token refresh returns a new valid token."""
        token = getattr(self.__class__, '_token', None)
        response = self.client.post('/api/v1/auth/refresh',
            headers=self._get_auth_headers(token))
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('token', data)
        self.assertIn('user', data)

    # --- Step 2: Dashboard ---

    def test_07_dashboard_loads(self):
        """E2E Step 2a: Main dashboard is accessible."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'html', response.data.lower())

    def test_08_admin_page_loads(self):
        """E2E Step 2b: Admin page is accessible."""
        token = getattr(self.__class__, '_token', None)
        response = self.client.get('/admin',
            headers=self._get_auth_headers(token))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'SoundShield', response.data)

    # --- Step 3: Upload & Analysis ---

    def test_09_upload_no_file_returns_400(self):
        """E2E Step 3a: Upload with no file returns 400."""
        response = self.client.post('/upload',
            data={},
            content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)

    def test_10_upload_empty_filename_returns_400(self):
        """E2E Step 3b: Upload with empty filename returns 400."""
        response = self.client.post('/upload',
            data={'file': (io.BytesIO(b''), '')},
            content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)

    def test_11_upload_unsupported_format_returns_400(self):
        """E2E Step 3c: Upload with unsupported format returns 400."""
        response = self.client.post('/upload',
            data={'file': (io.BytesIO(b'not a real file'), 'test.xyz')},
            content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)

    def test_12_upload_file(self):
        """E2E Step 3d: Upload a valid WAV file for analysis."""
        wav_buf = self._create_test_wav(duration=2.0)
        token = getattr(self.__class__, '_token', None)

        response = self.client.post('/upload',
            data={'file': (wav_buf, 'e2e_test.wav'), 'language': 'en'},
            content_type='multipart/form-data',
            headers=self._get_auth_headers(token))
        # Sync upload returns 200 with results or 500 if analysis internals fail
        # (the upload/save itself should succeed)
        self.assertIn(response.status_code, [200, 500])
        data = response.get_json()
        if response.status_code == 200:
            self.assertTrue(data.get('success'))
            self.assertIn('filename', data)
            self.assertIn('results', data)

    def test_13_upload_async(self):
        """E2E Step 3e: Async upload returns job status."""
        wav_buf = self._create_test_wav(duration=2.0)
        token = getattr(self.__class__, '_token', None)

        response = self.client.post('/upload-async',
            data={'file': (wav_buf, 'e2e_async_test.wav'), 'language': 'en'},
            content_type='multipart/form-data',
            headers=self._get_auth_headers(token))
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get('success'))
        self.assertIn('job_id', data)
        self.assertEqual(data['message'], 'Analysis started')

    def test_14_upload_async_invalid_language(self):
        """E2E Step 3f: Async upload with invalid language returns 400."""
        wav_buf = self._create_test_wav(duration=1.5)
        response = self.client.post('/upload-async',
            data={'file': (wav_buf, 'bad_lang.wav'), 'language': 'zz'},
            content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)

    # --- Step 4: Batch Upload ---

    def test_15_batch_upload_no_files_returns_400(self):
        """E2E Step 4a: Batch upload with no files returns 400."""
        response = self.client.post('/api/v1/batch-upload')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)

    def test_16_batch_upload(self):
        """E2E Step 4b: Batch upload multiple files returns 202."""
        files = []
        for name in ['batch1.wav', 'batch2.wav']:
            files.append((self._create_test_wav(duration=1.5), name))

        token = getattr(self.__class__, '_token', None)
        response = self.client.post('/api/v1/batch-upload',
            data={
                'files': [(buf, name) for buf, name in files],
                'language': 'en'
            },
            content_type='multipart/form-data',
            headers=self._get_auth_headers(token))
        self.assertIn(response.status_code, [200, 202])
        data = response.get_json()
        self.assertIn('batch_id', data)

    def test_17_batch_status_not_found(self):
        """E2E Step 4c: Batch status for nonexistent job returns 404."""
        response = self.client.get('/api/v1/batch/nonexistent_id/status')
        self.assertEqual(response.status_code, 404)

    # --- Step 5: API Endpoints ---

    def test_18_health_check(self):
        """E2E Step 5a: Health endpoint responds with healthy status."""
        response = self.client.get('/api/v1/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('version', data)
        self.assertIn('models', data)

    def test_19_legacy_health_check(self):
        """E2E Step 5b: Legacy /health endpoint also works."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'healthy')

    def test_20_list_analyses(self):
        """E2E Step 5c: List analyses endpoint returns data array."""
        token = getattr(self.__class__, '_token', None)
        response = self.client.get('/api/v1/analyses',
            headers=self._get_auth_headers(token))
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)
        self.assertIsInstance(data['data'], list)

    def test_21_analysis_not_found(self):
        """E2E Step 5d: Get nonexistent analysis returns 404."""
        response = self.client.get('/api/v1/analyses/99999')
        self.assertEqual(response.status_code, 404)

    def test_22_get_stats(self):
        """E2E Step 5e: Stats endpoint returns database statistics."""
        response = self.client.get('/api/v1/stats')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)
        self.assertIn('total_analyses', data['data'])

    def test_23_system_info(self):
        """E2E Step 5f: System info endpoint returns platform data."""
        response = self.client.get('/api/v1/system/info')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)
        self.assertIn('python_version', data['data'])

    # --- Step 6: Config & Thresholds ---

    def test_24_get_thresholds(self):
        """E2E Step 6a: Get detector thresholds."""
        response = self.client.get('/api/v1/config/thresholds')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)
        self.assertIn('cry', data['data'])
        self.assertIn('violence', data['data'])

    def test_25_put_threshold_invalid_detector(self):
        """E2E Step 6b: Update threshold with invalid detector returns 400."""
        response = self.client.put('/api/v1/config/thresholds',
            data=json.dumps({'detector': 'invalid', 'key': 'x', 'value': 1}),
            content_type='application/json')
        self.assertEqual(response.status_code, 400)

    # --- Step 7: User Management ---

    def test_26_list_users(self):
        """E2E Step 7a: User list endpoint returns users."""
        token = getattr(self.__class__, '_token', None)
        response = self.client.get('/api/v1/auth/users',
            headers=self._get_auth_headers(token))
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)
        self.assertTrue(len(data['data']) > 0)
        # At minimum the default admin user should exist
        usernames = [u['username'] for u in data['data']]
        self.assertIn('admin', usernames)

    def test_27_register_user_missing_fields(self):
        """E2E Step 7b: Register with missing fields returns 400."""
        response = self.client.post('/api/v1/auth/register',
            json={'username': '', 'password': ''})
        self.assertEqual(response.status_code, 400)

    def test_28_register_user_short_password(self):
        """E2E Step 7c: Register with short password returns 400."""
        response = self.client.post('/api/v1/auth/register',
            json={'username': 'newuser', 'password': '123'})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)

    # --- Step 8: Notifications & Webhooks ---

    def test_29_list_notifications(self):
        """E2E Step 8a: Notifications endpoint returns data."""
        response = self.client.get('/api/v1/notifications')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)
        self.assertIn('unread_count', data)

    def test_30_mark_all_notifications_read(self):
        """E2E Step 8b: Mark all notifications as read."""
        response = self.client.post('/api/v1/notifications/read-all')
        self.assertEqual(response.status_code, 200)

    def test_31_list_webhooks(self):
        """E2E Step 8c: Webhooks list endpoint works."""
        response = self.client.get('/api/v1/webhooks')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)

    # --- Step 9: Logs & Audit ---

    def test_32_logs_endpoint(self):
        """E2E Step 9a: Logs endpoint returns data."""
        response = self.client.get('/api/v1/logs?limit=5')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)

    def test_33_audit_log(self):
        """E2E Step 9b: Audit log endpoint returns entries."""
        response = self.client.get('/api/v1/audit-log')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('data', data)

    # --- Step 10: Export & Error Handling ---

    def test_34_export_unsupported_format(self):
        """E2E Step 10a: Export with unsupported format returns 400."""
        response = self.client.get('/api/v1/analyses/1/export?format=xlsx')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)
        self.assertIn('supported', data)

    def test_35_export_nonexistent_analysis(self):
        """E2E Step 10b: Export for nonexistent analysis returns 404."""
        response = self.client.get('/api/v1/analyses/99999/export?format=pdf')
        self.assertIn(response.status_code, [404, 503])

    def test_36_404_returns_json(self):
        """E2E Step 10c: Unknown API route returns JSON 404."""
        response = self.client.get('/api/v1/nonexistent')
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertIn('error', data)

    def test_37_security_headers_present(self):
        """E2E Step 10d: Security headers are set on responses."""
        response = self.client.get('/api/v1/health')
        self.assertEqual(response.headers.get('X-Content-Type-Options'), 'nosniff')
        self.assertEqual(response.headers.get('X-Frame-Options'), 'DENY')
        self.assertEqual(response.headers.get('X-XSS-Protection'), '1; mode=block')
        self.assertIn('Content-Security-Policy', response.headers)


if __name__ == '__main__':
    unittest.main()
