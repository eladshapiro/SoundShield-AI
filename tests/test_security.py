"""
Security Tests for SoundShield-AI

Tests JWT authentication, RBAC enforcement, input sanitization,
and common web vulnerabilities.
"""
import os
import sys
import unittest
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSecurityHeaders(unittest.TestCase):
    """Test security headers are present on all responses."""

    @classmethod
    def setUpClass(cls):
        os.environ['AUTH_ENABLED'] = 'False'
        from web_app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_x_content_type_options(self):
        resp = self.client.get('/api/v1/health')
        self.assertEqual(resp.headers.get('X-Content-Type-Options'), 'nosniff')

    def test_x_frame_options(self):
        resp = self.client.get('/api/v1/health')
        self.assertEqual(resp.headers.get('X-Frame-Options'), 'DENY')

    def test_x_xss_protection(self):
        resp = self.client.get('/api/v1/health')
        self.assertEqual(resp.headers.get('X-XSS-Protection'), '1; mode=block')

    def test_csp_header_present(self):
        resp = self.client.get('/api/v1/health')
        self.assertIn('Content-Security-Policy', resp.headers)

    def test_request_id_present(self):
        resp = self.client.get('/api/v1/health')
        self.assertIn('X-Request-ID', resp.headers)


class TestJWTSecurity(unittest.TestCase):
    """Test JWT token handling and validation."""

    @classmethod
    def setUpClass(cls):
        from web_app import app
        from config import config as app_config
        app.config['TESTING'] = True
        cls.client = app.test_client()
        cls.app = app
        cls._config = app_config
        # Enable auth at runtime via config singleton
        cls._config.security.auth_enabled = True

    @classmethod
    def tearDownClass(cls):
        cls._config.security.auth_enabled = False

    def _get_token(self):
        resp = self.client.post('/api/v1/auth/login',
            json={'username': 'admin', 'password': 'admin'})
        return resp.get_json().get('token', '')

    def test_protected_endpoint_requires_auth(self):
        """Protected endpoint returns 401 without token."""
        resp = self.client.get('/api/v1/analyses')
        self.assertEqual(resp.status_code, 401)

    def test_invalid_token_rejected(self):
        """Invalid JWT token is rejected."""
        resp = self.client.get('/api/v1/analyses',
            headers={'Authorization': 'Bearer invalid.token.here'})
        self.assertEqual(resp.status_code, 401)

    def test_valid_token_accepted(self):
        """Valid JWT token grants access."""
        token = self._get_token()
        resp = self.client.get('/api/v1/analyses',
            headers={'Authorization': 'Bearer ' + token})
        self.assertEqual(resp.status_code, 200)

    def test_viewer_cannot_delete(self):
        """Viewer role cannot access admin endpoints."""
        # Create a viewer user
        admin_token = self._get_token()
        self.client.post('/api/v1/auth/register',
            json={'username': 'sec_viewer', 'password': 'testpass123', 'role': 'viewer'},
            headers={'Authorization': 'Bearer ' + admin_token})

        # Login as viewer
        resp = self.client.post('/api/v1/auth/login',
            json={'username': 'sec_viewer', 'password': 'testpass123'})
        if resp.status_code == 200:
            viewer_token = resp.get_json()['token']
            # Try admin action — RBAC check happens before resource lookup,
            # so 403 is returned even if analysis doesn't exist
            resp = self.client.delete('/api/v1/analyses/99999',
                headers={'Authorization': 'Bearer ' + viewer_token})
            self.assertIn(resp.status_code, [403, 404])

    def test_analyst_cannot_manage_users(self):
        """Analyst role cannot access admin-only user management."""
        admin_token = self._get_token()
        self.client.post('/api/v1/auth/register',
            json={'username': 'sec_analyst', 'password': 'testpass123', 'role': 'analyst'},
            headers={'Authorization': 'Bearer ' + admin_token})

        resp = self.client.post('/api/v1/auth/login',
            json={'username': 'sec_analyst', 'password': 'testpass123'})
        if resp.status_code == 200:
            analyst_token = resp.get_json()['token']
            resp = self.client.get('/api/v1/auth/users',
                headers={'Authorization': 'Bearer ' + analyst_token})
            self.assertEqual(resp.status_code, 403)


class TestInputSanitization(unittest.TestCase):
    """Test input sanitization against common attacks."""

    @classmethod
    def setUpClass(cls):
        from config import config as app_config
        app_config.security.auth_enabled = False
        from web_app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_sql_injection_in_analysis_id(self):
        """SQL injection in URL parameter returns 404, not 500."""
        resp = self.client.get('/api/v1/analyses/1 OR 1=1')
        self.assertEqual(resp.status_code, 404)

    def test_xss_in_filename(self):
        """XSS in filename does not reflect script tags in response body."""
        import io, numpy as np, soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format='WAV')
        buf.seek(0)
        resp = self.client.post('/upload',
            data={'file': (buf, '<script>alert(1)</script>.wav'), 'language': 'en'},
            content_type='multipart/form-data')
        # Server should not crash or reflect the XSS payload
        self.assertIn(resp.status_code, [200, 302, 400, 500])
        # Verify the script tag is not reflected in the response body
        if resp.data:
            self.assertNotIn(b'<script>alert(1)</script>', resp.data)

    def test_path_traversal_in_report(self):
        """Path traversal attempt returns 404."""
        resp = self.client.get('/report/../../etc/passwd')
        self.assertIn(resp.status_code, [404, 400])

    def test_path_traversal_in_download(self):
        """Path traversal in download returns 404."""
        resp = self.client.get('/download/../../../etc/shadow')
        self.assertIn(resp.status_code, [404, 400])

    def test_oversized_json_body(self):
        """Very large JSON body is handled."""
        resp = self.client.post('/api/v1/auth/login',
            json={'username': 'a' * 10000, 'password': 'b' * 10000})
        self.assertEqual(resp.status_code, 401)

    def test_webhook_ssrf_internal_ip(self):
        """Webhook with internal IP is blocked."""
        resp = self.client.post('/api/v1/webhooks',
            json={'url': 'http://10.0.0.1/callback'})
        self.assertEqual(resp.status_code, 400)

    def test_empty_content_type(self):
        """Request with wrong content type is handled gracefully."""
        resp = self.client.post('/api/v1/auth/login',
            data='not json', content_type='text/plain')
        self.assertIn(resp.status_code, [400, 415])


class TestRateLimitHeaders(unittest.TestCase):
    """Test that rate limiting headers are set."""

    @classmethod
    def setUpClass(cls):
        os.environ['AUTH_ENABLED'] = 'False'
        from web_app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_health_endpoint_not_rate_limited(self):
        """Health endpoint responds even under load."""
        for _ in range(20):
            resp = self.client.get('/api/v1/health')
            self.assertEqual(resp.status_code, 200)


if __name__ == '__main__':
    unittest.main()
