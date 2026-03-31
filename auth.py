"""
Authentication & Role-Based Access Control for SoundShield-AI

Provides JWT-based authentication with three roles:
- viewer: Read dashboards and reports
- analyst: Upload and analyze files
- admin: Full access including settings, deletion, webhooks
"""

import os
import sqlite3
import logging
import functools
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from flask import request, jsonify, g

from config import config

logger = logging.getLogger(__name__)

# JWT settings
JWT_ALGORITHM = 'HS256'
JWT_EXPIRY_HOURS = 24

# Role hierarchy: higher roles inherit lower role permissions
ROLE_HIERARCHY = {
    'admin': ['admin', 'analyst', 'viewer'],
    'analyst': ['analyst', 'viewer'],
    'viewer': ['viewer'],
}


class UserStore:
    """SQLite-backed user storage with bcrypt password hashing."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.database.db_path
        self._init_table()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self):
        conn = self._get_conn()
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'viewer',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    last_login TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1
                )
            ''')
            conn.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username
                ON users(username)
            ''')
            conn.commit()

            # Create default admin if no users exist
            count = conn.execute('SELECT COUNT(*) as c FROM users').fetchone()['c']
            if count == 0:
                self.create_user('admin', 'admin', 'admin')
                logger.info("Created default admin user (username: admin, password: admin)")
        finally:
            conn.close()

    def create_user(self, username: str, password: str, role: str = 'viewer') -> dict:
        """Create a new user with bcrypt-hashed password."""
        if role not in ROLE_HIERARCHY:
            raise ValueError(f"Invalid role: {role}. Must be one of: {list(ROLE_HIERARCHY.keys())}")

        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        conn = self._get_conn()
        try:
            conn.execute(
                'INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)',
                (username, password_hash, role)
            )
            conn.commit()
            return {'username': username, 'role': role}
        except sqlite3.IntegrityError:
            raise ValueError(f"User '{username}' already exists")
        finally:
            conn.close()

    def authenticate(self, username: str, password: str) -> dict:
        """Verify credentials and return user info or None."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                'SELECT id, username, password_hash, role, is_active FROM users WHERE username = ?',
                (username,)
            ).fetchone()

            if not row or not row['is_active']:
                return None

            if bcrypt.checkpw(password.encode('utf-8'), row['password_hash'].encode('utf-8')):
                # Update last login
                conn.execute(
                    'UPDATE users SET last_login = datetime("now") WHERE id = ?',
                    (row['id'],)
                )
                conn.commit()
                return {
                    'id': row['id'],
                    'username': row['username'],
                    'role': row['role']
                }
            return None
        finally:
            conn.close()

    def authenticate_by_id(self, user_id: int) -> dict:
        """Look up a user by ID (for token refresh). Returns user dict or None."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                'SELECT id, username, role, is_active FROM users WHERE id = ? AND is_active = 1',
                (user_id,)
            ).fetchone()
            if not row:
                return None
            return {
                'id': row['id'],
                'username': row['username'],
                'role': row['role']
            }
        finally:
            conn.close()

    def list_users(self) -> list:
        """List all users (without password hashes)."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                'SELECT id, username, role, created_at, last_login, is_active FROM users'
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def generate_token(user: dict) -> str:
    """Generate a JWT token for an authenticated user."""
    payload = {
        'user_id': user['id'],
        'username': user['username'],
        'role': user['role'],
        'exp': datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        'iat': datetime.now(timezone.utc),
    }
    return jwt.encode(payload, config.security.secret_key, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token."""
    return jwt.decode(token, config.security.secret_key, algorithms=[JWT_ALGORITHM])


def get_current_user():
    """Extract current user from the Authorization header. Returns None if no auth."""
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return None

    token = auth_header[7:]
    try:
        payload = decode_token(token)
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def require_role(role: str):
    """Decorator to require a specific role for an endpoint.

    If AUTH_ENABLED is False (default for dev), allows all requests through.
    When enabled, checks JWT token and verifies role hierarchy.
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Skip auth if disabled (development mode)
            if not config.security.auth_enabled:
                g.current_user = {'id': 0, 'username': 'anonymous', 'role': 'admin'}
                return f(*args, **kwargs)

            user = get_current_user()
            if user is None:
                return jsonify({'error': 'Authentication required', 'hint': 'Provide Authorization: Bearer <token>'}), 401

            # Check role hierarchy
            user_permissions = ROLE_HIERARCHY.get(user.get('role', ''), [])
            if role not in user_permissions:
                return jsonify({'error': 'Insufficient permissions', 'required_role': role}), 403

            g.current_user = user
            return f(*args, **kwargs)
        return wrapper
    return decorator
