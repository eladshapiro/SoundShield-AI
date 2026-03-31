"""
Audit Logging for SoundShield-AI

Tracks all user actions on sensitive data: uploads, analyses,
report views, and deletions. Required for COPPA/GDPR compliance.
"""

import logging
import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, List
from config import config

logger = logging.getLogger(__name__)


class AuditLogger:
    """Append-only audit log for compliance tracking."""

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
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                    action TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id TEXT,
                    user_ip TEXT,
                    user_agent TEXT,
                    details TEXT,
                    status TEXT DEFAULT 'success'
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON audit_log(timestamp)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_audit_action
                ON audit_log(action)
            ''')
            conn.commit()
            logger.info("Audit log table initialized")
        finally:
            conn.close()

    def log(self, action: str, resource_type: str = None,
            resource_id: str = None, user_ip: str = None,
            user_agent: str = None, details: Dict = None,
            status: str = 'success'):
        """Record an audit event."""
        conn = self._get_conn()
        try:
            conn.execute('''
                INSERT INTO audit_log
                (action, resource_type, resource_id, user_ip, user_agent, details, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                action,
                resource_type,
                str(resource_id) if resource_id else None,
                user_ip,
                user_agent,
                json.dumps(details) if details else None,
                status,
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Audit log write failed: {e}")
        finally:
            conn.close()

    def log_upload(self, filename: str, user_ip: str = None, user_agent: str = None):
        self.log('upload', 'audio_file', filename, user_ip, user_agent)

    def log_analysis_start(self, filename: str, language: str,
                           user_ip: str = None):
        self.log('analysis_start', 'audio_file', filename, user_ip,
                 details={'language': language})

    def log_analysis_complete(self, filename: str, analysis_id: int,
                              risk_level: str):
        self.log('analysis_complete', 'analysis', str(analysis_id),
                 details={'filename': filename, 'risk_level': risk_level})

    def log_report_view(self, analysis_id: int, user_ip: str = None):
        self.log('report_view', 'analysis', str(analysis_id), user_ip)

    def log_report_download(self, filename: str, user_ip: str = None):
        self.log('download', 'report', filename, user_ip)

    def log_deletion(self, resource_type: str, resource_id: str,
                     user_ip: str = None):
        self.log('delete', resource_type, resource_id, user_ip)

    def get_recent_logs(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Retrieve recent audit entries."""
        conn = self._get_conn()
        try:
            rows = conn.execute('''
                SELECT * FROM audit_log
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset)).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def count_entries(self, action_filter: str = None) -> int:
        """Return total number of audit log entries, optionally filtered by action."""
        conn = self._get_conn()
        try:
            if action_filter:
                row = conn.execute(
                    'SELECT COUNT(*) AS cnt FROM audit_log WHERE action = ?',
                    (action_filter,)
                ).fetchone()
            else:
                row = conn.execute('SELECT COUNT(*) AS cnt FROM audit_log').fetchone()
            return row['cnt']
        finally:
            conn.close()

    def get_logs_for_resource(self, resource_type: str,
                              resource_id: str) -> List[Dict]:
        """Get all audit entries for a specific resource."""
        conn = self._get_conn()
        try:
            rows = conn.execute('''
                SELECT * FROM audit_log
                WHERE resource_type = ? AND resource_id = ?
                ORDER BY timestamp DESC
            ''', (resource_type, resource_id)).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
