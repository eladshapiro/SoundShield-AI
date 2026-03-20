"""
Database Module for SoundShield-AI
SQLite persistence for analysis history and incidents.
"""
import sqlite3
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'soundshield.db')


class Database:
    """SQLite database for persisting analysis history."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    original_filename TEXT,
                    upload_time TEXT NOT NULL DEFAULT (datetime('now')),
                    duration REAL,
                    risk_level TEXT,
                    overall_assessment TEXT,
                    total_incidents INTEGER DEFAULT 0,
                    critical_incidents INTEGER DEFAULT 0,
                    language TEXT DEFAULT 'en',
                    report_path TEXT,
                    models_used TEXT,
                    results_json TEXT
                );

                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    severity TEXT,
                    confidence REAL,
                    detected_value TEXT,
                    is_ml_backed INTEGER DEFAULT 0,
                    speaker_type TEXT,
                    details_json TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_analyses_upload_time ON analyses(upload_time);
                CREATE INDEX IF NOT EXISTS idx_incidents_analysis_id ON incidents(analysis_id);
                CREATE INDEX IF NOT EXISTS idx_incidents_type ON incidents(type);
            ''')
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
        finally:
            conn.close()

    def save_analysis(self, filename: str, original_filename: str,
                      analysis_results: Dict, report: Dict,
                      report_path: str = None, models_used: str = None) -> int:
        """Save analysis results to database. Returns analysis ID."""
        conn = self._get_conn()
        try:
            summary = report.get('summary', {})

            cursor = conn.execute('''
                INSERT INTO analyses
                (filename, original_filename, duration, risk_level, overall_assessment,
                 total_incidents, critical_incidents, language, report_path, models_used,
                 results_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                filename,
                original_filename,
                analysis_results.get('duration', 0),
                summary.get('risk_level', 'low'),
                summary.get('overall_assessment', 'normal'),
                summary.get('total_incidents', 0),
                summary.get('critical_incidents', 0),
                analysis_results.get('language', 'en'),
                report_path,
                models_used,
                json.dumps(self._make_serializable(analysis_results), ensure_ascii=False)
            ))
            analysis_id = cursor.lastrowid

            # Save individual incidents
            self._save_incidents(conn, analysis_id, analysis_results)

            conn.commit()
            logger.info(f"Saved analysis {analysis_id} for {original_filename}")
            return analysis_id
        finally:
            conn.close()

    def _save_incidents(self, conn: sqlite3.Connection, analysis_id: int,
                        analysis_results: Dict):
        """Save individual incidents for an analysis."""
        # Concerning emotions
        for emotion in analysis_results.get('concerning_emotions', []):
            conn.execute('''
                INSERT INTO incidents
                (analysis_id, type, start_time, end_time, severity, confidence,
                 detected_value, is_ml_backed, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id, 'emotion',
                emotion.get('start_time'), emotion.get('end_time'),
                emotion.get('severity', 'low'), emotion.get('confidence', 0),
                emotion.get('detected_emotion', ''),
                emotion.get('ml_backed', 0),
                json.dumps(emotion.get('all_scores', {}))
            ))

        # Violence segments
        for violence in analysis_results.get('violence_segments', []):
            conn.execute('''
                INSERT INTO incidents
                (analysis_id, type, start_time, end_time, severity, confidence,
                 detected_value, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id, 'violence',
                violence.get('start_time'), violence.get('end_time'),
                violence.get('adjusted_severity', 'low'),
                violence.get('confidence', 0),
                ', '.join(violence.get('violence_types', [])),
                json.dumps(violence.get('context', {}))
            ))

        # Cry segments
        for cry in analysis_results.get('cry_with_responses', []):
            conn.execute('''
                INSERT INTO incidents
                (analysis_id, type, start_time, end_time, severity, confidence,
                 detected_value, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id, 'cry',
                cry.get('start_time'), cry.get('end_time'),
                cry.get('intensity', 'medium'), 0,
                'response' if cry.get('response_detected') else 'no_response',
                json.dumps({'response_quality': cry.get('response_quality', 'none')})
            ))

        # Neglect - unanswered cries
        neglect = analysis_results.get('neglect_analysis', {})
        for unanswered in neglect.get('unanswered_cries', []):
            conn.execute('''
                INSERT INTO incidents
                (analysis_id, type, start_time, end_time, severity, confidence,
                 detected_value, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id, 'neglect',
                unanswered.get('cry_start_time'), unanswered.get('cry_end_time'),
                unanswered.get('neglect_severity', 'low'), 0,
                'unanswered_cry',
                json.dumps({'time_without_response': unanswered.get('time_without_response', 0)})
            ))

    def _make_serializable(self, obj):
        """Make analysis results JSON-serializable by converting numpy types."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()
                    if k not in ('audio_analysis',)}  # Skip raw audio data
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def get_analysis(self, analysis_id: int) -> Optional[Dict]:
        """Get a single analysis by ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                'SELECT * FROM analyses WHERE id = ?', (analysis_id,)
            ).fetchone()
            if not row:
                return None
            return dict(row)
        finally:
            conn.close()

    def get_analysis_history(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get analysis history, newest first."""
        conn = self._get_conn()
        try:
            rows = conn.execute('''
                SELECT id, filename, original_filename, upload_time, duration,
                       risk_level, overall_assessment, total_incidents,
                       critical_incidents, language, models_used
                FROM analyses
                ORDER BY upload_time DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset)).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_incidents(self, analysis_id: int) -> List[Dict]:
        """Get all incidents for an analysis."""
        conn = self._get_conn()
        try:
            rows = conn.execute('''
                SELECT * FROM incidents WHERE analysis_id = ? ORDER BY start_time
            ''', (analysis_id,)).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_comparison_data(self, analysis_ids: List[int]) -> List[Dict]:
        """Get data for comparing multiple analyses side by side."""
        conn = self._get_conn()
        try:
            placeholders = ','.join('?' * len(analysis_ids))
            analyses = conn.execute(f'''
                SELECT * FROM analyses WHERE id IN ({placeholders})
            ''', analysis_ids).fetchall()

            result = []
            for analysis in analyses:
                analysis_dict = dict(analysis)
                incidents = conn.execute('''
                    SELECT type, COUNT(*) as count,
                           AVG(confidence) as avg_confidence
                    FROM incidents WHERE analysis_id = ?
                    GROUP BY type
                ''', (analysis_dict['id'],)).fetchall()
                analysis_dict['incident_summary'] = [dict(i) for i in incidents]
                result.append(analysis_dict)

            return result
        finally:
            conn.close()

    def search_analyses(self, query: str = None, risk_level: str = None,
                        date_from: str = None, date_to: str = None) -> List[Dict]:
        """Search analyses with filters."""
        conn = self._get_conn()
        try:
            conditions = []
            params = []

            if query:
                conditions.append("(original_filename LIKE ? OR filename LIKE ?)")
                params.extend([f'%{query}%', f'%{query}%'])
            if risk_level:
                conditions.append("risk_level = ?")
                params.append(risk_level)
            if date_from:
                conditions.append("upload_time >= ?")
                params.append(date_from)
            if date_to:
                conditions.append("upload_time <= ?")
                params.append(date_to)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            rows = conn.execute(f'''
                SELECT id, filename, original_filename, upload_time, duration,
                       risk_level, overall_assessment, total_incidents,
                       critical_incidents, language
                FROM analyses {where}
                ORDER BY upload_time DESC
                LIMIT 100
            ''', params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def delete_analysis(self, analysis_id: int) -> bool:
        """Delete an analysis and its incidents."""
        conn = self._get_conn()
        try:
            conn.execute('DELETE FROM analyses WHERE id = ?', (analysis_id,))
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()
