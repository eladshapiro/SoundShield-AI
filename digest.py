"""
Digest Generator for SoundShield-AI

Generates daily and weekly summary digests of analysis activity,
risk distribution, and incident trends.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class DigestGenerator:
    """Generates summary digests from analysis data."""

    def __init__(self, db=None):
        self.db = db

    def generate_daily_digest(self) -> Dict:
        """Generate a digest for the last 24 hours."""
        return self._generate_digest(hours=24, period='daily')

    def generate_weekly_digest(self) -> Dict:
        """Generate a digest for the last 7 days."""
        return self._generate_digest(hours=168, period='weekly')

    def _generate_digest(self, hours: int, period: str) -> Dict:
        """Generate a digest for the given time window."""
        if not self.db:
            return {'error': 'Database not available', 'period': period}

        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        conn = self.db._get_conn()
        try:
            # Total analyses
            row = conn.execute(
                'SELECT COUNT(*) as total FROM analyses WHERE upload_time >= ?',
                (cutoff,)
            ).fetchone()
            total = row['total'] if row else 0

            # Risk distribution
            risk_rows = conn.execute(
                'SELECT risk_level, COUNT(*) as count FROM analyses WHERE upload_time >= ? GROUP BY risk_level',
                (cutoff,)
            ).fetchall()
            risk_dist = {r['risk_level']: r['count'] for r in risk_rows} if risk_rows else {}

            # Incident stats
            incident_row = conn.execute(
                'SELECT SUM(total_incidents) as total_inc, SUM(critical_incidents) as total_crit FROM analyses WHERE upload_time >= ?',
                (cutoff,)
            ).fetchone()
            total_incidents = incident_row['total_inc'] or 0 if incident_row else 0
            total_critical = incident_row['total_crit'] or 0 if incident_row else 0

            # Average duration
            dur_row = conn.execute(
                'SELECT AVG(duration) as avg_dur FROM analyses WHERE upload_time >= ? AND duration > 0',
                (cutoff,)
            ).fetchone()
            avg_duration = round(dur_row['avg_dur'] or 0, 1) if dur_row else 0

            # Top incident types from incidents table
            type_rows = conn.execute(
                '''SELECT i.type, COUNT(*) as count
                   FROM incidents i JOIN analyses a ON i.analysis_id = a.id
                   WHERE a.upload_time >= ?
                   GROUP BY i.type ORDER BY count DESC LIMIT 5''',
                (cutoff,)
            ).fetchall()
            top_types = {r['type']: r['count'] for r in type_rows} if type_rows else {}

        finally:
            conn.close()

        return {
            'period': period,
            'generated_at': datetime.now().isoformat(),
            'window_hours': hours,
            'total_analyses': total,
            'risk_distribution': risk_dist,
            'total_incidents': total_incidents,
            'total_critical': total_critical,
            'avg_audio_duration': avg_duration,
            'top_incident_types': top_types,
        }

    def format_as_text(self, digest: Dict) -> str:
        """Format digest as plain text summary."""
        lines = [
            f"SoundShield-AI {digest.get('period', 'Daily').title()} Digest",
            f"Generated: {digest.get('generated_at', 'N/A')}",
            f"Window: Last {digest.get('window_hours', 24)} hours",
            "",
            f"Total Analyses: {digest.get('total_analyses', 0)}",
            f"Total Incidents: {digest.get('total_incidents', 0)}",
            f"Critical Incidents: {digest.get('total_critical', 0)}",
            f"Avg Audio Duration: {digest.get('avg_audio_duration', 0)}s",
            "",
            "Risk Distribution:",
        ]
        for level, count in digest.get('risk_distribution', {}).items():
            lines.append(f"  {level}: {count}")

        if digest.get('top_incident_types'):
            lines.append("")
            lines.append("Top Incident Types:")
            for itype, count in digest.get('top_incident_types', {}).items():
                lines.append(f"  {itype}: {count}")

        return '\n'.join(lines)
