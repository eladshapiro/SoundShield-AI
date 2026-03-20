"""
Notification System for SoundShield-AI

Sends alerts when critical incidents are detected during analysis.
Supports: webhook (HTTP POST), console logging, and in-app notifications.
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from config import config

logger = logging.getLogger(__name__)


@dataclass
class Notification:
    """A single notification event."""
    id: str
    timestamp: str
    level: str  # info, warning, critical
    title: str
    message: str
    incident_type: str  # violence, cry, emotion, neglect, language
    analysis_id: Optional[int] = None
    filename: Optional[str] = None
    details: Dict = field(default_factory=dict)
    read: bool = False


class NotificationManager:
    """Manages alert notifications for detected incidents."""

    def __init__(self):
        self._notifications: List[Notification] = []
        self._webhooks: List[str] = []
        self._callbacks: List[Callable] = []
        self._max_stored = 500
        self._counter = 0

        # Load webhook URLs from env
        webhook_url = config.web.cors_origins  # placeholder — real impl uses dedicated env var
        import os
        webhook_env = os.getenv('NOTIFICATION_WEBHOOK_URL', '')
        if webhook_env:
            self._webhooks = [url.strip() for url in webhook_env.split(',') if url.strip()]

        logger.info(f"NotificationManager initialized ({len(self._webhooks)} webhooks)")

    def _next_id(self) -> str:
        self._counter += 1
        return f"notif_{int(time.time())}_{self._counter}"

    def notify(self, level: str, title: str, message: str,
               incident_type: str, analysis_id: int = None,
               filename: str = None, details: Dict = None) -> Notification:
        """Create and dispatch a notification."""
        notif = Notification(
            id=self._next_id(),
            timestamp=datetime.utcnow().isoformat() + 'Z',
            level=level,
            title=title,
            message=message,
            incident_type=incident_type,
            analysis_id=analysis_id,
            filename=filename,
            details=details or {},
        )

        # Store in-app
        self._notifications.insert(0, notif)
        if len(self._notifications) > self._max_stored:
            self._notifications = self._notifications[:self._max_stored]

        # Log
        log_fn = logger.critical if level == 'critical' else (
            logger.warning if level == 'warning' else logger.info
        )
        log_fn(f"ALERT [{level.upper()}] {title}: {message}")

        # Send webhooks
        self._send_webhooks(notif)

        # Call registered callbacks
        for cb in self._callbacks:
            try:
                cb(notif)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")

        return notif

    def notify_critical_incident(self, incident_type: str, severity: str,
                                  start_time: float, end_time: float,
                                  confidence: float, filename: str = None,
                                  analysis_id: int = None):
        """Convenience method for critical incident alerts."""
        type_labels = {
            'violence': 'Violence Detected',
            'cry': 'Unanswered Child Distress',
            'emotion': 'Aggressive Emotion Detected',
            'neglect': 'Neglect Pattern Detected',
            'language': 'Inappropriate Language Detected',
        }
        title = type_labels.get(incident_type, f'{incident_type.title()} Detected')

        level = 'critical' if severity in ('high', 'critical') else 'warning'

        message = (
            f"{incident_type.title()} incident ({severity} severity, "
            f"{confidence:.0%} confidence) at {start_time:.1f}s-{end_time:.1f}s"
        )
        if filename:
            message += f" in {filename}"

        return self.notify(
            level=level,
            title=title,
            message=message,
            incident_type=incident_type,
            analysis_id=analysis_id,
            filename=filename,
            details={
                'severity': severity,
                'confidence': confidence,
                'start_time': start_time,
                'end_time': end_time,
            },
        )

    def _send_webhooks(self, notif: Notification):
        """Send notification to registered webhooks."""
        if not self._webhooks:
            return

        payload = json.dumps({
            'event': 'soundshield_alert',
            'notification': {
                'id': notif.id,
                'timestamp': notif.timestamp,
                'level': notif.level,
                'title': notif.title,
                'message': notif.message,
                'incident_type': notif.incident_type,
                'filename': notif.filename,
                'details': notif.details,
            }
        })

        import urllib.request
        for url in self._webhooks:
            try:
                req = urllib.request.Request(
                    url,
                    data=payload.encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST',
                )
                urllib.request.urlopen(req, timeout=5)
                logger.debug(f"Webhook sent to {url}")
            except Exception as e:
                logger.error(f"Webhook failed for {url}: {e}")

    def register_callback(self, callback: Callable):
        """Register a callback function for notifications."""
        self._callbacks.append(callback)

    def add_webhook(self, url: str):
        """Add a webhook URL at runtime."""
        if url not in self._webhooks:
            self._webhooks.append(url)

    def remove_webhook(self, url: str):
        """Remove a webhook URL."""
        self._webhooks = [u for u in self._webhooks if u != url]

    def get_notifications(self, limit: int = 50, offset: int = 0,
                          level: str = None, unread_only: bool = False) -> List[Dict]:
        """Get stored notifications with filtering."""
        filtered = self._notifications
        if level:
            filtered = [n for n in filtered if n.level == level]
        if unread_only:
            filtered = [n for n in filtered if not n.read]

        page = filtered[offset:offset + limit]
        return [self._to_dict(n) for n in page]

    def mark_read(self, notification_id: str) -> bool:
        """Mark a notification as read."""
        for n in self._notifications:
            if n.id == notification_id:
                n.read = True
                return True
        return False

    def mark_all_read(self):
        """Mark all notifications as read."""
        for n in self._notifications:
            n.read = True

    def get_unread_count(self) -> int:
        return sum(1 for n in self._notifications if not n.read)

    def _to_dict(self, notif: Notification) -> Dict:
        return {
            'id': notif.id,
            'timestamp': notif.timestamp,
            'level': notif.level,
            'title': notif.title,
            'message': notif.message,
            'incident_type': notif.incident_type,
            'analysis_id': notif.analysis_id,
            'filename': notif.filename,
            'details': notif.details,
            'read': notif.read,
        }


# Global instance
notifications = NotificationManager()
