"""
Live Audio Monitoring via WebSocket for SoundShield-AI

Accepts streaming audio chunks over WebSocket and returns
real-time incident events as they are detected.

Usage:
    Client connects to /ws namespace via Socket.IO.
    Sends 'audio_chunk' events with base64-encoded PCM audio.
    Receives 'incident' events with structured detection results.
"""

import logging
import io
import base64
import numpy as np
import soundfile as sf
from typing import Dict, Optional

from config import config

logger = logging.getLogger(__name__)


class LiveAudioProcessor:
    """Processes streaming audio chunks for real-time detection."""

    def __init__(self, language: str = 'en'):
        self.language = language
        self.sr = config.audio.sample_rate
        self.buffer = np.array([], dtype=np.float32)
        self.chunk_duration = 2.0  # Process every 2 seconds
        self.chunk_samples = int(self.chunk_duration * self.sr)
        self.total_processed = 0.0  # seconds

        # Lazy-load detectors to avoid import overhead
        self._cry_detector = None
        self._violence_detector = None
        self._emotion_detector = None

    @property
    def cry_detector(self):
        if self._cry_detector is None:
            from cry_detector import CryDetector
            self._cry_detector = CryDetector()
        return self._cry_detector

    @property
    def violence_detector(self):
        if self._violence_detector is None:
            from violence_detector import ViolenceDetector
            self._violence_detector = ViolenceDetector()
        return self._violence_detector

    @property
    def emotion_detector(self):
        if self._emotion_detector is None:
            from emotion_detector import EmotionDetector
            self._emotion_detector = EmotionDetector()
        return self._emotion_detector

    def process_chunk(self, audio_b64: str) -> list:
        """Process a base64-encoded audio chunk.

        Returns list of incident dicts (may be empty).
        """
        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio_data, _ = sf.read(io.BytesIO(audio_bytes), dtype='float32')

            # Ensure mono
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            self.buffer = np.concatenate([self.buffer, audio_data])
        except Exception as e:
            logger.error(f"Failed to decode audio chunk: {e}")
            return []

        incidents = []

        # Process complete chunks from buffer
        while len(self.buffer) >= self.chunk_samples:
            chunk = self.buffer[:self.chunk_samples]
            self.buffer = self.buffer[self.chunk_samples:]

            chunk_start = self.total_processed
            chunk_end = chunk_start + self.chunk_duration
            self.total_processed = chunk_end

            # Run detectors on chunk
            chunk_incidents = self._analyze_chunk(chunk, chunk_start, chunk_end)
            incidents.extend(chunk_incidents)

        return incidents

    def _analyze_chunk(self, chunk: np.ndarray,
                       start_time: float, end_time: float) -> list:
        """Run all detectors on a single chunk."""
        incidents = []

        # Violence detection (fastest, most critical)
        try:
            violence = self.violence_detector.detect_violence_segments(chunk, self.sr)
            for v in violence:
                incidents.append({
                    'type': 'violence',
                    'category': ', '.join(v.get('violence_types', [])),
                    'confidence': v.get('confidence', 0),
                    'severity': v.get('severity', 'low'),
                    'timestamp_start': start_time + v.get('start_time', 0),
                    'timestamp_end': start_time + v.get('end_time', 0),
                    'ml_backed': False,
                })
        except Exception as e:
            logger.error(f"Live violence detection error: {e}")

        # Cry detection
        try:
            cries = self.cry_detector.detect_cry_segments(chunk, self.sr)
            for c in cries:
                incidents.append({
                    'type': 'cry',
                    'category': c.get('intensity', 'medium'),
                    'confidence': c.get('confidence', 0),
                    'severity': 'medium' if c.get('intensity') != 'high' else 'high',
                    'timestamp_start': start_time + c.get('start_time', 0),
                    'timestamp_end': start_time + c.get('end_time', 0),
                    'ml_backed': False,
                })
        except Exception as e:
            logger.error(f"Live cry detection error: {e}")

        # Emotion detection
        try:
            if len(chunk) >= self.sr * 0.5:
                features = self.emotion_detector.calculate_emotion_features(chunk, self.sr)
                emotion = self.emotion_detector.detect_emotion(features)
                if emotion['primary_emotion'] in ('anger', 'aggression', 'stress'):
                    if emotion['confidence'] > config.emotion.confidence_threshold:
                        incidents.append({
                            'type': 'emotion',
                            'category': emotion['primary_emotion'],
                            'confidence': emotion['confidence'],
                            'severity': 'medium' if emotion['confidence'] < 0.8 else 'high',
                            'timestamp_start': start_time,
                            'timestamp_end': end_time,
                            'ml_backed': False,
                        })
        except Exception as e:
            logger.error(f"Live emotion detection error: {e}")

        return incidents

    def reset(self):
        """Reset processor state for a new session."""
        self.buffer = np.array([], dtype=np.float32)
        self.total_processed = 0.0


def register_socketio_events(socketio, app):
    """Register Socket.IO event handlers on the given socketio instance."""
    sessions = {}  # sid -> LiveAudioProcessor

    @socketio.on('connect', namespace='/ws')
    def handle_connect():
        from flask import request as req
        sid = req.sid
        sessions[sid] = LiveAudioProcessor()
        logger.info(f"Live monitoring session started: {sid}")
        socketio.emit('status', {'message': 'Connected to SoundShield live monitor'},
                       namespace='/ws', to=sid)

    @socketio.on('disconnect', namespace='/ws')
    def handle_disconnect():
        from flask import request as req
        sid = req.sid
        sessions.pop(sid, None)
        logger.info(f"Live monitoring session ended: {sid}")

    @socketio.on('audio_chunk', namespace='/ws')
    def handle_audio_chunk(data):
        from flask import request as req
        sid = req.sid
        processor = sessions.get(sid)
        if not processor:
            processor = LiveAudioProcessor()
            sessions[sid] = processor

        audio_b64 = data.get('audio') if isinstance(data, dict) else data
        if not audio_b64:
            return

        incidents = processor.process_chunk(audio_b64)
        for incident in incidents:
            socketio.emit('incident', incident, namespace='/ws', to=sid)

        # Periodic heartbeat with processing status
        socketio.emit('heartbeat', {
            'processed_seconds': round(processor.total_processed, 1),
            'incidents_in_chunk': len(incidents),
        }, namespace='/ws', to=sid)

    @socketio.on('set_language', namespace='/ws')
    def handle_set_language(data):
        from flask import request as req
        sid = req.sid
        processor = sessions.get(sid)
        if processor:
            lang = data.get('language', 'en')
            if lang in config.pipeline.supported_languages:
                processor.language = lang

    @socketio.on('reset', namespace='/ws')
    def handle_reset():
        from flask import request as req
        sid = req.sid
        processor = sessions.get(sid)
        if processor:
            processor.reset()
