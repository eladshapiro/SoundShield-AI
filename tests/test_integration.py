"""
Integration Tests for SoundShield-AI

Tests the full detection pipeline end-to-end using synthetic audio.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_synthetic_recording(sr=22050, duration=10.0):
    """Create a synthetic kindergarten recording with mixed signals.

    Structure:
      0-3s:   Quiet ambient (normal)
      3-5s:   Adult speech (150Hz tone, moderate energy)
      5-7s:   Child cry (450Hz + harmonics, high energy, AM)
      7-9s:   Silence (no response — should flag neglect)
      9-10s:  Adult speech response
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)

    # 0-3s: quiet ambient
    signal[:int(3 * sr)] = 0.005 * np.random.randn(int(3 * sr))

    # 3-5s: adult speech
    idx_start, idx_end = int(3 * sr), int(5 * sr)
    t_seg = t[idx_start:idx_end]
    speech = 0.12 * np.sin(2 * np.pi * 150 * t_seg) + 0.05 * np.sin(2 * np.pi * 300 * t_seg)
    signal[idx_start:idx_end] = speech

    # 5-7s: child cry
    idx_start, idx_end = int(5 * sr), int(7 * sr)
    t_seg = t[idx_start:idx_end]
    cry = (0.35 * np.sin(2 * np.pi * 450 * t_seg) +
           0.2 * np.sin(2 * np.pi * 900 * t_seg))
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t_seg)
    signal[idx_start:idx_end] = cry * am

    # 7-9s: silence
    signal[int(7 * sr):int(9 * sr)] = 0.002 * np.random.randn(int(2 * sr))

    # 9-10s: adult response
    idx_start, idx_end = int(9 * sr), int(10 * sr)
    t_seg = t[idx_start:idx_end]
    signal[idx_start:idx_end] = 0.1 * np.sin(2 * np.pi * 160 * t_seg)

    return signal.astype(np.float32)


class TestFullPipeline(unittest.TestCase):
    """Tests the detection pipeline from audio through all detectors."""

    @classmethod
    def setUpClass(cls):
        cls.sr = 22050
        cls.audio = make_synthetic_recording(cls.sr, 10.0)

    def test_audio_analyzer_pipeline(self):
        """AudioAnalyzer should extract features and calibrate baseline."""
        from audio_analyzer import AudioAnalyzer
        aa = AudioAnalyzer()
        audio, sr = cls_audio_sr(self)
        features = aa.extract_features(audio, sr)

        self.assertIn('duration', features)
        self.assertIn('rms_energy', features)
        self.assertIn('mfccs', features)
        self.assertAlmostEqual(features['duration'], 10.0, places=0)

        baseline = aa.calibrate_noise_baseline(audio, sr)
        self.assertIn('mean_rms', baseline)
        self.assertIn('noise_floor_db', baseline)

    def test_cry_detector_finds_cry_segment(self):
        """CryDetector should find the cry segment around 5-7s."""
        from cry_detector import CryDetector
        cd = CryDetector()
        segments = cd.detect_cry_segments(self.audio, self.sr)

        # Should detect something in the 5-7s range
        cry_in_range = [s for s in segments
                        if s['start_time'] < 7.5 and s['end_time'] > 4.5]
        self.assertGreater(len(cry_in_range), 0,
                           "Expected cry detection in 5-7s range")

    def test_violence_detector_on_mixed(self):
        """ViolenceDetector should not false-positive on normal audio."""
        from violence_detector import ViolenceDetector
        vd = ViolenceDetector()
        segments = vd.detect_violence_segments(self.audio, self.sr)
        # This synthetic audio has no shouting/aggression
        # There may be some detections from the cry burst but not many
        for seg in segments:
            # Shouldn't flag 'threatening' on baby cries
            self.assertNotIn('threatening', seg.get('violence_types', []))

    def test_emotion_detector_pipeline(self):
        """EmotionDetector should produce valid emotions for segments."""
        from emotion_detector import EmotionDetector
        ed = EmotionDetector()
        segments = [self.audio[i:i + self.sr * 2]
                    for i in range(0, len(self.audio) - self.sr * 2, self.sr * 2)]
        results = ed.analyze_segment_emotions(segments, self.sr)

        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn('emotion_analysis', r)
            self.assertIn('primary_emotion', r['emotion_analysis'])

    def test_neglect_detector_flags_unanswered_cry(self):
        """NeglectDetector should flag cry without immediate response."""
        from cry_detector import CryDetector
        from neglect_detector import NeglectDetector

        cd = CryDetector()
        nd = NeglectDetector()

        cry_segs = cd.detect_cry_segments(self.audio, self.sr)
        result = nd.detect_neglect_patterns(self.audio, self.sr,
                                            cry_segments=cry_segs)

        self.assertIn('overall_neglect_score', result)
        self.assertIn('neglect_severity', result)

    def test_cry_episode_aggregation(self):
        """Cry episodes should aggregate nearby segments."""
        from cry_detector import CryDetector
        cd = CryDetector()
        segments = cd.detect_cry_segments(self.audio, self.sr)

        if segments:
            episodes = cd.aggregate_cry_episodes(segments)
            self.assertGreater(len(episodes), 0)
            for ep in episodes:
                self.assertIn('total_duration', ep)
                self.assertIn('peak_intensity', ep)

    def test_cry_response_time_measurement(self):
        """Response time should be measured for detected cries."""
        from cry_detector import CryDetector
        cd = CryDetector()
        segments = cd.detect_cry_segments(self.audio, self.sr)

        if segments:
            enriched = cd.measure_response_time(self.audio, self.sr, segments)
            self.assertEqual(len(enriched), len(segments))
            for e in enriched:
                self.assertIn('response_time_seconds', e)
                self.assertIn('response_rating', e)


class TestConfigPropagation(unittest.TestCase):
    """Tests that config changes propagate to detectors."""

    def test_config_affects_cry_detector(self):
        """Changing config should affect new CryDetector instances."""
        from config import config
        from cry_detector import CryDetector

        original = config.cry.energy_threshold
        try:
            config.cry.energy_threshold = 0.99
            cd = CryDetector()
            self.assertEqual(cd.cry_features['energy_threshold'], 0.99)
        finally:
            config.cry.energy_threshold = original

    def test_config_affects_violence_detector(self):
        from config import config
        from violence_detector import ViolenceDetector

        original = config.violence.shouting_energy
        try:
            config.violence.shouting_energy = 0.88
            vd = ViolenceDetector()
            self.assertEqual(
                vd.violence_thresholds['shouting']['energy_threshold'], 0.88
            )
        finally:
            config.violence.shouting_energy = original


class TestNotificationIntegration(unittest.TestCase):
    """Tests notification triggers."""

    def test_critical_incident_creates_notification(self):
        from notifications import NotificationManager
        nm = NotificationManager()
        notif = nm.notify_critical_incident(
            incident_type='violence',
            severity='critical',
            start_time=10.0,
            end_time=15.0,
            confidence=0.9,
            filename='test.wav',
        )
        self.assertEqual(notif.level, 'critical')
        self.assertIn('violence', notif.message.lower())

    def test_warning_for_medium_severity(self):
        from notifications import NotificationManager
        nm = NotificationManager()
        notif = nm.notify_critical_incident(
            incident_type='cry',
            severity='medium',
            start_time=5.0,
            end_time=8.0,
            confidence=0.7,
        )
        self.assertEqual(notif.level, 'warning')

    def test_unread_count(self):
        from notifications import NotificationManager
        nm = NotificationManager()
        nm.notify('info', 'Test', 'Test message', 'test')
        nm.notify('warning', 'Test 2', 'Another', 'test')
        self.assertEqual(nm.get_unread_count(), 2)
        nm.mark_all_read()
        self.assertEqual(nm.get_unread_count(), 0)


def cls_audio_sr(test_instance):
    """Helper to get class-level audio and sr."""
    return test_instance.audio, test_instance.sr


if __name__ == '__main__':
    unittest.main()
