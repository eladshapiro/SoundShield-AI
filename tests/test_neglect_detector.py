"""
Unit Tests for NeglectDetector Module
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neglect_detector import NeglectDetector


def make_silence(sr=22050, duration=5.0):
    return np.zeros(int(sr * duration))


def make_speech(sr=22050, duration=2.0, freq=150, amplitude=0.15):
    """Generate speech-like audio with low spectral centroid (<400Hz).

    Uses a band-limited pulse train filtered to keep most energy below 400Hz.
    """
    from scipy.signal import butter, lfilter
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental + harmonics
    signal = (
        amplitude * np.sin(2 * np.pi * freq * t) +
        amplitude * 0.5 * np.sin(2 * np.pi * 2 * freq * t) +
        amplitude * 0.25 * np.sin(2 * np.pi * 3 * freq * t)
    )
    # Low-pass filter at 400Hz to ensure spectral centroid stays low
    nyq = sr / 2
    b, a = butter(4, 400 / nyq, btype='low')
    signal = lfilter(b, a, signal)
    # Re-scale to target amplitude
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * amplitude
    return signal


class TestNeglectDetector(unittest.TestCase):
    """Test suite for NeglectDetector."""

    def setUp(self):
        self.detector = NeglectDetector()
        self.sr = 22050

    def test_initialization(self):
        """Detector should initialize with config-driven thresholds."""
        self.assertIn('unanswered_cry_duration', self.detector.neglect_thresholds)
        self.assertIn('min_cry_duration', self.detector.neglect_thresholds)
        self.assertIn('prolonged_silence', self.detector.neglect_patterns)

    def test_no_neglect_with_empty_inputs(self):
        """Should return clean analysis when no cry/violence segments provided."""
        audio = make_speech(self.sr, 10.0)
        result = self.detector.detect_neglect_patterns(audio, self.sr)
        self.assertIn('overall_neglect_score', result)
        self.assertIn('neglect_severity', result)
        self.assertEqual(result['unanswered_cries'], [])

    def test_unanswered_cry_detection(self):
        """Should flag unanswered cries when no staff response follows."""
        # Create audio: 5s cry + 15s silence
        cry = 0.2 * np.random.randn(int(self.sr * 5))
        silence = np.zeros(int(self.sr * 15))
        audio = np.concatenate([cry, silence])

        cry_segments = [{
            'start_time': 0,
            'end_time': 5.0,
            'duration': 5.0,
            'intensity': 'medium',
            'confidence': 0.8,
        }]
        result = self.detector.detect_neglect_patterns(audio, self.sr, cry_segments=cry_segments)
        self.assertGreater(len(result['unanswered_cries']), 0)

    def test_short_cry_not_flagged(self):
        """Cries shorter than min_cry_duration should not be flagged."""
        audio = make_silence(self.sr, 10.0)
        cry_segments = [{
            'start_time': 0,
            'end_time': 2.0,
            'duration': 2.0,  # < min_cry_duration (5.0)
            'intensity': 'low',
            'confidence': 0.5,
        }]
        result = self.detector.detect_neglect_patterns(audio, self.sr, cry_segments=cry_segments)
        self.assertEqual(len(result['unanswered_cries']), 0)

    def test_staff_response_detection(self):
        """Should detect adult speech as staff response."""
        speech = make_speech(self.sr, 3.0, freq=150, amplitude=0.15)
        self.assertTrue(self.detector._detect_staff_response(speech, self.sr))

    def test_silence_not_detected_as_response(self):
        """Silence should not be detected as staff response."""
        silence = make_silence(self.sr, 3.0)
        self.assertFalse(self.detector._detect_staff_response(silence, self.sr))

    def test_cry_neglect_severity_by_duration(self):
        """Severity should scale with cry duration."""
        self.assertEqual(
            self.detector._calculate_cry_neglect_severity({'duration': 8, 'intensity': 'low'}),
            'low'
        )
        self.assertEqual(
            self.detector._calculate_cry_neglect_severity({'duration': 25, 'intensity': 'low'}),
            'high'
        )
        self.assertEqual(
            self.detector._calculate_cry_neglect_severity({'duration': 50, 'intensity': 'low'}),
            'critical'
        )

    def test_silence_severity_by_duration(self):
        """Silence severity should scale with duration."""
        self.assertEqual(self.detector._assess_silence_severity(100), 'low')
        self.assertEqual(self.detector._assess_silence_severity(200), 'medium')
        self.assertEqual(self.detector._assess_silence_severity(400), 'high')
        self.assertEqual(self.detector._assess_silence_severity(700), 'critical')

    def test_interaction_severity_by_ratio(self):
        """Interaction severity should scale inversely with adult speech ratio."""
        self.assertEqual(self.detector._assess_interaction_severity(0.08), 'low')
        self.assertEqual(self.detector._assess_interaction_severity(0.03), 'medium')
        self.assertEqual(self.detector._assess_interaction_severity(0.01), 'high')
        self.assertEqual(self.detector._assess_interaction_severity(0.001), 'critical')

    def test_neglect_score_zero_when_clean(self):
        """Score should be 0 when no neglect indicators found."""
        analysis = {
            'unanswered_cries': [],
            'prolonged_silence_periods': [],
            'lack_of_interaction_periods': [],
            'ignored_distress_episodes': [],
        }
        self.assertEqual(self.detector._calculate_neglect_score(analysis), 0.0)

    def test_neglect_severity_levels(self):
        """Severity levels should map correctly from score."""
        self.assertEqual(self.detector._determine_neglect_severity(0.05), 'none')
        self.assertEqual(self.detector._determine_neglect_severity(0.25), 'low')
        self.assertEqual(self.detector._determine_neglect_severity(0.45), 'medium')
        self.assertEqual(self.detector._determine_neglect_severity(0.65), 'high')
        self.assertEqual(self.detector._determine_neglect_severity(0.85), 'critical')

    def test_distress_neglect_severity_increases_with_multi_types(self):
        """Multiple violence types should increase neglect severity."""
        single_type = {'severity': 'low', 'violence_types': ['shouting']}
        multi_type = {'severity': 'low', 'violence_types': ['shouting', 'aggressive_tone']}
        self.assertEqual(self.detector._assess_distress_neglect_severity(single_type), 'low')
        self.assertEqual(self.detector._assess_distress_neglect_severity(multi_type), 'medium')


if __name__ == '__main__':
    unittest.main()
