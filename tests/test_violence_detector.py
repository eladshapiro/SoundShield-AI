"""
Unit Tests for ViolenceDetector Module
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from violence_detector import ViolenceDetector


def make_silence(sr=22050, duration=2.0):
    return np.zeros(int(sr * duration))


def make_loud_noise(sr=22050, duration=2.0, amplitude=0.6):
    """Generate loud broadband noise (high energy, high temporal instability)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = amplitude * np.random.randn(len(t))
    return noise


def make_quiet_tone(sr=22050, duration=2.0):
    """Generate a quiet, stable tone."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 0.02 * np.sin(2 * np.pi * 200 * t)


class TestViolenceDetector(unittest.TestCase):
    """Test suite for ViolenceDetector."""

    def setUp(self):
        self.detector = ViolenceDetector()
        self.sr = 22050

    def test_initialization(self):
        """Detector should initialize with config-driven thresholds."""
        self.assertIn('shouting', self.detector.violence_thresholds)
        self.assertIn('aggressive_tone', self.detector.violence_thresholds)
        self.assertIn('threatening', self.detector.violence_thresholds)
        self.assertIn('physical_violence_indicators', self.detector.violence_thresholds)

    def test_silence_no_violence(self):
        """Silent audio should produce no violence segments."""
        audio = make_silence(self.sr, 3.0)
        segments = self.detector.detect_violence_segments(audio, self.sr)
        self.assertEqual(len(segments), 0)

    def test_quiet_tone_no_violence(self):
        """Quiet tone should not trigger violence detection."""
        audio = make_quiet_tone(self.sr, 3.0)
        segments = self.detector.detect_violence_segments(audio, self.sr)
        self.assertEqual(len(segments), 0)

    def test_energy_gate_blocks_low_energy(self):
        """Energy gate should block violence detection for low-energy audio."""
        features = {'mean_energy': 0.01}
        types = self.detector._detect_violence_types(features)
        self.assertEqual(types, [])

    def test_feature_extraction(self):
        """Feature dict should contain all expected keys."""
        audio = make_loud_noise(self.sr, 1.5)
        features = self.detector._calculate_violence_features(audio, self.sr)
        expected = [
            'mean_energy', 'max_energy', 'energy_variance', 'energy_skewness',
            'mean_frequency', 'frequency_variance', 'frequency_range',
            'mean_spectral_rolloff', 'mean_spectral_bandwidth',
            'mean_pitch_variance', 'mean_zero_crossing_rate',
            'temporal_instability', 'high_frequency_ratio',
        ]
        for key in expected:
            self.assertIn(key, features, f"Missing key: {key}")

    def test_severity_calculation(self):
        """Severity should increase with violence type severity."""
        features = {'mean_energy': 0.15, 'max_energy': 0.2, 'temporal_instability': 0.3}
        self.assertEqual(self.detector._calculate_severity(features, ['shouting']), 'low')
        self.assertEqual(self.detector._calculate_severity(features, ['threatening']), 'high')
        self.assertEqual(self.detector._calculate_severity(features, ['potential_physical_violence']), 'critical')

    def test_confidence_empty_types(self):
        """Confidence should be 0 for no detected types."""
        features = {'mean_energy': 0.1, 'frequency_variance': 0.1, 'temporal_instability': 0.1}
        self.assertEqual(self.detector._calculate_confidence(features, []), 0.0)

    def test_confidence_with_types(self):
        """Confidence should be > 0 when types are detected."""
        features = {'mean_energy': 0.3, 'frequency_variance': 0.5, 'temporal_instability': 0.5}
        conf = self.detector._calculate_confidence(features, ['shouting', 'aggressive_tone'])
        self.assertGreater(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_merge_violence_segments(self):
        """Close segments should merge; far segments should not."""
        segments = [
            {'start_time': 0, 'end_time': 1, 'duration': 1, 'features': {},
             'violence_types': ['shouting'], 'severity': 'low', 'confidence': 0.5},
            {'start_time': 2, 'end_time': 3, 'duration': 1, 'features': {},
             'violence_types': ['aggressive_tone'], 'severity': 'medium', 'confidence': 0.6},
            {'start_time': 10, 'end_time': 11, 'duration': 1, 'features': {},
             'violence_types': ['shouting'], 'severity': 'low', 'confidence': 0.4},
        ]
        merged = self.detector._merge_violence_segments(segments)
        # First two should merge (gap = 1s < 1.5s), third stays separate
        self.assertEqual(len(merged), 2)
        self.assertIn('aggressive_tone', merged[0]['violence_types'])

    def test_context_assessment_patterns(self):
        """Context assessment should identify known patterns."""
        ctx = {
            'before_violence': {'activity_type': 'distress', 'intensity': 'high'},
            'after_violence': {'activity_type': 'silence'},
        }
        self.assertEqual(self.detector._assess_overall_context(ctx), 'escalation_then_silence')

        ctx2 = {
            'before_violence': {'activity_type': 'silence'},
            'after_violence': {'activity_type': 'adult_speech'},
        }
        self.assertEqual(self.detector._assess_overall_context(ctx2), 'immediate_adult_response')

    def test_adjust_severity_by_context(self):
        """Severity should increase for escalation, decrease for adult response."""
        ctx_escalation = {'overall_assessment': 'escalation_then_silence'}
        self.assertEqual(self.detector._adjust_severity_by_context('medium', ctx_escalation), 'high')

        ctx_response = {'overall_assessment': 'immediate_adult_response'}
        self.assertEqual(self.detector._adjust_severity_by_context('medium', ctx_response), 'low')

    def test_skewness_uniform(self):
        """Skewness of uniform data should be near zero."""
        data = np.ones(100) * 0.5
        self.assertEqual(self.detector._calculate_skewness(data), 0)

    def test_temporal_instability(self):
        """Stable audio should have low instability; noisy high."""
        stable = np.zeros(self.sr)
        self.assertEqual(self.detector._calculate_temporal_instability(stable, self.sr), 0)

        noisy = np.random.randn(self.sr)
        instability = self.detector._calculate_temporal_instability(noisy, self.sr)
        self.assertGreater(instability, 0)


if __name__ == '__main__':
    unittest.main()
