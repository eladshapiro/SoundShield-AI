"""
Unit Tests for CryDetector Module
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cry_detector import CryDetector


def make_silence(sr=22050, duration=2.0):
    """Generate silent audio."""
    return np.zeros(int(sr * duration))


def make_cry_like(sr=22050, duration=2.0):
    """Generate audio that mimics a baby cry: high energy, 200-800 Hz, harmonic."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental at 450 Hz with harmonics + amplitude modulation (sobbing)
    f0 = 450
    signal = (
        0.5 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.15 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    # Amplitude modulation for sobbing pattern
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    signal *= am
    # Normalize to high energy
    signal = signal / np.max(np.abs(signal)) * 0.4
    return signal


def make_adult_speech(sr=22050, duration=2.0):
    """Generate audio mimicking adult speech: lower frequency, speech-like ZCR."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = 150  # Adult voice fundamental
    signal = 0.15 * np.sin(2 * np.pi * f0 * t)
    # Add some noise to create speech-like ZCR
    signal += 0.02 * np.random.randn(len(t))
    return signal


class TestCryDetector(unittest.TestCase):
    """Test suite for CryDetector."""

    def setUp(self):
        self.detector = CryDetector()
        self.sr = 22050

    def test_initialization(self):
        """Detector initializes with expected thresholds from config."""
        self.assertIn('frequency_range', self.detector.cry_features)
        self.assertIn('energy_threshold', self.detector.cry_features)
        self.assertIn('response_window', self.detector.response_features)

    def test_silence_returns_no_cries(self):
        """Silent audio should produce no cry segments."""
        audio = make_silence(self.sr, 5.0)
        segments = self.detector.detect_cry_segments(audio, self.sr)
        self.assertEqual(len(segments), 0)

    def test_cry_features_extraction(self):
        """Feature dict should contain all expected keys."""
        audio = make_cry_like(self.sr, 2.0)
        features = self.detector._calculate_cry_features(audio, self.sr)
        expected_keys = [
            'mean_energy', 'max_energy', 'energy_variance',
            'mean_frequency', 'in_cry_frequency_range',
            'f0_mean', 'f0_std', 'voiced_ratio',
            'am_depth', 'mean_spectral_flatness',
            'mean_pitch_variance', 'mean_zero_crossing_rate',
        ]
        for key in expected_keys:
            self.assertIn(key, features, f"Missing key: {key}")

    def test_cry_intensity_levels(self):
        """Intensity levels should map correctly."""
        low_features = {'mean_energy': 0.09}
        med_features = {'mean_energy': 0.15}
        high_features = {'mean_energy': 0.3}
        self.assertEqual(self.detector._calculate_cry_intensity(low_features), 'low')
        self.assertEqual(self.detector._calculate_cry_intensity(med_features), 'medium')
        self.assertEqual(self.detector._calculate_cry_intensity(high_features), 'high')

    def test_merge_overlapping_segments(self):
        """Overlapping segments should merge correctly."""
        segments = [
            {'start_time': 0, 'end_time': 2, 'duration': 2, 'features': {}, 'confidence': 0.8, 'intensity': 'medium'},
            {'start_time': 1.5, 'end_time': 4, 'duration': 2.5, 'features': {}, 'confidence': 0.9, 'intensity': 'high'},
        ]
        merged = self.detector._merge_overlapping_segments(segments)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['start_time'], 0)
        self.assertEqual(merged[0]['end_time'], 4)
        self.assertEqual(merged[0]['confidence'], 0.9)

    def test_non_overlapping_segments_not_merged(self):
        """Non-overlapping segments should stay separate."""
        segments = [
            {'start_time': 0, 'end_time': 2, 'duration': 2, 'features': {}, 'confidence': 0.8, 'intensity': 'low'},
            {'start_time': 5, 'end_time': 7, 'duration': 2, 'features': {}, 'confidence': 0.7, 'intensity': 'low'},
        ]
        merged = self.detector._merge_overlapping_segments(segments)
        self.assertEqual(len(merged), 2)

    def test_empty_segments_merge(self):
        """Empty segment list should return empty."""
        self.assertEqual(self.detector._merge_overlapping_segments([]), [])

    def test_response_detection_with_speech(self):
        """Should detect response when adult speech follows cry."""
        cry_audio = make_cry_like(self.sr, 3.0)
        speech_audio = make_adult_speech(self.sr, 3.0)
        combined = np.concatenate([cry_audio, speech_audio])

        cry_segments = [{'start_time': 0, 'end_time': 3.0, 'duration': 3.0,
                         'confidence': 0.8, 'intensity': 'medium', 'features': {}}]
        results = self.detector.detect_response_to_cry(combined, self.sr, cry_segments)
        self.assertEqual(len(results), 1)
        # Result should include response analysis
        self.assertIn('response_analysis', results[0])

    def test_response_detection_with_silence(self):
        """Should not detect response when silence follows cry."""
        cry_audio = make_cry_like(self.sr, 3.0)
        silence_audio = make_silence(self.sr, 5.0)
        combined = np.concatenate([cry_audio, silence_audio])

        cry_segments = [{'start_time': 0, 'end_time': 3.0, 'duration': 3.0,
                         'confidence': 0.8, 'intensity': 'medium', 'features': {}}]
        results = self.detector.detect_response_to_cry(combined, self.sr, cry_segments)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]['response_detected'])

    def test_response_quality_assessment(self):
        """Response quality should depend on energy."""
        self.assertEqual(self.detector._assess_response_quality({'mean_energy': 0.03}), 'poor')
        self.assertEqual(self.detector._assess_response_quality({'mean_energy': 0.10}), 'adequate')
        self.assertEqual(self.detector._assess_response_quality({'mean_energy': 0.20}), 'good')


if __name__ == '__main__':
    unittest.main()
