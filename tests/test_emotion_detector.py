"""
Unit Tests for Emotion Detection Module

Tests the EmotionDetector class functionality.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_detector import (
    EmotionDetector,
    EmotionDetectionError,
    SUPPORTED_EMOTIONS,
    MIN_SEGMENT_LENGTH_SECONDS
)


class TestEmotionDetector(unittest.TestCase):
    """Test suite for EmotionDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = EmotionDetector()
        self.sample_rate = 16000
        self.duration = 2.0  # 2 seconds
        self.sample_audio = np.random.randn(int(self.sample_rate * self.duration))
    
    def tearDown(self):
        """Clean up after tests."""
        self.detector = None
    
    def test_initialization(self):
        """Test detector initializes correctly."""
        self.assertIsNotNone(self.detector.emotion_thresholds)
        self.assertIn('anger', self.detector.emotion_thresholds)
        self.assertIn('stress', self.detector.emotion_thresholds)
        self.assertIn('calm', self.detector.emotion_thresholds)
        self.assertIn('aggression', self.detector.emotion_thresholds)
    
    def test_calculate_emotion_features_valid_audio(self):
        """Test feature calculation with valid audio."""
        features = self.detector.calculate_emotion_features(
            self.sample_audio,
            self.sample_rate
        )
        
        # Check all expected features are present
        self.assertIn('mean_energy', features)
        self.assertIn('energy_variance', features)
        self.assertIn('mean_pitch_variance', features)
        self.assertIn('pitch_stability', features)
        self.assertIn('mean_spectral_centroid', features)
        self.assertIn('spectral_centroid_variance', features)
        self.assertIn('mean_zero_crossing_rate', features)
        self.assertIn('mfcc_mean', features)
        self.assertIn('mfcc_variance', features)
    
    def test_calculate_emotion_features_empty_audio(self):
        """Test feature calculation fails with empty audio."""
        empty_audio = np.array([])
        
        with self.assertRaises(ValueError) as context:
            self.detector.calculate_emotion_features(empty_audio, self.sample_rate)
        self.assertIn('empty', str(context.exception))
    
    def test_calculate_emotion_features_invalid_sample_rate(self):
        """Test feature calculation fails with invalid sample rate."""
        with self.assertRaises(ValueError) as context:
            self.detector.calculate_emotion_features(self.sample_audio, -1)
        self.assertIn('sample rate', str(context.exception))
    
    def test_calculate_emotion_features_too_short(self):
        """Test feature calculation fails with too short audio."""
        short_audio = np.random.randn(100)  # Very short
        
        with self.assertRaises(ValueError) as context:
            self.detector.calculate_emotion_features(short_audio, self.sample_rate)
        self.assertIn('too short', str(context.exception))
    
    def test_detect_emotion_valid_features(self):
        """Test emotion detection with valid features."""
        features = self.detector.calculate_emotion_features(
            self.sample_audio,
            self.sample_rate
        )
        
        result = self.detector.detect_emotion(features)
        
        # Check result structure
        self.assertIn('emotion_scores', result)
        self.assertIn('primary_emotion', result)
        self.assertIn('confidence', result)
        
        # Check emotion scores
        emotion_scores = result['emotion_scores']
        self.assertIn('anger', emotion_scores)
        self.assertIn('stress', emotion_scores)
        self.assertIn('calm', emotion_scores)
        self.assertIn('aggression', emotion_scores)
        
        # Check confidence is between 0 and 1
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        
        # Check primary emotion is a supported emotion
        self.assertIn(result['primary_emotion'], SUPPORTED_EMOTIONS)
    
    def test_detect_emotion_empty_features(self):
        """Test emotion detection fails with empty features."""
        with self.assertRaises(ValueError) as context:
            self.detector.detect_emotion({})
        self.assertIn('empty', str(context.exception))
    
    def test_detect_emotion_high_energy(self):
        """Test detection of high-energy emotions."""
        # Create features suggesting anger
        features = {
            'mean_energy': 0.25,  # High energy
            'energy_variance': 0.15,
            'mean_pitch_variance': 0.5,  # High pitch variance
            'pitch_stability': 0.5,
            'mean_spectral_centroid': 2000,
            'spectral_centroid_variance': 0.3,
            'mean_spectral_rolloff': 0.7,  # High rolloff
            'mean_spectral_bandwidth': 0.8,
            'mean_zero_crossing_rate': 0.06,
            'zero_crossing_rate_variance': 0.02,
            'mfcc_mean': [0] * 13,
            'mfcc_variance': [0] * 13
        }
        
        result = self.detector.detect_emotion(features)
        
        # Should detect anger or aggression due to high energy and pitch variance
        self.assertIn(result['primary_emotion'], ['anger', 'aggression'])
    
    def test_detect_emotion_calm(self):
        """Test detection of calm emotion."""
        # Create features suggesting calm
        features = {
            'mean_energy': 0.03,  # Low energy
            'energy_variance': 0.02,
            'mean_pitch_variance': 0.1,  # Low pitch variance
            'pitch_stability': 0.9,  # High stability
            'mean_spectral_centroid': 800,
            'spectral_centroid_variance': 0.1,
            'mean_spectral_rolloff': 0.3,  # Low rolloff
            'mean_spectral_bandwidth': 0.3,
            'mean_zero_crossing_rate': 0.02,
            'zero_crossing_rate_variance': 0.01,
            'mfcc_mean': [0] * 13,
            'mfcc_variance': [0] * 13
        }
        
        result = self.detector.detect_emotion(features)
        
        # Should detect calm
        self.assertEqual(result['primary_emotion'], 'calm')
    
    def test_analyze_segment_emotions(self):
        """Test analysis of multiple audio segments."""
        # Create list of audio segments
        segments = [
            np.random.randn(int(self.sample_rate * 1.0))
            for _ in range(3)
        ]
        
        results = self.detector.analyze_segment_emotions(segments, self.sample_rate)
        
        # Check results
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        for result in results:
            self.assertIn('segment_index', result)
            self.assertIn('start_time', result)
            self.assertIn('end_time', result)
            self.assertIn('emotion_analysis', result)
    
    def test_detect_concerning_emotions(self):
        """Test detection of concerning emotion segments."""
        # Create segment with concerning emotion
        segment_emotions = [
            {
                'segment_index': 0,
                'start_time': 0.0,
                'end_time': 5.0,
                'emotion_analysis': {
                    'primary_emotion': 'anger',
                    'confidence': 0.8,
                    'emotion_scores': {'anger': 0.8, 'stress': 0.2, 'calm': 0.1, 'aggression': 0.3}
                }
            },
            {
                'segment_index': 1,
                'start_time': 5.0,
                'end_time': 10.0,
                'emotion_analysis': {
                    'primary_emotion': 'calm',
                    'confidence': 0.7,
                    'emotion_scores': {'anger': 0.1, 'stress': 0.1, 'calm': 0.7, 'aggression': 0.1}
                }
            }
        ]
        
        concerning = self.detector.detect_concerning_emotions(segment_emotions, threshold=0.6)
        
        # Should detect the anger segment
        self.assertGreater(len(concerning), 0)
        self.assertEqual(concerning[0]['detected_emotion'], 'anger')
        self.assertIn('severity', concerning[0])
    
    def test_calculate_severity(self):
        """Test severity calculation."""
        # Test anger with high confidence
        severity = self.detector._calculate_severity('anger', 0.9)
        self.assertEqual(severity, 'high')
        
        # Test stress with low confidence
        severity = self.detector._calculate_severity('stress', 0.6)
        self.assertEqual(severity, 'low')
        
        # Test aggression with high confidence
        severity = self.detector._calculate_severity('aggression', 0.85)
        self.assertEqual(severity, 'critical')


class TestEmotionDetectorConstants(unittest.TestCase):
    """Test module constants."""
    
    def test_supported_emotions(self):
        """Test SUPPORTED_EMOTIONS constant."""
        self.assertIsInstance(SUPPORTED_EMOTIONS, list)
        self.assertIn('anger', SUPPORTED_EMOTIONS)
        self.assertIn('stress', SUPPORTED_EMOTIONS)
        self.assertIn('calm', SUPPORTED_EMOTIONS)
        self.assertIn('aggression', SUPPORTED_EMOTIONS)
    
    def test_min_segment_length(self):
        """Test MIN_SEGMENT_LENGTH_SECONDS constant."""
        self.assertIsInstance(MIN_SEGMENT_LENGTH_SECONDS, (int, float))
        self.assertGreater(MIN_SEGMENT_LENGTH_SECONDS, 0)


if __name__ == '__main__':
    unittest.main()
