"""
Unit Tests for SoundShield-AI Main Module

Tests the core functionality of the KindergartenRecordingAnalyzer class.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import numpy as np

# Import modules to test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    KindergartenRecordingAnalyzer,
    AudioAnalysisError,
    InvalidAudioFormatError,
    AudioFileTooLongError,
    SUPPORTED_LANGUAGES,
    SUPPORTED_FORMATS
)


class TestKindergartenRecordingAnalyzer(unittest.TestCase):
    """Test suite for KindergartenRecordingAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = KindergartenRecordingAnalyzer(language='en')
    
    def tearDown(self):
        """Clean up after tests."""
        self.analyzer = None
    
    def test_initialization_valid_language(self):
        """Test successful initialization with valid language."""
        analyzer = KindergartenRecordingAnalyzer(language='en')
        self.assertEqual(analyzer.language, 'en')
        self.assertIsNotNone(analyzer.audio_analyzer)
        self.assertIsNotNone(analyzer.emotion_detector)
        self.assertIsNotNone(analyzer.cry_detector)
    
    def test_initialization_invalid_language(self):
        """Test initialization fails with invalid language."""
        with self.assertRaises(ValueError) as context:
            KindergartenRecordingAnalyzer(language='invalid')
        self.assertIn('Unsupported language', str(context.exception))
    
    def test_validate_audio_file_not_found(self):
        """Test validation fails for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.analyzer._validate_audio_file('nonexistent_file.wav')
    
    def test_validate_audio_file_unsupported_format(self):
        """Test validation fails for unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_file = f.name
        
        try:
            with self.assertRaises(InvalidAudioFormatError):
                self.analyzer._validate_audio_file(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validate_audio_file_supported_format(self):
        """Test validation passes for supported format."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_file = f.name
            f.write(b'RIFF' + b'\x00' * 100)  # Minimal WAV header
        
        try:
            self.analyzer._validate_audio_file(temp_file)  # Should not raise
        finally:
            os.unlink(temp_file)
    
    @patch('main.AudioAnalyzer')
    def test_analyze_audio_file_with_progress_callback(self, mock_audio_analyzer):
        """Test analysis with progress callback."""
        progress_calls = []
        
        def progress_callback(current, total, message):
            progress_calls.append((current, total, message))
        
        # Mock audio analyzer
        mock_instance = Mock()
        mock_instance.analyze_audio_file.return_value = {
            'duration': 10.0,
            'sample_rate': 16000,
            'segments': [np.zeros(16000)]
        }
        mock_instance.load_audio.return_value = (np.zeros(160000), 16000)
        mock_audio_analyzer.return_value = mock_instance
        
        # Create analyzer with mocked components
        analyzer = KindergartenRecordingAnalyzer(language='en')
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_file = f.name
            f.write(b'RIFF' + b'\x00' * 1000)
        
        try:
            # This would fail without mocking all dependencies,
            # but tests progress callback is called
            try:
                analyzer.analyze_audio_file(
                    temp_file,
                    progress_callback=progress_callback
                )
            except Exception:
                pass  # Expected to fail, we're testing callback
            
            # Verify progress callback was called
            self.assertGreater(len(progress_calls), 0)
        finally:
            os.unlink(temp_file)
    
    def test_supported_languages_constant(self):
        """Test supported languages constant is defined."""
        self.assertIn('en', SUPPORTED_LANGUAGES)
        self.assertIn('he', SUPPORTED_LANGUAGES)
    
    def test_supported_formats_constant(self):
        """Test supported formats constant is defined."""
        self.assertIn('.wav', SUPPORTED_FORMATS)
        self.assertIn('.mp3', SUPPORTED_FORMATS)
        self.assertIn('.m4a', SUPPORTED_FORMATS)


class TestAudioAnalysisErrors(unittest.TestCase):
    """Test custom exception classes."""
    
    def test_audio_analysis_error(self):
        """Test AudioAnalysisError can be raised."""
        with self.assertRaises(AudioAnalysisError):
            raise AudioAnalysisError("Test error")
    
    def test_invalid_audio_format_error(self):
        """Test InvalidAudioFormatError inherits from AudioAnalysisError."""
        with self.assertRaises(AudioAnalysisError):
            raise InvalidAudioFormatError("Invalid format")
    
    def test_audio_file_too_long_error(self):
        """Test AudioFileTooLongError inherits from AudioAnalysisError."""
        with self.assertRaises(AudioAnalysisError):
            raise AudioFileTooLongError("File too long")


if __name__ == '__main__':
    unittest.main()
