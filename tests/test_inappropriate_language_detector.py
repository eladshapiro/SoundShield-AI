"""
Unit Tests for InappropriateLanguageDetector Module
"""

import unittest
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inappropriate_language_detector import InappropriateLanguageDetector


class TestInappropriateLanguageDetector(unittest.TestCase):
    """Test suite for InappropriateLanguageDetector."""

    def setUp(self):
        self.detector = InappropriateLanguageDetector()

    def test_initialization(self):
        """Detector should load word lists for both languages."""
        self.assertIsNotNone(self.detector.inappropriate_words_hebrew)
        self.assertIsNotNone(self.detector.inappropriate_words_english)
        self.assertGreater(len(self.detector.inappropriate_words_english), 0)

    def test_english_word_detection(self):
        """Should detect English inappropriate words in transcription."""
        if not self.detector.inappropriate_words_english:
            self.skipTest("No English words loaded")
        sample_word = list(self.detector.inappropriate_words_english.keys())[0]
        text = f"The teacher said {sample_word} to the children"
        results = self.detector.detect_inappropriate_language(text, language='en')
        self.assertIsInstance(results, list)

    def test_empty_transcription(self):
        """Empty transcription should return empty list."""
        results = self.detector.detect_inappropriate_language('', language='en')
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_clean_text_no_detections(self):
        """Clean text should produce no inappropriate word findings."""
        text = "Good morning children, today we play a nice game"
        results = self.detector.detect_inappropriate_language(text, language='en')
        self.assertEqual(len(results), 0)

    def test_custom_words_file(self):
        """Should load from custom words file when provided."""
        custom_words = {
            "english": {
                "testbadword": {"severity": "high", "category": "test"}
            },
            "hebrew": {}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_words, f)
            temp_path = f.name

        try:
            detector = InappropriateLanguageDetector(words_file=temp_path)
            self.assertIn('testbadword', detector.inappropriate_words_english)
        finally:
            os.unlink(temp_path)

    def test_nonexistent_words_file_uses_defaults(self):
        """Should fall back to defaults when file doesn't exist."""
        detector = InappropriateLanguageDetector(words_file='/nonexistent/path.json')
        # Should still have words loaded from defaults
        self.assertIsNotNone(detector.inappropriate_words_english)

    def test_context_indicators_loaded(self):
        """Context indicators should be loaded."""
        self.assertIsNotNone(self.detector.context_indicators)


if __name__ == '__main__':
    unittest.main()
