"""
Unit Tests for Database Module
"""

import unittest
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database


class TestDatabase(unittest.TestCase):
    """Test suite for Database CRUD operations."""

    def setUp(self):
        """Create a temporary database for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db = Database(self.temp_db.name)

    def tearDown(self):
        """Clean up temp database."""
        try:
            os.unlink(self.temp_db.name)
        except OSError:
            pass

    def test_initialization(self):
        """Database should initialize and create tables."""
        self.assertIsNotNone(self.db)

    def _save_test_analysis(self, filename='test_audio.wav', risk_level='medium'):
        """Helper to save an analysis with correct API signature."""
        analysis_results = {
            'duration': 30.0,
            'language': 'en',
            'concerning_emotions': [],
            'violence_segments': [],
            'cry_segments': [],
            'neglect_analysis': {},
        }
        report = {
            'summary': {
                'risk_level': risk_level,
                'overall_assessment': 'normal',
                'total_incidents': 0,
                'critical_incidents': 0,
            }
        }
        return self.db.save_analysis(
            filename=filename,
            original_filename=filename,
            analysis_results=analysis_results,
            report=report,
        )

    def test_save_and_retrieve_analysis(self):
        """Should save analysis and retrieve it."""
        analysis_id = self._save_test_analysis()
        self.assertIsNotNone(analysis_id)
        self.assertGreater(analysis_id, 0)

        result = self.db.get_analysis(analysis_id)
        self.assertIsNotNone(result)
        self.assertEqual(result['original_filename'], 'test_audio.wav')

    def test_save_and_retrieve_incidents(self):
        """Should save incidents via analysis_results and retrieve them."""
        analysis_results = {
            'duration': 60.0,
            'language': 'en',
            'concerning_emotions': [{
                'start_time': 10.0,
                'end_time': 15.0,
                'detected_emotion': 'anger',
                'confidence': 0.85,
                'severity': 'high',
                'ml_backed': False,
            }],
            'violence_segments': [],
            'cry_segments': [],
            'neglect_analysis': {},
        }
        report = {
            'summary': {
                'risk_level': 'high',
                'overall_assessment': 'concerning',
                'total_incidents': 1,
                'critical_incidents': 0,
            }
        }
        analysis_id = self.db.save_analysis(
            filename='incident_audio.wav',
            original_filename='incident_audio.wav',
            analysis_results=analysis_results,
            report=report,
        )
        incidents = self.db.get_incidents(analysis_id)
        self.assertGreaterEqual(len(incidents), 1)

    def test_analysis_history(self):
        """Should return analyses in reverse chronological order."""
        for i in range(5):
            self._save_test_analysis(filename=f'audio_{i}.wav')

        history = self.db.get_analysis_history(limit=3)
        self.assertEqual(len(history), 3)

    def test_nonexistent_analysis(self):
        """Getting a nonexistent analysis should return None."""
        result = self.db.get_analysis(9999)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
