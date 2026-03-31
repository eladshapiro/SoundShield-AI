"""Tests for the digest generator."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digest import DigestGenerator


class TestDigestGenerator(unittest.TestCase):

    def test_daily_digest_no_db(self):
        gen = DigestGenerator(db=None)
        result = gen.generate_daily_digest()
        self.assertEqual(result['period'], 'daily')
        self.assertIn('error', result)

    def test_weekly_digest_no_db(self):
        gen = DigestGenerator(db=None)
        result = gen.generate_weekly_digest()
        self.assertEqual(result['period'], 'weekly')

    def test_format_as_text(self):
        gen = DigestGenerator()
        digest = {
            'period': 'daily',
            'generated_at': '2026-03-31',
            'window_hours': 24,
            'total_analyses': 5,
            'total_incidents': 3,
            'total_critical': 1,
            'avg_audio_duration': 45.2,
            'risk_distribution': {'low': 3, 'high': 2},
            'top_incident_types': {'violence': 2, 'cry': 1}
        }
        text = gen.format_as_text(digest)
        self.assertIn('Total Analyses: 5', text)
        self.assertIn('violence: 2', text)

    def test_format_empty_digest(self):
        gen = DigestGenerator()
        digest = {'period': 'daily', 'generated_at': '', 'window_hours': 24,
                  'total_analyses': 0, 'total_incidents': 0, 'total_critical': 0,
                  'avg_audio_duration': 0, 'risk_distribution': {}, 'top_incident_types': {}}
        text = gen.format_as_text(digest)
        self.assertIn('Total Analyses: 0', text)


if __name__ == '__main__':
    unittest.main()
