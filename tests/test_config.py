"""
Unit Tests for Configuration Module
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SoundShieldConfig, AudioAnalyzerConfig, CryDetectorConfig,
    ViolenceDetectorConfig, EmotionDetectorConfig, NeglectDetectorConfig,
    WebAppConfig, DatabaseConfig, PipelineConfig, config
)


class TestConfig(unittest.TestCase):
    """Test suite for centralized configuration."""

    def test_global_config_singleton(self):
        """Global config should be a SoundShieldConfig instance."""
        self.assertIsInstance(config, SoundShieldConfig)

    def test_version_set(self):
        """Config should have a version string."""
        self.assertIsNotNone(config.version)
        self.assertTrue(config.version.count('.') >= 1)

    def test_audio_defaults(self):
        """Audio config should have sensible defaults."""
        self.assertEqual(config.audio.sample_rate, 22050)
        self.assertEqual(config.audio.calibration_seconds, 30.0)
        self.assertEqual(config.audio.n_mfcc, 13)

    def test_cry_defaults(self):
        """Cry config should have expected defaults."""
        self.assertEqual(config.cry.frequency_range, (200, 800))
        self.assertEqual(config.cry.energy_threshold, 0.08)
        self.assertEqual(config.cry.response_window, 10.0)

    def test_violence_defaults(self):
        """Violence config should have expected defaults."""
        self.assertEqual(config.violence.min_energy_gate, 0.05)
        self.assertEqual(config.violence.shouting_energy, 0.25)

    def test_emotion_defaults(self):
        """Emotion config should have expected defaults."""
        self.assertEqual(config.emotion.confidence_threshold, 0.6)
        self.assertEqual(config.emotion.anger_energy, 0.15)

    def test_neglect_defaults(self):
        """Neglect config should have expected defaults."""
        self.assertEqual(config.neglect.min_cry_duration, 5.0)
        self.assertEqual(config.neglect.response_window, 15.0)

    def test_web_defaults(self):
        """Web config should have expected defaults."""
        self.assertEqual(config.web.port, 5000)
        self.assertIn('wav', config.web.allowed_extensions)

    def test_pipeline_defaults(self):
        """Pipeline config should have expected defaults."""
        self.assertIn('en', config.pipeline.supported_languages)
        self.assertIn('he', config.pipeline.supported_languages)
        self.assertEqual(config.pipeline.max_audio_length_seconds, 3600)

    def test_env_override(self):
        """_env_float should read from environment when present."""
        from config import _env_float
        original = os.environ.get('TEST_SOUNDSHIELD_THRESHOLD')
        try:
            os.environ['TEST_SOUNDSHIELD_THRESHOLD'] = '0.99'
            val = _env_float('TEST_SOUNDSHIELD_THRESHOLD', 0.08)
            self.assertAlmostEqual(val, 0.99, places=5)
        finally:
            if original is not None:
                os.environ['TEST_SOUNDSHIELD_THRESHOLD'] = original
            else:
                os.environ.pop('TEST_SOUNDSHIELD_THRESHOLD', None)

    def test_bool_env_parsing(self):
        """Boolean env vars should parse correctly."""
        original = os.environ.get('USE_ADVANCED_MODELS')
        try:
            os.environ['USE_ADVANCED_MODELS'] = 'false'
            from config import _env_bool
            self.assertFalse(_env_bool('USE_ADVANCED_MODELS', True))
            os.environ['USE_ADVANCED_MODELS'] = '1'
            self.assertTrue(_env_bool('USE_ADVANCED_MODELS', False))
        finally:
            if original is not None:
                os.environ['USE_ADVANCED_MODELS'] = original
            else:
                os.environ.pop('USE_ADVANCED_MODELS', None)

    def test_logging_config_defaults(self):
        """LoggingConfig should have expected defaults."""
        from config import LoggingConfig
        lc = LoggingConfig()
        self.assertEqual(lc.log_level, 'INFO')
        self.assertEqual(lc.log_format, 'json')
        self.assertTrue(lc.enable_correlation_id)

    def test_logging_config_in_main_config(self):
        """SoundShieldConfig should include logging_config."""
        self.assertTrue(hasattr(config, 'logging_config'))
        self.assertEqual(config.logging_config.log_format, 'json')

    def test_correlation_id_generation(self):
        """Test correlation ID is generated and retrievable."""
        from structured_logging import set_correlation_id, get_correlation_id
        cid = set_correlation_id('test-123')
        self.assertEqual(get_correlation_id(), 'test-123')

    def test_step_timer_context_manager(self):
        """Test StepTimer works as context manager."""
        from structured_logging import StepTimer
        import time
        with StepTimer('test_step', step_number=1) as timer:
            time.sleep(0.01)
        # No assertion needed -- just verify it doesn't throw


if __name__ == '__main__':
    unittest.main()
